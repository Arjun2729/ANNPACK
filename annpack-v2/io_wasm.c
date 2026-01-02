#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten.h>
#include "include/ann_io.h"

/*
  We implement io_reader_http(url) + read_batch().

  Core idea:
  - JS fetch is async -> wrap in Asyncify.handleAsync so C can "block" until done.
  - Always return actual bytes read into req->result_len.
*/

EM_JS(void, js_init_range_fetch, (), {
  if (Module.__annpack_fetch_ready) return;
  Module.__annpack_fetch_ready = true;

  Module.__annpack_fetch_range = async function(url, start, end, len, dstPtr) {
    const headers = { "Range": `bytes=${start}-${end}` };
    const resp = await fetch(url, { headers });
    if (!resp.ok && resp.status !== 206) {
      // allow 200 only if the server ignores Range (not desired here)
      return 0;
    }
    const buf = new Uint8Array(await resp.arrayBuffer());
    const n = Math.min(buf.length, len >>> 0);
    if (n === 0) return 0;
    if (buf.length > n && resp.status === 200) {
      console.warn("[annpack] Server ignored Range; truncating to requested length.");
    }
    Module.HEAPU8.set(buf.subarray(0, n), dstPtr);
    return n;
  };
});

// Note: function name is prefixed with '_' to match ASYNCIFY_IMPORTS and generated import names.
EM_JS(int, _js_fetch_range_blocking, (const char* url_c, uint32_t offset, uint32_t len, uint8_t* dst), {
  js_init_range_fetch();
  const url = UTF8ToString(url_c);
  return Asyncify.handleAsync(async () => {
    const start = offset >>> 0;
    const end = start + (len ? (len - 1) : 0);
    const n = await Module.__annpack_fetch_range(url, start, end, len, dst);
    return n|0;
  });
});

typedef struct {
  char* url;
} http_ctx_t;

static void http_read_batch(void* vctx, io_req_t* reqs, int n) {
  http_ctx_t* ctx = (http_ctx_t*)vctx;
  if (!ctx || !ctx->url || !reqs || n <= 0) return;
  for (int i = 0; i < n; i++) {
    io_req_t* r = &reqs[i];
    int got = _js_fetch_range_blocking(ctx->url, (uint32_t)r->offset, (uint32_t)r->len, (uint8_t*)r->dst);
    r->result_len = (got < 0) ? 0 : (uint64_t)got;
  }
}

static void http_destroy(void* vctx) {
  http_ctx_t* ctx = (http_ctx_t*)vctx;
  if (ctx) {
    free(ctx->url);
    free(ctx);
  }
}

io_reader_t* io_reader_http(const char* url) {
  if (!url) return NULL;
  http_ctx_t* ctx = (http_ctx_t*)calloc(1, sizeof(http_ctx_t));
  ctx->url = strdup(url);

  io_reader_t* r = (io_reader_t*)calloc(1, sizeof(io_reader_t));
  r->ctx = ctx;
  r->read_batch = http_read_batch;
  r->destroy = http_destroy;
  // Self-check: ensure fetch path works
  uint8_t tmp[16];
  int got = _js_fetch_range_blocking(ctx->url, 0, sizeof(tmp), tmp);
  if (got != (int)sizeof(tmp)) {
    r->destroy(r->ctx);
    free(r);
    return NULL;
  }
  return r;
}
