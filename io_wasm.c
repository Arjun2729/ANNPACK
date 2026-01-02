#include <emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ann_io.h"

EM_ASYNC_JS(int, js_fetch_batch, (const char* url, int count, uint32_t* off_ptr, uint32_t* len_ptr, uint32_t* dst_ptrs), {
    try {
        const offsets = HEAPU32.subarray(off_ptr >> 2, (off_ptr >> 2) + count);
        const lengths = HEAPU32.subarray(len_ptr >> 2, (len_ptr >> 2) + count);
        const dests   = HEAPU32.subarray(dst_ptrs >> 2, (dst_ptrs >> 2) + count);
        const urlStr  = UTF8ToString(url);
        
        const promises = [];
        for (let i = 0; i < count; i++) {
            const start = offsets[i];
            const end = start + lengths[i] - 1;
            
            const p = fetch(urlStr, {
                headers: { 'Range': `bytes=${start}-${end}` }
            }).then(async (resp) => {
                if (resp.status !== 206) return 0;
                const buf = await resp.arrayBuffer();
                if (buf.byteLength !== lengths[i]) return 0;
                HEAPU8.set(new Uint8Array(buf), dests[i]);
                return 1;
            }).catch(e => 0);
            promises.push(p);
        }
        
        const results = await Promise.all(promises);
        for(let r of results) if(r === 0) return 0;
        return 1;
    } catch (e) {
        console.error(e);
        return 0;
    }
});

typedef struct { char *url; } wasm_ctx_t;

static void wasm_read_batch(void *c, io_req_t *reqs, int count) {
    wasm_ctx_t *ctx = (wasm_ctx_t*)c;
    uint32_t *offsets = malloc(count * 4);
    uint32_t *lens    = malloc(count * 4);
    uint32_t *dsts    = malloc(count * 4);
    
    for(int i=0; i<count; i++) {
        offsets[i] = (uint32_t)reqs[i].offset;
        lens[i]    = (uint32_t)reqs[i].len;
        dsts[i]    = (uint32_t)reqs[i].dst;
    }
    
    int ok = js_fetch_batch(ctx->url, count, offsets, lens, dsts);
    
    for(int i=0; i<count; i++) reqs[i].result_len = (ok == 1) ? reqs[i].len : -1;
    
    free(offsets); free(lens); free(dsts);
}

static void wasm_destroy(void *c) {
    wasm_ctx_t *ctx = (wasm_ctx_t*)c;
    free(ctx->url); free(ctx);
}

io_reader_t *io_reader_http(const char *url) {
    wasm_ctx_t *ctx = malloc(sizeof(*ctx));
    ctx->url = strdup(url);
    io_reader_t *r = malloc(sizeof(*r));
    r->ctx = ctx; r->read_batch = wasm_read_batch; r->destroy = wasm_destroy;
    return r;
}
