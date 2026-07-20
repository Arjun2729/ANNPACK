#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ann_io.h"
#include <emscripten/fetch.h>

typedef struct {
    char *url;
} http_ctx_t;

static void http_read_batch(void *ctx, io_req_t *reqs, int n_reqs) {
    http_ctx_t *c = (http_ctx_t *)ctx;
    for (int i = 0; i < n_reqs; i++) {
        uint64_t start = reqs[i].offset;
        uint64_t end = reqs[i].offset + reqs[i].len - 1;
        char range_header[64];
        snprintf(range_header, sizeof(range_header), "bytes=%llu-%llu", (unsigned long long)start, (unsigned long long)end);

        emscripten_fetch_attr_t attr;
        emscripten_fetch_attr_init(&attr);
        strcpy(attr.requestMethod, "GET");
        attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY | EMSCRIPTEN_FETCH_SYNCHRONOUS;
        const char *headers[] = {
            "Range",
            range_header,
            NULL
        };
        attr.requestHeaders = headers;

        emscripten_fetch_t *fetch = emscripten_fetch(&attr, c->url);
        if (fetch && (fetch->status == 200 || fetch->status == 206)) {
            size_t to_copy = fetch->numBytes < reqs[i].len ? fetch->numBytes : reqs[i].len;
            memcpy(reqs[i].dst, fetch->data, to_copy);
            reqs[i].result_len = fetch->numBytes;
        } else {
            fprintf(stderr, "fetch failed for range %s\n", range_header);
            reqs[i].result_len = 0;
        }

        if (fetch) {
            emscripten_fetch_close(fetch);
        }
    }
}

static void http_destroy(void *ctx) {
    http_ctx_t *c = (http_ctx_t *)ctx;
    if (c) {
        free(c->url);
        free(c);
    }
}

io_reader_t *io_reader_http(const char *url) {
    if (!url) return NULL;
    http_ctx_t *ctx = (http_ctx_t *)malloc(sizeof(http_ctx_t));
    if (!ctx) return NULL;
    ctx->url = (char *)malloc(strlen(url) + 1);
    if (!ctx->url) {
        free(ctx);
        return NULL;
    }
    strcpy(ctx->url, url);

    io_reader_t *reader = (io_reader_t *)malloc(sizeof(io_reader_t));
    if (!reader) {
        free(ctx->url);
        free(ctx);
        return NULL;
    }

    reader->ctx = ctx;
    reader->read_batch = http_read_batch;
    reader->destroy = http_destroy;
    return reader;
}
