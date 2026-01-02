#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "ann_io.h"

typedef struct {
    int fd;
} file_ctx_t;

static void http_read_batch(void *vctx, io_req_t *reqs, int n_reqs) {
    file_ctx_t *ctx = (file_ctx_t *)vctx;
    if (!ctx || ctx->fd < 0 || !reqs || n_reqs <= 0) return;
    for (int i = 0; i < n_reqs; i++) {
        io_req_t *r = &reqs[i];
        r->result_len = 0;
        if (!r->dst || r->len == 0) continue;
        ssize_t got = pread(ctx->fd, r->dst, (size_t)r->len, (off_t)r->offset);
        if (got > 0) r->result_len = (uint64_t)got;
    }
}

static void http_destroy(void *vctx) {
    file_ctx_t *ctx = (file_ctx_t *)vctx;
    if (!ctx) return;
    if (ctx->fd >= 0) close(ctx->fd);
    free(ctx);
}

io_reader_t *io_reader_http(const char *url) {
    if (!url) return NULL;
    const char *path = url;
    if (strncmp(url, "file://", 7) == 0) path = url + 7;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    file_ctx_t *ctx = (file_ctx_t *)calloc(1, sizeof(file_ctx_t));
    io_reader_t *r = (io_reader_t *)calloc(1, sizeof(io_reader_t));
    if (!ctx || !r) {
        if (fd >= 0) close(fd);
        free(ctx);
        free(r);
        return NULL;
    }
    ctx->fd = fd;
    r->ctx = ctx;
    r->read_batch = http_read_batch;
    r->destroy = http_destroy;
    return r;
}
