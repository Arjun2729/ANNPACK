#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t offset;
    size_t len;
    void *dst;
    int64_t result_len;
} io_req_t;

typedef struct io_reader {
    void *ctx;
    void (*read_batch)(void *ctx, io_req_t *reqs, int count);
    void (*destroy)(void *ctx);
} io_reader_t;

void ann_bench_run(const char *url);
io_reader_t *io_reader_http(const char *url);
