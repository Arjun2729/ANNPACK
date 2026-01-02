#ifndef ANN_IO_H
#define ANN_IO_H

#include <stdint.h>

typedef struct io_req {
    uint64_t offset;
    uint64_t len;
    void *dst;
    uint64_t result_len;
} io_req_t;

typedef struct io_reader {
    void *ctx;
    void (*read_batch)(void *ctx, io_req_t *reqs, int n_reqs);
    void (*destroy)(void *ctx);
} io_reader_t;

io_reader_t *io_reader_http(const char *url);

#endif
