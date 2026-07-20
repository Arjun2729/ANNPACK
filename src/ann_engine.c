#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ann_format.h"
#include "ann_io.h"

#define PROBE 8
#define HEADER_READ 256

typedef struct {
    ann_header_t header;
    float       *centroids;    // n_lists * dim
    uint64_t    *list_offsets; // n_lists
    uint64_t    *list_lengths; // n_lists
    io_reader_t *reader;
} ann_index_t;

static ann_index_t *G_IDX = NULL;

static inline float half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    uint32_t exp = (uint32_t)(h & 0x7C00u) >> 10;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; // zero
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            exp = (exp + (127 - 15)) << 23;
            f = sign | exp | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13); // inf or NaN
    } else {
        exp = (exp + (127 - 15)) << 23;
        f = sign | exp | (mant << 13);
    }

    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static void insert_topk(ann_result_t *res, int *count, int k, uint64_t id, float score) {
    if (*count < k || score > res[*count - 1].score) {
        int pos = (*count < k) ? *count : k - 1;
        while (pos > 0 && score > res[pos - 1].score) pos--;
        if (*count < k) (*count)++;
        for (int m = (*count) - 1; m > pos; m--) {
            res[m] = res[m - 1];
        }
        res[pos].id = id;
        res[pos].score = score;
    }
}

int ann_load_index(const char *url) {
    if (G_IDX) return 1;
    if (!url) return 0;

    io_reader_t *reader = io_reader_http(url);
    if (!reader) return 0;

    uint8_t hdr_buf[HEADER_READ];
    memset(hdr_buf, 0, sizeof(hdr_buf));
    io_req_t req = {0};
    req.offset = 0;
    req.len = HEADER_READ;
    req.dst = hdr_buf;
    reader->read_batch(reader->ctx, &req, 1);

    if (req.result_len < sizeof(ann_header_t)) {
        reader->destroy(reader->ctx);
        return 0;
    }

    ann_header_t *h = (ann_header_t *)hdr_buf;
    if (h->magic != ANN_MAGIC ||
        h->version != ANN_VERSION ||
        h->endian != ANN_ENDIAN_LITTLE ||
        h->header_size != ANN_HEADER_SIZE) {
        reader->destroy(reader->ctx);
        return 0;
    }

    ann_index_t *idx = (ann_index_t *)calloc(1, sizeof(ann_index_t));
    if (!idx) {
        reader->destroy(reader->ctx);
        return 0;
    }

    idx->header = *h;
    idx->reader = reader;

    size_t cent_sz = (size_t)h->n_lists * h->dim * sizeof(float);
    size_t table_sz = (size_t)h->n_lists * sizeof(ann_list_meta_t);

    idx->centroids = (float *)malloc(cent_sz);
    ann_list_meta_t *table = (ann_list_meta_t *)malloc(table_sz);
    idx->list_offsets = (uint64_t *)malloc((size_t)h->n_lists * sizeof(uint64_t));
    idx->list_lengths = (uint64_t *)malloc((size_t)h->n_lists * sizeof(uint64_t));

    if (!idx->centroids || !table || !idx->list_offsets || !idx->list_lengths) {
        free(idx->centroids);
        free(table);
        free(idx->list_offsets);
        free(idx->list_lengths);
        free(idx);
        reader->destroy(reader->ctx);
        return 0;
    }

    io_req_t reqs[2];
    memset(reqs, 0, sizeof(reqs));
    reqs[0].offset = h->header_size;
    reqs[0].len = cent_sz;
    reqs[0].dst = idx->centroids;

    reqs[1].offset = h->offset_table_pos;
    reqs[1].len = table_sz;
    reqs[1].dst = table;

    reader->read_batch(reader->ctx, reqs, 2);

    for (uint32_t i = 0; i < h->n_lists; i++) {
        idx->list_offsets[i] = table[i].offset;
        idx->list_lengths[i] = table[i].length;
    }

    free(table);
    G_IDX = idx;
    return 1;
}

int ann_search(void *ctx_unused, const float *query, ann_result_t *out_results, int k) {
    (void)ctx_unused;
    if (!G_IDX || !query || !out_results || k <= 0) return 0;

    const uint32_t dim = G_IDX->header.dim;
    const uint32_t n_lists = G_IDX->header.n_lists;

    float best_scores[PROBE];
    int best_ids[PROBE];
    for (int i = 0; i < PROBE; i++) {
        best_scores[i] = -1e30f;
        best_ids[i] = -1;
    }

    // coarse search over centroids
    for (uint32_t c = 0; c < n_lists; c++) {
        float dot = 0.0f;
        float *cent = G_IDX->centroids + ((size_t)c * dim);
        for (uint32_t j = 0; j < dim; j++) {
            dot += query[j] * cent[j];
        }
        for (int p = 0; p < PROBE; p++) {
            if (dot > best_scores[p]) {
                for (int m = PROBE - 1; m > p; m--) {
                    best_scores[m] = best_scores[m - 1];
                    best_ids[m] = best_ids[m - 1];
                }
                best_scores[p] = dot;
                best_ids[p] = (int)c;
                break;
            }
        }
    }

    int top_count = 0;
    for (int i = 0; i < k; i++) {
        out_results[i].id = 0;
        out_results[i].score = -1e30f;
    }

    // fine search within top PROBE lists
    for (int idx = 0; idx < PROBE; idx++) {
        int list_id = best_ids[idx];
        if (list_id < 0) continue;

        uint64_t off = G_IDX->list_offsets[list_id];
        uint64_t len = G_IDX->list_lengths[list_id];
        if (len == 0) continue;

        uint8_t *buf = (uint8_t *)malloc(len);
        if (!buf) continue;

        io_req_t req = {0};
        req.offset = off;
        req.len = len;
        req.dst = buf;
        G_IDX->reader->read_batch(G_IDX->reader->ctx, &req, 1);

        if (req.result_len < 4) {
            free(buf);
            continue;
        }

        uint32_t count = *(uint32_t *)buf;
        uint64_t *ids = (uint64_t *)(buf + 4);
        uint16_t *vecs = (uint16_t *)(buf + 4 + (size_t)count * 8);
        size_t needed = 4 + (size_t)count * 8 + (size_t)count * dim * 2;
        if (count == 0 || needed > len) {
            free(buf);
            continue;
        }

        for (uint32_t i = 0; i < count; i++) {
            float dot = 0.0f;
            uint16_t *v = vecs + (size_t)i * dim;
            for (uint32_t j = 0; j < dim; j++) {
                dot += query[j] * half_to_float(v[j]);
            }
            insert_topk(out_results, &top_count, k, ids[i], dot);
        }

        free(buf);
    }

    return top_count;
}

int ann_result_size_bytes(void) { return (int)sizeof(ann_result_t); }
