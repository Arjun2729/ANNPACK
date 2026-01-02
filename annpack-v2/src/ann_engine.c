#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif
#include "ann_format.h"
#include "ann_io.h"
#include "ann_api.h"

#ifndef ANN_DEBUG
#define ANN_DEBUG 0
#endif

#define DEFAULT_PROBE 8
#define MAX_PROBE 4096
#define MIN_DIM 1
#define MAX_DIM 4096
#define MAX_LISTS 1000000

static char g_last_error[256] = {0};
static int g_probe = DEFAULT_PROBE;
static unsigned long long g_last_scan_count = 0;
static void set_error(const char *msg) {
    strncpy(g_last_error, msg ? msg : "", sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

EMSCRIPTEN_KEEPALIVE
const char *ann_last_error(void) { return g_last_error; }
EMSCRIPTEN_KEEPALIVE
unsigned long long ann_last_scan_count(void) { return g_last_scan_count; }

typedef struct {
    ann_header_t header;
    float *centroids;
    uint64_t *list_offsets;
    uint64_t *list_lengths;
    io_reader_t *reader;
} ann_index_t;

EMSCRIPTEN_KEEPALIVE
int ann_get_n_lists(void *ctx) {
    if (!ctx) return 0;
    ann_index_t *idx = (ann_index_t *)ctx;
    return (int)idx->header.n_lists;
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void minheap_sift_down(float *scores, uint64_t *ids, int heap_size, int idx) {
    while (1) {
        int left = 2 * idx + 1;
        int right = left + 1;
        int smallest = idx;
        if (left < heap_size && scores[left] < scores[smallest]) smallest = left;
        if (right < heap_size && scores[right] < scores[smallest]) smallest = right;
        if (smallest == idx) break;
        float ts = scores[idx]; scores[idx] = scores[smallest]; scores[smallest] = ts;
        uint64_t tid = ids[idx]; ids[idx] = ids[smallest]; ids[smallest] = tid;
        idx = smallest;
    }
}

static void minheap_sift_up(float *scores, uint64_t *ids, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (scores[parent] <= scores[idx]) break;
        float ts = scores[idx]; scores[idx] = scores[parent]; scores[parent] = ts;
        uint64_t tid = ids[idx]; ids[idx] = ids[parent]; ids[parent] = tid;
        idx = parent;
    }
}

static int validate_header(const ann_header_t *h) {
    if (!h) return 0;
    if (h->magic != ANN_MAGIC || h->version != ANN_VERSION || h->endian != ANN_ENDIAN_LITTLE || h->header_size != ANN_HEADER_SIZE) {
        set_error("Bad header magic/version/endian/size");
        return 0;
    }
    if (h->dim < MIN_DIM || h->dim > MAX_DIM) { set_error("Bad dim"); return 0; }
    if (h->metric != ANN_METRIC_DOT) { set_error("Unsupported metric"); return 0; }
    if (h->n_lists == 0 || h->n_lists > MAX_LISTS) { set_error("Bad n_lists"); return 0; }
    if (h->offset_table_pos == 0) { set_error("Bad offset table pos"); return 0; }
    return 1;
}

EMSCRIPTEN_KEEPALIVE int ann_result_size_bytes(void) { return (int)sizeof(ann_result_t); }
EMSCRIPTEN_KEEPALIVE void ann_set_probe(int probe) { if (probe < 1) probe = 1; if (probe > MAX_PROBE) probe = MAX_PROBE; g_probe = probe; }
static int g_max_k = ANN_MAX_K;
static int g_max_scan = 0;
EMSCRIPTEN_KEEPALIVE void ann_set_max_k(int cap) { if (cap < 1) cap = 1; if (cap > ANN_MAX_K) cap = ANN_MAX_K; g_max_k = cap; }
EMSCRIPTEN_KEEPALIVE void ann_set_max_scan(int cap) { if (cap < 0) cap = 0; g_max_scan = cap; }

EMSCRIPTEN_KEEPALIVE
void *ann_load_index(const char *url) {
    set_error(NULL);
    io_reader_t *r = io_reader_http(url);
    if (!r) { set_error("io_reader_http failed"); return NULL; }

    uint8_t head_buf[256];
    io_req_t req = { .offset = 0, .len = sizeof(head_buf), .dst = head_buf, .result_len = 0 };
    r->read_batch(r->ctx, &req, 1);
#if ANN_DEBUG
    printf("[C] header fetch: requested=%zu got=%llu\n", sizeof(head_buf), (unsigned long long)req.result_len);
    printf("[C] first8 = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           head_buf[0], head_buf[1], head_buf[2], head_buf[3],
           head_buf[4], head_buf[5], head_buf[6], head_buf[7]);
#endif
    if (req.result_len < sizeof(ann_header_t)) { set_error("Header short read"); r->destroy(r->ctx); free(r); return NULL; }

    const ann_header_t *h = (const ann_header_t *)head_buf;
    if (!validate_header(h)) { r->destroy(r->ctx); free(r); return NULL; }

    ann_index_t *idx = (ann_index_t *)calloc(1, sizeof(ann_index_t));
    if (!idx) { set_error("alloc idx failed"); r->destroy(r->ctx); free(r); return NULL; }
    idx->reader = r;
    memcpy(&idx->header, h, sizeof(ann_header_t));

    size_t cent_sz = (size_t)h->n_lists * h->dim * sizeof(float);
    size_t table_sz = (size_t)h->n_lists * sizeof(ann_list_meta_t);
    idx->centroids = (float *)malloc(cent_sz);
    ann_list_meta_t *table = (ann_list_meta_t *)malloc(table_sz);
    if (!idx->centroids || !table) {
        set_error("alloc metadata failed");
        free(idx->centroids); free(table); r->destroy(r->ctx); free(r); free(idx);
        return NULL;
    }

    io_req_t reqs[2] = {
        { .offset = h->header_size, .len = cent_sz, .dst = idx->centroids, .result_len = 0 },
        { .offset = h->offset_table_pos, .len = table_sz, .dst = table, .result_len = 0 }
    };
    r->read_batch(r->ctx, reqs, 2);
    if (reqs[0].result_len != (uint64_t)cent_sz || reqs[1].result_len != (uint64_t)table_sz) {
        set_error("Metadata read failed");
        free(idx->centroids); free(table); r->destroy(r->ctx); free(r); free(idx);
        return NULL;
    }

    idx->list_offsets = (uint64_t *)malloc(h->n_lists * sizeof(uint64_t));
    idx->list_lengths = (uint64_t *)malloc(h->n_lists * sizeof(uint64_t));
    if (!idx->list_offsets || !idx->list_lengths) {
        set_error("alloc offsets failed");
        free(idx->centroids); free(table); free(idx->list_offsets); free(idx->list_lengths); r->destroy(r->ctx); free(r); free(idx);
        return NULL;
    }

    for (uint32_t i = 0; i < h->n_lists; i++) {
        idx->list_offsets[i] = table[i].offset;
        idx->list_lengths[i] = table[i].length;
    }
    free(table);
    return idx;
}

EMSCRIPTEN_KEEPALIVE
void ann_free_index(void *ctx) {
    if (!ctx) return;
    ann_index_t *idx = (ann_index_t *)ctx;
    if (idx->reader && idx->reader->destroy) idx->reader->destroy(idx->reader->ctx);
    free(idx->reader);
    free(idx->centroids);
    free(idx->list_offsets);
    free(idx->list_lengths);
    free(idx);
}

EMSCRIPTEN_KEEPALIVE
int ann_search(void *ctx, const float *query, ann_result_t *out_results, int k) {
    set_error(NULL);
    g_last_scan_count = 0;
    if (!ctx || !query || !out_results || k <= 0) { set_error("bad args"); return 0; }
    ann_index_t *idx = (ann_index_t *)ctx;
    const uint32_t dim = idx->header.dim;
    int K = (k > 0) ? ((k > g_max_k) ? g_max_k : k) : 1;
    if (K < 1) K = 1;

    // Coarse search
    int probe = (g_probe > 0) ? g_probe : DEFAULT_PROBE;
    if (probe > (int)idx->header.n_lists) probe = (int)idx->header.n_lists;
    if (probe > MAX_PROBE) probe = MAX_PROBE;
    float *best_scores = (float *)malloc(sizeof(float) * probe);
    int *best_ids = (int *)malloc(sizeof(int) * probe);
    if (!best_scores || !best_ids) { free(best_scores); free(best_ids); set_error("alloc probe failed"); return 0; }
    for (int i = 0; i < probe; i++) { best_scores[i] = -1e9f; best_ids[i] = -1; }
    for (uint32_t c = 0; c < idx->header.n_lists; c++) {
        const float *cent = idx->centroids + c * dim;
        float dot = 0.0f;
        for (uint32_t j = 0; j < dim; j++) dot += query[j] * cent[j];
        if (dot > best_scores[probe - 1]) {
            int pos = probe - 1;
            while (pos > 0 && dot > best_scores[pos - 1]) pos--;
            for (int m = probe - 1; m > pos; m--) { best_scores[m] = best_scores[m - 1]; best_ids[m] = best_ids[m - 1]; }
            best_scores[pos] = dot; best_ids[pos] = (int)c;
        }
    }

    float *heap_scores = (float *)malloc(sizeof(float) * K);
    uint64_t *heap_ids = (uint64_t *)malloc(sizeof(uint64_t) * K);
    if (!heap_scores || !heap_ids) { free(heap_scores); free(heap_ids); set_error("alloc heap failed"); free(best_scores); free(best_ids); return 0; }
    int heap_size = 0;

    // Simple 2-slot LRU cache scoped per search
    typedef struct { int list_id; uint8_t *data; uint64_t len; uint64_t off; uint64_t last_use; } cache_entry_t;
    cache_entry_t cache[2] = { { -1, NULL, 0, 0, 0 }, { -1, NULL, 0, 0, 0 } };
    uint64_t use_tick = 1;

    io_req_t *reqs = (io_req_t *)calloc(probe, sizeof(io_req_t));
    uint8_t **buffers = (uint8_t **)calloc(probe, sizeof(uint8_t *));
    uint64_t *lens = (uint64_t *)calloc(probe, sizeof(uint64_t));
    int req_count = 0;
    int *req_map = (int *)calloc(probe, sizeof(int)); // map probe idx -> req idx or -1
    for (int i = 0; i < probe; i++) req_map[i] = -1;

    for (int bi = 0; bi < probe; bi++) {
        int list_id = best_ids[bi];
        if (list_id < 0) continue;
        uint64_t off = idx->list_offsets[list_id];
        uint64_t len = idx->list_lengths[list_id];
        if (len < 4 || len > (1ULL << 32)) continue;

        // cache lookup
        int hit = -1;
        for (int c = 0; c < 2; c++) {
            if (cache[c].list_id == list_id && cache[c].len == len && cache[c].off == off && cache[c].data) {
                hit = c;
                cache[c].last_use = use_tick++;
                break;
            }
        }
        if (hit >= 0) {
            buffers[bi] = cache[hit].data;
            lens[bi] = cache[hit].len;
            continue;
        }

        uint8_t *buf = (uint8_t *)malloc(len);
        if (!buf) continue;
        buffers[bi] = buf;
        lens[bi] = len;
        reqs[req_count].offset = off;
        reqs[req_count].len = len;
        reqs[req_count].dst = buf;
        reqs[req_count].result_len = 0;
        req_map[bi] = req_count;
        req_count++;
    }

    if (req_count > 0) {
        idx->reader->read_batch(idx->reader->ctx, reqs, req_count);
    }

    for (int bi = 0; bi < probe; bi++) {
        uint8_t *buf = buffers[bi];
        if (!buf) continue;
        int ridx = req_map[bi];
        if (ridx >= 0) {
            if (reqs[ridx].result_len < 4) { free(buf); continue; }
            lens[bi] = reqs[ridx].result_len;
        }
        uint64_t len = lens[bi];

        uint32_t count = *(uint32_t *)buf;
        size_t needed = 4ull + (size_t)count * 8ull + (size_t)count * dim * 2ull;
        if (needed > len || count == 0) { if (ridx >= 0) free(buf); continue; }

        uint64_t *ids = (uint64_t *)(buf + 4);
        uint16_t *vecs = (uint16_t *)(buf + 4 + (size_t)count * 8);

        uint32_t scan_limit = (g_max_scan > 0 && (uint32_t)g_max_scan < count) ? (uint32_t)g_max_scan : count;
        g_last_scan_count += (unsigned long long)scan_limit;
        for (uint32_t i = 0; i < scan_limit; i++) {
            const uint16_t *v = vecs + i * dim;
            float dot = 0.0f;
            for (uint32_t j = 0; j < dim; j++) dot += query[j] * half_to_float(v[j]);

            if (heap_size < K) {
                heap_scores[heap_size] = dot;
                heap_ids[heap_size] = ids[i];
                minheap_sift_up(heap_scores, heap_ids, heap_size);
                heap_size++;
            } else if (dot > heap_scores[0]) {
                heap_scores[0] = dot;
                heap_ids[0] = ids[i];
                minheap_sift_down(heap_scores, heap_ids, heap_size, 0);
            }
        }

        if (ridx >= 0) {
            int victim = (cache[0].list_id < 0) ? 0 : (cache[1].list_id < 0 ? 1 : (cache[0].last_use <= cache[1].last_use ? 0 : 1));
            if (cache[victim].data) free(cache[victim].data);
            cache[victim].data = buf;
            cache[victim].len = len;
            cache[victim].off = reqs[ridx].offset;
            cache[victim].list_id = best_ids[bi];
            cache[victim].last_use = use_tick++;
        }
    }

    free(buffers);
    free(reqs);
    free(lens);
    free(req_map);
    for (int c = 0; c < 2; c++) {
        if (cache[c].data) free(cache[c].data);
    }
    free(best_scores);
    free(best_ids);

    int ret_count = heap_size;
    for (int i = heap_size - 1; i >= 0; i--) {
        out_results[i].id = heap_ids[0];
        out_results[i].score = heap_scores[0];
        heap_scores[0] = heap_scores[heap_size - 1];
        heap_ids[0] = heap_ids[heap_size - 1];
        heap_size--;
        if (heap_size > 0) minheap_sift_down(heap_scores, heap_ids, heap_size, 0);
    }

    free(heap_scores);
    free(heap_ids);
    return ret_count;
}
