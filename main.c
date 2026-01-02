#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten.h>
#include "ann_io.h"

#define HEADER_SIZE 256

typedef struct {
    uint32_t dim;
    uint32_t n_lists;
    uint32_t n_vectors;
    uint64_t offset_table_pos;
    float *centroids;
    uint64_t *list_offsets;
    uint64_t *list_lengths;
    io_reader_t *reader;
} ann_index_t;

static ann_index_t *G_INDEX = NULL;

typedef struct __attribute__((packed)) {
    uint64_t id;    // 8 bytes
    float    score; // 4 bytes
} ann_result_t;

_Static_assert(sizeof(ann_result_t) == 12, "ann_result_t must remain packed to 12 bytes");

EMSCRIPTEN_KEEPALIVE
int ann_result_size_bytes(void) {
    return (int)sizeof(ann_result_t);
}

// Helpers
static inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 0x0001;
    uint32_t e = (h >> 10) & 0x001f;
    uint32_t m = h & 0x03ff;
    if (e == 0) return (m == 0) ? (s ? -0.0f : 0.0f) : (s ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * ((float)m / 1024.0f);
    if (e == 31) return (m == 0) ? (s ? -INFINITY : INFINITY) : NAN;
    return (s ? -1.0f : 1.0f) * powf(2.0f, (float)e - 15.0f) * (1.0f + ((float)m / 1024.0f));
}

static float calc_dot(const float *a, const float *b, int dim) {
    float d = 0.0f;
    for(int i=0; i<dim; i++) d += a[i] * b[i];
    return d;
}

EMSCRIPTEN_KEEPALIVE
int ann_load_index(const char *url) {
    if (G_INDEX) return 1;
    printf("[C] Loading Index from %s...\n", url);

    io_reader_t *reader = io_reader_http(url);
    if (!reader) return 0;

    // 1. Fetch Header
    uint8_t head_buf[HEADER_SIZE];
    io_req_t req = { .offset = 0, .len = HEADER_SIZE, .dst = head_buf };
    reader->read_batch(reader->ctx, &req, 1);

    if (req.result_len != HEADER_SIZE) {
        printf("[Error] Failed to fetch header\n");
        return 0;
    }

    // 2. Validate Magic
    uint32_t magic = *(uint32_t*)head_buf;
    if (magic != 0x504E4E41) {
        printf("[Error] Invalid Magic: %08X\n", magic);
        return 0;
    }

    ann_index_t *idx = calloc(1, sizeof(ann_index_t));
    idx->reader = reader;

    // 3. Parse values from header (trust the file)
    // Layout: Magic(8) | Ver(4) | Endian(4) | HSize(4) | Dim(4) | Metric(4) | Lists(4) | Vecs(4)
    // Offset Table Pointer is at 36.
    idx->dim = *(uint32_t*)(head_buf + 20);
    idx->n_lists = *(uint32_t*)(head_buf + 28);
    idx->n_vectors = *(uint32_t*)(head_buf + 32);
    idx->offset_table_pos = *(uint64_t*)(head_buf + 36);

    printf("[C] Config: Dim=%d, Lists=%d, Vecs=%d\n", idx->dim, idx->n_lists, idx->n_vectors);
    printf("[C] Offsets Table @ %llu\n", idx->offset_table_pos);

    // Safety Checks
    if (idx->dim > 4096 || idx->n_lists > 1000000 || idx->offset_table_pos == 0) {
        printf("[Error] Header values out of bounds. Corrupt file?\n");
        free(idx);
        return 0;
    }

    // 4. Fetch Metadata
    size_t centroids_size = idx->n_lists * idx->dim * sizeof(float);
    size_t table_size = idx->n_lists * 16;

    idx->centroids = malloc(centroids_size);
    uint8_t *table_buf = malloc(table_size);

    // Calculate start of centroids (Header Size is at offset 16)
    uint32_t header_size = *(uint32_t*)(head_buf + 16);

    io_req_t setup_reqs[2] = {
        { .offset = header_size, .len = centroids_size, .dst = idx->centroids },
        { .offset = idx->offset_table_pos, .len = table_size, .dst = table_buf }
    };

    double t0 = emscripten_get_now();
    reader->read_batch(reader->ctx, setup_reqs, 2);
    double t1 = emscripten_get_now();
    printf("[C] Metadata Loaded in %.2f ms\n", t1 - t0);

    // 5. Parse Table
    idx->list_offsets = malloc(idx->n_lists * 8);
    idx->list_lengths = malloc(idx->n_lists * 8);
    uint64_t *raw_table = (uint64_t*)table_buf;

    for(uint32_t i=0; i<idx->n_lists; i++) {
        idx->list_offsets[i] = raw_table[2*i];
        idx->list_lengths[i] = raw_table[2*i+1];
    }
    free(table_buf);

    G_INDEX = idx;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
int ann_search(const float *query_vec, uint8_t *out_bytes, int max_k) {
    if (!G_INDEX || !query_vec || !out_bytes || max_k <= 0) return 0;
    ann_index_t *idx = G_INDEX;

    printf("[C] [ann_search] Starting search (max_k=%d, result_t=%zu bytes)\n", max_k, sizeof(ann_result_t));

    // 1. Coarse search: find top-N probe lists by centroid score
    const int PROBE = 8;
    int best_lists[PROBE];
    float best_list_scores[PROBE];
    for (int i = 0; i < PROBE; i++) { best_lists[i] = -1; best_list_scores[i] = -1e9; }

    for(uint32_t c = 0; c < idx->n_lists; c++) {
        float dot = 0;
        const float *cent = idx->centroids + (c * idx->dim);
        for(int i = 0; i < idx->dim; i++) dot += query_vec[i] * cent[i];
        if (dot > best_list_scores[PROBE - 1]) {
            int pos = PROBE - 1;
            while (pos > 0 && dot > best_list_scores[pos - 1]) pos--;
            for (int m = PROBE - 1; m > pos; m--) {
                best_list_scores[m] = best_list_scores[m - 1];
                best_lists[m] = best_lists[m - 1];
            }
            best_list_scores[pos] = dot;
            best_lists[pos] = (int)c;
        }
    }

    int probe_count = 0;
    for (int i = 0; i < PROBE; i++) {
        if (best_lists[i] >= 0) probe_count++;
    }
    if (probe_count == 0) {
        printf("[C] [ann_search] No probe lists found\n");
        return 0;
    }

    printf("[C] [ann_search] Probing %d lists. Best centroid: %d (score %.4f)\n",
           probe_count, best_lists[0], best_list_scores[0]);

    // Global top-k buffers
    float *top_scores = malloc(sizeof(float) * max_k);
    uint64_t *top_ids = malloc(sizeof(uint64_t) * max_k);
    if (!top_scores || !top_ids) {
        free(top_scores);
        free(top_ids);
        return 0;
    }
    int top_count = 0;

    for (int p = 0; p < probe_count; p++) {
        int list_id = best_lists[p];
        uint64_t off = idx->list_offsets[list_id];
        uint64_t len_bytes = idx->list_lengths[list_id];

        if (len_bytes == 0 || len_bytes > 100 * 1024 * 1024) {
            printf("[C] [ann_search] List %d invalid size (%llu). Skipping.\n", list_id, (unsigned long long)len_bytes);
            continue;
        }

        uint8_t *list_data = malloc(len_bytes);
        if (!list_data) continue;
        io_req_t req = { .offset = off, .len = len_bytes, .dst = list_data };

        double t0 = emscripten_get_now();
        idx->reader->read_batch(idx->reader->ctx, &req, 1);
        double t1 = emscripten_get_now();
        printf("[C] [ann_search] List %d fetch latency: %.2f ms\n", list_id, t1 - t0);

        uint32_t count = *(uint32_t*)list_data;
        size_t needed = 4 + (count * 8) + (count * idx->dim * 2);
        if (needed > len_bytes) {
            printf("[C] [ann_search] Error: List %d corrupted (Count %d needs %zu bytes, have %llu)\n",
                   list_id, count, needed, (unsigned long long)len_bytes);
            free(list_data);
            continue;
        }

        if (count == 0) {
            printf("[C] [ann_search] Empty list %d\n", list_id);
            free(list_data);
            continue;
        }

        uint64_t *ids = (uint64_t*)(list_data + 4);
        uint16_t *vecs = (uint16_t*)(list_data + 4 + count * 8);

        printf("[C] [ann_search] List %d: count=%u, len_bytes=%llu\n",
               list_id, count, (unsigned long long)len_bytes);

        for(uint32_t i = 0; i < count; i++) {
            float dot = 0;
            uint16_t *v = vecs + (i * idx->dim);
            for(int k = 0; k < idx->dim; k++) dot += query_vec[k] * half_to_float(v[k]);

            if (top_count < max_k || dot > top_scores[top_count - 1]) {
                int pos = (top_count < max_k) ? top_count : max_k - 1;
                while (pos > 0 && dot > top_scores[pos - 1]) pos--;
                if (top_count < max_k) top_count++;
                for(int m = top_count - 1; m > pos; m--) {
                    top_scores[m] = top_scores[m - 1];
                    top_ids[m] = top_ids[m - 1];
                }
                top_scores[pos] = dot;
                top_ids[pos] = ids[i];
            }
        }

        free(list_data);
    }

    ann_result_t *out = (ann_result_t*)out_bytes;
    for(int i = 0; i < top_count; i++) {
        out[i].id = top_ids[i];
        out[i].score = top_scores[i];
    }

    free(top_scores);
    free(top_ids);
    printf("[C] [ann_search] Returning top-%d results\n", top_count);
    return top_count;
}

// Legacy random-test API retained for compatibility.
EMSCRIPTEN_KEEPALIVE
void ann_query_bytes(uint8_t *bytes, int len) {
    if (!G_INDEX) return;
    float *query_vec = (float*)bytes;
    ann_index_t *idx = G_INDEX;

    // Validation
    if (len != idx->dim * 4) {
        printf("[Error] Query Dim Mismatch. JS sent %d bytes, C expected %d\n", len, idx->dim * 4);
        return;
    }

    // 1. Coarse Search
    int best_c = 0;
    float best_score = -1e9;
    for(uint32_t c=0; c<idx->n_lists; c++) {
        float dot = 0;
        for(int i=0; i<idx->dim; i++) dot += query_vec[i] * idx->centroids[c*idx->dim + i];
        if (dot > best_score) { best_score = dot; best_c = c; }
    }
    printf("[C] Nearest list: %d (score %.4f)\n", best_c, best_score);

    // 2. Fetch List
    uint64_t off = idx->list_offsets[best_c];
    uint64_t len_bytes = idx->list_lengths[best_c];

    // Safety: allow large lists but cap at 100MB
    if (len_bytes == 0 || len_bytes > 100 * 1024 * 1024) {
        printf("[C] List %d invalid size (%llu). Skipping.\n", best_c, len_bytes);
        return;
    }

    uint8_t *list_data = malloc(len_bytes);
    io_req_t req = { .offset = off, .len = len_bytes, .dst = list_data };

    double t0 = emscripten_get_now();
    idx->reader->read_batch(idx->reader->ctx, &req, 1);
    double t1 = emscripten_get_now();
    printf("[C] List fetch latency: %.2f ms\n", t1 - t0);

    // 3. Scan & Score
    uint32_t count = *(uint32_t*)list_data;

    // Safety check size
    size_t needed = 4 + (count * 8) + (count * idx->dim * 2);
    if (needed > len_bytes) {
        printf("[Error] List corrupted (Count %d needs %zu bytes)\n", count, needed);
        free(list_data);
        return;
    }

    uint64_t *ids = (uint64_t*)(list_data + 4);
    uint16_t *vecs = (uint16_t*)(list_data + 4 + count*8);

    uint64_t best_id = 0;
    float max_sim = -1e9;

    for(uint32_t i=0; i<count; i++) {
        float dot = 0;
        uint16_t *v = vecs + (i * idx->dim);
        for(int k=0; k<idx->dim; k++) dot += query_vec[k] * half_to_float(v[k]);
        if (dot > max_sim) { max_sim = dot; best_id = ids[i]; }
    }

    free(list_data);
    printf(">>> RESULT: ID %llu (Score %.4f) <<<\n", best_id, max_sim);
}
