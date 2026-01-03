#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ann_api.h"
#include "ann_format.h"

static int test_positive(const char *base_dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny.annpack", base_dir);
    void *ctx = ann_load_index(path);
    if (!ctx) {
        fprintf(stderr, "positive: load failed (%s)\n", ann_last_error());
        return 1;
    }
    if (ann_result_size_bytes() != 12) {
        fprintf(stderr, "positive: result size mismatch\n");
        return 2;
    }
    ann_set_probe(2);
    ann_set_max_k(8);
    ann_set_max_scan(0);
    float q[4] = {1.f, 0.f, 0.f, 0.f};
    ann_result_t out[4];
    int ret = ann_search(ctx, q, out, 3);
    ann_free_index(ctx);
    if (ret <= 0) {
        fprintf(stderr, "positive: search returned %d\n", ret);
        return 3;
    }
    if (out[0].id != 100) {
        fprintf(stderr, "positive: expected id 100 got %llu\n", (unsigned long long)out[0].id);
        return 4;
    }
    return 0;
}

static int test_bad(const char *path) {
    void *ctx = ann_load_index(path);
    if (ctx) {
        fprintf(stderr, "negative: load unexpectedly succeeded for %s\n", path);
        ann_free_index(ctx);
        return 1;
    }
    const char *err = ann_last_error();
    if (!err || strlen(err) == 0) {
        fprintf(stderr, "negative: ann_last_error empty for %s\n", path);
        return 2;
    }
    return 0;
}

static int test_loop(const char *base_dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny.annpack", base_dir);
    for (int i = 0; i < 50; i++) {
        void *ctx = ann_load_index(path);
        if (!ctx) {
            fprintf(stderr, "loop: load failed on iter %d (%s)\n", i, ann_last_error());
            return 1;
        }
        float q[4] = {0.f, 1.f, 0.f, 0.f};
        ann_result_t out[2];
        int ret = ann_search(ctx, q, out, 2);
        ann_free_index(ctx);
        if (ret <= 0) {
            fprintf(stderr, "loop: search failed on iter %d\n", i);
            return 2;
        }
    }
    return 0;
}

static int write_bad_header(const char *path, uint32_t n_lists, uint32_t dim) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    ann_header_t h = {0};
    h.magic = ANN_MAGIC;
    h.version = ANN_VERSION;
    h.endian = ANN_ENDIAN_LITTLE;
    h.header_size = ANN_HEADER_SIZE;
    h.dim = dim;
    h.metric = ANN_METRIC_DOT;
    h.n_lists = n_lists;
    h.n_vectors = 0;
    h.offset_table_pos = 1;
    fwrite(&h, 1, sizeof(h), f);
    if (sizeof(h) < ANN_HEADER_SIZE) {
        char zero = 0;
        for (size_t i = sizeof(h); i < ANN_HEADER_SIZE; i++) {
            fwrite(&zero, 1, 1, f);
        }
    }
    fclose(f);
    return 0;
}

static int test_bad_header_guard(const char *base_dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_badlists.annpack", base_dir);
    if (write_bad_header(path, 1000001u, 4u) != 0) {
        fprintf(stderr, "bad_header: failed to write %s\n", path);
        return 1;
    }
    return test_bad(path);
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "tests/tmp";
    int rc = 0;
    rc |= test_positive(dir);
    char bad_magic[512]; snprintf(bad_magic, sizeof(bad_magic), "%s/tiny_badmagic.annpack", dir);
    char bad_trunc[512]; snprintf(bad_trunc, sizeof(bad_trunc), "%s/tiny_trunc.annpack", dir);
    char bad_table[512]; snprintf(bad_table, sizeof(bad_table), "%s/tiny_badtable.annpack", dir);
    rc |= test_bad(bad_magic);
    rc |= test_bad(bad_trunc);
    rc |= test_bad(bad_table);
    rc |= test_bad_header_guard(dir);
    rc |= test_loop(dir);
    if (rc != 0) {
        fprintf(stderr, "native_test failed with code %d\n", rc);
        return rc;
    }
    printf("native_test ok\n");
    return 0;
}
