#ifndef ANN_FORMAT_H
#define ANN_FORMAT_H

#include <stdint.h>

#define ANN_MAGIC          0x504E4E41ULL
#define ANN_VERSION        1
#define ANN_ENDIAN_LITTLE  1
#define ANN_METRIC_DOT     1
#define ANN_HEADER_SIZE    72

#pragma pack(push, 1)
typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t endian;
    uint32_t header_size;
    uint32_t dim;
    uint32_t metric;
    uint32_t n_lists;
    uint32_t n_vectors;
    uint64_t offset_table_pos;
    uint8_t  reserved[28];
} ann_header_t;

typedef struct {
    uint64_t offset;
    uint64_t length;
} ann_list_meta_t;

typedef struct {
    uint64_t id;   // 8 bytes
    float    score;// 4 bytes
} ann_result_t;    // packed => 12 bytes
#pragma pack(pop)

int ann_load_index(const char *url);
int ann_search(void *ctx, const float *query, ann_result_t *out_results, int k);
int ann_result_size_bytes(void);

#endif // ANN_FORMAT_H
