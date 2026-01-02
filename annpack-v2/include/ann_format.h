#ifndef ANN_FORMAT_H
#define ANN_FORMAT_H

#include <stdint.h>

#define ANN_MAGIC 0x504E4E41ULL
#define ANN_VERSION 1
#define ANN_ENDIAN_LITTLE 1
#define ANN_METRIC_DOT 1
#define ANN_HEADER_SIZE 72

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
    uint8_t reserved[28];
} ann_header_t;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint64_t offset;
    uint64_t length;
} ann_list_meta_t;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint64_t id;   // 8
    float score;   // 4
} ann_result_t;    // packed: 12 bytes
#pragma pack(pop)

_Static_assert(sizeof(ann_result_t) == 12, "ann_result_t must be 12 bytes (packed)");

// Max k to guard allocations in wasm runtime
#define ANN_MAX_K 256

#endif
