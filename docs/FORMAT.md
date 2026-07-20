# ANNPack v2 Binary Format

This document describes the static IVF-based ANNPack v2 format for L2-normalized vectors stored with dot-product scoring.

## Header (72 bytes, little-endian)

```c
uint64_t magic;           // "ANNP" = 0x504E4E41
uint32_t version;         // 1
uint32_t endian;          // 1 for little-endian
uint32_t header_size;     // 72
uint32_t dim;             // vector dimension
uint32_t metric;          // 1 = dot-product / cosine
uint32_t n_lists;         // number of IVF lists
uint32_t n_vectors;       // total number of vectors
uint64_t offset_table_pos;// absolute file offset of list offset table
uint8_t  reserved[28];    // zero padding
```

Immediately after the header are the IVF centroids:

- `n_lists * dim` float32 values in row-major order.

## Lists

Each list lives at its own absolute file offset. At that offset:

```c
uint32 count;
uint64 ids[count];
float16 vecs[count][dim];  // row-major, little-endian
```

Vectors are stored as IEEE 754 half-precision. All vectors are expected to be L2-normalized.

## Offset Table

At `offset_table_pos` there are `n_lists` entries of:

```c
typedef struct {
    uint64_t offset;
    uint64_t length;
} ann_list_meta_t;
```

## Semantics

- The index is static and read-only.
- All vectors must be L2-normalized before writing.
- Search flow: coarse search over centroids using dot-product (cosine), pick the top `PROBE` lists, then brute-force scan within those lists keeping the top-K results by score.
