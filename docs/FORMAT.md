# ANNPack File Format

ANNPack is a static IVF-based ANN index for L2‑normalized vectors. It is stored as a single binary file containing:

1. A fixed-size header.
2. A centroid matrix.
3. Per-list blobs (counts, ids, fp16 vectors).
4. A per-list offset table.

All multi-byte values are little-endian. Vectors are assumed to be L2‑normalized and searched with dot-product/cosine.

## Header Layout (72 bytes)

| Offset | Size | Field              | Type   | Notes                                                                               |
|--------|------|--------------------|--------|-------------------------------------------------------------------------------------|
| 0      | 8    | magic              | uint64 | Must be `0x00000000504E4E41` (ASCII "ANNP" in low 32 bits).                         |
| 8      | 4    | version            | uint32 | `1`                                                                                 |
| 12     | 4    | endian             | uint32 | `1` = little-endian                                                                 |
| 16     | 4    | header_size        | uint32 | `72`                                                                                |
| 20     | 4    | dim                | uint32 | Vector dimensionality                                                               |
| 24     | 4    | metric             | uint32 | `1` = dot-product / cosine on L2-normalized vectors                                 |
| 28     | 4    | n_lists            | uint32 | Number of IVF lists                                                                 |
| 32     | 4    | n_vectors          | uint32 | Total number of vectors in the index                                                |
| 36     | 8    | offset_table_pos   | uint64 | Absolute byte offset of the list offset table                                       |
| 44     | 28   | reserved           | bytes  | Zero padding for future extensions (header_size - 44 bytes)                         |

## Centroid Matrix

- Starts at byte offset `header_size` (typically 72).
- Contains `n_lists * dim` float32 values (little-endian), row-major.
  - Centroid `i` occupies the contiguous block `[i * dim : (i + 1) * dim]`.

## List Blobs

For each list `i` in order `0 .. n_lists-1`, stored at absolute file offset `offset_i` (from the offset table):

1. `count_i` (uint32)
2. `ids_i[count_i]` (uint64 array)
3. `vecs_i[count_i][dim]` (float16 array), row-major, little-endian IEEE-754 half

There is no padding between these segments. The total byte size of the blob is recorded in the offset table.

## Offset Table

- Located at `offset_table_pos`.
- Contains `n_lists` entries of:

```c
struct list_meta {
    uint64_t offset; // absolute byte offset of the list blob
    uint64_t length; // total byte length of that blob
};
```

## Semantics

- All vectors are expected to be L2‑normalized before writing.
- Search (as implemented in the C/WASM runtime and Python reader):
  - Coarse: compute dot products between the query and all centroids; select top `PROBE` lists (default 8).
  - Fine: within those lists, decode fp16 vectors, compute dot products, and maintain a global top‑K.
  - Similarity is dot-product (cosine for normalized vectors).
- Endianness is fixed to little-endian (`endian = 1`). The `version` and `header_size` fields allow future evolution without breaking this spec.

## Compatibility

The format described here is exactly what the current Python builder writes (`annpack.build.write_annpack`) and what the C/WASM runtime (`ann_load_index`, `ann_search`) reads. Any other language (e.g., Rust, Go) can implement a reader by following this layout.***
