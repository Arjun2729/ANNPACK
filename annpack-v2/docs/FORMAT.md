# ANNPack File Format
Static IVF index for L2-normalized vectors.

Header (72 bytes, little-endian):
- uint64 magic           @0   must be 0x00000000504E4E41 ("ANNP")
- uint32 version         @8   1
- uint32 endian          @12  1 (little)
- uint32 header_size     @16  72
- uint32 dim             @20  vector dimension
- uint32 metric          @24  1 = dot-product/cosine
- uint32 n_lists         @28  IVF lists
- uint32 n_vectors       @32  total vectors
- uint64 offset_table    @36  absolute offset of list offset table
- padding/reserved       @44..71 zero

Centroids:
- start @ header_size
- n_lists * dim float32 (little), row-major.

Lists (for each i in 0..n_lists-1, at offset_table[i].offset):
- uint32 count
- uint64 ids[count]
- float16 vecs[count][dim] (little), row-major.
- total length stored in offset_table[i].length

Offset table:
- at offset_table_pos
- n_lists entries of:
  struct { uint64_t offset; uint64_t length; }

Semantics:
- Vectors are L2-normalized before write.
- Search: dot-product coarse on centroids (top PROBE), fine scan within those lists, keep top-K.
- Endianness is little-endian; header_size/version allow future evolution.
