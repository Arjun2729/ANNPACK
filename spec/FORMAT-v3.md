# ANNPack Binary Wire Format v3

Status: frozen wire draft. Core conformance is defined by [ANNPack Core v1.0-draft](CORE-v1.0-draft.md); optional behavior is isolated in [numbered extensions](extensions/README.md).

All multibyte integers are unsigned little-endian. Offsets are absolute from the first byte of the artifact. Parsers MUST use checked arithmetic and MUST reject overlapping or out-of-bounds regions.

## 1. Header

The fixed header is 128 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | ASCII magic `ANNPACK3` |
| 8 | 4 | Format version, currently `3` |
| 12 | 4 | Header size, currently `128` |
| 16 | 8 | Container flags |
| 24 | 8 | Section-directory offset |
| 32 | 8 | Section-directory length |
| 40 | 4 | Manifest section ID |
| 44 | 4 | Section count |
| 48 | 32 | BLAKE3 content root |
| 80 | 48 | Reserved, zero in v3 |

The directory length MUST equal `section_count * 80`. Reserved header bytes MUST be zero. A conforming reference reader accepts at most 16,384 sections.

## 2. Section directory

Each entry is 80 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | Section ID |
| 4 | 2 | Section type |
| 6 | 2 | Section-format version |
| 8 | 2 | Codec |
| 10 | 2 | Flags |
| 12 | 8 | Stored offset |
| 20 | 8 | Stored length |
| 28 | 8 | Logical length |
| 36 | 8 | Item count |
| 44 | 32 | BLAKE3 hash of stored bytes |
| 76 | 4 | Reserved |

Flag bit zero means the section is required. An unknown required section or required codec MUST be rejected. Unknown optional sections MUST be ignored safely.

Codec zero is uncompressed. Its stored and logical lengths MUST match. Codec one is zlib-wrapped DEFLATE. Readers MUST bound decompression to the declared logical length, reject any length mismatch, and impose a decompression-ratio limit before allocating. The reference reader allows at most 256:1 once a section's logical length exceeds 16 MiB.

Initial section types:

| ID | Name | Core requirement or owner |
|---:|---|---|
| 1 | Manifest | Core, required |
| 2 | Documents | Core, required |
| 3 | Passage index | Core, required |
| 4 | Passage data | Core, required |
| 5 | Lexical dictionary | Core, required |
| 6 | Lexical postings | Core, required |
| 7 | Vector profile | ANN-1, optional |
| 8 | Vector data | ANN-1, optional |
| 9 | Vector index | ANN-1, optional |
| 10 | Signature | Core, optional artifact content |
| 11 | Policy extension | ANN-5, optional |
| 12 | Delta manifest | reserved; ANN-2 uses a separate update artifact |

The numeric section ID is artifact-local and independent of section type. Directory entries MUST be encoded in strictly increasing section-ID order. Reserved entry bytes MUST be zero. V3-defined section types are singletons except Signature, which may appear more than once for key rotation and multi-party attestation.

## 3. Content root

The content root is:

```text
BLAKE3(
  UTF8("ANNPACK3-CONTENT-ROOT\\0") ||
  encoded_directory_entry_1 ||
  ...
)
```

Directory entries whose section type is Signature are excluded. All other encoded entries, including offsets, lengths, flags, and stored-byte hashes, are included in section-ID order.

Excluding signature entries allows signatures to be added, replaced, or mirrored without changing the immutable identity of the knowledge content. Signatures authenticate the resulting content root.

## 4. Manifest

The v3 reference profile uses deterministic UTF-8 JSON with stable struct field order and lexicographically ordered maps. The manifest describes:

- Pack name and version
- Description
- Source revision and base URL
- Optional explicit build time
- Builder identity
- Document and passage counts
- Capabilities
- Embedding profiles
- Policy
- Dependencies

Policy may declare public, authenticated, licensed, or organization-restricted access; redistribution terms; expiry; payment discovery; and encryption descriptors. These declarations communicate acquisition and handling requirements. They do not themselves implement payment settlement, access control, or DRM.

Builders MUST NOT inject a clock value into deterministic builds unless explicitly requested.

## 5. Documents and passages

The Documents section is a deterministic JSON array. Public document IDs are lowercase BLAKE3 hexadecimal strings derived from normalized source paths.

The Passage Index contains records with:

- Passage ID
- Passage-block ordinal
- Offset and length inside the logical block

It also contains a block table with stored offset, stored and logical lengths, and a BLAKE3 hash of each compressed block.

Passage Data concatenates independently compressed, zlib-wrapped DEFLATE blocks. Each logical block contains concatenated deterministic JSON records. This permits a reader to verify the compressed block and inflate it with a declared bound while fetching only the blocks containing selected results.

Reference IDs are derived as:

```text
document_id = BLAKE3("document\\0" || normalized_source_path)

passage_id = BLAKE3(
  "passage\\0" || document_id || "\\0" ||
  heading_path || "\\0" || normalized_passage_text
)
```

Sequential ordinals are private index coordinates and MUST NOT be exposed as persistent identities.

## 6. Lexical index

The Lexical Dictionary contains passage lengths, average passage length, and a lexicographically ordered term map. Each term maps to:

- Offset relative to Lexical Postings
- Length
- Document frequency

Each posting is two unsigned varints:

```text
passage_ordinal_delta
term_frequency
```

The first ordinal is stored directly; subsequent values are positive deltas. The reference profile stores the complete postings section using section-level DEFLATE, so the section hash authenticates all posting lists in one range read. Decoders MUST reject unterminated, overflowing, zero-frequency, trailing-byte, or out-of-range data.

The reference ranker uses BM25 with `k1=1.2` and `b=0.75`. Terms containing digits or technical punctuation receive an explicit exact-token boost. Ranking ties resolve by passage ordinal.

## 7. Vector profiles and data

This section is owned by optional extension [ANN-1](extensions/ANN-1-vector-retrieval.md), not Core.

Vector representations are optional. A profile includes:

- Profile ID
- Model name and exact revision
- Dimensions
- Scalar type
- Pooling
- Normalization
- Query prefix/template
- Document prefix/template
- Ordered passage IDs

A model name and dimensions alone are not a reproducible embedding-space descriptor.

The flat-vector data section begins with:

```text
u32 vector_count
u32 dimensions
f32 values[vector_count][dimensions]
```

Non-finite values and shape mismatches MUST be rejected.

The reference Vector Index section implements deterministic `ivf-flat-v1`. It stores its exact distance function (`dot`), dimensions, default probe count, centroids, and a complete partition of vector ordinals. Readers MUST reject non-finite or dimension-mismatched centroids, duplicate or missing ordinals, and out-of-range lists. Query execution ranks centroids, probes a caller-bounded number of lists, then scores the selected exact vectors. HNSW, quantized, sparse, graph, and structured indexes belong in separate optional section formats.

The reference hybrid ranker uses reciprocal-rank fusion with constant 60. Raw BM25 and vector scores are not presumed comparable.

## 8. Signatures

Signature sections contain deterministic JSON envelopes with:

- Algorithm (`Ed25519`)
- Public key
- Signature
- Signed root
- Key ID
- Optional asserted identity
- Optional expiration
- Optional transparency-log reference

The signed bytes are:

```text
UTF8("ANNPACK3-SIGNATURE\\0") || content_root
```

Key ID is BLAKE3 of the raw public key. A valid signature does not independently prove the asserted publisher identity.

## 9. Limits and errors

Readers MUST impose implementation limits before allocation. The reference implementation limits manifests to 4 MiB, individual sections to 64 GiB, independently compressed passage blocks to 1 MiB logical, results to 1,000, and embedding dimensions to 65,536.

Malformed input MUST produce a bounded error. It MUST NOT panic, hang, read outside the artifact, or allocate an unchecked attacker-controlled length.
