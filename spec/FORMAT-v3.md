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

Flag bit zero means the section is required. Flag bit one means the section is **derived**: its contents are produced from passage text by an offline model and are matching-only. Derived sections MUST NOT be marked required, and a reader MUST NOT let a derived section contribute any citable text to an evidence envelope (see [ANN-7](extensions/ANN-7-query-expansion.md)). An unknown required section or required codec MUST be rejected. Unknown optional sections, derived or not, MUST be ignored safely.

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
| 13 | Term overlay | ANN-7 / ANN-8, optional, derived |
| 14 | Anchor set | ANN-9, optional |
| 15 | Anchor coordinates | ANN-9, optional, derived |

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

### 3.1 What the artifact root does and does not commit to

This value is the **artifact root**. It is a commitment to *these exact bytes*.
Because the directory entries it hashes include each section's stored-byte hash,
offset, and length, the artifact root transitively commits to:

- the DEFLATE encoder, its compression level, and its exact output bytes
- passage-block packing and block boundaries
- section ordering, padding, and absolute offsets
- the precise JSON serialization of every structured section

Several of those are implementation choices this specification does not
normatively fix. **Two conformant implementations can therefore compile the same
source into logically equivalent packs with different artifact roots.** The
artifact root is an identity for one artifact, not a semantic identity for a
corpus, and it MUST NOT be described as builder-independent.

The manifest carries no builder or tool identifier, so the artifact root does not
change merely because the *version* of a given builder changed. That is a
narrower and weaker property than cross-implementation reproducibility, and the
reference project verifies only that narrower one (see the
`same-builder-determinism` CI job, which rebuilds the golden artifact across
operating systems and toolchain versions). Implementation provenance, if
recorded, belongs in a signature or external attestation, never in rooted
content.

For a commitment that *is* independent of compression and layout, use the
logical content root in §4.1.

> **History.** v0.3.0 embedded the builder version in the manifest, coupling the
> root to the tool version. v0.3.1 removed it and briefly claimed cross-builder
> reproducibility, which was never true for the reasons above. v0.4.0 corrects the
> claim and adds the logical content root.

## 4. Manifest

The v3 reference profile uses deterministic UTF-8 JSON with stable struct field order and lexicographically ordered maps. The manifest describes:

- Pack name and version
- Description
- Source revision and base URL
- Optional explicit build time
- Document and passage counts
- Capabilities
- Embedding profiles
- Policy
- Dependencies
- The logical content root (§4.1), from manifest section format 2

Policy may declare public, authenticated, licensed, or organization-restricted access; redistribution terms; expiry; payment discovery; and encryption descriptors. These declarations communicate acquisition and handling requirements. They do not themselves implement payment settlement, access control, or DRM.

Builders MUST NOT inject a clock value into deterministic builds unless explicitly requested.

The manifest carries no builder or tool identifier. Implementation provenance
belongs in a signature or an external attestation, never in rooted content.

### 4.1 Logical content root (`passage_merkle_root`)

Manifest section format 2 and later MUST carry:

```json
"passage_merkle_root": "<64 lowercase hex characters>"
```

It is a Merkle root over the per-passage evidence hashes, in deterministic corpus
order:

```text
leaf_i   = BLAKE3(UTF8("ANNPACK3-PASSAGE-EVIDENCE\\0") || passage_record_json_i)
parent   = BLAKE3(UTF8("ANNPACK3-EVIDENCE-NODE\\0") || left || right)
```

Build the tree pairwise from the left. When a level has an odd number of nodes,
**promote** the final node unchanged to the next level; do **not** duplicate it
(duplication would make an N-leaf tree collide with an (N+1)-leaf tree whose last
two leaves are equal). A single leaf is its own root. An empty corpus has no
logical content root, and builders MUST reject empty corpora.

The leaf separator is the same one Core already uses for `passage_hash`, so a
leaf is exactly the evidence hash a reader computes for that passage. The
distinct node separator ensures a leaf can never be reinterpreted as an interior
node.

Unlike the artifact root, this value commits **only** to canonicalized passage
records. It is invariant under DEFLATE settings, block packing, section ordering,
and offsets. Two implementations that agree on ingestion and chunking produce the
same `passage_merkle_root` even when their artifact roots differ, which makes it
the correct basis for cross-builder comparison and for standalone evidence
receipts (see [`EVIDENCE-v1.md`](EVIDENCE-v1.md)).

### 4.2 Manifest section format versions

The manifest schema is versioned by the **section-format version** field of its
directory entry, independently of the `ANNPACK3` wire version.

| Version | Introduced | Change |
|---:|---|---|
| 1 | v0.1 | Original schema, including a required `builder` string. |
| 2 | v0.4.0 | `builder` removed; `passage_merkle_root` added and required. |

Readers MUST accept every manifest format version they implement and MUST reject
an unrecognized one with an explicit version error at the container boundary,
before attempting to deserialize the payload. Readers MUST ignore unknown
manifest fields so a later minor addition stays readable.

> **v0.3.1 compatibility defect.** v0.3.1 removed the required `builder` field
> while leaving the section-format version at 1, the wire version at `ANNPACK3`,
> and the media type unchanged. New readers accepted old packs (unknown fields
> are ignored) but old readers failed on new packs with a bare
> `missing field \`builder\`` deserialization error rather than a version
> refusal — a one-way break shipped as a patch release. v0.4.0 makes the boundary
> explicit by bumping the manifest section format to 2. Artifacts published under
> v0.3.x remain readable and retain their original artifact roots; they carry no
> logical content root and therefore cannot issue standalone receipts.

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

### 6.1 Normative tokenization

Core scoring is normative: two conformant readers MUST return the same ranking
and the same scores for the same query against the same pack. Everything in this
section is therefore a requirement, not a description of the reference
implementation.

Both indexing and querying MUST tokenize identically:

1. Apply Unicode normalization form **NFKC** to the input.
2. Lowercase using Unicode simple lowercase mapping.
3. Split on Unicode whitespace.
4. From each resulting token, trim leading and trailing characters that are
   neither Unicode alphanumeric (`\p{L}` or `\p{N}`) nor a member of the
   **technical punctuation set** defined below. Trimming affects only the edges;
   interior characters are never removed and a token is never split further.
5. Discard tokens that are empty after trimming.

The **technical punctuation set** is exactly these seven characters:

```text
_  -  .  :  /  @  #
```

Preserving them inside tokens is what keeps `std::move`, `foo_bar`,
`package.module`, `AP-104`, and `@scope/pkg` addressable as single terms.

Worked example — this input:

```text
AP-104 std::move useEffect foo_bar package.module
```

MUST produce exactly these five tokens:

```text
ap-104   std::move   useeffect   foo_bar   package.module
```

A tokenizer that splits on `:` or `_`, or that drops them, is **not conformant**.

### 6.2 Normative BM25 profile

```text
k1 = 1.2
b  = 0.75

idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)) * boost(t)
score    = Σ over unique query terms:
             idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
```

where `N` is the passage count, `df(t)` the term's document frequency, `tf` the
term frequency in the passage, `dl` the passage length in tokens, and `avgdl` the
average passage length (floored at 1.0).

`boost(t)` is exactly:

```text
boost(t) = 3.0  if t contains an ASCII digit or any character in the
                technical punctuation set
           1.0  otherwise
```

The boost is applied to the idf term, as written above.

Ranking ties resolve by ascending passage ordinal. Scores are computed in
IEEE-754 double precision.

> Interoperability note. A clean-room reader written against the pre-v0.4.0 text
> — which said only "terms containing digits or technical punctuation receive an
> explicit exact-token boost" — chose boost `2.0`, a three-character punctuation
> set, and a regex tokenizer that split `std::move` into `std` and `move`. It
> passed the then-current conformance suite because the golden queries did not
> exercise those tokens. That is why this section is now fully specified and why
> the conformance vectors assert exact scores.

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
