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
| 11 | *retired* | was ANN-5 policy; withdrawn, number not reused |
| 12 | Delta manifest | reserved; ANN-2 uses a separate update artifact |
| 13 | Term overlay | ANN-7 / ANN-8, optional, derived |
| 14 | *retired* | was ANN-9 anchor set; withdrawn, number not reused |
| 15 | *retired* | was ANN-9 anchor coordinates; withdrawn, number not reused |
| 16 | Lexical terms | block-addressable term table; required in lexical index format 2 |
| 17 | Passage records | block-addressable record table; required in passage index format 2 |

The numeric section ID is artifact-local and independent of section type. Early reference-builder IDs happen to equal their section types and later ones do not: section ID 14 identifies a section of type 13 even though section *type* 14 is retired, and IDs 17 and 18 identify types 16 and 17. A reader MUST resolve sections by type or by a declared ID, never by assuming the two numbers agree. Directory entries MUST be encoded in strictly increasing section-ID order. Reserved entry bytes MUST be zero.

V3-defined section types are singletons except `Signature` and `TermOverlay`, which MAY occur more than once — `Signature` for key rotation and multi-party attestation, `TermOverlay` because a pack may carry both an ANN-7 expansion and an ANN-8 SPLADE overlay, each a separate section of the same type. Unknown optional section types MAY also occur more than once unless the extension defining them specifies otherwise.

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

This value is the **artifact root**. It commits to the non-signature directory
entries and, through the stored-byte hash each entry carries, to the stored
section bytes those entries reference. It is **not** a whole-file hash: it does
not authenticate unreferenced trailing bytes, inter-section padding, or the
excluded signature sections.

Because the entries it hashes include each section's stored-byte hash, offset,
and length, the artifact root transitively commits to:

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
- The logical content root (§4.1), required from manifest section format 2
- The authenticated source descriptor (§4.3), required from manifest section format 4

Policy may declare public, authenticated, licensed, or organization-restricted access; redistribution terms; expiry; and a policy URL. These declarations communicate acquisition and handling requirements. They do not themselves implement access control or DRM.

A `dependencies` list, and the policy `payment` and `encryption` descriptors, existed through manifest section format 2 and were removed in format 3 with ANN-5 and ANN-6. Readers ignore unknown manifest fields, so a format 1 or 2 artifact carrying them stays readable.

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
| 2 | v0.4.0 | `builder` removed; `passage_merkle_root` added and required (§4.1). |
| 3 | v0.5.0 | `dependencies` and the policy `payment` and `encryption` descriptors removed with ANN-5 and ANN-6. No field became required; a format-3 manifest is a format-2 manifest with those fields absent. |
| 4 | v0.7.0-rc1 | `source` added and required: the authenticated source descriptor (§4.3), for every input format rather than OKF alone (ADR-0005). |

The current format emitted by the reference builder is **4**. A reader that
implements only formats 1-3 MUST refuse a format-4 manifest explicitly at the
container boundary, as it would any unrecognized version, rather than parsing it
and silently ignoring the descriptor it cannot validate.

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

### 4.3 Authenticated source descriptor (`source`)

Manifest section format 4 and later MUST carry:

```json
"source": {
  "format": "<resolved input format>",
  "version": "<format version, or null>",
  "digest_algorithm": "blake3",
  "digest": "<64 lowercase hex characters>"
}
```

`digest` is over the exact source bytes the compiler consumed, so a source claim
cannot diverge from what was actually built (ADR-0005). Readers MUST reject a
format-4 manifest when:

- `source` is absent;
- `digest_algorithm` is anything other than `blake3`;
- `digest` is not exactly 64 lowercase hexadecimal characters; or
- `format` is empty or the literal `auto`.

`auto` is a request to detect an input format, not a resolved one. Recording it
would leave a verifier unable to tell which ingestion rules produced the digest,
so it MUST NOT appear in a manifest even though it is a valid builder argument.

Formats 1-3 legitimately predate this requirement. A missing descriptor below
format 4 is history, not corruption, and such an artifact retains its original
artifact root.

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

### 5.1 Passage identifiers must be unique within a pack

Passage IDs index the passage table, so a reader MUST reject a pack containing
two passages with the same ID.

The derivation above is content-addressed, so two passages in one document with
the same heading path and the same normalized text collide. That is not rare in
practice: a repeated warning, a boilerplate disclaimer, a template section, or
generated documentation can all produce identical text under an identical
heading. **Builders MUST reject such a corpus rather than emit a pack no
conforming reader can open.** The error should name the source file and the
heading context so the author can act on it.

Authors resolve a collision by making one occurrence distinguishable — a
distinct heading, a cross-reference instead of a repeated block, or any wording
difference. Normalization collapses whitespace only, so any change to the words
is sufficient.

> This is a pre-1.0 constraint, and it is the conservative half of a real
> trade-off. Widening the derivation (for example by mixing in the ordinal or a
> source byte offset) would let duplicates coexist, but it would change every
> existing passage ID and therefore every logical content root and published
> receipt. Rejecting at build time keeps published identities stable. If the
> constraint proves too costly for real corpora, the derivation is the thing to
> revisit, in a version that says so.

### 5.1a Why Documents stays whole

The Documents section is deliberately *not* block-addressed, unlike the passage record table (§5.2) and the lexical index (§6).

An [Evidence v1](EVIDENCE-v1.md) receipt that authenticates a canonical URL embeds the complete stored Documents section, because the verifier runs with no pack and no network and must re-derive the section hash from bytes it carries. Block-addressing Documents would therefore save a reader nothing on the receipt path — the whole section travels regardless — while requiring the offline verifier to understand block structure. It would move complexity into the one component whose correctness the entire evidence chain rests on, to save a small and bounded fraction of transfer.

If a corpus ever makes this section large enough to matter, the right move is to shrink what a receipt must embed, not to partition the section.

### 5.2 Passage record table

Where the record table lives depends on the passage index format version, carried in the **Passage Index** section's `format_version`. Readers MUST support both.

**Format 1.** Records are inline in the Passage Index as JSON objects with hex passage ids. Resolving any single result requires downloading and parsing the whole table.

**Format 2.** Records move to the **Passage Records** section (type 17, required), whose payload is two independently addressable regions of independently deflated, independently hashed blocks. The block tables live in the Passage Index under `record_blocks`:

```text
record_blocks.stride      fixed record width in bytes (12)
record_blocks.per_block   records per block, uniform except in the final block
record_blocks.records[]   offset, stored_length, logical_length, hash
record_blocks.ids[]       offset, stored_length, logical_length, hash, first_term
```

The `records` region holds fixed-width records in passage-ordinal order:

```text
block          u32 little-endian
offset         u32 little-endian
length         u32 little-endian
```

A record carries **no passage identifier**. The identifier is already present in
the `ids` region and in the passage payload itself, and a third copy costs 32
incompressible bytes per passage — enough to make a pack larger than the source
it was compiled from. A reader that needs the identifier for an ordinal reads
the payload, or inverts the `ids` region.

Because the record no longer carries an identifier to compare against, a reader
MUST instead check that the payload's own `ordinal` field equals the ordinal it
sought. That is the property the identifier comparison actually provided: it
detects a mis-seek, whether from a wrong stride, a wrong block, or a malformed
block table.

Because records are fixed width and uniformly packed, the block holding ordinal *n* is `n / per_block` and its position within that block is `(n % per_block) * stride`. A reader MUST compute it rather than search.

The `ids` region holds the same passages keyed by identifier and sorted by it, as `32-byte passage_id || u32 ordinal` (36 bytes). Each block's `first_term` carries its first id as lowercase hex, reusing the sparse-index field §6 defines. This region exists only to answer lookup-by-identifier, which passage order cannot serve; a reader that never resolves a passage by id never fetches it.

Both regions MUST tile the Passage Records section exactly, in the order given, with no gap or overlap. `records` MUST total `passage_count * stride` logical bytes and `ids` MUST total `passage_count * 36`; a table that does not cover every passage would make some ordinals silently unreachable rather than fail. `ids` block `first_term` values MUST be strictly increasing.

Block hashes carry the same requirement as §6: a section hash authenticates a section only in full, so a reader MUST verify a block against its own hash before using it.

## 6. Lexical index

The Lexical Dictionary contains passage lengths and average passage length. Where the term map lives depends on the lexical index format version, which is carried in the **Lexical Postings** section's `format_version` field. Readers MUST support both.

Each term maps to:

- Offset relative to Lexical Postings
- Length
- Document frequency

**Format 1.** The term map is stored inline in the Lexical Dictionary, and Lexical Postings is a single section-level DEFLATE payload. Resolving one term therefore requires downloading and inflating both sections in full.

**Format 2.** The term map moves to the **Lexical Terms** section (type 16, required) and both it and Lexical Postings use section-level codec `None`, with their payloads partitioned into independently deflated, independently hashed blocks. The block tables live in the Passage Index under `lexical_blocks`, which a reader already fetches at open:

```text
lexical_blocks.dictionary[]  offset, stored_length, logical_length, hash, first_term
lexical_blocks.postings[]    offset, stored_length, logical_length, hash
```

`offset` is relative to the start of its section and `hash` is BLAKE3 over the *stored* block bytes. Because a section hash authenticates a section only in full, a block's own hash is what makes a partial read trustworthy; a reader MUST verify it before using the block, and MUST NOT accept a block on the strength of the section hash alone.

Blocks MUST tile their section exactly: contiguous, in order, no gap and no overlap, together covering `stored_length`. Dictionary blocks MUST each carry `first_term` and those values MUST be strictly increasing, since that is the sparse index a reader searches. The block that can contain a term is the last one whose `first_term` is not greater than it; a term sorting before every block is absent. A posting list is reassembled from exactly those postings blocks its byte range intersects, and a reader MUST reject a list whose reassembled length differs from the term's recorded length.

A format-2 pack MUST NOT be read by a format-1-only reader: the Lexical Terms section is marked required precisely so such a reader refuses the pack rather than searching an index it only partly understands.

Each posting is two unsigned varints:

```text
passage_ordinal_delta
term_frequency
```

The first ordinal is stored directly; subsequent values are positive deltas. Decoders MUST reject unterminated, overflowing, zero-frequency, trailing-byte, or out-of-range data.

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

The reference hybrid ranker places both modes on a comparable absolute scale and sums them. Lexical scores are divided by the query's maximum achievable BM25 score — the total `idf * (k1 + 1)` over query terms — so a score expresses the fraction of the achievable total a passage accounts for. Vector scores are dot products of normalized embeddings, already cosine similarities on a fixed scale.

It does not use reciprocal-rank fusion. RRF scores by rank position and discards score magnitude, which is the signal separating a retriever that found a match from one that did not; measured on this repository's hard-negative corpus it ranked a lexical-47th passage above a vector-1st passage and scored 0.556 recall@5 against vector-only at 0.794, where absolute-scale fusion scores 0.730. Hybrid is not enabled by default for a separate reason, recorded with the measurements in `rust/src/search.rs`.

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

### 8.1 What the signature authenticates

The signature covers the artifact root and nothing else. Signature sections are
excluded from that root (§3), so no field of the envelope is committed by it
either.

Verification therefore rests on exactly four fields: the algorithm, the public
key, the signature, and the signed root — checked against the reader's own
computed root — plus the key ID, checked against BLAKE3 of the public key.

The asserted identity, expiration, transparency-log URL, revocation URL and
build-attestation fields are **unauthenticated metadata**. Anyone who can
rewrite the artifact can rewrite them without invalidating the signature.
Readers MUST NOT make any security decision from them, MUST NOT report them as
signed, and MUST NOT treat expiration as enforcement. Presenting them at all is
optional; presenting them as verified is a conformance failure.

Binding them would require a signed-envelope format in which the signature
covers the envelope as well as the root. This version does not define one.

## 9. Limits and errors

Readers MUST impose implementation limits before allocation. The reference implementation limits manifests to 4 MiB, individual sections to 64 GiB, independently compressed passage blocks to 1 MiB logical, results to 1,000, and embedding dimensions to 65,536. Its tools apply the same discipline to input that arrives outside a container: one MCP JSON-RPC request line is capped at 8 MiB, and a receipt file is size-checked against a 64 MiB limit before it is read.

Malformed input MUST produce a bounded error. It MUST NOT panic, hang, read outside the artifact, or allocate an unchecked attacker-controlled length.
