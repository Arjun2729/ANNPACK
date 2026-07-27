# ADR-0003: Two roots — artifact identity and logical content identity

Status: accepted, 2026-07-27. Supersedes the cross-builder reproducibility claim
introduced in v0.3.1.

## Context

v0.3.1 removed the builder identifier from the manifest and claimed the result
was a content root "reproducible by any conformant builder." That claim was
false. The root is computed over section directory entries, which include each
section's stored-byte hash, offset, and length, so it transitively commits to:

- DEFLATE implementation, compression level, and exact output bytes
- passage block packing and boundaries
- section ordering, padding, and absolute offsets
- the precise JSON serialization of every structured section

None of those is normatively specified. Two conformant implementations can
compile identical source into logically equivalent packs with different roots.
The CI matrix added alongside the claim proves only that *the same builder* is
deterministic across operating systems and toolchain versions — a real and
worthwhile property, but a much weaker one.

Meanwhile the product claim — tamper-evident provenance of a retrieved span —
needs a commitment a third party can check *without the artifact*. The artifact
root cannot serve that on its own: verifying one passage against it requires the
containing block, the passage index, and the directory.

## Decision

Define two roots with separate jobs, and never conflate them.

**Artifact root** (unchanged bytes, corrected wording). BLAKE3 over the
non-signature directory entries. Identity of *one artifact*. Reproducible by the
same builder across environments; explicitly **not** a cross-builder semantic
identity.

**Logical content root** (`manifest.passage_merkle_root`, new in manifest format
2). A Merkle root over per-passage evidence hashes in corpus order, using the
existing Core passage-evidence separator for leaves and a distinct separator for
interior nodes. Invariant under compression, block packing, and layout. Stable
across any two builders that agree on ingestion and chunking.

## Consequences

- Evidence receipts become possible: a ~2–5 KB document proves a passage was in
  an artifact, offline, in ⌈log₂ n⌉ hash operations. This is the basis of
  [EVIDENCE-v1](../EVIDENCE-v1.md) and the `verify-evidence` tool.
- Cross-builder comparison has a well-defined target. Two implementations
  disagreeing on `passage_merkle_root` disagree about *content*; disagreeing only
  on the artifact root is a layout difference and may be acceptable.
- Requiring the field is a manifest schema change, hence manifest section format
  2 and the explicit compatibility boundary in FORMAT-v3 §4.2.
- v0.3.x packs carry no logical content root and cannot issue receipts. They
  remain readable and keep their original artifact roots. `receipt_for_passage`
  refuses rather than emitting a receipt whose chain cannot close.
- The reference builder must compute leaves from the same serialized bytes it
  stores, or a leaf could disagree with the record a reader hashes. `encode_passages`
  therefore derives both from one serialization.

## Alternatives rejected

**Normatively specify DEFLATE and layout so artifact roots match across
builders.** This pins every implementation to one compressor's exact output
forever, making the format hostile to reimplementation — the opposite of the
goal. Compression is an encoding detail; it should not be load-bearing for
identity.

**Bao / BLAKE3 subtree proofs against existing section hashes.** Attractive
because it needs no new field: BLAKE3 is internally a Merkle tree, so a byte range
can be proven against a section hash. But the passage index is DEFLATE-compressed
as a whole, so no range proof is possible over it, and the record-to-block mapping
lives there. It would also add a dependency and couple receipts to BLAKE3's
internal chunking rather than to passage semantics.

**Put the Merkle root in a new section instead of the manifest.** Binding it to
the artifact root would then require shipping the section bytes in the receipt.
In the manifest it is covered by the root already, and the manifest is small
enough to carry inside the receipt.
