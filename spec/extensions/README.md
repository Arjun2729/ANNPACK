# Adyar Note registry

An **Adyar Note (AN)** is a normative extension document. Each one specifies wire behavior outside the Core profile, is independently conformant, and carries its own number for citation.

Core-only clients are conformant. Extensions are independent opt-ins and must be named explicitly in conformance reports.

| ID | Implemented specification | Scope |
|---|---|---|
| [AN-1](AN-1-vector-retrieval.md) | yes | Reproducible vector profiles and IVF-flat retrieval |
| [AN-2](AN-2-deltas.md) | yes | Verified copy/add update artifacts |
| [AN-3](AN-3-oci-distribution.md) | yes | OCI Distribution transport |
| AN-4 | unassigned | No compatibility promise exists |
| AN-5 | withdrawn | Declarative payment and encryption metadata. Removed in v0.5.0: it described requirements it could not enforce and had no consumer. |
| AN-6 | withdrawn | Root-pinned pack dependencies. Removed in v0.5.0: a name and a version string with no resolution, installation, or conflict policy behind them. |
| [AN-7](AN-7-query-expansion.md) | yes | Build-time query expansion (doc2query lineage), disabled by default |
| [AN-8](AN-8-vocabulary-expansion.md) | yes | Vocabulary-space expansion (SPLADE lineage), disabled by default |
| AN-9 | withdrawn | Anchor-based relative representations. Dominated by simpler methods; removed in v0.5.0. Section types 14 and 15 are retired and not reused. |
| [AN-10](AN-10-multi-profile-packs.md) | yes | Multi-profile "fat packs" with deterministic fallback |

An extension number is assigned only when its wire behavior exists in the reference implementation and has conformance tests. Unassigned numbers are not roadmap promises.

## Identifier migration

These documents were numbered `ANN-N` when the project was called ANNPack, where the prefix carried an approximate-nearest-neighbour reading the format never earned: ranking is BM25-first and vector retrieval is one optional extension among several. The numbers are unchanged; only the prefix is.

| Former | Current | Subject |
|---|---|---|
| ANN-1 | [AN-1](AN-1-vector-retrieval.md) | Vector retrieval |
| ANN-2 | [AN-2](AN-2-deltas.md) | Deltas |
| ANN-3 | [AN-3](AN-3-oci-distribution.md) | OCI Distribution |
| ANN-4 | AN-4 | Unassigned |
| ANN-5 | AN-5 | Withdrawn |
| ANN-6 | AN-6 | Withdrawn |
| ANN-7 | [AN-7](AN-7-query-expansion.md) | Query expansion |
| ANN-8 | [AN-8](AN-8-vocabulary-expansion.md) | Vocabulary expansion |
| ANN-9 | AN-9 | Withdrawn |
| ANN-10 | [AN-10](AN-10-multi-profile-packs.md) | Multi-profile packs |

`ANN-N` identifiers are historical. Cite `AN-N`.

One consequence is visible on the wire: `ConformanceReport.extensions` now reports `AN-1` where it previously reported `ANN-1`. A consumer matching those strings exactly needs updating. This is a deliberate break taken while the Core profile is still `v1.0-draft`, because the alternative — reporting both spellings, or carrying a parallel legacy field — would outlive the transition it was meant to serve. It is the only serialized identifier the rename changes. The `ANNPACK3` and `ANNDELT1` magic, the `annpack-core-v1.0-draft` profile id, and the `https://annpack.dev/attestations/build/v1` predicate type are frozen and unaffected; those name a format version, not a project.

## Assignment policy

AN-4 is unassigned and AN-5, AN-6, and AN-9 are withdrawn. No withdrawn number is reused: a reader that encounters one is looking at an artifact from an older generation, and treating it as something new would be worse than ignoring it. AN-7 and AN-8 share the same determinism discipline: semantic understanding is produced by a separate offline command into a pinned, hashed sidecar, and the deterministic `build` consumes the sidecar (recording its digest in `manifest.derived_inputs`) without running any model. None of these extensions is enabled by default, and none carries a measured retrieval-quality claim in this repository.
