# ANNPack extension registry

Core-only clients are conformant. Extensions are independent opt-ins and must be named explicitly in conformance reports.

| ID | Implemented specification | Scope |
|---|---|---|
| [ANN-1](ANN-1-vector-retrieval.md) | yes | Reproducible vector profiles and IVF-flat retrieval |
| [ANN-2](ANN-2-deltas.md) | yes | Verified copy/add update artifacts |
| [ANN-3](ANN-3-oci-distribution.md) | yes | OCI Distribution transport |
| ANN-4 | unassigned | No compatibility promise exists |
| [ANN-5](ANN-5-policy-descriptors.md) | yes | Declarative payment and encryption metadata |
| [ANN-6](ANN-6-pack-dependencies.md) | yes | Root-pinned pack dependencies |
| [ANN-7](ANN-7-query-expansion.md) | yes | Build-time query expansion (doc2query lineage), disabled by default |
| [ANN-8](ANN-8-vocabulary-expansion.md) | yes | Vocabulary-space expansion (SPLADE lineage), disabled by default |
| [ANN-9](ANN-9-anchor-representations.md) | yes | Anchor-based relative representations, research-grade and unvalidated |
| [ANN-10](ANN-10-multi-profile-packs.md) | yes | Multi-profile "fat packs" with deterministic fallback |

An extension number is assigned only when its wire behavior exists in the reference implementation and has conformance tests. Unassigned numbers are not roadmap promises.

ANN-4 remains deliberately unassigned; ANN-7 through ANN-10 continue the sequence without reusing it. ANN-7, ANN-8, and ANN-9 share the same determinism discipline: semantic understanding is produced by a separate offline command into a pinned, hashed sidecar, and the deterministic `build` consumes the sidecar (recording its digest in `manifest.derived_inputs`) without running any model. None of these extensions is enabled by default, and none carries a measured retrieval-quality claim in this repository.
