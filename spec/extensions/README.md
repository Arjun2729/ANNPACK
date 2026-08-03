# ANNPack extension registry

Core-only clients are conformant. Extensions are independent opt-ins and must be named explicitly in conformance reports.

| ID | Implemented specification | Scope |
|---|---|---|
| [ANN-1](ANN-1-vector-retrieval.md) | yes | Reproducible vector profiles and IVF-flat retrieval |
| [ANN-2](ANN-2-deltas.md) | yes | Verified copy/add update artifacts |
| [ANN-3](ANN-3-oci-distribution.md) | yes | OCI Distribution transport |
| ANN-4 | unassigned | No compatibility promise exists |
| ANN-5 | withdrawn | Declarative payment and encryption metadata. Removed in v0.5.0: it described requirements it could not enforce and had no consumer. |
| ANN-6 | withdrawn | Root-pinned pack dependencies. Removed in v0.5.0: a name and a version string with no resolution, installation, or conflict policy behind them. |
| [ANN-7](ANN-7-query-expansion.md) | yes | Build-time query expansion (doc2query lineage), disabled by default |
| [ANN-8](ANN-8-vocabulary-expansion.md) | yes | Vocabulary-space expansion (SPLADE lineage), disabled by default |
| ANN-9 | withdrawn | Anchor-based relative representations. Dominated by simpler methods; removed in v0.5.0. Section types 14 and 15 are retired and not reused. |
| [ANN-10](ANN-10-multi-profile-packs.md) | yes | Multi-profile "fat packs" with deterministic fallback |

An extension number is assigned only when its wire behavior exists in the reference implementation and has conformance tests. Unassigned numbers are not roadmap promises.

ANN-4 is unassigned and ANN-5, ANN-6, and ANN-9 are withdrawn. No withdrawn number is reused: a reader that encounters one is looking at an artifact from an older generation, and treating it as something new would be worse than ignoring it. ANN-7 and ANN-8 share the same determinism discipline: semantic understanding is produced by a separate offline command into a pinned, hashed sidecar, and the deterministic `build` consumes the sidecar (recording its digest in `manifest.derived_inputs`) without running any model. None of these extensions is enabled by default, and none carries a measured retrieval-quality claim in this repository.
