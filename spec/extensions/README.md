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

An extension number is assigned only when its wire behavior exists in the reference implementation and has conformance tests. Unassigned numbers are not roadmap promises.
