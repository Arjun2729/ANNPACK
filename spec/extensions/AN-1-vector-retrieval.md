# AN-1: Vector retrieval

Status: implemented draft. Requires ANNPack Core v1.0-draft.

AN-1 consists of optional section types 7, 8, and 9. A pack is AN-1 conformant only when all three are present and mutually consistent. Their formats, bounds, exact passage-order requirement, `ivf-flat-v1` validation, dot scoring, and hybrid fusion behavior are defined in [the v3 wire format](../FORMAT-v3.md#7-vector-profiles-and-data).

An embedding profile identifies the model and exact revision, output dimensions and dtype, pooling, normalization, query/document prefixes, and—when a portable runtime is claimed—the runtime library version, weight dtype, and input-token bound. Clients reject a provider whose declared descriptor differs from the pack profile. CPU, WASM, and WebGPU are deployment providers for the same pinned ONNX space; their numerical/ranking parity must be measured, not assumed or encoded as different model identities.

The first golden-path candidate is [`../examples/default-embedding-profile.json`](../examples/default-embedding-profile.json). It pins `mixedbread-ai/mxbai-embed-xsmall-v1` to a specific revision and Transformers.js runtime. It does not become the release default until the real-corpus retrieval evaluation shows that it improves or preserves hybrid recall.

Alternative vector indexes require new AN-1 section-format versions or later numbered extensions. Core and AN-1 readers must not infer HNSW, quantization, sparse retrieval, or another metric from `ivf-flat-v1`.
