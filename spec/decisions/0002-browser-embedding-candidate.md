# ADR-0002: Use mxbai-embed-xsmall-v1 as the browser candidate

Status: candidate accepted; promotion to default is blocked on real-corpus evaluation.

## Decision

Use `mixedbread-ai/mxbai-embed-xsmall-v1` as the first golden-path browser embedding candidate. Pin model revision `e6ac24e5d6efb8782b59de1647b3ececb4ece94e`, Transformers.js `3.8.1`, q8 weights, mean pooling, normalized float32 output, 384 dimensions, and a 4096-token bound. That revision's q8/quantized ONNX artifact is 24,448,010 bytes with SHA-256 `952f996d8cf46c311ee8654a750fa942b71c8b94aabe69d043dbb2bcaff5528e`.

The execution provider is not part of the embedding-space ID. Offline passage generation uses the Node CPU provider while browsers may use WASM or WebGPU over the same pinned weights. Cross-provider cosine and ranking parity are release measurements; provider differences must not be hidden by calling them different semantic models.

## Why

- The publisher describes a 24.1M-parameter, 384-dimensional, 4096-context English retrieval model under Apache-2.0: <https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1>.
- The official Transformers.js WebGPU embedding guide uses this model as its text-embedding example: <https://huggingface.co/docs/transformers.js/en/guides/webgpu>.
- Transformers.js documents exact model-revision loading, feature-extraction pooling/normalization, and browser quantization: <https://huggingface.co/docs/transformers.js/pipelines> and <https://huggingface.co/docs/transformers.js/guides/dtypes>.

EmbeddingGemma is not the first browser default. Its multilingual quality and flexible dimensions are attractive, but Google's current model overview lists roughly 308M parameters: <https://ai.google.dev/gemma/docs/embeddinggemma>. That is excessive cold-download and initialization risk before Adyar has proven its retrieval evaluation and adoption path.

## Promotion gate

The candidate becomes the blessed default only after a pinned real documentation corpus and 50–100 human-adjudicated queries show acceptable vector and hybrid recall, and an actual browser cold-load test shows acceptable download, initialization, and query latency. If it loses, the descriptor remains supported and another profile may replace it without changing Core.
