# ANN-1: Vector retrieval

Status: implemented draft. Requires ANNPack Core v1.0-draft.

ANN-1 consists of optional section types 7, 8, and 9. A pack is ANN-1 conformant only when all three are present and mutually consistent. Their formats, bounds, exact passage-order requirement, `ivf-flat-v1` validation, dot scoring, and hybrid fusion behavior are defined in [the v3 wire format](../FORMAT-v3.md#7-vector-profiles-and-data).

An embedding profile identifies the model and exact revision, output dimensions and dtype, pooling, normalization, query/document prefixes, and—when a portable runtime is claimed—the runtime library version, weight dtype, and input-token bound. Clients reject a provider whose declared descriptor differs from the pack profile. CPU, WASM, and WebGPU are deployment providers for the same pinned ONNX space; their numerical/ranking parity must be measured, not assumed or encoded as different model identities.

The first golden-path candidate is [`../examples/default-embedding-profile.json`](../examples/default-embedding-profile.json). It pins `mixedbread-ai/mxbai-embed-xsmall-v1` to a specific revision and Transformers.js runtime. It does not become the release default until the real-corpus retrieval evaluation shows that it improves or preserves hybrid recall.

## Canonical execution

An embedding profile that names model semantics does not determine an embedding.
Measured on an identical corpus — same content digest, same passage order —
macOS arm64 and Linux x64 produce different vectors under every 8-bit variant
tested:

| weights | min self-cosine | max dimension delta |
|---|---|---|
| q8 (`model_quantized.onnx`) | 0.995439 | 1.57e-02 |
| int8 (`model_int8.onnx`, byte-identical output to q8) | 0.995439 | 1.57e-02 |
| uint8 (`model_uint8.onnx`) | 0.993729 | 2.00e-02 |
| fp32 (`model.onnx`) | 0.999999 | 1.10e-07 |

ONNX Runtime documents U8S8 saturation via `VPMADDUBSW` on x86-64 with
AVX2/AVX512 but without VNNI, and recommends U8U8 as free of it. U8U8 avoids
that instruction and diverges slightly more, so saturation is not the
explanation. What the data supports is broader: native runtimes select kernels
from the host instruction set, and architecture-specific 8-bit kernels do not
reproduce a vector across architectures.

A profile therefore identifies a **canonical execution machine**, not only a
model. The current canonical candidate:

```text
model            model_quantized.onnx   sha256 952f996d…   23.3 MB
execution        ort-wasm-simd-threaded.wasm   sha256 f061472c…
max_tokens       512
batch            1
numThreads       1
output           token_embeddings
pooling          attention-mask-weighted mean
normalization    L2
```

Under that machine the embedding reproduces byte-identically, not
approximately:

```text
                  passage vectors      query vectors
Mac Node WASM     ef8c5aab…08a9        a6886ff2…7b66
Linux Node WASM   ef8c5aab…08a9        a6886ff2…7b66
Chrome WASM       ef8c5aab…08a9        —
```

**`batch = 1` is canonical semantics, not a performance setting.** An
embedding must depend on its input and not on what was embedded beside it.
Batch composition is a real input: on one host and one runtime, batch=32
against batch=1 moves 15 of 168 passages below 0.99 self-cosine and changes 7
query outcomes — a larger perturbation than the cross-architecture runtime
difference it was hiding behind.

`max_tokens = 512` is the tokenizer's `model_max_length`. Inputs beyond it must
be truncated; feeding longer sequences is outside the profile.

### Accelerated providers

Native CPU, WebGPU and future accelerated paths remain available. They are
implementations *relative to* the canonical reference, and their numerical and
ranking parity must be measured against it rather than assumed. Measured on the
63-query corpus, vector recall@5:

| | recall@5 | MRR@5 | technical | hard-negative |
|---|---|---|---|---|
| native q8, batch=32 | 46/63 | 0.591 | 22/28 | 24/35 |
| native q8, batch=1 | 45/63 | 0.590 | 21/28 | 24/35 |
| canonical WASM, batch=1 | 44/63 | 0.582 | 21/28 | 23/35 |

Decomposed, the two steps are not comparable in kind:

| | query flips | min self-cosine | passages below 0.99 |
|---|---|---|---|
| batch=32 → batch=1 | 7 | 0.987674 | 15/168 |
| native → WASM, batch=1 | 1 | 0.996201 | 0/168 |

Moving to the canonical runtime changes one hit/miss result and leaves every
passage above 0.99 cosine. Batch=32 is not ground truth, so the batch=1 step is
a change in semantics rather than a loss of quality. This corpus is small,
machine-authored and unadjudicated; single-query differences here do not
support product-quality claims in either direction.

Cost, measured over 2000 passages:

```text
bulk embedding, unbatched    native 41.7s     wasm 235.1s    ~5.6x
model load                   native  127ms    wasm   209ms
single query                 native    0.7ms  wasm     6.3ms   p95 8.6ms
```

Batching is slower for both on variable-length text, because padding to the
longest item wastes work. The slowdown is concentrated in offline bulk
embedding; the interactive query-embedding penalty is a few milliseconds in
absolute terms.

Not established, and not claimed: that canonical WASM is universally optimal, or
that fp32 should become the default. `model.onnx` is 91.5 MB against 23.3 MB,
and a candidate becomes the default when a real-corpus evaluation says so.

Alternative vector indexes require new ANN-1 section-format versions or later numbered extensions. Core and ANN-1 readers must not infer HNSW, quantization, sparse retrieval, or another metric from `ivf-flat-v1`.
