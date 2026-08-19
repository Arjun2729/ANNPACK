# Canonical ANN-1 execution

A profile that names a model does not determine an embedding. Native ONNX
Runtime selects kernels from the host instruction set, so the same model
produces different vectors on arm64 and x64 — and no choice of 8-bit
quantization fixes that, including the U8U8 variant the runtime documents as
free of the saturation usually blamed for it.

Pinning the execution machine does fix it:

```text
model            model_quantized.onnx          sha256 952f996d…
execution        ort-wasm-simd-threaded.wasm   sha256 f061472c…
max_tokens       512
batch            1
numThreads       1
output           token_embeddings
pooling          attention-mask-weighted mean
normalization    L2
```

```text
                  passage vectors      query vectors
Mac Node WASM     ef8c5aab…08a9        a6886ff2…7b66
Linux Node WASM   ef8c5aab…08a9        a6886ff2…7b66
Chrome WASM       ef8c5aab…08a9        —
```

Byte-identical, not approximate.

`batch = 1` is semantics, not tuning: an embedding must depend on its input and
not on what was embedded beside it. On one host and runtime, batch=32 against
batch=1 moves 15 of 168 passages below 0.99 self-cosine — a larger perturbation
than the cross-architecture difference it was masking.

## Files

| | |
|---|---|
| `embed-canonical.mjs` | the reference implementation; emits the digest it is the authority on |
| `check-canonical-parity.py` | asserts model, runtime and vector digests against `canonical-pins.json` |
| `canonical-pins.json` | the pinned chain |
| `browser-parity.html` | third leg: the same computation in a browser |

## Running

```bash
python3 evals/canonical/check-canonical-parity.py
```

Requires `npm install --prefix evals` and the corpus reproduced by
[`../corpora/reproduce-okf-hard-negatives.sh`](../corpora/reproduce-okf-hard-negatives.sh).

For the browser leg, stage the model, tokenizer and `dist/` runtime beside
`browser-parity.html`, serve the directory, and load it headless. The page sets
`allowRemoteModels = false` so it cannot silently fetch different bytes.

## What this does not establish

That canonical WASM is universally optimal, or that it should replace native
execution. Native remains available as an accelerated provider whose parity is
measured against this reference rather than assumed. Bulk embedding is ~5.6x
slower; a single query costs ~6ms against ~0.7ms.
