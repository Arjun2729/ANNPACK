# Browser parity probe

Runs the pinned q8 model through `onnxruntime-web` in a browser, against the
same model, tokenizer and `.wasm` bytes the Node runs use, and prints a SHA-256
over the resulting vector payload.

It exists to close the third leg of a portability claim. Native ONNX Runtime
selects kernels from the host instruction set, so the same model yields
different vectors on arm64 and x64 — and no choice of 8-bit quantization fixed
that. Pinning the execution machine instead does:

    Mac Node WASM     7672ae503d5929d66c635731cf505ad0a87148b58db96035c27565e952b18648
    Linux Node WASM   7672ae503d5929d66c635731cf505ad0a87148b58db96035c27565e952b18648
    Chrome WASM       7672ae503d5929d66c635731cf505ad0a87148b58db96035c27565e952b18648

    model_quantized.onnx         952f996d8cf46c31…
    ort-wasm-simd-threaded.wasm  f061472c6e77d6d5…
    numThreads                   1

Threading is pinned because parallel reductions vary in order. Fixed-width WASM
SIMD is deterministic by specification; relaxed SIMD is not, and ORT's
`wasm_simd` kernels do not use it.

## Running it

```bash
# Stage identical bytes, then serve and drive a headless browser.
python3 -m http.server 8899 --directory <staged-dir>
chrome --headless=new --disable-gpu --virtual-time-budget=300000 \
  --dump-dom http://127.0.0.1:8899/index.html
```

The page requires the model, tokenizer and `dist/` runtime staged beside it;
`allowRemoteModels` is off so it cannot silently fetch different bytes from a
hub.

## What this does not establish

The Node path reimplements mean pooling and normalization outside
Transformers.js, because its Node build refuses `device: 'wasm'`. WASM-to-WASM
comparison is unaffected — every side runs identical code — but WASM-to-native
is not a like-for-like comparison of the same implementation.
