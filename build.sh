#!/usr/bin/env bash
set -euo pipefail

EMCC_BIN=${EMCC:-emcc}

$EMCC_BIN -O3 src/ann_engine.c src/io_wasm.c -Iinclude -o annpack.js \
  -s ASYNCIFY=1 \
  -s FETCH=1 \
  -s "EXPORTED_FUNCTIONS=['_malloc','_free','_ann_load_index','_ann_search','_ann_result_size_bytes']" \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32','HEAPU8']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s NO_EXIT_RUNTIME=1

echo "Built annpack.js with Emscripten"
