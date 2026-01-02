#!/usr/bin/env bash
set -e
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/web/wasm_dist"

if ! command -v emcc >/dev/null 2>&1; then
  echo "[wasm] emcc not found. Install emsdk and activate it first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

SRC_MAIN="$ROOT_DIR/main.c"
SRC_IO="$ROOT_DIR/io_wasm.c"

if [[ ! -f "$SRC_MAIN" || ! -f "$SRC_IO" ]]; then
  echo "[wasm] missing sources: $SRC_MAIN or $SRC_IO" >&2
  exit 1
fi

echo "[wasm] building into $OUT_DIR"

emcc -O3 \
  "$SRC_MAIN" "$SRC_IO" \
  -I"$ROOT_DIR" \
  -o "$OUT_DIR/annpack.js" \
  -s ASYNCIFY \
  -s FETCH=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s "EXPORTED_FUNCTIONS=['_malloc','_free','_ann_load_index','_ann_search','_ann_result_size_bytes']" \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32','HEAPU8']" \
  -s NO_EXIT_RUNTIME=1

echo "[wasm] wrote: $OUT_DIR/annpack.js and $OUT_DIR/annpack.wasm"
