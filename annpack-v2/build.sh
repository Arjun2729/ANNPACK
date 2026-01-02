#!/usr/bin/env bash
set -euo pipefail

EMCC=${EMCC:-emcc}
EMCACHE=${EMCACHE:-.emcache}
if [[ "$EMCACHE" = /* ]]; then
  EM_CACHE="$EMCACHE"
else
  EM_CACHE="$(pwd)/$EMCACHE"
fi
mkdir -p "$EMCACHE"

EM_CACHE="$EM_CACHE" \
$EMCC -O3 \
  src/ann_engine.c io_wasm.c \
  -Iinclude -I. \
  -o annpack.js \
  -s ASYNCIFY -s ASYNCIFY_IMPORTS='["_js_fetch_range_blocking"]' \
  -s FETCH=1 \
  -s "EXPORTED_FUNCTIONS=['_malloc','_free','_ann_load_index','_ann_free_index','_ann_search','_ann_result_size_bytes','_ann_last_error','_ann_last_scan_count','_ann_get_n_lists','_ann_set_probe','_ann_set_max_k','_ann_set_max_scan']" \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32','HEAPU8']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s NO_EXIT_RUNTIME=1

# Verify exports are present in generated JS
for sym in _ann_load_index _ann_free_index _ann_search _ann_result_size_bytes _ann_last_error _ann_last_scan_count _ann_get_n_lists _ann_set_probe _ann_set_max_k _ann_set_max_scan _js_fetch_range_blocking; do
  if ! grep -q "$sym" annpack.js; then
    echo "Missing export: $sym" >&2
    exit 1
  fi
done
