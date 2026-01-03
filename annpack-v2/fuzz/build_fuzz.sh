#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${CC:-clang}"
CFLAGS=("-O1" "-g" "-fsanitize=fuzzer,address" "-DANN_FUZZ" "-I${ROOT}/include" "-I${ROOT}")

"$CC" "${CFLAGS[@]}" \
  "${ROOT}/src/ann_engine.c" \
  "${ROOT}/tests/io_posix.c" \
  "${ROOT}/fuzz/fuzz_annpack.c" \
  -o "${ROOT}/fuzz/fuzz_annpack"

echo "[fuzz] built ${ROOT}/fuzz/fuzz_annpack"
