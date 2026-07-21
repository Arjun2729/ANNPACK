#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUSTUP_BIN=${RUSTUP_BIN:-/opt/homebrew/opt/rustup/bin}
WASM_BINDGEN=${WASM_BINDGEN:-$HOME/.cargo/bin/wasm-bindgen}

cd "$ROOT"
PATH="$RUSTUP_BIN:$PATH" cargo build \
  --lib \
  --target wasm32-unknown-unknown \
  --no-default-features \
  --features wasm \
  --release

"$WASM_BINDGEN" \
  --target web \
  --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/annpack.wasm

echo "Built web/pkg/annpack_bg.wasm and browser bindings"

