#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUSTUP_BIN=${RUSTUP_BIN:-/opt/homebrew/opt/rustup/bin}
WASM_BINDGEN=${WASM_BINDGEN:-$HOME/.cargo/bin/wasm-bindgen}

cd "$ROOT"
# Without this, rustc bakes absolute source paths into panic strings, so the
# published .wasm ships the builder's home directory to every visitor of the
# demo page. Remap the repository and cargo registry to stable placeholders.
REMAP="--remap-path-prefix=$ROOT=/annpack --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"
PATH="$RUSTUP_BIN:$PATH" RUSTFLAGS="${RUSTFLAGS:-} $REMAP" cargo build \
  --lib \
  --target wasm32-unknown-unknown \
  --no-default-features \
  --features wasm \
  --release

"$WASM_BINDGEN" \
  --target web \
  --out-dir web/pkg \
  --out-name adyar \
  target/wasm32-unknown-unknown/release/adyar.wasm

echo "Built web/pkg/adyar_bg.wasm and browser bindings"

