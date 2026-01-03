#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/examples/frontend_static_demo/out"
CSV="$OUT_DIR/tiny_docs.csv"

mkdir -p "$OUT_DIR"

cat > "$CSV" <<'CSV'
id,text
0,hello from annpack
1,paris is france
CSV

export ANNPACK_OFFLINE=1

annpack build --input "$CSV" --text-col text --output "$OUT_DIR/pack" --lists 4
echo "wrote $OUT_DIR/pack.manifest.json"
