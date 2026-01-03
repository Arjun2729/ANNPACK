#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/examples/frontend_static_demo/out"
PORT="${PORT:-9000}"

if [ ! -f "$OUT_DIR/pack.manifest.json" ]; then
  echo "pack.manifest.json not found; run build_pack.sh first"
  exit 1
fi

echo "Serving $OUT_DIR on http://127.0.0.1:$PORT/"
echo "Manifest URL: http://127.0.0.1:$PORT/pack.manifest.json"
python3 -m http.server "$PORT" --directory "$OUT_DIR"
