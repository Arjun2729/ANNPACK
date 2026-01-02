#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-fiqa}"
MANIFEST="${2:-$ROOT/data/$DATASET/${DATASET}.manifest.json}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  echo "Build it first, e.g.: python tools/build_beir.py --dataset fiqa --outdir data/fiqa --shard-size 50000"
  exit 2
fi

python "$ROOT/tools/fidelity_gate.py" --manifest "$MANIFEST" --queries "$DATASET" --sample 200 --k 10 --seed 42
