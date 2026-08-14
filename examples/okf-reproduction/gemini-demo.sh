#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <pack.annpack> [question]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ADYAR=${ADYAR:-${ANNPACK:-$ROOT/target/release/adyar}}
PACK=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
QUESTION=${2:-Which BigQuery table contains ecommerce events, and what exact knowledge artifact supports the answer?}
WORK=${WORK:-$ROOT/target/gemini-okf-demo}

command -v gemini >/dev/null || {
  echo "Gemini CLI is not installed" >&2
  exit 1
}
if [[ ! -x "$ADYAR" ]]; then
  echo "ANNPack release binary not found at $ADYAR" >&2
  exit 1
fi

mkdir -p "$WORK"
"$ADYAR" integrate gemini "$PACK" \
  --output "$WORK/.gemini/settings.json" \
  --force \
  --json

cd "$WORK"
gemini --prompt \
  "Use the ANNPack knowledge tools before answering. $QUESTION Include the pack root, passage hash, source revision, and canonical URL returned by the tool." \
  --output-format json
