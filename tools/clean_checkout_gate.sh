#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d /tmp/annpack_clean_checkout_XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

log() {
  echo "[clean_checkout] $*"
}

log "cloning repo into $WORK"

git clone --no-hardlinks --depth 1 "$ROOT" "$WORK/repo" >/dev/null

cd "$WORK/repo"
export ANNPACK_OFFLINE=1

log "creating venv for clean checkout"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$PYTHON_BIN" -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install -U pip >/dev/null
python -m pip install -e .[dev] >/dev/null
export PYTHON_BIN="$(command -v python)"

log "running stage_all from clean checkout"

bash tools/stage_all.sh

log "PASS clean checkout"
