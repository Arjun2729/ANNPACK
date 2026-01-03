#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d /tmp/annpack_clean_checkout_XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

log() {
  echo "[clean_checkout] $*"
}

log "cloning repo into $WORK"

git clone --no-hardlinks --depth 1 "file://$ROOT" "$WORK/repo" >/dev/null

cd "$WORK/repo"
export ANNPACK_OFFLINE=1

log "creating venv for clean checkout"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$PYTHON_BIN" -m venv --system-site-packages "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install -U pip >/dev/null 2>&1 || true
if ! python -m pip install -e .[dev] >/dev/null 2>&1; then
  echo "[clean_checkout] pip install -e .[dev] failed; retrying without deps and isolation"
  python -m pip install -e . --no-deps --no-build-isolation >/dev/null 2>&1 || true
fi
if ! python -c "import annpack" >/dev/null 2>&1; then
  echo "[clean_checkout] install failed (likely no network to fetch build deps). Run with network or preinstall setuptools/build."
  exit 1
fi
export PYTHON_BIN="$(command -v python)"

log "running stage_all from clean checkout"

bash tools/stage_all.sh

log "PASS clean checkout"
