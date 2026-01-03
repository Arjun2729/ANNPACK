#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

log() {
  echo "[hygiene] $*"
}

check_tracked() {
  local pattern="$1"
  local found
  found=$(git ls-files -z | tr '\0' '\n' | grep -E "$pattern" || true)
  if [ -n "$found" ]; then
    log "tracked artifacts found for pattern: $pattern"
    echo "$found" | sed 's/^/  - /'
    fail=1
  fi
}

log "checking tracked artifacts"
check_tracked '\\.(annpack|meta\\.jsonl|manifest\\.json|parquet|feather|arrow|npy|npz|wasm)$'
check_tracked '^(dist/|build/|out/|site/|node_modules/|web/.*/dist/|web/.*/node_modules/|\.venv/|venv/|\.emcache/|registry_storage/)'

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
if ! "$PYTHON_BIN" -c "import build" >/dev/null 2>&1; then
  log "missing build module; install with: $PYTHON_BIN -m pip install build"
  exit 1
fi

log "checking sdist contents"
SDIST_DIR="$(mktemp -d /tmp/annpack_hygiene_dist_XXXXXX)"
trap 'rm -rf "$SDIST_DIR"' EXIT

"$PYTHON_BIN" -m build --sdist --no-isolation --outdir "$SDIST_DIR" >/dev/null
SDIST_FILE=$(ls -t "$SDIST_DIR"/*.tar.gz | head -n1 || true)
if [ -z "$SDIST_FILE" ]; then
  log "sdist build failed"
  exit 1
fi

"$PYTHON_BIN" - "$SDIST_FILE" <<'PY'
import sys, tarfile, re
sdist = sys.argv[1]
patterns = [
    r"^dist/", r"^build/", r"^out/", r"^site/", r"^node_modules/", r"^web/.*/dist/", r"^web/.*/node_modules/",
    r"\.venv/", r"/venv/", r"\.emcache/", r"registry_storage/",
    r"\.annpack$", r"\.parquet$", r"\.feather$", r"\.arrow$", r"\.npy$", r"\.npz$", r"\.wasm$",
]
compiled = [re.compile(p) for p in patterns]
found = []
with tarfile.open(sdist, "r:gz") as tf:
    for m in tf.getmembers():
        name = m.name
        for c in compiled:
            if c.search(name):
                found.append(name)
                break
if found:
    print("[hygiene] sdist contains forbidden artifacts:")
    for n in sorted(set(found)):
        print("  -", n)
    sys.exit(1)
PY

if [ "$fail" -ne 0 ]; then
  log "failed"
  exit 1
fi

rm -rf build python/annpack.egg-info

log "PASS repo hygiene"
