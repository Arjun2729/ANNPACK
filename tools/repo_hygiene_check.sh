#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_BYTES=$((5 * 1024 * 1024))
fail=0

log() {
  echo "[hygiene_check] $*"
}

forbidden_patterns=(
  '\.(annpack|meta\.jsonl|manifest\.json|parquet|feather|arrow|npy|npz|wasm)$'
  '^(dist/|build/|out/|site/|node_modules/|web/.*/dist/|web/.*/node_modules/|\.venv/|venv/|\.emcache/|registry_storage/)'
)

log "checking forbidden tracked patterns"
for pat in "${forbidden_patterns[@]}"; do
  found=$(git ls-files -z | tr '\0' '\n' | grep -E "$pat" || true)
  if [ -n "$found" ]; then
    log "tracked forbidden files for pattern: $pat"
    echo "$found" | sed 's/^/  - /'
    fail=1
  fi
done

log "checking tracked file sizes > ${MAX_BYTES} bytes"
git ls-files -z | while IFS= read -r -d '' path; do
  if [ -f "$path" ]; then
    size=$(stat -f "%z" "$path" 2>/dev/null || stat -c "%s" "$path" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_BYTES" ]; then
      log "tracked file too large: $path ($size bytes)"
      fail=1
    fi
  fi
done

if [ "$fail" -ne 0 ]; then
  log "FAILED"
  exit 1
fi

log "PASS repo hygiene check"
