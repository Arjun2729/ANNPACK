#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

VERSION="$($PYTHON_BIN - <<'PY'
import tomllib
from pathlib import Path
p = Path('pyproject.toml')
with p.open('rb') as f:
    data = tomllib.load(f)
print(data['project']['version'])
PY
)"

if git tag -l "v$VERSION" | grep -q .; then
  echo "[release_check] tag v$VERSION already exists"
  exit 1
fi

if ls dist/*"$VERSION"* >/dev/null 2>&1; then
  echo "[release_check] dist contains artifacts for $VERSION; remove dist/ or bump version"
  exit 1
fi

echo "[release_check] ok for version $VERSION"
