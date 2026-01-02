#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

version=$(python - <<'PY'
import tomllib
from pathlib import Path
p = Path("pyproject.toml")
with p.open("rb") as f:
    data = tomllib.load(f)
print(data["project"]["version"])
PY
)

echo "[release] stage_all gate"
bash tools/stage_all.sh

echo "[release] building dist artifacts"
python -m build

if python -m twine --version >/dev/null 2>&1; then
  echo "[release] twine check"
  python -m twine check dist/*
else
  echo "[release] twine not installed (pip install twine)"
fi

tmp_env="$(mktemp -d)"
trap 'rm -rf "$tmp_env"' EXIT
python -m venv "$tmp_env/venv"
source "$tmp_env/venv/bin/activate"
pip install -U pip >/dev/null
pip install dist/*.whl >/dev/null
annpack --version
deactivate

echo "[release] tag: git tag v${version}"
echo "[release] push tags: git push --tags"
echo "[release] upload: TWINE_USERNAME=__token__ TWINE_PASSWORD=*** python -m twine upload dist/*"
