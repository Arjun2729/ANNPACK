#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${PYTHON_BIN:-}" ] && [ -n "${pythonLocation:-}" ] && [ -x "${pythonLocation}/bin/python" ]; then
  PYTHON_BIN="${pythonLocation}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
BUILD_CWD="$(mktemp -d /tmp/annpack_stage4_buildcwd_XXXXXX)"
if ! "$PYTHON_BIN" -c "import setuptools.build_meta" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install -U pip setuptools >/dev/null 2>&1 || true
fi
if ! "$PYTHON_BIN" -c "import setuptools.build_meta" >/dev/null 2>&1; then
  echo "[stage4] missing setuptools.build_meta; install with: $PYTHON_BIN -m pip install setuptools"
  exit 1
fi
if ! (cd "$BUILD_CWD" && "$PYTHON_BIN" -c "import build") >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -U build >/dev/null 2>&1 || true
fi
if ! (cd "$BUILD_CWD" && "$PYTHON_BIN" -c "import build") >/dev/null 2>&1; then
  echo "[stage4] missing build module; install with: $PYTHON_BIN -m pip install build"
  exit 1
fi
if ! (cd "$BUILD_CWD" && "$PYTHON_BIN" -c "import twine") >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -U twine >/dev/null 2>&1 || true
fi
if ! (cd "$BUILD_CWD" && "$PYTHON_BIN" -c "import twine") >/dev/null 2>&1; then
  echo "[stage4] missing twine module; install with: $PYTHON_BIN -m pip install twine"
  exit 1
fi

export ANNPACK_OFFLINE=1

echo "[stage4] building dist artifacts..."
BUILD_DIST="$(mktemp -d /tmp/annpack_stage4_dist_XXXXXX)"
(cd "$BUILD_CWD" && "$PYTHON_BIN" -m build --sdist --wheel --no-isolation --outdir "$BUILD_DIST" "$ROOT")
WHEEL=$(ls -t "$BUILD_DIST"/*.whl | head -n1 || true)
SDIST=$(ls -t "$BUILD_DIST"/*.tar.gz | head -n1 || true)
if [ -z "$WHEEL" ] || [ -z "$SDIST" ]; then
  echo "[stage4] missing wheel or sdist in $BUILD_DIST"
  exit 1
fi

echo "[stage4] twine check..."
"$PYTHON_BIN" -m twine check "$BUILD_DIST"/*

run_smoke() {
  local label="$1"
  local artifact="$2"
  local work_dir
  work_dir="$(mktemp -d /tmp/annpack_stage4_${label}_XXXXXX)"
  "$PYTHON_BIN" -m venv --system-site-packages "$work_dir/venv"
  source "$work_dir/venv/bin/activate"
  if [[ "$artifact" == *.tar.gz ]]; then
    python -m pip install --no-deps --no-build-isolation "$artifact"
  else
    python -m pip install --no-deps "$artifact"
  fi
  python - <<'PY'
missing = []
for name in ("polars", "faiss", "numpy"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing base deps in system site-packages: " + ", ".join(missing))
PY
  if [ "${ANNPACK_EMBED_CHECK:-0}" = "1" ]; then
    python - <<'PY'
missing = []
for name in ("sentence_transformers", "torch", "datasets"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing embed deps for check: " + ", ".join(missing))
print("OK embed imports")
PY
  fi
  annpack --help >/dev/null
  annpack --version >/dev/null
  python -c "import annpack; print(annpack.__version__)" >/dev/null

  cd "$work_dir"
  cat > tiny_docs.csv <<'CSV'
id,text
0,hello
1,paris is france
CSV
  out_dir="$work_dir/out"
  out_prefix="$out_dir/pack"
  annpack build --input "$work_dir/tiny_docs.csv" --text-col text --output "$out_prefix" --lists 4 --device cpu
  PACK_DIR="$out_dir" python - <<'PY'
import os
from annpack.api import open_pack
pack = open_pack(os.environ["PACK_DIR"])
hits = pack.search("hello", top_k=2)
pack.close()
if not hits:
    raise SystemExit("empty results from search")
PY
  deactivate
}

echo "[stage4] wheel install smoke..."
run_smoke wheel "$WHEEL"

echo "[stage4] sdist install smoke..."
run_smoke sdist "$SDIST"

echo "PASS stage4 acceptance"
