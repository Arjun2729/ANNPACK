#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
if ! "$PYTHON_BIN" -c "import setuptools.build_meta" >/dev/null 2>&1; then
  for cand in /opt/homebrew/Caskroom/miniforge/base/bin/python /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3 /usr/bin/python3; do
    if [ -x "$cand" ] && "$cand" -c "import setuptools.build_meta" >/dev/null 2>&1; then
      PYTHON_BIN="$cand"
      break
    fi
  done
fi
if ! "$PYTHON_BIN" -c "import setuptools.build_meta" >/dev/null 2>&1; then
  echo "[stage2] no Python with setuptools.build_meta found; set PYTHON_BIN to a Python that has it."
  exit 1
fi
export ANNPACK_OFFLINE=1

echo "[stage2] building wheel..."
BUILD_DIST="$(mktemp -d /tmp/annpack_stage2_dist_XXXXXX)"
BUILD_ENV="$(mktemp -d /tmp/annpack_stage2_build_XXXXXX)"
"$PYTHON_BIN" -m venv --system-site-packages "$BUILD_ENV/venv"
"$BUILD_ENV/venv/bin/python" -m pip wheel "$ROOT" -w "$BUILD_DIST" --no-deps --no-build-isolation
WHEEL=$(ls -t "$BUILD_DIST"/*.whl | head -n1)

echo "[stage2] installing wheel..."
WORK_DIR="$(mktemp -d /tmp/annpack_stage2_work_XXXXXX)"
"$PYTHON_BIN" -m venv --system-site-packages "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"
python -m pip install "$WHEEL"

echo "[stage2] cli checks..."
annpack --help >/dev/null
annpack --version

cd "$WORK_DIR"
cat > tiny_docs.csv <<'CSV'
id,text
0,hello
1,paris is france
CSV

PACK_DIR1="$(mktemp -d /tmp/annpack_stage2_pack_XXXXXX)"
PACK_DIR2="$(mktemp -d /tmp/annpack_stage2_pack_XXXXXX)"
export PACK_DIR1 PACK_DIR2

echo "[stage2] building pack (1)..."
python - <<'PY'
import os
from annpack.api import build_pack
build_pack("tiny_docs.csv", output_dir=os.environ["PACK_DIR1"], text_col="text", id_col="id", lists=4, seed=123, offline=True)
PY

echo "[stage2] building pack (2)..."
python - <<'PY'
import os
from annpack.api import build_pack
build_pack("tiny_docs.csv", output_dir=os.environ["PACK_DIR2"], text_col="text", id_col="id", lists=4, seed=123, offline=True)
PY

diff "$PACK_DIR1/pack.manifest.json" "$PACK_DIR2/pack.manifest.json"
diff "$PACK_DIR1/pack.meta.jsonl" "$PACK_DIR2/pack.meta.jsonl"
diff "$PACK_DIR1/pack.annpack" "$PACK_DIR2/pack.annpack"

echo "[stage2] api search..."
python - <<'PY'
import os
from annpack.api import open_pack
pack = open_pack(os.environ["PACK_DIR1"])
results = pack.search("hello", top_k=2)
pack.close()
if not isinstance(results, list) or not results:
    raise SystemExit("Empty results from pack.search")
for row in results:
    if "id" not in row or "score" not in row or "shard" not in row:
        raise SystemExit("Result missing required fields")
print("OK api search")
PY

echo "PASS stage2 acceptance"
