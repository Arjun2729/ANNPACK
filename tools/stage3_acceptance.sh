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
  echo "[stage3] no Python with setuptools.build_meta found; set PYTHON_BIN to a Python that has it."
  exit 1
fi
export ANNPACK_OFFLINE=1

echo "[stage3] building wheel..."
BUILD_DIST="$(mktemp -d /tmp/annpack_stage3_dist_XXXXXX)"
BUILD_ENV="$(mktemp -d /tmp/annpack_stage3_build_XXXXXX)"
"$PYTHON_BIN" -m venv --system-site-packages "$BUILD_ENV/venv"
"$BUILD_ENV/venv/bin/python" -m pip wheel "$ROOT" -w "$BUILD_DIST" --no-deps --no-build-isolation
WHEEL=$(ls -t "$BUILD_DIST"/*.whl | head -n1)

echo "[stage3] installing wheel..."
WORK_DIR="$(mktemp -d /tmp/annpack_stage3_work_XXXXXX)"
"$PYTHON_BIN" -m venv --system-site-packages "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"
python -m pip install "$WHEEL"

cd "$WORK_DIR"
cat > tiny_docs.csv <<'CSV'
id,text
0,hello
1,paris is france
CSV

PACKSET_DIR="$(mktemp -d /tmp/annpack_packset_XXXXXX)"
DELTA1_DIR="$PACKSET_DIR/deltas/0001.delta"
mkdir -p "$DELTA1_DIR"
export PACKSET_DIR DELTA1_DIR

echo "[stage3] building base packset..."
python - <<'PY'
import os
from annpack.packset import build_packset_base
build_packset_base(
    input_csv="tiny_docs.csv",
    packset_dir=os.environ["PACKSET_DIR"],
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
PY

cat > delta_add.csv <<'CSV'
id,text
0,hello updated
2,delta add
CSV

echo "[stage3] building delta..."
python - <<'PY'
import os
from annpack.packset import build_delta, update_packset_manifest

info = build_delta(
    base_dir=os.path.join(os.environ["PACKSET_DIR"], "base"),
    add_csv="delta_add.csv",
    delete_ids=[1],
    out_delta_dir=os.environ["DELTA1_DIR"],
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
update_packset_manifest(os.environ["PACKSET_DIR"], os.environ["DELTA1_DIR"], seq=1)
PY

echo "[stage3] packset query checks..."
python - <<'PY'
import os
from annpack.api import open_pack

pack = open_pack(os.environ["PACKSET_DIR"])
hits_add = pack.search("delta add", top_k=3)
hits_update = pack.search("hello updated", top_k=3)
hits_deleted = pack.search("paris is france", top_k=3)
pack.close()

if not any(h["id"] == 2 for h in hits_add):
    raise SystemExit("delta add not found")
if not any(h["id"] == 0 for h in hits_update):
    raise SystemExit("updated id 0 not found")
updated = next((h for h in hits_update if h["id"] == 0), None)
if not updated or not updated.get("meta") or updated["meta"].get("text") != "hello updated":
    raise SystemExit("updated meta not returned for id 0")
if any(h["id"] == 1 for h in hits_deleted):
    raise SystemExit("deleted id 1 returned in results")
print("OK packset search")
PY

echo "[stage3] delta determinism..."
DELTA_A="$(mktemp -d /tmp/annpack_deltaA_XXXXXX)"
DELTA_B="$(mktemp -d /tmp/annpack_deltaB_XXXXXX)"
export DELTA_A DELTA_B
python - <<'PY'
import os
from annpack.packset import build_delta
build_delta(
    base_dir=os.path.join(os.environ["PACKSET_DIR"], "base"),
    add_csv="delta_add.csv",
    delete_ids=[1],
    out_delta_dir=os.environ["DELTA_A"],
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
build_delta(
    base_dir=os.path.join(os.environ["PACKSET_DIR"], "base"),
    add_csv="delta_add.csv",
    delete_ids=[1],
    out_delta_dir=os.environ["DELTA_B"],
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
PY
diff "$DELTA_A/pack.annpack" "$DELTA_B/pack.annpack"
diff "$DELTA_A/pack.meta.jsonl" "$DELTA_B/pack.meta.jsonl"
diff "$DELTA_A/tombstones.jsonl" "$DELTA_B/tombstones.jsonl"
diff "$DELTA_A/delta.manifest.json" "$DELTA_B/delta.manifest.json"

echo "[stage3] packset manifest determinism..."
PACKSET_DIR2="$(mktemp -d /tmp/annpack_packset_XXXXXX)"
DELTA2_DIR="$PACKSET_DIR2/deltas/0001.delta"
mkdir -p "$DELTA2_DIR"
export PACKSET_DIR2 DELTA2_DIR
python - <<'PY'
import os
from annpack.packset import build_packset_base, build_delta, update_packset_manifest

build_packset_base(
    input_csv="tiny_docs.csv",
    packset_dir=os.environ["PACKSET_DIR2"],
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
build_delta(
    base_dir=os.path.join(os.environ["PACKSET_DIR2"], "base"),
    add_csv="delta_add.csv",
    delete_ids=[1],
    out_delta_dir=os.environ["DELTA2_DIR"],
    text_col="text",
    id_col="id",
    lists=4,
    seed=123,
    offline=True,
)
update_packset_manifest(os.environ["PACKSET_DIR2"], os.environ["DELTA2_DIR"], seq=1)
PY
diff "$PACKSET_DIR/pack.manifest.json" "$PACKSET_DIR2/pack.manifest.json"

echo "PASS stage3 acceptance"
