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
  echo "[stage1] no Python with setuptools.build_meta found; set PYTHON_BIN to a Python that has it."
  exit 1
fi
export ANNPACK_OFFLINE=1

echo "[stage1] building wheel..."
BUILD_DIST="$(mktemp -d /tmp/annpack_dist_XXXXXX)"
BUILD_ENV="$(mktemp -d /tmp/annpack_build_XXXXXX)"
"$PYTHON_BIN" -m venv --system-site-packages "$BUILD_ENV/venv"
"$BUILD_ENV/venv/bin/python" -m pip wheel "$ROOT" -w "$BUILD_DIST" --no-deps --no-build-isolation
WHEEL=$(ls -t "$BUILD_DIST"/*.whl | head -n1)

echo "[stage1] installing wheel..."
WORK_DIR="$(mktemp -d /tmp/annpack_work_XXXXXX)"
"$PYTHON_BIN" -m venv --system-site-packages "$WORK_DIR/venv"
source "$WORK_DIR/venv/bin/activate"
python -m pip install "$WHEEL"

echo "[stage1] cli help check..."
python "$ROOT/tools/test_cli_help.py"

echo "[stage1] version check..."
annpack --version

cd "$WORK_DIR"
cp "$ROOT/tiny_docs.csv" .
PACK_DIR="$(mktemp -d /tmp/annpack_pack_XXXXXX)"
OUT_PREFIX="$PACK_DIR/pack"

echo "[stage1] building tiny_docs -> $OUT_PREFIX ..."
annpack build --input "$WORK_DIR/tiny_docs.csv" --text-col text --output "$OUT_PREFIX" --lists 256 --device cpu

echo "[stage1] serve + smoke test on $PACK_DIR ..."
SERVE_LOG="$(mktemp /tmp/annpack_stage1_serve.XXXXXX)"
SERVE_PORT="$("$PYTHON_BIN" - <<'PY'
import socket
try:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
    s.close()
except OSError as e:
    raise SystemExit(f"[stage1] failed to allocate port: {e}")
PY
)"
if [ -z "$SERVE_PORT" ]; then
  echo "[stage1] failed to allocate serve port"
  exit 1
fi
PYTHONUNBUFFERED=1 annpack serve "$PACK_DIR" --port "$SERVE_PORT" >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
cleanup_serve() {
  if [ -n "${SERVE_PID:-}" ]; then
    kill "$SERVE_PID" >/dev/null 2>&1 || true
    wait "$SERVE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_serve EXIT
TRIES=0
ACTUAL_PORT="$SERVE_PORT"
until false; do
  if [ -n "$ACTUAL_PORT" ] && curl -sf "http://127.0.0.1:$ACTUAL_PORT/index.html" >/dev/null 2>&1; then
    break
  fi
  TRIES=$((TRIES+1))
  if ! kill -0 "$SERVE_PID" >/dev/null 2>&1; then
    echo "[stage1] serve exited early. Last log lines:"
    tail -n 200 "$SERVE_LOG"
    cleanup_serve
    exit 1
  fi
  if [ "$TRIES" -ge 20 ]; then
    echo "[stage1] server did not start"
    tail -n 200 "$SERVE_LOG"
    cleanup_serve
    exit 1
  fi
  sleep 0.5
done
echo "[stage1] serve ready on port $ACTUAL_PORT"
SMOKE_PORT="$("$PYTHON_BIN" - <<'PY'
import socket
try:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
    s.close()
except OSError as e:
    raise SystemExit(f"[stage1] failed to allocate port: {e}")
PY
)"
if [ -z "$SMOKE_PORT" ]; then
  echo "[stage1] failed to allocate smoke port"
  exit 1
fi
if ! annpack smoke "$PACK_DIR" --port "$SMOKE_PORT"; then
  echo "[stage1] smoke failed. Serve log tail:"
  tail -n 200 "$SERVE_LOG"
  cleanup_serve
  exit 1
fi
cleanup_serve
trap - EXIT

echo "[stage1] deterministic manifest/meta check..."
PACK_DIR2="$(mktemp -d /tmp/annpack_pack_XXXXXX)"
OUT_PREFIX2="$PACK_DIR2/pack"
ANNPACK_DEVICE=cpu annpack build --input "$WORK_DIR/tiny_docs.csv" --text-col text --output "$OUT_PREFIX2" --lists 256
PACK_DIR3="$(mktemp -d /tmp/annpack_pack_XXXXXX)"
OUT_PREFIX3="$PACK_DIR3/pack"
ANNPACK_DEVICE=cpu annpack build --input "$WORK_DIR/tiny_docs.csv" --text-col text --output "$OUT_PREFIX3" --lists 256
diff "$OUT_PREFIX2.manifest.json" "$OUT_PREFIX3.manifest.json"
diff "$OUT_PREFIX2.meta.jsonl" "$OUT_PREFIX3.meta.jsonl"

echo "PASS stage1 acceptance"
