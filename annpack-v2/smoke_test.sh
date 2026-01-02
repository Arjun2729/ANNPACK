#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fail() { echo "FAIL: $1" >&2; exit 1; }
pass() { echo "PASS: $1"; }

command -v emcc >/dev/null 2>&1 || fail "emcc not found in PATH"
command -v curl >/dev/null 2>&1 || fail "curl not found in PATH"
command -v xxd  >/dev/null 2>&1 || fail "xxd not found in PATH"
command -v python3 >/dev/null 2>&1 || fail "python3 not found in PATH"
command -v cc >/dev/null 2>&1 || fail "cc (clang) not found in PATH"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
if [[ -z "$PORT" ]]; then
  if PORT=$(python3 - <<'PY'
import socket
try:
    s = socket.socket()
    s.bind(("", 0))
    print(s.getsockname()[1])
    s.close()
except Exception:
    raise SystemExit(1)
PY
); then
    :
  else
    PORT=8765
    echo "[warn] Could not auto-select free port; falling back to ${PORT}" >&2
  fi
fi

TARGET=""
for cand in wikipedia_en_50k.annpack generic_index.annpack; do
  if [[ -f "$cand" ]]; then TARGET="$cand"; break; fi
done
[[ -n "$TARGET" ]] || fail "No .annpack file found (expected wikipedia_en_50k.annpack or generic_index.annpack)"

echo "[test] Rebuilding wasm..."
./build.sh >/tmp/annpack_build.log 2>&1 || (cat /tmp/annpack_build.log; fail "build.sh failed")

for sym in _ann_load_index _ann_free_index _ann_search _ann_result_size_bytes _ann_last_error _ann_last_scan_count _ann_get_n_lists _ann_set_probe _ann_set_max_k _ann_set_max_scan _js_fetch_range_blocking; do
  if ! grep -q "$sym" annpack.js; then
    fail "Missing symbol in annpack.js: $sym"
  fi
done
pass "Exports present in annpack.js"

# Deterministic tiny index + native harness tests
TMP_DIR="$ROOT/tests/tmp"
mkdir -p "$TMP_DIR"
python3 tests/gen_tiny_index.py "$TMP_DIR" >/tmp/annpack_gen.log 2>&1 || (cat /tmp/annpack_gen.log; fail "tiny index generation failed")
cc -std=c11 -O2 -Iinclude tests/native_test.c tests/io_posix.c src/ann_engine.c -o /tmp/annpack_native_test >/tmp/annpack_native_build.log 2>&1 || (cat /tmp/annpack_native_build.log; fail "native test build failed")
/tmp/annpack_native_test "$TMP_DIR" >/tmp/annpack_native_run.log 2>&1 || (cat /tmp/annpack_native_run.log; fail "native test run failed")
pass "Native determinism + negative + loop tests passed"

echo "[test] Starting server on ${HOST}:${PORT} ..."
HOST="$HOST" PORT="$PORT" python3 server.py >/tmp/annpack_srv.log 2>&1 &
SRV_PID=$!
cleanup() { kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

sleep 1

HDR=$(curl -s -H "Range: bytes=0-15" "http://${HOST}:${PORT}/${TARGET}") || { cat /tmp/annpack_srv.log; fail "Range fetch (header) failed"; }
MAGIC=$(printf '%s' "$HDR" | xxd -p | head -c 8)
[[ "$MAGIC" == "414e4e50" ]] || { cat /tmp/annpack_srv.log; fail "Bad magic: $MAGIC"; }
pass "Header magic ANNP ok for $TARGET"

MID=$(curl -s -H "Range: bytes=1024-1039" "http://${HOST}:${PORT}/${TARGET}") || { cat /tmp/annpack_srv.log; fail "Range fetch (mid) failed"; }
[[ $(printf '%s' "$MID" | wc -c | tr -d ' ') -eq 16 ]] || fail "Mid-range fetch length != 16"
pass "Mid-range fetch length ok"

pass "smoke_test.sh completed"
