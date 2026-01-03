#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[quickstart] $*"
}

WORK="$(mktemp -d /tmp/annpack_quickstart_XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

if ! command -v annpack >/dev/null 2>&1; then
  log "annpack not found; creating temp venv"
  "$PYTHON_BIN" -m venv "$WORK/venv"
  source "$WORK/venv/bin/activate"
  python -m pip install -U pip >/dev/null
  python -m pip install annpack >/dev/null
fi

PORT="$($PYTHON_BIN - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)"

export ANNPACK_OFFLINE=1

cat > "$WORK/tiny_docs.csv" <<'CSV'
id,text
0,hello
1,paris is france
CSV

log "building tiny pack (offline)"
annpack build --input "$WORK/tiny_docs.csv" --text-col text --output "$WORK/out/pack" --lists 4

log "starting serve on http://127.0.0.1:$PORT"
annpack serve "$WORK/out" --host 127.0.0.1 --port "$PORT" > "$WORK/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" >/dev/null 2>&1 || true' EXIT

sleep 0.5
annpack smoke "$WORK/out" --port "$PORT"

log "READY: open http://127.0.0.1:$PORT/"
log "Press Ctrl+C to stop the server"
wait "$SERVE_PID"
