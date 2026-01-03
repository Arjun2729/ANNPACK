#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d /tmp/annpack_demo_XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

log() {
  echo "[demo] $*"
}

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
export ANNPACK_OFFLINE=1

cat > "$WORK/tiny_docs.csv" <<'CSV'
id,text
0,hello from annpack
1,paris is france
CSV

log "building tiny pack (offline)"
annpack build --input "$WORK/tiny_docs.csv" --text-col text --output "$WORK/pack" --lists 4

PORT="$($PYTHON_BIN - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)"

log "starting annpack serve on http://127.0.0.1:$PORT/"
annpack serve "$WORK" --host 127.0.0.1 --port "$PORT" > "$WORK/serve.log" 2>&1 &
SERVE_PID=$!
sleep 0.5

if ! curl -fs "http://127.0.0.1:$PORT/index.html" >/dev/null 2>&1; then
  log "serve failed; last log:"
  tail -n 50 "$WORK/serve.log" || true
  kill "$SERVE_PID" >/dev/null 2>&1 || true
  exit 1
fi

log "READY: open http://127.0.0.1:$PORT/"
log "UI dev server (optional):"
log "  cd web && npm --workspace annpack-ui run dev"
log "Press Ctrl+C to stop the server"
wait "$SERVE_PID"
