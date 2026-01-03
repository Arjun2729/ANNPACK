#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[quickstart] $*"
}

WORK="$(mktemp -d /tmp/annpack_quickstart_XXXXXX)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'rm -rf "$WORK"' EXIT

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

if ! command -v annpack >/dev/null 2>&1; then
  log "annpack not found; creating temp venv"
  "$PYTHON_BIN" -m venv --system-site-packages "$WORK/venv"
  source "$WORK/venv/bin/activate"
  python -m ensurepip --upgrade >/dev/null 2>&1 || true
  python -m pip install -U pip >/dev/null 2>&1 || true
  if ! python -m pip install annpack >/dev/null 2>&1; then
    log "pip install annpack failed; falling back to local editable install"
    python -m pip install -e "$ROOT" --no-deps --no-build-isolation >/dev/null 2>&1 || true
  fi
  if ! python -c "import annpack" >/dev/null 2>&1; then
    log "install failed (likely no network to fetch build deps). Run with network or preinstall setuptools/build."
    exit 1
  fi
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

for attempt in 1 2 3; do
  SMOKE_PORT="$($PYTHON_BIN - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)"
  log "running smoke on port $SMOKE_PORT (attempt $attempt)"
  if annpack smoke "$WORK/out" --port "$SMOKE_PORT" >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 3 ]; then
    log "smoke failed; check output with: annpack smoke \"$WORK/out\" --port <free-port>"
    exit 1
  fi
done

SERVE_PID=""
set +e
for attempt in 1 2 3 4 5; do
  PORT="$($PYTHON_BIN - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)"
  log "starting serve on http://127.0.0.1:$PORT (attempt $attempt)"
  annpack serve "$WORK/out" --host 127.0.0.1 --port "$PORT" > "$WORK/serve.log" 2>&1 &
  SERVE_PID=$!
  trap 'kill "$SERVE_PID" >/dev/null 2>&1 || true' EXIT
  sleep 0.5
  if ! kill -0 "$SERVE_PID" >/dev/null 2>&1; then
    log "serve exited early (attempt $attempt); retrying"
    tail -n 20 "$WORK/serve.log" >/dev/null 2>&1 || true
    SERVE_PID=""
    continue
  fi
  if curl -fs "http://127.0.0.1:$PORT/index.html" >/dev/null 2>&1; then
    log "READY: open http://127.0.0.1:$PORT/"
    log "PASS quickstart (server running)"
    break
  fi
  kill "$SERVE_PID" >/dev/null 2>&1 || true
  SERVE_PID=""
done
set -e

if [ -z "$SERVE_PID" ]; then
  log "failed to start server; last log:"
  tail -n 50 "$WORK/serve.log" || true
  exit 1
fi
log "Press Ctrl+C to stop the server"
wait "$SERVE_PID"
