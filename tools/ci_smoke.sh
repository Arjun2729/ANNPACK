#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[ci_smoke] $*"
}

pick_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

export ANNPACK_OFFLINE=1

cat > "$work_dir/tiny_docs.csv" <<'CSV'
id,text
0,hello
1,paris is france
CSV

pack_dir="$work_dir/pack"
mkdir -p "$pack_dir"

log "building tiny pack"
annpack build --input "$work_dir/tiny_docs.csv" --text-col text --id-col id --output "$pack_dir/pack" --lists 4

serve_port="$(pick_port)"
serve_log="$work_dir/serve.log"

log "starting serve on port $serve_port"
annpack serve "$pack_dir" --host 127.0.0.1 --port "$serve_port" >"$serve_log" 2>&1 &
serve_pid=$!

cleanup() {
  kill "$serve_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$serve_port/index.html" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$serve_pid" >/dev/null 2>&1; then
    echo "[ci_smoke] serve exited early"
    tail -n 200 "$serve_log"
    exit 1
  fi
  sleep 0.2
  if [[ $i -eq 30 ]]; then
    echo "[ci_smoke] serve did not become ready"
    tail -n 200 "$serve_log"
    exit 1
  fi
 done

curl -sf "http://127.0.0.1:$serve_port/pack/pack.manifest.json" >/dev/null
log "smoke ok"
