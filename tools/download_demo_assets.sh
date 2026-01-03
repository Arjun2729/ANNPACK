#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[demo_assets] $*"
}

DEST="${1:-./demo_assets}"
URL="${ANNPACK_DEMO_URL:-https://github.com/Arjun2729/ANNPACK/releases/latest/download/annpack_demo_pack.tar.gz}"
ARCHIVE="$DEST/annpack_demo_pack.tar.gz"

mkdir -p "$DEST"

log "downloading from $URL"
if ! curl -fL "$URL" -o "$ARCHIVE"; then
  log "download failed; set ANNPACK_DEMO_URL to a valid asset URL"
  exit 1
fi

log "verifying archive"
if ! tar -tzf "$ARCHIVE" >/dev/null; then
  log "invalid archive"
  exit 1
fi

log "extracting to $DEST"
tar -xzf "$ARCHIVE" -C "$DEST"

log "done"
