#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[determinism] $*"
}

hash_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    sha256sum "$path" | awk '{print $1}'
  fi
}

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

export ANNPACK_OFFLINE=1

cat > "$work_dir/tiny_docs.csv" <<'CSV'
id,text
0,hello
1,paris is france
CSV

build_once() {
  local out_dir="$1"
  mkdir -p "$out_dir"
  annpack build --input "$work_dir/tiny_docs.csv" --text-col text --id-col id --output "$out_dir/pack" --lists 4
}

out_a="$work_dir/a"
out_b="$work_dir/b"

log "build A"
build_once "$out_a"
log "build B"
build_once "$out_b"

ann_a="$out_a/pack.annpack"
ann_b="$out_b/pack.annpack"
manifest_a="$out_a/pack.manifest.json"
manifest_b="$out_b/pack.manifest.json"
meta_a="$out_a/pack.meta.jsonl"
meta_b="$out_b/pack.meta.jsonl"

hash_ann_a="$(hash_file "$ann_a")"
hash_ann_b="$(hash_file "$ann_b")"

if [[ "$hash_ann_a" != "$hash_ann_b" ]]; then
  echo "[determinism] annpack hash mismatch"
  exit 1
fi

if ! diff -q "$manifest_a" "$manifest_b" >/dev/null; then
  echo "[determinism] manifest mismatch"
  exit 1
fi

if ! diff -q "$meta_a" "$meta_b" >/dev/null; then
  echo "[determinism] meta mismatch"
  exit 1
fi

log "determinism ok"
