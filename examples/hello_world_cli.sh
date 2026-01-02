#!/usr/bin/env bash
set -euo pipefail

export ANNPACK_OFFLINE=1

work_dir="$(mktemp -d)"
pack_dir="$work_dir/pack"
mkdir -p "$pack_dir"

cat > "$work_dir/tiny_docs.csv" <<'CSV'
id,text
0,hello
1,paris is france
CSV

annpack build --input "$work_dir/tiny_docs.csv" --text-col text --id-col id --output "$pack_dir/pack" --lists 4

echo "Pack built at: $pack_dir"
echo "Next: annpack serve \"$pack_dir\" --port 8000"
echo "Or:   annpack smoke \"$pack_dir\" --port 8000"
