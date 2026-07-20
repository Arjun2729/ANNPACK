#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ANNPACK=${ANNPACK:-$ROOT/target/release/annpack}
WORK=${WORK:-$ROOT/target/google-okf-reproduction}
REPOSITORY=https://github.com/GoogleCloudPlatform/knowledge-catalog.git
REVISION=d44368c15e38e7c92481c5992e4f9b5b421a801d

if [[ ! -x "$ANNPACK" ]]; then
  echo "ANNPack release binary not found at $ANNPACK" >&2
  echo "Run: cargo build --release" >&2
  exit 1
fi

mkdir -p "$WORK"
if [[ ! -d "$WORK/knowledge-catalog/.git" ]]; then
  git clone "$REPOSITORY" "$WORK/knowledge-catalog"
fi
git -C "$WORK/knowledge-catalog" fetch origin "$REVISION"
git -C "$WORK/knowledge-catalog" checkout --detach "$REVISION"

build_bundle() {
  local bundle=$1
  local artifact_name=$2
  "$ANNPACK" build "$WORK/knowledge-catalog/okf/bundles/$bundle" \
    --source-format okf \
    --output "$WORK/$artifact_name.annpack" \
    --name "google-okf-$artifact_name" \
    --version 0.1.0 \
    --source-revision "git:$REVISION" \
    --license Apache-2.0 \
    --redistributable true \
    --json > "$WORK/$artifact_name.build.json"
}

build_bundle ga4 ga4
build_bundle crypto_bitcoin crypto-bitcoin
build_bundle stackoverflow stackoverflow

python3 - "$ANNPACK" "$WORK" "$ROOT/launch/google-okf/expected-roots.json" <<'PY'
import json
import pathlib
import subprocess
import sys

annpack, work, expected_path = sys.argv[1:]
work = pathlib.Path(work)
expected = json.loads(pathlib.Path(expected_path).read_text())["artifacts"]
for name, wanted in expected.items():
    result = subprocess.run(
        [annpack, "inspect", str(work / f"{name}.annpack"), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)["root_hash"]
    if actual != wanted:
        raise SystemExit(f"{name}: root {actual} != expected {wanted}")
    print(f"{name}: {actual} verified")
PY
