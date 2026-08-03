#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ANNPACK=${ANNPACK:-$ROOT/target/release/annpack}
WORK=${WORK:-$ROOT/target/google-okf-reproduction}
REPOSITORY=https://github.com/GoogleCloudPlatform/knowledge-catalog.git
REVISION=3fcbb9f828c2f23d109c855ee403c3a4c81f3a96
EXPECTED=$ROOT/examples/okf-reproduction/expected-roots.json

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
    --version 0.2.0 \
    --source-revision "git:$REVISION" \
    --license Apache-2.0 \
    --redistributable true \
    --json > "$WORK/$artifact_name.build.json"
}

build_bundle ga4 ga4
build_bundle crypto_bitcoin crypto-bitcoin
build_bundle stackoverflow stackoverflow

python3 - "$ANNPACK" "$WORK" "$EXPECTED" "$REPOSITORY" "$REVISION" "${UPDATE_EXPECTED_ROOTS:-0}" <<'PY'
import json
import pathlib
import subprocess
import sys

annpack, work, expected_path, repository, revision, update = sys.argv[1:]
compiler_version = subprocess.run(
    [annpack, "--version"], capture_output=True, text=True, check=True
).stdout.strip().split()[-1]
work = pathlib.Path(work)
expected_path = pathlib.Path(expected_path)
artifacts = {}
for name in ("ga4", "crypto-bitcoin", "stackoverflow"):
    result = subprocess.run(
        [annpack, "inspect", str(work / f"{name}.annpack"), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    artifacts[name] = json.loads(result.stdout)["root_hash"]

if update == "1":
    payload = {
        "schema": "annpack-reproduction-v1",
        "source": {
            "repository": repository,
            "revision": revision,
            "format": "okf",
            "format_version": "0.2",
            "license": "Apache-2.0",
        },
        # Derived from the binary that actually produced these roots. A literal
        # here drifts away from the bytes it identifies, which is precisely the
        # failure this file exists to prevent.
        "compiler": f"annpack-reference/{compiler_version}",
        "root_scheme": (
            "Artifact root: BLAKE3 over the non-signature section directory. "
            "It identifies these exact bytes for this builder and is not a "
            "cross-implementation semantic identity. Compare the manifest's "
            "passage_merkle_root for the layout-independent passage commitment."
        ),
        "note": (
            "Generated from the pinned OKF v0.2 repository revision. Any source, "
            "ingestion, chunking, compression, or layout change can change an "
            "artifact root; review regenerated values rather than updating them "
            "by hand."
        ),
        "artifacts": artifacts,
    }
    expected_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {expected_path}")
else:
    expected = json.loads(expected_path.read_text())
    if expected["source"]["revision"] != revision:
        raise SystemExit(
            f"expected-roots revision {expected['source']['revision']} != pinned {revision}"
        )
    if expected["source"]["format_version"] != "0.2":
        raise SystemExit("expected-roots does not identify OKF v0.2")
    for name, actual in artifacts.items():
        wanted = expected["artifacts"][name]
        if actual != wanted:
            raise SystemExit(f"{name}: root {actual} != expected {wanted}")
        print(f"{name}: {actual} verified")
PY
