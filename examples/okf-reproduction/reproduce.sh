#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ADYAR=${ADYAR:-${ANNPACK:-$ROOT/target/release/adyar}}
WORK=${WORK:-$ROOT/target/google-okf-reproduction}
VENDOR=$ROOT/examples/okf-reproduction/vendor
REPOSITORY=https://github.com/GoogleCloudPlatform/knowledge-catalog.git
REVISION=3fcbb9f828c2f23d109c855ee403c3a4c81f3a96
EXPECTED=$ROOT/examples/okf-reproduction/expected-roots.json

if [[ ! -x "$ADYAR" ]]; then
  echo "ANNPack release binary not found at $ADYAR" >&2
  echo "Run: cargo build --release" >&2
  exit 1
fi

# This used to clone $REPOSITORY and check out $REVISION live. Stopped:
# that commit's own branch was deleted after its PR merged, and fetching
# it turned out to be unreliable in three different ways depending on
# exactly how it was asked for -- a bare-SHA fetch was refused ("not our
# ref"), an explicit fetch of the still-listed pull ref worked once
# locally and then failed on CI minutes later ("couldn't find remote
# ref"), and even a full plain clone (which should already contain the
# commit as an ordinary ancestor of the default branch) came back on CI
# with an incomplete object ("unable to read tree"). That pattern --
# succeeding locally and failing differently on GitHub-hosted runners --
# points at inconsistent object availability on GitHub's own backend for
# this specific orphaned commit, not anything a git invocation on this
# side can work around.
#
# The three bundles below are vendored from that exact commit instead
# (examples/okf-reproduction/vendor/, Apache-2.0, upstream license
# preserved as UPSTREAM-LICENSE.md in the same directory). $REPOSITORY
# and $REVISION are kept as recorded provenance of where they came from,
# not as something this script still fetches.
#
# WORK sits under target/, which CI caches. Runs from before this fix
# left a cloned knowledge-catalog checkout there; a stale cache can still
# hand that back on restore, and re-saving that leftover tree (nested
# Cargo target/trybuild dirs mid-write) is what broke the cache step, not
# anything this script does. WORK is scratch output owned entirely by
# this script, so clearing it first guarantees a clean save regardless of
# what an old cache restored.
rm -rf "$WORK"
mkdir -p "$WORK"

build_bundle() {
  local bundle=$1
  local artifact_name=$2
  "$ADYAR" build "$VENDOR/$bundle" \
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

python3 - "$ADYAR" "$WORK" "$EXPECTED" "$REPOSITORY" "$REVISION" "${UPDATE_EXPECTED_ROOTS:-0}" <<'PY'
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
        "compiler": f"adyar-reference/{compiler_version}",
        "root_scheme": (
            "Artifact root: BLAKE3 over the non-signature section directory. "
            "It identifies these exact bytes for this builder and is not a "
            "cross-implementation semantic identity. Compare the manifest's "
            "passage_merkle_root for the layout-independent passage commitment."
        ),
        "note": (
            "Generated from the pinned OKF v0.2 repository revision, vendored "
            "under examples/okf-reproduction/vendor/ rather than fetched live "
            "(see reproduce.sh for why). Any source, ingestion, chunking, "
            "compression, or layout change can change an artifact root; review "
            "regenerated values rather than updating them by hand."
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
