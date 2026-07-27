#!/usr/bin/env bash
# Rebuild every tracked demo pack from its pinned source.
#
# These artifacts are committed so GitHub Pages can serve them, which means they
# are the one place unreproducible bytes could hide. Regenerate with this script
# whenever the format or builder changes; never hand-edit them.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ANNPACK=${ANNPACK:-$ROOT/target/release/annpack}
cd "$ROOT"

[ -x "$ANNPACK" ] || { echo "build the release binary first: cargo build --release" >&2; exit 1; }

"$ANNPACK" build fixtures/docs-v1 --output docs/docs-v1.annpack \
  --name vendor-docs --version 1.0.0 --source-revision git:v1 \
  --base-url https://vendor.example/docs/v1 >/dev/null
"$ANNPACK" build fixtures/docs-v2 --output docs/docs-v2.annpack \
  --name vendor-docs --version 2.0.0 --source-revision git:v2 \
  --base-url https://vendor.example/docs/v2 >/dev/null

# The Google OKF reproduction demo. Requires the pinned upstream checkout that
# launch/google-okf/reproduce.sh creates.
OKF=target/google-okf-reproduction/knowledge-catalog/okf/bundles/ga4
if [ -d "$OKF" ]; then
  mkdir -p docs/packs
  "$ANNPACK" build "$OKF" --source-format okf --output docs/packs/google-okf-ga4.annpack \
    --name google-okf-ga4 --version 0.1.0 \
    --source-revision git:d44368c15e38e7c92481c5992e4f9b5b421a801d \
    --license Apache-2.0 --redistributable true >/dev/null
else
  echo "skipping OKF demo pack: run launch/google-okf/reproduce.sh first" >&2
fi

for pack in docs/docs-v1.annpack docs/docs-v2.annpack docs/packs/*.annpack; do
  [ -e "$pack" ] || continue
  printf '%-44s %s\n' "$pack" "$("$ANNPACK" inspect "$pack" | python3 -c 'import json,sys;print(json.load(sys.stdin)["root_hash"])')"
done
