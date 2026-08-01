#!/usr/bin/env bash
# Rebuild every tracked demo pack from pinned inputs.
#
# Tracked demo artifacts must be reproducible. The GA4 signature uses an
# intentionally public, insecure test seed committed under fixtures/demo-signing;
# it proves signature mechanics only and MUST NOT be used as a publisher key.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ANNPACK=${ANNPACK:-$ROOT/target/release/annpack}
cd "$ROOT"

[ -x "$ANNPACK" ] || {
  echo "build the release binary first: cargo build --release" >&2
  exit 1
}

"$ANNPACK" build fixtures/docs-v1 --output docs/docs-v1.annpack \
  --name vendor-docs --version 1.0.0 --source-revision git:v1 \
  --base-url https://vendor.example/docs/v1 >/dev/null
"$ANNPACK" build fixtures/docs-v2 --output docs/docs-v2.annpack \
  --name vendor-docs --version 2.0.0 --source-revision git:v2 \
  --base-url https://vendor.example/docs/v2 >/dev/null

OKF=target/google-okf-reproduction/knowledge-catalog/okf/bundles/ga4
if [ -d "$OKF" ]; then
  mkdir -p docs/packs
  "$ANNPACK" build "$OKF" --source-format okf \
    --output docs/packs/google-okf-ga4.annpack \
    --name google-okf-ga4 --version 0.2.0 \
    --source-revision git:3fcbb9f828c2f23d109c855ee403c3a4c81f3a96 \
    --license Apache-2.0 --redistributable true >/dev/null

  KEYDIR=$(mktemp -d)
  trap 'rm -rf "$KEYDIR"' EXIT
  grep -E '^[0-9a-f]{64}$' \
    fixtures/demo-signing/INSECURE-PUBLIC-TEST-SEED.txt > "$KEYDIR/demo.key"

  # Public key for the intentionally public seed above. The subsequent verify
  # command proves this file matches the key that produced the signature.
  printf '%s\n' \
    '03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8' \
    > docs/packs/google-okf-ga4.pub

  "$ANNPACK" sign docs/packs/google-okf-ga4.annpack \
    --output "$KEYDIR/signed.annpack" \
    --key "$KEYDIR/demo.key" \
    --identity 'demo:public-test-key (identity untrusted)' >/dev/null
  mv "$KEYDIR/signed.annpack" docs/packs/google-okf-ga4.annpack

  "$ANNPACK" verify docs/packs/google-okf-ga4.annpack \
    --public-key docs/packs/google-okf-ga4.pub >/dev/null

  rm -rf "$KEYDIR"
  trap - EXIT
else
  echo "skipping OKF demo pack: run launch/google-okf/reproduce.sh first" >&2
fi

for pack in docs/docs-v1.annpack docs/docs-v2.annpack docs/packs/*.annpack; do
  [ -e "$pack" ] || continue
  printf '%-44s %s\n' "$pack" \
    "$("$ANNPACK" inspect "$pack" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["root_hash"])')"
done
