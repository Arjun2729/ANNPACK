#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <bucket-name> <pack.annpack>" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUCKET=${1#gs://}
PACK=$2
ANNPACK=${ANNPACK:-$ROOT/target/release/annpack}

if [[ ! -f "$PACK" ]]; then
  echo "pack not found: $PACK" >&2
  exit 1
fi
if [[ ! -x "$ANNPACK" ]]; then
  echo "ANNPack release binary not found at $ANNPACK" >&2
  exit 1
fi

ROOT_HASH=$(
  "$ANNPACK" inspect "$PACK" --json \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["root_hash"])'
)
OBJECT="packs/$ROOT_HASH.annpack"

gcloud storage buckets update "gs://$BUCKET" \
  --cors-file="$ROOT/examples/okf-reproduction/gcs-cors.json" \
  --web-main-page-suffix=index.html

gcloud storage cp "$PACK" "gs://$BUCKET/$OBJECT" \
  --content-type=application/vnd.annpack \
  --cache-control=public,max-age=31536000,immutable,no-transform

gcloud storage cp "$ROOT/web/index.html" "gs://$BUCKET/index.html" \
  --content-type=text/html \
  --cache-control=no-cache
gcloud storage cp "$ROOT/web/annpack-browser.js" "gs://$BUCKET/annpack-browser.js" \
  --content-type=text/javascript \
  --cache-control=public,max-age=300,no-transform
gcloud storage cp "$ROOT/web/pkg/annpack.js" "gs://$BUCKET/pkg/annpack.js" \
  --content-type=text/javascript \
  --cache-control=public,max-age=300,no-transform
gcloud storage cp "$ROOT/web/pkg/annpack_bg.wasm" "gs://$BUCKET/pkg/annpack_bg.wasm" \
  --content-type=application/wasm \
  --cache-control=public,max-age=300,no-transform

echo "Artifact root: $ROOT_HASH"
echo "Demo URL: https://storage.googleapis.com/$BUCKET/index.html?pack=./$OBJECT&root=$ROOT_HASH"
echo "The bucket objects must be publicly readable for an unauthenticated launch demo."
