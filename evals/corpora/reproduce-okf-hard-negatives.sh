#!/usr/bin/env bash
# Reproduce the okf-hard-negatives evaluation corpus, and assert its identity.
#
# This exists because the README used to describe the procedure in prose and
# drifted from it: it named the pre-vendoring checkout path, so following the
# documented steps from a clean tree produced zero documents and a build error.
# The benchmark data was reproducible; the documented procedure was not.
#
# The 47/153/63 counts below are benchmark-identity checks, not ANNPack
# invariants. If ingestion or chunking changes and this corpus becomes 157
# passages, every published number silently stops describing the same thing.
# Failing here forces that to be a decision rather than a drift.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

VENDOR="examples/okf-reproduction/vendor"
WORK="${WORK:-target/okf-eval}"
BUNDLES=(ga4 crypto_bitcoin stackoverflow)
REVISION="git:3fcbb9f828c2f23d109c855ee403c3a4c81f3a96"
QRELS="evals/corpora/okf-hard-negatives.jsonl"
EXPECT_DOCS=47
EXPECT_PASSAGES=153
EXPECT_QUERIES=63

BINARY="${ANNPACK:-$ROOT/target/release/annpack}"
[ -x "$BINARY" ] || { echo "::error::release binary missing at $BINARY; run cargo build --release" >&2; exit 1; }

fail() { echo "::error::$*" >&2; exit 1; }

# The pinned upstream source is vendored rather than cloned, so reproduction
# does not depend on a remote repository still serving that revision.
[ -d "$VENDOR" ] || fail "vendored OKF source missing at $VENDOR"

rm -rf "$WORK"
mkdir -p "$WORK/corpus"
for bundle in "${BUNDLES[@]}"; do
  [ -d "$VENDOR/$bundle" ] || fail "vendored bundle missing: $VENDOR/$bundle"
  # Flattened with a bundle prefix so same-named files across bundles coexist,
  # and so cross-domain distractors land in one corpus.
  while IFS= read -r file; do
    cp "$file" "$WORK/corpus/${bundle}__$(basename "$file")"
  done < <(find "$VENDOR/$bundle" -name '*.md')
done

docs=$(find "$WORK/corpus" -name '*.md' | wc -l | tr -d ' ')
[ "$docs" -eq "$EXPECT_DOCS" ] || fail "corpus has $docs documents, expected $EXPECT_DOCS"

"$BINARY" build "$WORK/corpus" --output "$WORK/core.annpack" \
  --name okf-eval --version 0.2.0 --source-revision "$REVISION" --json >/dev/null
"$BINARY" export-passages "$WORK/core.annpack" --output "$WORK/passages.json"

passages=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$WORK/passages.json")
[ "$passages" -eq "$EXPECT_PASSAGES" ] || fail "corpus has $passages passages, expected $EXPECT_PASSAGES"

queries=$(grep -c . "$QRELS")
[ "$queries" -eq "$EXPECT_QUERIES" ] || fail "$QRELS has $queries queries, expected $EXPECT_QUERIES"

# A query whose target passage no longer exists is silently unanswerable, which
# depresses every mode equally and looks like a retrieval result.
python3 - "$WORK/passages.json" "$QRELS" <<'PY'
import json, sys
ids = {p["id"] for p in json.load(open(sys.argv[1]))}
bad = []
for line in open(sys.argv[2]):
    if not line.strip(): continue
    q = json.loads(line)
    if not set(q["relevant_passage_ids"]) & ids:
        bad.append(q["id"])
if bad:
    print(f"::error::{len(bad)} queries reference passages absent from the corpus: {bad[:5]}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "okf-hard-negatives reproduced: ${docs} documents, ${passages} passages, ${queries} queries, all targets present"
echo "  corpus:   $WORK/corpus"
echo "  pack:     $WORK/core.annpack"
echo "  passages: $WORK/passages.json"
