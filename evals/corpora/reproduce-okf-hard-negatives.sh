#!/usr/bin/env bash
# Reproduce the okf-hard-negatives evaluation corpus, and verify its identity.
#
# The previous version of this script flattened source paths to their basename:
#
#     ga4/tables/index.md   ->  ga4__index.md
#     ga4/datasets/index.md ->  ga4__index.md      <- overwrote the first
#
# Sixty-two source files collapsed onto forty-seven names. Fourteen were
# silently discarded, and which fourteen depended on `find` traversal order,
# which is filesystem-dependent. macOS and Linux therefore built corpora that
# agreed on 150 of 153 passages and differed on three.
#
# The counts matched on both machines, so every assertion passed. Hours went
# into attributing the resulting embedding differences to ONNX int8 saturation
# and then to tokenizer portability, when the two hosts were simply not
# embedding the same documents. Count is not identity; this script now checks
# identity.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

VENDOR="examples/okf-reproduction/vendor"
WORK="${WORK:-target/okf-eval}"
BUNDLES=(ga4 crypto_bitcoin stackoverflow)
REVISION="git:3fcbb9f828c2f23d109c855ee403c3a4c81f3a96"
QRELS="evals/corpora/okf-hard-negatives.jsonl"
IDENTITY="evals/corpora/okf-hard-negatives.identity.json"
UPDATE="${UPDATE:-0}"

BINARY="${ANNPACK:-$ROOT/target/release/annpack}"
[ -x "$BINARY" ] || { echo "::error::release binary missing at $BINARY; run cargo build --release" >&2; exit 1; }
fail() { echo "::error::$*" >&2; exit 1; }
[ -d "$VENDOR" ] || fail "vendored OKF source missing at $VENDOR"

rm -rf "$WORK"
mkdir -p "$WORK/corpus"

# Flatten the full relative path, not the basename, so nothing collides. Sorted
# so the assembly order does not depend on the filesystem either.
for bundle in "${BUNDLES[@]}"; do
  [ -d "$VENDOR/$bundle" ] || fail "vendored bundle missing: $VENDOR/$bundle"
  while IFS= read -r file; do
    relative="${file#"$VENDOR/$bundle/"}"
    destination="$WORK/corpus/${bundle}__${relative//\//__}"
    # Belt and braces: even with path-preserving names, a collision must stop
    # the run rather than overwrite. That is the failure this script exists to
    # prevent, and it was invisible precisely because it was silent.
    [ -e "$destination" ] && fail "corpus name collision: $file -> $destination"
    cp "$file" "$destination"
  done < <(find "$VENDOR/$bundle" -name '*.md' | LC_ALL=C sort)
done

"$BINARY" build "$WORK/corpus" --output "$WORK/core.annpack" \
  --name okf-eval --version 0.3.0 --source-revision "$REVISION" --json >/dev/null
"$BINARY" export-passages "$WORK/core.annpack" --output "$WORK/passages.json"

# Identity, not shape: a digest over sorted (passage_id, text), plus a manifest
# of the source files that produced it. If the digest moves, the manifest says
# whether collection or passage generation changed.
python3 - "$VENDOR" "$WORK" "$QRELS" "$IDENTITY" "$UPDATE" <<'PY'
import hashlib, json, pathlib, sys

vendor, work, qrels_path, identity_path, update = sys.argv[1:6]
work, vendor = pathlib.Path(work), pathlib.Path(vendor)

sources = {}
for path in sorted(vendor.rglob("*.md")):
    if path.parent == vendor:
        continue
    sources[str(path.relative_to(vendor))] = hashlib.sha256(path.read_bytes()).hexdigest()

passages = json.loads((work / "passages.json").read_text())
digest = hashlib.sha256(
    json.dumps(sorted((p["id"], p["text"]) for p in passages)).encode()
).hexdigest()

documents = len(list((work / "corpus").glob("*.md")))
queries = [json.loads(l) for l in pathlib.Path(qrels_path).read_text().splitlines() if l.strip()]
ids = {p["id"] for p in passages}
unresolved = [q["id"] for q in queries if not set(q["relevant_passage_ids"]) & ids]

observed = {
    "documents": documents,
    "passages": len(passages),
    "queries": len(queries),
    "corpus_sha256": digest,
    "sources": sources,
}

if update == "1":
    pathlib.Path(identity_path).write_text(json.dumps(observed, indent=2) + "\n")
    print(f"wrote {identity_path}: {documents} documents, {len(passages)} passages, "
          f"{len(queries)} queries, corpus {digest[:16]}")
    raise SystemExit(0)

expected = json.loads(pathlib.Path(identity_path).read_text())
for key in ("documents", "passages", "queries"):
    if observed[key] != expected[key]:
        raise SystemExit(f"::error::{key}: {observed[key]}, expected {expected[key]}")
if observed["sources"] != expected["sources"]:
    added = set(observed["sources"]) - set(expected["sources"])
    removed = set(expected["sources"]) - set(observed["sources"])
    changed = {k for k in set(observed["sources"]) & set(expected["sources"])
               if observed["sources"][k] != expected["sources"][k]}
    raise SystemExit(f"::error::source files changed: +{sorted(added)[:3]} "
                     f"-{sorted(removed)[:3]} ~{sorted(changed)[:3]}")
if observed["corpus_sha256"] != expected["corpus_sha256"]:
    raise SystemExit(
        f"::error::corpus digest {observed['corpus_sha256'][:16]}, expected "
        f"{expected['corpus_sha256'][:16]} -- source files are unchanged, so "
        f"ingestion or chunking moved")
if unresolved:
    raise SystemExit(f"::error::{len(unresolved)} queries reference absent passages: {unresolved[:5]}")

print(f"okf-hard-negatives verified: {documents} documents, {len(passages)} passages, "
      f"{len(queries)} queries, corpus {digest[:16]}, all targets present")
PY
