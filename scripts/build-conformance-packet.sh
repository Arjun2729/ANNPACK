#!/usr/bin/env bash
# Regenerate spec/conformance/ from the pinned corpus and the reference binary.
#
# The packet is the contract handed to an independent implementer, so every
# expected value in it is generated from the reference implementation and then
# pinned by tests/conformance_vectors.rs. If the reference drifts from the
# committed vectors, that test fails — the vectors do not silently follow it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ANNPACK=${ANNPACK:-$ROOT/target/release/annpack}
cd "$ROOT"
PACKET=spec/conformance
ART=$PACKET/artifacts
VEC=$PACKET/vectors

[ -x "$ANNPACK" ] || { echo "build the release binary first: cargo build --release" >&2; exit 1; }
mkdir -p "$ART" "$VEC"

# ---------------------------------------------------------------- artifacts
"$ANNPACK" build "$PACKET/corpus" \
  --output "$ART/conformance-v2.annpack" \
  --name annpack-conformance --version 1.0.0 \
  --description "ANNPack Core conformance corpus" \
  --source-revision spec:conformance-v1 \
  --base-url https://conformance.test \
  --license CC0-1.0 >/dev/null

# A signed copy. The private key is intentionally NOT committed: implementers
# verify signatures, they do not produce them. The public key is committed.
KEYDIR=$(mktemp -d)
"$ANNPACK" keygen --output "$KEYDIR/test.key" --public-output "$KEYDIR/test.pub" >/dev/null
rm -f "$ART/conformance-v2-signed.annpack"
"$ANNPACK" sign "$ART/conformance-v2.annpack" \
  --output "$ART/conformance-v2-signed.annpack" \
  --key "$KEYDIR/test.key" --identity conformance.test >/dev/null
cp "$KEYDIR/test.pub" "$ART/conformance-v2-signed.pub"
rm -rf "$KEYDIR"

# Corruption corpus and the v0.3-era compatibility fixture travel with the packet.
mkdir -p "$ART/corruption"
cp launch/evidence/2026-07-20/workstream8-conformance/invalid-corpus/*.annpack "$ART/corruption/"
cp spec/test-vectors/compat/manifest-v1-legacy.annpack "$ART/"
cp spec/test-vectors/minimal-v3.annpack "$ART/"

# ------------------------------------------------------------------ vectors
python3 - "$ANNPACK" "$ROOT" <<'PY'
import json, struct, subprocess, sys, pathlib

annpack, root = sys.argv[1], pathlib.Path(sys.argv[2])
packet = root / "spec/conformance"
art, vec = packet / "artifacts", packet / "vectors"
pack = str(art / "conformance-v2.annpack")

def run(*args):
    return json.loads(subprocess.run([annpack, *args], check=True,
                                     capture_output=True, text=True).stdout)

inspect = run("inspect", pack)
pack_root = inspect["root_hash"]

# --- tokenizer vectors -----------------------------------------------------
# Expected outputs are written from the normative rules in FORMAT-v3 §6.1, not
# scraped from the implementation; tests/conformance_vectors.rs asserts the
# reference tokenizer agrees with them.
tokenizer = {
    "$comment": "FORMAT-v3 §6.1. NFKC, lowercase, split on whitespace, trim edge "
                "characters that are neither Unicode alphanumeric nor in the "
                "technical punctuation set _-.:/@#. Interior characters are never "
                "removed and tokens are never split further.",
    "technical_punctuation": ["_", "-", ".", ":", "/", "@", "#"],
    "cases": [
        {"input": "AP-104 std::move useEffect foo_bar package.module",
         "expected": ["ap-104", "std::move", "useeffect", "foo_bar", "package.module"],
         "why": "the five identifiers a divergent tokenizer gets wrong"},
        {"input": "@scope/pkg v1.2.3",
         "expected": ["@scope/pkg", "v1.2.3"],
         "why": "at, slash and dot are interior-preserving"},
        {"input": "(parenthesised) \"quoted\" trailing... ",
         "expected": ["parenthesised", "quoted", "trailing..."],
         "why": "edge trimming removes non-member punctuation only; '.' is a member so it survives at the edge"},
        {"input": "CamelCase MIXED Case",
         "expected": ["camelcase", "mixed", "case"],
         "why": "simple lowercase mapping"},
        {"input": "  spaced\tout\nlines  ",
         "expected": ["spaced", "out", "lines"],
         "why": "all Unicode whitespace splits"},
        {"input": "ﬁ ①",
         "expected": ["fi", "1"],
         "why": "NFKC folds the ligature and the circled digit"},
        {"input": "--- ...",
         "expected": ["---", "..."],
         "why": "tokens made only of technical punctuation are not empty after trimming"},
    ],
}
(vec / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n")

# --- scoring vectors -------------------------------------------------------
# Exact scores, not merely ranking order. A reader with the wrong boost value
# ranks identically here but scores differently; a reader with the wrong
# tokenizer returns a different top result.
queries = [
    ("std::move",      "colon-interior identifier; a splitting tokenizer also matches separate-words.md"),
    ("foo_bar",        "underscore identifier; a splitting tokenizer ranks separate-words.md FIRST"),
    ("package.module", "dot identifier"),
    ("AP-104",         "digit+hyphen identifier, boost applies"),
    ("@scope/pkg",     "at and slash interior"),
    ("cache",          "ordinary prose term, boost 1.0"),
    ("std move",       "the same characters as std::move but as two plain tokens"),
]
scoring = {
    "$comment": "FORMAT-v3 §6.2. Scores are IEEE-754 doubles and MUST match "
                "exactly, not merely produce the same ordering. Ties resolve by "
                "ascending passage ordinal. Compare `score_bits` (the big-endian "
                "IEEE-754 bit pattern, hex) rather than the decimal `score`: many "
                "JSON parsers, including serde_json without the float_roundtrip "
                "feature, lose up to 1 ULP when reading a decimal double.",
    "pack": "artifacts/conformance-v2.annpack",
    "pack_root": pack_root,
    "queries": [],
}
for query, why in queries:
    response = run("search", pack, query, "--limit", "10", "--mode", "lexical", "--json")
    scoring["queries"].append({
        "query": query,
        "why": why,
        "result_count": len(response["results"]),
        "results": [
            {
                "rank": hit["rank"],
                "score": hit["score"],
                "score_bits": struct.pack(">d", hit["score"]).hex(),
                "passage_id": hit["passage_id"],
                "source_path": hit["source_path"],
                "heading_path": hit["heading_path"],
                "passage_hash": hit["evidence"]["passage_hash"],
            }
            for hit in response["results"]
        ],
    })
(vec / "scoring.json").write_text(json.dumps(scoring, indent=2) + "\n")

# --- evidence and receipt vectors -----------------------------------------
passages = run("export-passages", pack)
target = next(p for p in passages if "AP-104" in p["text"])
receipt = run("receipt", pack, target["id"])
evidence = {
    "$comment": "CORE §evidence-envelope and EVIDENCE-v1. The receipt below MUST "
                "verify with no pack, no network, and no trust in the issuer.",
    "pack_root": pack_root,
    "passage_merkle_root": inspect["manifest"]["passage_merkle_root"],
    "passage_id": target["id"],
    "passage_hash": receipt["passage_hash"],
    "inclusion_proof_steps": len(receipt["inclusion_proof"]),
    "receipt": receipt,
}
(vec / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

# --- signature vectors -----------------------------------------------------
signed = str(art / "conformance-v2-signed.annpack")
signed_inspect = run("inspect", signed)
signature = {
    "$comment": "Signatures are excluded from the artifact root, so signing MUST "
                "NOT change it. identity_trusted MUST remain false without an "
                "externally supplied key binding.",
    "unsigned_root": pack_root,
    "signed_root": signed_inspect["root_hash"],
    "roots_match": signed_inspect["root_hash"] == pack_root,
    "public_key_file": "artifacts/conformance-v2-signed.pub",
    "signatures": [
        {
            "key_id": s["key_id"],
            "identity": s["identity"],
            "cryptographically_valid": s["cryptographically_valid"],
            "identity_trusted_without_external_binding": s["identity_trusted"],
        }
        for s in signed_inspect["signatures"]
    ],
}
(vec / "signature.json").write_text(json.dumps(signature, indent=2) + "\n")

# --- compatibility vectors -------------------------------------------------
legacy = run("inspect", str(art / "manifest-v1-legacy.annpack"))
compat = {
    "$comment": "FORMAT-v3 §4.2. A reader MUST open manifest format 1 and 2, and "
                "MUST refuse an unknown version with an explicit version error at "
                "the container boundary, not a deserialization error.",
    "manifest_v1_legacy": {
        "artifact": "artifacts/manifest-v1-legacy.annpack",
        "manifest_format_version": next(s["format_version"] for s in legacy["sections"] if s["type"] == "manifest"),
        "root": legacy["root_hash"],
        "carries_passage_merkle_root": legacy["manifest"].get("passage_merkle_root") is not None,
        "must_open": True,
        "may_issue_receipts": False,
    },
    "manifest_v2_current": {
        "artifact": "artifacts/conformance-v2.annpack",
        "manifest_format_version": next(s["format_version"] for s in inspect["sections"] if s["type"] == "manifest"),
        "root": pack_root,
        "carries_passage_merkle_root": True,
        "must_open": True,
        "may_issue_receipts": True,
    },
    "unknown_manifest_format_version": {
        "must_reject": True,
        "error_kind": "unsupported manifest section format version",
    },
}
(vec / "compatibility.json").write_text(json.dumps(compat, indent=2) + "\n")

# --- corruption vectors ----------------------------------------------------
corruption = {
    "$comment": "Every artifact MUST be rejected with an error. A reader MUST NOT "
                "panic, hang, or return results. Rejection MAY occur at open or on "
                "first use of the affected section: section hashes are verified "
                "before decoding each payload, so section-hash-mismatch surfaces "
                "when that section is read rather than when the file is opened.",
    "artifacts": {
        "empty.annpack": "file shorter than the 128-byte header",
        "magic-only.annpack": "file shorter than the 128-byte header",
        "wrong-magic.annpack": "bad magic bytes",
        "wrong-version.annpack": "unsupported wire format version",
        "truncated-at-header.annpack": "directory range exceeds the file",
        "directory-bit-flip.annpack": "artifact root does not match the directory",
        "section-hash-mismatch.annpack": "section BLAKE3 hash mismatch",
        "reserved-header-set.annpack": "reserved header bytes are nonzero",
    },
}
(vec / "corruption.json").write_text(json.dumps(corruption, indent=2) + "\n")

# --- range vectors ---------------------------------------------------------
range_vectors = {
    "$comment": "PROTOCOL-v1. A lexical query over HTTP MUST use exact byte "
                "ranges and MUST reject a range-ignoring server. Counts are "
                "informative, not normative: they scale with passage block count.",
    "reference_counts": {
        "conformance-v2.annpack": {"head": 1, "range_gets": 8},
    },
    "must_reject": [
        "a 200 response to a range request that did not elect a full download",
        "an incorrect Content-Range header",
        "a truncated or oversized range body",
        "a validator (ETag) change during one read session",
    ],
}
(vec / "range.json").write_text(json.dumps(range_vectors, indent=2) + "\n")

print(f"conformance pack root: {pack_root}")
print(f"vectors written to {vec}")
PY

echo "conformance packet regenerated"
