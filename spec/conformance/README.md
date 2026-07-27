# ANNPack Core conformance packet

Everything needed to implement and validate an independent ANNPack Core reader.
One command runs every check and writes a machine-readable report.

```bash
./run.py --adapter ./my-adapter --implementation "go/acme-reader" --output report.json
```

Reference implementation, for comparison: **42/42 checks pass**
(`reference-report.json`).

---

## The normativity rule

**The specification is normative. The reference implementation is what changes
when they disagree.**

This is a commitment, not a preference. If your reader and `rust/` disagree and
the specification permits your reading, the reference implementation has the bug
and we will fix it. We will not retroactively edit the specification to match our
code.

The corollary matters more: **a disagreement you find is a deliverable, not a
failure.** Record it. The vectors here are pinned by `tests/conformance_vectors.rs`
so the reference cannot silently drift into agreement with itself.

### Why this packet looks the way it does

An earlier clean-room reader passed the previous conformance suite while
disagreeing with the reference on tokenization and ranking. It chose boost `2.0`
instead of `3.0`, a three-character punctuation set instead of seven, and a
regex that split `std::move` into `std` and `move`. It passed because the golden
corpus had one passage and no technical tokens.

So this packet is built to discriminate:

- The corpus contains `separate-words.md`, a decoy holding `std`, `move`, `foo`
  and `bar` as **separate words**. A conformant tokenizer matches exactly one
  passage for `std::move` and one for `foo_bar`. A splitting tokenizer matches
  two — and for `foo_bar` it ranks the **wrong page first**. The failure is a
  different top result, not a rounding difference.
- Scores are asserted **exactly**, as IEEE-754 bit patterns. A reader with the
  wrong boost constant produces identical rankings and different scores; only
  exact comparison catches it.

---

## Contents

| Path | What it is |
|---|---|
| `corpus/` | Source Markdown. Deliberately contains technical identifiers and a decoy page. |
| `artifacts/conformance-v2.annpack` | The pack under test (manifest format 2). |
| `artifacts/conformance-v2-signed.annpack` | Same content, one signature section. |
| `artifacts/conformance-v2-signed.pub` | Public key for the above. The private key is deliberately not published — you verify signatures, you do not produce them. |
| `artifacts/manifest-v1-legacy.annpack` | A v0.3-era pack, manifest format 1. Must still open. |
| `artifacts/minimal-v3.annpack` | The historical golden artifact. |
| `artifacts/corruption/` | Eight malformed artifacts. Every one must be rejected. |
| `vectors/tokenizer.json` | Normative tokenization cases. |
| `vectors/scoring.json` | Exact scores as decimal **and** IEEE-754 bit pattern. |
| `vectors/compatibility.json` | Manifest format 1 / 2 / unknown behaviour. |
| `vectors/corruption.json` | Expected rejection reasons. |
| `vectors/signature.json` | Signing must not change the artifact root. |
| `vectors/evidence.json` | A published receipt that must verify offline. |
| `vectors/range.json` | HTTP range requirements. |
| `run.py` | The runner. |
| `reference-report.json` | The reference implementation's own report. |

Regenerate with `scripts/build-conformance-packet.sh`.

---

## Specifications to implement

Read, in order:

1. [`../CORE-v1.0-draft.md`](../CORE-v1.0-draft.md) — normative requirements
2. [`../FORMAT-v3.md`](../FORMAT-v3.md) — binary encoding; **§6.1–6.2 are the
   tokenizer and BM25 profile and are fully normative**
3. [`../SECURITY.md`](../SECURITY.md) — every invariant is mandatory
4. [`../PROTOCOL-v1.md`](../PROTOCOL-v1.md) — HTTP range access
5. [`../EVIDENCE-v1.md`](../EVIDENCE-v1.md) — receipts (optional; `--skip-evidence`)

Do **not** read `rust/`, `web/annpack-browser.js`, or `bindings/`. If the
specification is ambiguous, choose, and write the ambiguity down.

Standard libraries for JSON, BLAKE3, Ed25519, zlib/DEFLATE and HTTP do not count
toward the ~500-line target.

---

## Adapter contract

Supply one executable taking four verbs. Nothing else is assumed about your
implementation.

| Invocation | Must print | Must exit |
|---|---|---|
| `<adapter> tokenize <text>` | JSON array of tokens | 0 |
| `<adapter> search <pack> <query>` | `{"results":[{"passage_id":…,"score":…}, …]}`, lexical mode, limit 10 | 0 |
| `<adapter> open <pack>` | anything | 0 if the artifact is acceptable **and its sections verify**, non-zero otherwise |
| `<adapter> verify-receipt <file>` | anything | 0 if the receipt verifies, non-zero otherwise |

`search` results must be in ranked order. Extra fields are ignored.

Two gotchas that have already bitten us:

- Some vectors begin with `-`. Guard your argument parsing (the reference
  adapter passes `--` before the text).
- `open` must fail for a section-hash mismatch. Section hashes are verified
  lazily, before decoding each payload, so opening the header alone is not
  enough — verify sections before reporting success.

See [`../../scripts/reference-adapter.sh`](../../scripts/reference-adapter.sh)
for a four-line example.

---

## Report

`run.py` writes `annpack-conformance-report-v1`:

```json
{
  "schema": "annpack-conformance-report-v1",
  "implementation": "go/acme-reader",
  "packet_pack_root": "9a0723f8…",
  "total": 42, "passed": 42, "failed": 0,
  "conformant": true,
  "results": [{"check": "tokenize 'AP-104 …'", "pass": true, "detail": ""}]
}
```

Exit status is 0 only when `failed` is 0.

---

## What we ask you to submit

1. Source, permissively licensed, in a repository we do not control.
2. `report.json` from this runner.
3. **An ambiguity log** — every place the specification admitted more than one
   reading, what you chose, and why. This is the most valuable artifact you can
   give us.
4. Anything that was unnecessarily hard to implement.

We publish all of it, including findings we have not fixed.

## One thing to check hard

`SECURITY.md` requires bounded allocation. Our own clean-room reader checked the
declared decompression ratio from the directory and then called
`zlib.decompress()` with no output bound — so a pack declaring a small
`logical_length` while shipping a bomb would exhaust memory before the length
check ran. It still reported implementing every invariant.

Treat "bounded" as meaning bounded *during* inflation, and tell us if the
specification does not say so clearly enough. We suspect it does not.
