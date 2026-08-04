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

Where an implementation and `rust/` disagree and the specification permits the
implementation's reading, the defect is in the reference implementation and is
corrected there. The specification is not retroactively amended to match the
reference code.

A recorded disagreement is a deliverable of conformance testing, not a failure
of it. The vectors here are pinned by `tests/conformance_vectors.rs` so the
reference implementation cannot drift into agreement with itself.

### Discrimination requirements

An earlier clean-room reader passed the previous conformance suite while
disagreeing with the reference on tokenization and ranking. It chose boost `2.0`
instead of `3.0`, a three-character punctuation set instead of seven, and a
regex that split `std::move` into `std` and `move`. It passed because the golden
corpus had one passage and no technical tokens.

The packet is therefore constructed to detect these cases:

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
| `artifacts/conformance-v2-signed.pub` | Public key for the above. The private key is not published: conformance requires verifying signatures, not producing them. |
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

Standard libraries for JSON, BLAKE3, Ed25519, zlib/DEFLATE, and HTTP do not
count toward the 600-line Core budget.

---

## Adapter contract

The runner requires one executable accepting four verbs. Nothing else is assumed
about the implementation.

| Invocation | Must print | Must exit |
|---|---|---|
| `<adapter> tokenize <text>` | JSON array of tokens | 0 |
| `<adapter> search <pack> <query>` | `{"results":[{"passage_id":…,"score":…}, …]}`, lexical mode, limit 10 | 0 |
| `<adapter> open <pack>` | anything | 0 if the artifact is acceptable **and its sections verify**, non-zero otherwise |
| `<adapter> verify-receipt <file>` | anything | 0 if the receipt verifies, non-zero otherwise |

`search` results must be in ranked order. Extra fields are ignored.

Two implementation notes, both derived from observed failures:

- Some tokenizer vectors begin with `-`. Argument parsing must handle this; the
  reference adapter passes `--` before the text.
- `open` must fail on a section-hash mismatch. Section hashes are verified
  lazily, before each payload is decoded, so parsing the header alone is
  insufficient — sections must be verified before success is reported.

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

## Submission

A conformance submission consists of:

1. Source, permissively licensed, in an independently controlled repository.
2. `report.json` from this runner.
3. An ambiguity log: each point at which the specification admitted more than
   one reading, the reading chosen, and the reasoning. This is the primary
   deliverable of an independent implementation.
4. Any requirement that was disproportionately difficult to implement.

All submitted material is published, including findings that remain unfixed.

## Known difficult requirement

`SECURITY.md` requires bounded allocation. A prior clean-room reader validated
the declared decompression ratio from the directory and then called
`zlib.decompress()` without an output bound, so an artifact declaring a small
`logical_length` while carrying a decompression bomb would exhaust memory before
the length check executed. That reader reported implementing every invariant.

"Bounded" is to be read as bounded *during* inflation. Implementations that find
the specification insufficiently explicit on this point should record it in the
ambiguity log.
