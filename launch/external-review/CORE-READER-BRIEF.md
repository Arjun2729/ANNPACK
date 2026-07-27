# Brief: independent ANNPack Core reader (paid engagement)

**Deliverable:** a read-only ANNPack Core v1.0-draft reader in TypeScript or Go,
written from the specification, plus a conformance report and a written record of
every ambiguity found.

**Budget guide:** $8,000–15,000. **Expected effort:** 3–5 days for a competent
systems engineer. **Timeline:** 3 weeks from start.

---

## Why this engagement exists

The specification cannot lose its `-draft` marker until a genuinely independent
implementation passes the conformance suite. We have a clean-room Python reader,
but it was produced inside this repository with AI assistance, so it does not
qualify. We are not asking you to confirm that the format works — we are asking
you to find the places where the specification is ambiguous enough that two
honest implementers would disagree.

**We already know this happens.** Our own clean-room reader diverged from the
reference on tokenization and the ranking boost, because the spec text was
underspecified. It nonetheless passed the then-current suite, because the golden
queries did not exercise those tokens. That is the failure mode you are hired to
surface.

**Finding disagreements is the deliverable, not a problem.** We will publish your
report including everything you found, unedited.

---

## Rules of engagement

Read only:

- `spec/CORE-v1.0-draft.md`
- `spec/FORMAT-v3.md`
- `spec/SECURITY.md`
- `spec/MEDIA-TYPES.md`
- `spec/PROTOCOL-v1.md`
- `spec/EVIDENCE-v1.md` (optional — receipts are a stretch goal)
- [`spec/conformance/`](../../spec/conformance/) — the packet, its artifacts, and the runner
- [`spec/COMPATIBILITY.md`](../../spec/COMPATIBILITY.md) — what happens to a format defect you report

Do **not** read `rust/`, `web/annpack-browser.js`, `bindings/`, or any existing
reader in this repository. If the specification is unclear, **write the ambiguity
down and make a choice** — do not resolve it by looking at our code. The
ambiguity log is as valuable to us as the reader.

You may use standard libraries for JSON, BLAKE3, Ed25519, zlib/DEFLATE, and HTTP.
These do not count toward the size target of roughly 500 source lines.

---

## Scope

1. Parse and bounds-check the 128-byte header and the 80-byte section directory.
2. Verify the artifact root before interpreting any section, and verify each
   section's BLAKE3 hash before decoding it.
3. Read Core sections 1–6 and accept manifest section format versions 1 and 2.
4. Fetch passage blocks by exact HTTP byte range.
5. Rank with the normative BM25 profile in FORMAT-v3 §6.1–6.2 — including exact
   tokenization, the technical punctuation set, and boost `3.0`.
6. Emit the Core evidence envelope for each result.
7. Implement every security invariant in `SECURITY.md`.

Stretch goal (priced separately if you want it): verify an `annpack-receipt-v1`
document per `EVIDENCE-v1.md`.

## How you will be measured

```bash
cd spec/conformance
./run.py --adapter ./your-adapter --implementation "go/your-name" --output report.json
```

42 checks. The reference implementation passes 42/42
(`spec/conformance/reference-report.json`). The adapter contract is four verbs
and is documented in `spec/conformance/README.md`.

## Acceptance criteria

- `run.py` reports `"conformant": true`.
- Independently computes the conformance artifact root
  `9a0723f89f21a060fc9f3458466199baa27a755c4e611943a6e8d401874f70ef`.
- Matches every score as an **IEEE-754 bit pattern**, not merely the same ranking
  order. (Note: many JSON parsers, including serde_json without
  `float_roundtrip`, lose up to 1 ULP reading a decimal double. Compare bits.)
- Returns exactly one hit for `std::move` and one for `foo_bar`. The corpus
  contains a decoy page holding those tokens split apart; a splitting tokenizer
  matches two and ranks the wrong page first for `foo_bar`.
- Rejects all eight corruption artifacts without panicking.
- Opens the manifest-v1 fixture and refuses an unknown manifest format version
  with a version error.

## One specific thing to check hard

`SECURITY.md` requires bounded decompression. Our own clean-room Python reader
checked the declared ratio limit from the directory and then called
`zlib.decompress()` with no output bound — so a pack declaring a small
`logical_length` while shipping a decompression bomb would exhaust memory before
the length check ran. It still claimed to implement all invariants.

Please treat "bounded allocation" as meaning bounded *during* inflation, and tell
us if the specification does not say so clearly enough. We suspect it does not.

## Deliverables

1. Source, permissively licensed, in a repository we do not control.
2. The conformance report JSON.
3. **An ambiguity log**: every place the spec admitted more than one reading, what
   you chose, and why.
4. A short note on anything that was unnecessarily hard to implement.

Payment is not contingent on the reader passing. It is contingent on the reader
being genuinely independent and the ambiguity log being honest.
