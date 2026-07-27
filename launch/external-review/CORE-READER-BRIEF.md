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
- the conformance packet artifacts

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

## Acceptance criteria

- Independently computes the golden artifact root
  `b1f63b4acdbee0a89de5c3455505be279845b4eda644c0d6c931814355a9d70b`.
- Reproduces every expected score in the tokenizer and scoring vectors **exactly**
  — not merely the same ranking order.
- Correctly handles `std::move`, `foo_bar`, `package.module`, `AP-104`.
- Rejects every artifact in the corruption corpus without panicking.
- Opens the manifest-v1 compatibility fixture and refuses an unknown manifest
  format version with a version error.
- Answers the conformance queries with the expected passages and passage hashes.
- Emits the machine-readable conformance report.

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
