# Reader implementations

Implementations of ANNPack Core other than `rust/`, with their conformance
reports. Both run in CI on every build.

| Reader | Result | Purpose |
|---|---|---|
| [`browser-adapter.sh`](browser-adapter.sh) → `web/annpack-browser.js` | 44/44 | The shipped browser runtime, held to the conformance contract |
| [`python-adapter.sh`](python-adapter.sh) → [`annpack_reader.py`](annpack_reader.py) | 44/44 | A reader implemented from the specification alone |

All three implementations run the complete suite, including the two Evidence v1
receipt checks. Those were previously skipped, which meant the receipt chain —
the format's central capability — had been validated against one implementation
only.

## `browser-reader.mjs`

An adapter over the runtime that `web/` serves, not a separate implementation.

The browser implements tokenization, BM25 scoring, and container parsing
independently of `rust/`. The two diverged once without detection: after a
fusion change in `rust/`, all browser smoke tests continued to pass while the
two runtimes returned different hybrid orderings for the same query against the
same artifact. Driving the browser through the conformance contract applies the
tokenizer vectors, exactly asserted IEEE-754 scores, manifest compatibility
cases, and the corruption corpus to it.

Verified to discriminate: modifying the browser tokenizer to split on `:` causes
the suite to fail.

## `annpack_reader.py`

A Core reader implemented against the specification text alone.

```bash
pip install blake3
./spec/conformance/run.py \
  --adapter ./spec/conformance/readers/python-adapter.sh \
  --implementation python/second-reader
```

44/44 checks pass ([report](python-report.json)), including exactly asserted
IEEE-754 scores for `std::move`, `foo_bar`, `package.module`, `AP-104`, and
`@scope/pkg` against the decoy corpus, both manifest format generations, and
rejection of all eight corruption artifacts, and offline verification of a
published receipt plus rejection of a tampered one. It measures 566 executable
lines against the 600-line Core budget.

Evidence v1 receipt verification is implemented: the chain from passage record
through Merkle path, logical content root, manifest, and directory to the
artifact root, plus optional Ed25519 signature verification. Ed25519 requires
the `cryptography` package; the conformance receipt is unsigned and verifies
without it.

### Scope

This reader does not establish interoperability, and Core retains its `-draft`
marker.

It was written by the same author, in the same working session, as the reference
implementation changes it validates. Shared authorship permits shared
assumptions: behavior the reference leaves implicit may be reproduced here
without either detecting it. That is the failure mode an independent
implementation exists to detect.

What it does establish:

- **The specification is sufficient to implement from.** Every constant, layout,
  bound, and algorithm required is stated in the specification text.
- **The reference implementation depends on no undocumented behavior.** A reader
  built from the specification agrees with it on ranking and on exact scores —
  the point at which an earlier clean-room attempt diverged undetected.
- **The size budget is achievable.** 566 lines under a stated counting method,
  against a 600-line budget, with receipt verification included.
- **Five specification ambiguities are recorded** in the reader's header
  comment. These are the points at which two implementers could reasonably
  diverge.

### Remaining requirement

Core v1.0-draft loses `-draft` when a reader written without access to this
repository's implementations passes `spec/conformance/`. The packet in
[`../README.md`](../README.md) is the material to supply for that work.
