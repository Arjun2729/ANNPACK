# Second readers

Implementations of ANNPack Core other than `rust/`, kept here with their
conformance reports.

## `annpack_reader.py` — spec-derived Python reader

Written against the specification text only, and it passes.

```bash
pip install blake3
./spec/conformance/run.py \
  --adapter ./spec/conformance/readers/python-adapter.sh \
  --implementation python/second-reader \
  --skip-evidence
```

**40/40 checks pass** ([report](python-report.json)). That includes exact
IEEE-754 scores for `std::move`, `foo_bar`, `package.module`, `AP-104` and
`@scope/pkg` against the decoy corpus, both manifest format generations, and all
eight corruption artifacts rejected. It measures **459 executable lines**,
against the 600-line Core budget.

It skips Evidence v1 receipts, which Core does not require.

### What this does and does not establish

**It does not make Core interoperable, and the `-draft` marker stays on.**

This reader was written in the same session, by the same author, as the changes
to the reference implementation it is checked against. Shared authorship means
shared blind spots: an assumption the reference makes silently is one this
reader is likely to make silently too, and neither would notice. That is
precisely the failure mode an independent implementation exists to catch, so
this one cannot be counted as one.

What it does establish, which was not established before:

- **The specification is sufficient.** A working reader can be built from the
  prose alone. Every constant, layout, bound, and algorithm needed is written
  down somewhere a reader can find it.
- **The reference is not relying on undocumented behaviour.** A reader built
  only from the specification agrees with it on ranking *and on exact scores* —
  the place where the previous clean-room attempt silently diverged.
- **The size budget is real.** 459 lines, measured with a stated counting
  method, against a 600-line budget.
- **Five ambiguities are now written down** in the reader's header comment.
  Those are the deliverable for whoever writes the genuinely independent reader:
  they are the places where two honest implementers could still disagree.

### The bar that remains

Core v1.0-draft loses `-draft` when a reader written by someone with **no access
to this repository's implementations** passes `spec/conformance/`. The packet in
[`../README.md`](../README.md) is what to hand them. This reader is a lower bound
on that work, not a substitute for it.
