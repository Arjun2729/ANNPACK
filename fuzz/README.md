# Fuzzing

```bash
cargo +nightly fuzz run open_consistent_pack -- -max_total_time=120
```

Corpora are not tracked. libFuzzer rediscovers them from the targets; committing
them adds files without making a run reproducible.

## Targets

| Target | Input | Reaches |
|---|---|---|
| `open_pack` | arbitrary bytes | header and magic rejection |
| `open_pack_prefixed` | directory bytes behind a constructed header | directory and entry validation |
| `open_consistent_pack` | mutated and repaired artifact | section decoding, block tables, record table, search |
| `search_query` | arbitrary query text | tokenizer, BM25, sparse block search, posting reassembly |
| `decode_varint` | arbitrary bytes | varint decoder |
| `inspect_delta` | arbitrary bytes | delta envelope parser |

## Reachability

Two gates stop byte-mutation inputs before the parser, and they had to be
addressed separately.

**The header.** Bytes 80..128 are reserved and must be zero. An earlier version
of `open_pack_prefixed` prepended only the magic and format version and let the
input supply the rest, so an input had to contain 48 consecutive zero bytes at
an exact offset to get past validation. Measured against its own corpus, 48 of
53 inputs died there. The target now constructs the whole fixed header, so the
input drives the directory instead.

**The content root.** `PackReader::open` recomputes the BLAKE3 root over the
directory and compares it to the header. Random mutation does not produce a
256-bit hash match. With the header fixed but the root check active, 59 of 72
corpus inputs die at it.

Removing both, the same corpus reaches 31 distinct validation paths: entry
ordering, reserved entry bytes, section-directory and section-header overlap,
range overflow, size limits, codec dispatch, unknown required section types,
derived-and-required conflicts, and stored/logical length mismatch.

| Corpus | Inputs | Reaches |
|---|---|---|
| `open_pack_prefixed` | 72 | directory and entry validation; 31 distinct rejection paths |
| `open_consistent_pack` | 1,199 | 1,114 (92.9%) past the root check; 692 also pass `verify_all` |

This is why `format.rs` region coverage measured near 10% from the original
byte-mutation entry points: those targets could not exercise the code behind
either gate.

## The `fuzzing-unsafe` bypass

The content-root comparison in `PackReader::open` is removed when both
`cfg(fuzzing)` and the crate feature `fuzzing-unsafe` are set. cargo-fuzz sets
the first; `fuzz/Cargo.toml` sets the second. A build with both performs no
artifact integrity verification and must never be published or deployed.

Both conditions are required because the feature alone was not a safe gate.
Cargo features are additive and cannot be excluded from `--all-features`, so
gating on the feature by itself meant `cargo test --all-features` and
`cargo build --all-features` silently produced a runtime with no integrity
verification. CI ran both, and `every_corruption_artifact_is_rejected` failed
there for three releases before anyone read the log.

An earlier version of this file claimed `open_pack` runs without the bypass, so
that the root check itself remained fuzzed. That was wrong. `fuzz/Cargo.toml`
enables the feature for the whole fuzz crate and cargo-fuzz sets `cfg(fuzzing)`
for every target in it, so **every** target here runs with the check removed,
`open_pack` included. Cargo has no per-target feature selection that would
change this.

The root check is therefore not fuzzed at all. That is an acceptable gap rather
than a hidden one: it is a 256-bit hash comparison, and a byte-mutation fuzzer
cannot satisfy it by construction, which is the whole reason the bypass exists.
It is covered by `tests/corruption.rs` and `tests/conformance_vectors.rs`, which
assert that a mismatched root is rejected — under `--all-features`, which is now
what makes that assertion meaningful.

## `open_consistent_pack`

Splices fuzzer bytes into a valid artifact, then restores the container's
self-consistency: every section hash is recomputed from the bytes its directory
entry references, then the content root is recomputed over the non-signature
entries. The envelope is well-formed; its contents are arbitrary.

The splice may land in the directory as well as in section payloads, so offsets,
lengths, codecs, flags, and section-format versions are exercised against a
container that hashes correctly.

This target is additive. `open_pack` covers the reject-early paths that
`open_consistent_pack` skips by construction; both should run.

## `search_query`

Holds the artifact fixed and varies the query. This is the input surface a
deployed reader is most exposed to, since artifacts are typically pinned and
queries are not.

Exercises normative tokenization (NFKC, the technical punctuation set, edge
trimming), BM25 scoring, the sparse dictionary-block search, posting reassembly
across block boundaries, ordinal arithmetic into the record table, and the
id-sorted binary search used by `get_passage`.

## Status

No crashes found. Scheduled CI runs 60–120 seconds per target; `deep-fuzz.yml`
runs longer. At those durations the result is weak evidence. The supported claim
is narrower: the parser is now reachable by fuzzing, which it previously was not.
