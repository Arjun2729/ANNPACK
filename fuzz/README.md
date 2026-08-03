# Fuzzing

```bash
cargo +nightly fuzz run open_consistent_pack -- -max_total_time=120
```

Corpora are not tracked. libFuzzer rediscovers them from the targets, and
committing them added hundreds of files without making a run reproducible.

## Targets

| Target | Input | Reaches |
|---|---|---|
| `open_pack` | arbitrary bytes | header and magic rejection |
| `open_pack_prefixed` | bytes behind a valid magic | directory bounds checks |
| `open_consistent_pack` | a mutated *and repaired* artifact | section decoding, block tables, record table, search |
| `search_query` | arbitrary query text | tokenizer, BM25, sparse block search, posting reassembly |
| `decode_varint` | arbitrary bytes | varint decoder |
| `inspect_delta` | arbitrary bytes | delta envelope parser |

## Why the first two cannot reach the parser

Opening a pack recomputes the BLAKE3 content root over the section directory and
compares it to the header. Random mutation does not produce a 256-bit hash
match, so every input dies at that gate — and everything behind it is
unreachable: section decoding, codec dispatch, the lexical block tables, the
passage record table, and the entire search path.

This is measured, not assumed. Replaying each corpus through `PackReader::open`:

| Corpus | Inputs | Past the root gate |
|---|---|---|
| `open_pack_prefixed` | 53 | **0 (0.0%)** — after 8.1M executions |
| `open_consistent_pack` | 1,199 | **1,114 (92.9%)**, 692 also pass `verify_all` |

That gap is the reason `format.rs` region coverage sat near 10% and why the
number was reported as a limitation rather than a target to optimize: the
targets were structurally incapable of improving it.

## What `open_consistent_pack` does

It splices fuzzer bytes into a valid artifact, then **repairs the container's
self-consistency** — recomputing every section hash from the bytes its entry
points at, then the content root over the non-signature entries, exactly as a
writer would. The envelope ends up well-formed; everything inside it is
arbitrary.

The splice deliberately may land in the directory as well as in section
payloads, so offsets, lengths, codecs, flags and section-format versions all get
exercised against a container that still hashes correctly.

It is additive, not a replacement: `open_pack` still covers the reject-early
paths that this target skips by construction. Both should run.

## What `search_query` does

Holds the artifact fixed and makes the *query* arbitrary. That is the other half
of the attack surface, and the half a deployed reader is most exposed to — a
pack is typically pinned and trusted, while queries arrive from users and agents.

It exercises normative tokenization (NFKC, the technical punctuation set, edge
trimming), BM25 scoring, the sparse dictionary-block search, posting reassembly
across block boundaries, ordinal arithmetic into the record table, and the
id-sorted binary search behind `get_passage`.

## Status

No crashes found. That is a weak statement at these durations — 60–120s per
target in scheduled CI, longer in `deep-fuzz.yml`. The useful claim is narrower:
the parser is now actually being fuzzed, which it previously was not.
