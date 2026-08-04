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
| `open_pack_prefixed` | bytes behind a valid magic | directory bounds checks |
| `open_consistent_pack` | mutated and repaired artifact | section decoding, block tables, record table, search |
| `search_query` | arbitrary query text | tokenizer, BM25, sparse block search, posting reassembly |
| `decode_varint` | arbitrary bytes | varint decoder |
| `inspect_delta` | arbitrary bytes | delta envelope parser |

## Reachability

`PackReader::open` recomputes the BLAKE3 content root over the section directory
and compares it to the header. Random mutation does not produce a 256-bit hash
match, so byte-mutation inputs terminate at that check. Section decoding, codec
dispatch, the lexical block tables, the passage record table, and the search
path are unreachable behind it.

Measured by replaying each corpus through `PackReader::open`:

| Corpus | Inputs | Past the root check |
|---|---|---|
| `open_pack_prefixed` | 53 | 0 (0.0%), after 8.1M executions |
| `open_consistent_pack` | 1,199 | 1,114 (92.9%); 692 also pass `verify_all` |

This is why `format.rs` region coverage measured near 10% from the byte-mutation
entry points: those targets cannot exercise the code behind the root check.

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
