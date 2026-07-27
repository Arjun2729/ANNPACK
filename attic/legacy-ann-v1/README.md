# Attic: legacy ANN v1 vector engine (not ANNPack)

**This code is not part of ANNPack and does not implement the ANNPack format.**
It is retained only for history. Nothing here is built, tested, or shipped.

## What it is

An earlier, unrelated experiment: a small C IVF vector-search engine with its own
container format, compiled to WebAssembly through Emscripten.

| | legacy engine (this directory) | ANNPack (`rust/`, `spec/`) |
|---|---|---|
| Magic | `ANNP` (`0x504E4E41`) | `ANNPACK3` |
| Version | 1 | 3 |
| Header | 72 bytes | 128 bytes |
| Content | f16 IVF vectors only | documents, passages, BM25, optional vectors |
| Integrity | none | BLAKE3 per section + content root |
| Provenance | none | evidence envelopes and receipts |

The two formats share a name prefix and nothing else. Reading `ann_format.h`
expecting the ANNPack wire format will mislead you; read
[`spec/FORMAT-v3.md`](../../spec/FORMAT-v3.md) instead.

## Why it was moved here

It sat in `src/` and `include/` at the repository root — the first place any
reviewer looks — presenting an obsolete format as if it were the product. It also
contradicts the current security model: it uses `#pragma pack` and casts
attacker-controlled input directly into native structs, which
[`spec/SECURITY.md`](../../spec/SECURITY.md) explicitly forbids
("Does not cast arbitrary input into native structs").

## Do not

- Build it (`build-emscripten.sh` is kept only so the history is legible).
- Cite it as an ANNPack implementation.
- Copy its parsing patterns into anything.
