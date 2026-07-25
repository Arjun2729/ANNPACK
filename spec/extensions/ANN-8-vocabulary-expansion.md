# ANN-8: Vocabulary-space expansion

Status: implemented draft, disabled by default. Requires ANNPack Core v1.0-draft.

## Thesis

ANN-8 stores, for each passage, weighted terms over a **shared vocabulary**
instead of opaque dense coordinates (the SPLADE lineage). A shared vocabulary is
a shared space; a private coordinate system is not — so the representation is
model-portable, drops into the existing inverted index, and is
human-inspectable. A reviewer can read *why* a passage matched, which
strengthens the evidence story rather than weakening it. Like ANN-7, no model
runs at query time.

## Determinism

Identical discipline to [ANN-7](ANN-7-query-expansion.md): an external model
emits raw term weights offline; `annpack generate splade` canonicalizes and
quantizes them into a pinned, hashed sidecar recording `generator`, `model`,
`revision`, and the vocabulary/quantization descriptor; `annpack build --splade
<sidecar>` consumes the sidecar, records `{kind, section_id, generator, model,
params, sidecar_digest}` in `manifest.derived_inputs`, and writes the derived
section. No model executes during `build`; a second build from identical inputs
is byte-identical.

## Wire format

Section type **13, Term overlay** (`format_version = 1`), codec 1, flags
`required=0`, `derived=1`, with `kind = "splade-v1"`. Same JSON envelope as
ANN-7, with a populated `vocabulary` object specifying vocabulary identity and
weight quantization in the section header:

```json
{
  "kind": "splade-v1",
  "generator": "splade-ref",
  "model": "<model name>",
  "revision": "<exact revision>",
  "threshold": null,
  "vocabulary": {
    "id": "<vocabulary identity, e.g. bert-base-uncased-wordpiece>",
    "size": 30522,
    "quantization": "linear-u16",
    "scale": 0.001
  },
  "terms": { "<vocab term>": [[<passage_ordinal>, <quantized_weight>], ...] }
}
```

`quantization` and `scale` make the integer weights reproducible: the real
weight is `quantized_weight * scale`. Terms are lexicographically ordered;
posting ordinals are strictly increasing; weights are non-negative integers.
Vocabulary identity is required — two packs are comparable only if their
`vocabulary.id` matches.

## Query path

Pure BM25 overlay, identical mechanism to ANN-7 (`splade_weight * idf(term) * (w
/ (w+1))`, `w` the dequantized weight), tunable at query time via
`--splade-weight`, default 0.0. Disabled by default; default reproduces Core.

## Costs

- Index size: comparable to ANN-7; a dense SPLADE expansion can be larger than a
  doc2query expansion because it may touch more vocabulary entries per passage.
- Build time: negligible (copied from sidecar). Generation is offline.
- Precision risk: over-expansion dilutes precision; the shared vocabulary bounds
  the term space but does not eliminate the risk. Disabled by default.

## Required runtime support

None for Core. An ANN-8 reader adds the same overlay-scoring path as ANN-7 plus
`vocabulary.id` matching if a caller supplies query-side term weights.

## Degradation

A reader that ignores section 13 reproduces Core results exactly. Optional
skipping applies; unknown-required rejection is unchanged.

## Rejection rules (each has a negative fixture)

- ordinal out of range; non-increasing ordinal; negative/non-finite weight;
- missing or empty `vocabulary.id` when `kind = "splade-v1"`;
- unknown `quantization`;
- provenance digest mismatch.

## Honesty

Unmeasured. Human-inspectability is a real property of the format; any retrieval
improvement is not claimed and not measured. Disabled by default.

## Open questions

- Whether a shared open vocabulary can be pinned across model revisions without
  drift is speculative.
- Interaction between ANN-7 and ANN-8 overlays in the same pack is unspecified
  and unmeasured.
