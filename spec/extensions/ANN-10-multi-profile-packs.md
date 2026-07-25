# ANN-10: Multi-profile packs ("fat packs")

Status: implemented draft, disabled by default. Requires ANNPack Core v1.0-draft.

## Thesis

A single pack may carry several retrieval representations at once — Core
lexical, ANN-1 vectors, ANN-7 expansion, ANN-8 vocabulary expansion, ANN-9
anchors — and the runtime selects whichever profile it supports, falling back
deterministically, always ending at Core lexical.

## Why this is uniquely cheap here

A hosted vector service pays for every representation it indexes: more RAM, more
CPU, a running cost per method per day. ANNPack is range-addressed and served
from a static file server, so an unused profile costs **storage** and
**near-zero transfer** — its byte ranges are simply never requested. Adding a
second or third representation does not add a serving cost, only a one-time build
cost and some bytes at rest.

This is the same asymmetry as a macOS universal binary or a multi-arch OCI image
manifest (`application/vnd.oci.image.index.v1+json`): one artifact carries
several targets, each consumer fetches only the slice it runs. Universal binaries
and multi-arch manifests are the precedent; ANN-10 applies it to retrieval
representations under range serving.

## Wire format

ANN-10 is a manifest descriptor, consistent with how ANN-5 and ANN-6 extend the
manifest. No new binary section type. `manifest.retrieval_profiles` is an
ordered array; order **is** the fallback order and the last entry MUST be the
Core lexical profile:

```json
"retrieval_profiles": [
  {"id": "vectors",   "kind": "vector",    "section_ids": [7,8,9], "requires": ["vector-ivf-flat-dot"]},
  {"id": "expansion", "kind": "expansion", "section_ids": [13],    "requires": ["term-overlay-expansion"]},
  {"id": "lexical",   "kind": "lexical",   "section_ids": [3,4,5,6], "requires": ["lexical-bm25"]}
]
```

Selection is deterministic: the runtime walks the array in order and picks the
first profile whose `requires` capabilities it all supports. Because the final
entry is Core lexical and every conformant reader supports `lexical-bm25`,
selection always terminates. `section_ids` lets a client compute exactly which
byte ranges a chosen profile needs, and therefore which it may skip.

## `inspect`

`annpack inspect` reports `retrieval_profiles` and, for each, whether the
reference runtime supports it, so an operator can see which profiles a pack
carries without searching it.

## Costs

- Index size: the sum of the included profiles' sections.
- Build time: the sum of the included profiles' build costs (all paid once,
  offline for the derived ones).
- Transfer / query p95: a client that selects one profile fetches only that
  profile's ranges. Unused profiles are never fetched (see the range-request
  conformance test).

## Required runtime support

None for Core: a Core reader ignores `retrieval_profiles` and searches
lexically. An ANN-10 reader implements the deterministic selection walk.

## Degradation

A reader that ignores `retrieval_profiles` reproduces Core results exactly. The
manifest field is optional and additive, exactly as ANN-5/ANN-6 manifest fields
are.

## Rejection rules (each has a negative fixture)

- `retrieval_profiles` whose last entry is not the Core lexical profile;
- a profile referencing a `section_id` that is not present in the pack;
- duplicate profile `id`.

## Honesty

No profile is enabled by default and none is measured against another here.
ANN-10 is a packaging mechanism; it makes no retrieval-quality claim. The
"near-zero transfer for unused profiles" property is a consequence of range
serving and is demonstrated by a range-request test, not a performance claim
about any specific corpus.

## Open questions

- Whether clients should be allowed to fuse multiple profiles (rather than
  select one) is unspecified and unmeasured.
