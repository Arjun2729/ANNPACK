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

### Selection

A conforming runtime MUST implement three request modes:

| Request | Behavior |
|---|---|
| `lexical` (**default**) | Force the Core lexical profile. Never activates a vector or derived profile, even on a fat pack. |
| `<id>` | Activate that profile if present and runtime-supported; otherwise fall back to Core lexical — never to a *different* derived profile the caller did not ask for. |
| `auto` | Walk the array in order and activate the first supported profile. May activate a derived profile; the caller opted in. |

Because the final entry is Core lexical and every conformant reader supports
`lexical-bm25`, selection always terminates. The selected profile, the reason,
and the effective weights MUST be reported to the caller.

A profile is **supported** only if its `requires` list is non-empty, every named
capability is one the runtime can execute, and its `kind` is recognized. An empty
`requires` MUST NOT be treated as satisfied: an `all()` test over an empty list is
vacuously true, which would make any profile appear supported. An unrecognized
`kind` MUST NOT be selected and silently executed as lexical, which would report a
retrieval strategy that never ran.

### Section scoping

`section_ids` declares exactly the sections a profile needs. A runtime that
activates a profile MUST read only that profile's declared sections and MUST NOT
read another profile's. Selecting `expansion` therefore never fetches the SPLADE
ranges, and vice versa.

> Before v0.4.0 the reference runtime ignored `section_ids` and loaded every term
> overlay, so selecting one derived profile also fetched the other. The
> "unused profiles are never fetched" property held only for Core lexical, and
> the conformance test cited as evidence covered only that case. Profile-to-
> profile isolation is now asserted directly by
> `selecting_expansion_never_fetches_the_splade_ranges` and its SPLADE twin.

### Safety boundary

The descriptor is optional metadata and MUST NOT be able to affect Core.
A runtime MUST evaluate Core conformance independently of extension conformance.
If the ANN-10 descriptor fails validation, the runtime MUST ignore the descriptor
entirely and serve Core lexical, and MUST refuse any profile-enabled request
rather than falling back to some other profile. Default lexical retrieval MUST
NOT be reachable from an invalid descriptor.

> Before v0.4.0 `core_conformant` was computed after extension checks had already
> appended to a shared issue list, and the lexical fallback selected the *last*
> profile in the array when no profile of kind `lexical` existed — so a malformed
> descriptor could steer the default path onto a derived profile.

## `inspect`

`annpack inspect` reports `retrieval_profiles` and, for each, whether the
reference runtime supports it, so an operator can see which profiles a pack
carries without searching it.

## Costs

- Index size: the sum of the included profiles' sections.
- Build time: the sum of the included profiles' build costs (all paid once,
  offline for the derived ones).
- Transfer / query p95: a client that selects one profile fetches only that
  profile's declared `section_ids`. Asserted by three range-request conformance
  tests: default lexical touches no optional-profile range, selecting
  `expansion` touches no SPLADE range, and selecting `splade` touches no
  expansion range.

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
- a profile referencing a section whose type its `kind` cannot use;
- duplicate profile `id`;
- an empty `requires` list;
- a `requires` entry naming an unknown capability;
- an unrecognized `kind`;
- an empty `section_ids` list.

Each of these marks the pack `extensions_conformant: false` while leaving
`core_conformant` untouched.

## Honesty

No profile is enabled by default and none is measured against another here.
ANN-10 is a packaging mechanism; it makes no retrieval-quality claim. The
"near-zero transfer for unused profiles" property is a consequence of range
serving and is demonstrated by a range-request test, not a performance claim
about any specific corpus.

## Open questions

- Whether clients should be allowed to fuse multiple profiles (rather than
  select one) is unspecified and unmeasured.
