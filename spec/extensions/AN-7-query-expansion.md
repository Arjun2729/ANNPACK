# AN-7: Build-time query expansion

Status: implemented draft, disabled by default. Requires ANNPack Core v1.0-draft.

## Thesis

Dense retrieval is symmetric: it needs the same model at query time as at build
time, on both ends, forever. AN-7 instead pushes semantic understanding into
build time, where compute is unlimited and no cross-party model agreement is
required. For each passage, an offline model generates the questions that
passage answers (the doc2query / docTTTTTquery lineage). The generated terms are
matched with the existing BM25 machinery. **No model runs at query time.**

## Determinism

Generation is not part of the build. It is a separate offline step:

1. An external model emits raw candidate questions per passage into a JSON file.
2. `annpack generate expansion` deterministically filters, tokenizes, sorts, and
   canonicalizes those candidates into a pinned **sidecar** and records a
   manifest (`generator`, `model`, `revision`, `threshold`) plus the sidecar's
   own BLAKE3 digest.
3. `annpack build --expansion <sidecar>` consumes the sidecar as an input,
   validates it against the deterministic corpus order, writes the derived
   section, and records `{kind, section_id, generator, model, params,
   sidecar_digest}` in `manifest.derived_inputs`.

Because no model executes during `build`, a second build from identical inputs
produces a byte-identical artifact and root. If the sidecar is absent the build
produces a valid Core-only pack.

## Filtering (less is more)

Unfiltered generation injects hallucinated terms and *lowers* precision; the
Doc2Query-- "when less is more" finding is that dropping low-relevance generated
queries improves the index. `annpack generate expansion` therefore requires a
`--threshold` in `[0,1]`; candidates whose generator-reported relevance is below
it are discarded before inclusion. The threshold and generation model are
recorded in the sidecar manifest and copied into `manifest.derived_inputs`.

## Wire format

Section type **13, Term overlay** (`format_version = 1`), codec 1 (zlib
DEFLATE), flags `required=0`, `derived=1`. Derived sections are matching-only
and are never citable (see Evidence integrity below). The section is
deterministic UTF-8 JSON:

```json
{
  "kind": "expansion-v1",
  "generator": "docTTTTTquery-ref",
  "model": "<model name>",
  "revision": "<exact revision>",
  "threshold": 0.30,
  "vocabulary": null,
  "terms": {
    "<term>": [[<passage_ordinal>, <weight>], ...]
  }
}
```

`terms` is a lexicographically ordered map. Each posting list is ordered by
strictly increasing passage ordinal. `weight` is a **positive** integer (the
count of surviving generated questions for that passage that contain the term).
Expansion terms are decoupled from raw passage text — they live only here, never
appended to Passage Data — because naive appending degrades the lexical length
model and breaks evidence integrity.

The section is validated on open: ordinals in range, strictly increasing per
list, weights finite and strictly greater than zero, `kind` recognized. A zero
weight MUST be rejected: it carries no matching signal, and permitting it would
give one retrieval state two legal encodings, posting-absent and
posting-present-with-zero. No allocation occurs
before the declared logical length and decompression-ratio bounds are checked by
the Core container reader.

## Query path

The query path stays pure BM25 with no query-time model. Each query term that
appears in the overlay dictionary contributes

```
expansion_weight * idf(term) * (w / (w + 1))
```

added to that passage's BM25 accumulator, where `w` is the stored integer
weight and `idf` is the Core BM25 idf for the term. `expansion_weight` is a
runtime parameter (`--expansion-weight`), so scoring is tuned without a rebuild.

**`expansion_weight` defaults to 0.0.** With the default, AN-7 has no effect on
ranking and Core results are reproduced exactly. The extension is never enabled
by default.

## Costs

- Index size: one extra derived section; grows with vocabulary and per-passage
  question count. Deflate-compressed; unfetched under lexical-only serving.
- Build time: negligible (the section is copied from the sidecar). Generation
  cost is paid offline, out of band, and is not part of the build.
- Precision risk: unfiltered or over-weighted expansion can inject hallucinated
  terms and reduce precision. This is why filtering is mandatory and the weight
  defaults to zero.

## Required runtime support

None for Core readers. An AN-7 reader adds the overlay-scoring path above.

## Degradation

A reader that ignores section 13 opens, verifies, and lexically searches the
pack with identical Core results. Section 13 is optional; unknown-optional
skipping applies. The existing unknown-**required** rejection is unchanged.

## Rejection rules (each has a negative fixture)

- overlay ordinal out of passage range;
- non-increasing or duplicate ordinal within a posting list;
- zero, negative or non-finite weight;
- unrecognized `kind`;

### Not a rejection rule: `sidecar_digest`

Earlier drafts listed "`derived_inputs` provenance digest that does not match the
section bytes" as a rejection rule. That rule was **impossible as stated and was
never implemented**, and it contradicted
[`SECURITY.md`](../SECURITY.md#derived-retrieval-sections).

`sidecar_digest` is the BLAKE3 of the *pinned sidecar file* the builder consumed.
It is not a hash of the emitted section, and no function of the section bytes
reproduces it, so a reader cannot check one against the other. It is recorded
provenance — an attestation of what the builder claims it consumed — not proof of
derivation. A builder could in principle record one digest and emit unrelated
section bytes.

It is covered by the artifact root like any other manifest field, so it cannot be
altered after signing. A consumer who needs to verify that a section really came
from that sidecar MUST re-run the deterministic `annpack generate` step and
compare the resulting section bytes. Derived sections are matching-only and
non-citable precisely so this gap can never affect evidence integrity.

## Honesty

None of this is measured. The current FastAPI evaluation corpus is too easy to
differentiate methods (lexical already hits the ceiling), so a harder corpus is
a prerequisite to evaluating AN-7 at all. No improvement numbers are reported
and the extension is disabled by default.

## Open questions

- What generator and threshold actually improve a hard corpus is unknown and
  speculative until measured.
- Whether expansion weight should be global or per-term is unresolved.
