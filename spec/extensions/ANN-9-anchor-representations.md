# ANN-9: Anchor-based relative representations

Status: **REJECTED / withdrawn (2026-07-25).** Retained for the historical record.

Empirical evaluation on real model pairs (relevance retrieval over FastAPI docs,
7 runs) showed anchor **relative coordinates are strictly dominated** in every
regime: for same-dimension model pairs raw cross-model comparison already
recovers ~99% of same-model retrieval (no bridge needed), and for the rare
different-dimension case an **anchor-supervised linear adapter** recovers ~95% of
recall while relative coordinates recover ~44%. The relative-coordinate
*retrieval path* is therefore withdrawn: it is never advertised as a capability
and never selected by ANN-10.

What is retained: the anchor **set** (the anchor texts shipped in the pack) is
kept as decode-only scaffolding, because those shared anchor texts are exactly
the supervision an anchor-supervised cross-model *adapter* needs — the capability
this extension aimed at, delivered by a different mechanism.

The original research-grade design follows, unchanged, for reference.

---

Status (original): **research-grade, unvalidated**, disabled by default. Requires
ANNPack Core v1.0-draft. Format and reader path only; no quality is claimed.

## Thesis

A dense vector is a coordinate in one model's private space; it is meaningless
to any other model. ANN-9 instead ships a fixed **anchor set** inside the pack
and stores each passage's similarity to those anchors. Because the anchors
travel with the pack, any model can embed the same anchors and compute
comparable coordinates in the *relative* space the anchors define.

The analogy is explicit and load-bearing. An ICC colour profile does not ship
the same sensor; it declares the space a device's numbers live in, and a
canonical connection space (CIEXYZ/Lab) makes translation between devices
possible. Unicode does not ship the same font; it declares code points that any
font can render. ANN-9 content declares its space (the anchor set), and the
anchors are the canonical connection space that makes translation possible.

State plainly: this frees a consumer from needing the **same** model. It does
**not** free a consumer from needing **a** model — the consumer must still embed
the anchors with something. Whether the resulting relative coordinates preserve
enough retrieval signal to be useful is **unknown and unmeasured**. This
extension is research-grade.

## Wire format

Two sections:

- Section type **14, Anchor set** (`format_version = 1`, codec 1, `required=0`,
  **not** derived — the anchors are canonical reference inputs shipped in the
  pack, not derived from passages):

  ```json
  { "space_id": "<identifier>", "anchors": ["<anchor text>", ...] }
  ```

- Section type **15, Anchor coordinates** (`format_version = 1`, codec 1,
  `required=0`, `derived=1` — coordinates are derived from passages and are
  matching-only, never citable):

  ```json
  {
    "space_id": "<matches section 14>",
    "metric": "cosine",
    "quantization": "linear-i16",
    "scale": 0.0001,
    "coordinates": [[<q_0>, <q_1>, ...], ...]
  }
  ```

`coordinates` has one row per passage in deterministic corpus order, each row of
length `anchors.len()`. Values are quantized signed integers; the real
similarity is `value * scale`. On open the reader checks: `space_id` agreement
between sections 14 and 15, one row per passage, uniform row length equal to the
anchor count, all values finite, recognized `metric` and `quantization`, before
any allocation of attacker-controlled size.

## Query path (reader path only)

A consumer embeds the in-pack anchors with any model, builds a query row of the
same length, and scores passages by the declared `metric` over the relative
coordinates. The reference reader implements decoding, validation, and cosine
scoring behind an explicit opt-in mode. It makes no quality claim and is not
enabled by default.

## Costs

- Index size: `passages * anchors` quantized values, plus the anchor texts.
- Build time: negligible (copied from a sidecar built exactly as ANN-7/ANN-8).
- Precision risk: entirely unknown. This is research-grade.

## Required runtime support

None for Core. An ANN-9 reader adds anchor-coordinate decoding and scoring, and
must be able to embed the anchor texts with *some* model to form a query row.

## Degradation

A reader that ignores sections 14 and 15 reproduces Core results exactly.
Optional skipping applies; unknown-required rejection is unchanged.

## Rejection rules (each has a negative fixture)

- `space_id` mismatch between sections 14 and 15;
- coordinate row count not equal to passage count;
- ragged rows (row length != anchor count);
- non-finite value; unknown `metric` or `quantization`.

## Honesty

Nothing here is measured. This is speculative and marked research-grade in this
sentence, not in a footnote. No adoption, no "solves", no percentages. Disabled
by default.

## Open questions

- Do relative coordinates retain usable retrieval signal? Unknown.
- How many anchors, and chosen how? Unknown.
- Cross-model agreement in the relative space is asserted as *possible*, not
  demonstrated.
