# ANNPack retrieval quality report — FastAPI 0.115.12

**Generated:** 2026-07-26
**Binary:** `target/release/annpack` v0.3.0
**Corpus:** FastAPI 0.115.12 docs (`docs/en/docs/`), source commit `628c34e0`, MIT, 141 documents / 1864 passages
**Vectors-enabled pack root:** `4d3ebb105ba937148da8bc7702edadf5633e7fa1a145b8f04897a2095b6878af`
**Lexical-only pack root (no vectors):** `c7147550fb7a2e0ff65af4030d730b3fad923fe0f548692b868cd26369a1cc7a`
**Embedding profile:** `mixedbread-ai/mxbai-embed-xsmall-v1` rev `e6ac24e5`, 384-dim, q8/CPU (Transformers.js candidate)
**Judgments:** 65 labeled queries (`qrels-labeled.jsonl`, in this directory), relevance recorded as `relevant_source_paths`
**Command:**
```bash
python3 evals/evaluate.py --binary target/release/annpack \
  --pack fastapi-vec.annpack --queries qrels-labeled.jsonl \
  --k 5 --require-vectors --compare-extensions
```

> **Claim scope.** Metrics are valid only for this pinned corpus, pack root, query set, and
> these relevance judgments. They are not a general benchmark.

## Results (macro, k=5, 65 queries)

| Mode | recall@5 | hit@5 | MRR@5 | queries missed |
|---|---|---|---|---|
| **Lexical (BM25 Core)** | **1.000** | **1.000** | **0.895** | 0 |
| Vector (mxbai-xsmall candidate) | 0.426 | 0.862 | 0.730 | 9 |
| Hybrid (reciprocal-rank fusion) | 0.604 | 0.892 | 0.814 | 7 |
| ANN-7 expansion overlay | 1.000 | 1.000 | 0.895 | 0 |
| ANN-8 splade overlay | 1.000 | 1.000 | 0.895 | 0 |

No losing mode is hidden: vector and hybrid both underperform lexical and are reported in full.

### By category (recall@5)

| Category (n) | Lexical | Vector | Hybrid |
|---|---|---|---|
| conceptual (7) | 1.00 | 0.37 | 0.53 |
| distractor (8) | 1.00 | 0.42 | 0.56 |
| natural-language (18) | 1.00 | 0.40 | 0.59 |
| technical-token (27) | 1.00 | 0.44 | 0.64 |
| version-sensitive (5) | 1.00 | 0.55 | 0.63 |

## Findings

1. **BM25 lexical retrieval is excellent on structured developer docs** — 100% recall@5 and
   hit@5 across every category, MRR@5 0.895. recall@5 is saturated, so MRR@5 is the honest
   discriminator among methods that reach the ceiling.

2. **The candidate embedding profile is not good enough to promote.** On this real corpus the
   24M-parameter `mxbai-embed-xsmall` candidate reaches only 0.426 recall@5 and misses 9 of 65
   queries outright. It loses to BM25 in every category.

3. **Reciprocal-rank hybrid reduces quality here.** Fusing a strong lexical ranker with a weak
   vector ranker drags recall@5 from 1.000 (lexical) down to 0.604. `evaluate.py
   --require-hybrid-not-worse` would **fail**: fusion does not clear the "no worse than the
   better single mode" bar with this embedding.

4. **The build-time extensions (ANN-7 expansion, ANN-8 splade) add nothing on real docs** —
   identical to lexical because lexical is already at the recall ceiling. Consistent with the
   project's "optional, off by default, none measured to improve retrieval" posture.

## Decisions this report supports

- **Gate 5 (quality table):** produced above, tied to pack root `4d3ebb10…`, all modes shown.
- **Gate 6 (embedding promotion):** **Do NOT promote** `mxbai-embed-xsmall` to the release
  default. The real-corpus table does not support it. Vector/hybrid remain opt-in/experimental
  until a stronger embedding model clears BM25 on an adjudicated real corpus.
- **Product default:** ship **lexical-only** as the quality default. This simplifies the pack
  (no vector section required) and the retrieval story, and is fully defensible because the
  losing modes are measured and published rather than omitted.

## Open items before this closes Gates 4–6 publicly

- **Label independence (Gate 4).** `qrels-labeled.jsonl` must be confirmed as human-authored
  queries with human relevance judgments produced independently of the retrieval implementation
  (see `evals/README.md`). Until that provenance is confirmed and the file is committed with it,
  treat this as a strong internal result, not a closed public gate.
- **Corpus difficulty.** recall@5 saturates for lexical; a harder query set (more distractors,
  paraphrase, negation) would sharpen the discrimination between methods.
