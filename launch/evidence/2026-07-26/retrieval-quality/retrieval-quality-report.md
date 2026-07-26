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

## Interpretation caveat (read before the findings)

**This eval is saturated and cannot support a comparative claim.** Lexical recall@5 = 1.000
means the benchmark hit its ceiling — a metric that maxes out cannot discriminate a good ranker
from a keyword `grep`, and it gives vector/hybrid no headroom to win. FastAPI's BigQuery-export
schema is close to the friendliest possible case for BM25 (queries share exact tokens with the
target passage). So the honest reading is **"our benchmark was too easy to tell the methods
apart,"** not "embeddings lost." The one defensible statement is narrow: *for the lexical-friendly
doc-site case we tested, BM25 is sufficient.* This says nothing about paraphrase / semantic-gap
queries where lexical fails and embeddings are supposed to earn their keep — which this corpus
does not contain. A hard-negative eval (paraphrase-only queries with zero lexical overlap with the
target span) is a prerequisite before any "lexical vs. vector" claim.

## Findings

1. **BM25 lexical retrieval is sufficient on this lexical-friendly corpus** — recall@5 and hit@5
   at ceiling, MRR@5 0.895. Because recall@5 is saturated, MRR@5 is the only non-degenerate
   number here, and even it is not a fair cross-method comparison (see caveat).

2. **The candidate embedding underperforms *on this corpus* — not a general verdict.** The
   24M-parameter `mxbai-embed-xsmall` candidate reaches 0.426 recall@5 here. That is a reason not
   to ship it as the default *blindly*, but on a corpus this lexical it is an unfair test; it is
   not evidence that embeddings don't help on harder queries.

3. **Reciprocal-rank hybrid is worse than lexical *here*.** Fusing a strong lexical ranker with a
   weak vector ranker on lexical-friendly queries drags recall@5 to 0.604. This is expected RRF
   behavior when one retriever is much weaker on the test distribution; it is not a general
   indictment of hybrid.

4. **ANN-7/ANN-8 overlays add nothing here** — identical to lexical because lexical is at ceiling.
   Consistent with the "optional, off by default, none measured to improve retrieval" posture.

## Decisions this report supports

- **Gate 5 (quality table):** the table is produced with all modes shown, but it is **not
  publishable as a quality claim** while saturated — the discriminating hard-negative eval is
  still owed.
- **Gate 6 (embedding promotion):** do not promote `mxbai-embed-xsmall` to default *on this
  evidence* — but the evidence is insufficient to judge embeddings, so this is a "not yet, and we
  can't tell here" decision, not "embeddings lose."
- **Product default:** shipping **lexical-only** is the right *default* (works on every pack, no
  vector section required); it is a default choice, not a proof that vectors are unnecessary.

## Open items before this closes Gates 4–6 publicly

- **Hard-negative eval (the real blocker).** Build paraphrase-only queries with no lexical overlap
  with the target span, so lexical can actually fail and the comparison discriminates. Only then
  is a lexical/vector/hybrid table a publishable quality claim.
- **Label independence (Gate 4).** `qrels-labeled.jsonl` must be confirmed as human-authored
  queries with human judgments produced independently of the implementation (see `evals/README.md`),
  and committed with that provenance.
- **Roots predate the v0.3.1 root-scheme reset** — the `4d3ebb10…` / `c71475…` roots above were
  built under the pre-v0.3.1 builder-in-root scheme; rebuild before citing them alongside v0.3.1.
