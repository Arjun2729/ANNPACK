# WITHDRAWN — 2026-07-26 FastAPI 3-mode retrieval quality report

**Status: withdrawn 2026-07-27. Do not cite any number in this directory.**

The report is retained unedited so the withdrawal is auditable. Its numbers must
not appear in the README, RELEASE-READINESS, outreach, or any external claim.

## Why it was withdrawn

**1. The vector and hybrid rows are not reproducible from committed artifacts.**
`qrels-labeled.jsonl` contains 65 queries and **zero** `query_vector` fields, yet
the report publishes vector and hybrid tables. The run's actual input was a
vector-augmented file in a temporary scratch directory that was never committed,
along with a vector pack that was never committed. The documented command cannot
reproduce the published table.

**2. The ANN-7 / ANN-8 rows did not evaluate ANN-7 or ANN-8.**
The evaluated pack declares `extensions: ["ANN-1"]` and carries no term overlay
of either kind. `--compare-extensions` ran lexical search with a non-zero overlay
weight against a pack with no overlay, so those rows necessarily reproduced Core
lexical exactly. The report's statement that "ANN-7/ANN-8 overlays add nothing
here" is therefore unsupported: the overlays were never present.
`evals/evaluate.py` now refuses `--compare-extensions` unless the pack actually
declares the requested extension.

**3. The benchmark is saturated.** Lexical recall@5 = 1.000 cannot discriminate
between rankers. The report said so itself, honestly and prominently — but a
saturated benchmark still cannot support the comparative table it presents.

**4. The roots predate the v0.4.0 manifest boundary.** The `4d3ebb10…` and
`c714755…` roots were produced under the v0.3.x root scheme and no current
builder reproduces them.

## What survives

`qrels-labeled.jsonl` — 65 labeled queries — remains potentially useful **once its
provenance is established**: it must be confirmed as human-authored queries with
human judgments made independently of the implementation, and committed with that
attestation. Until then it is unverified input, not evidence.

## What is owed before any retrieval-quality claim

1. A **hard-negative eval**: paraphrase-only queries with no lexical overlap with
   the target span, so lexical can actually fail and the comparison discriminates.
2. **Committed, complete inputs**: qrels including query vectors, the exact pack,
   and a one-command recipe that reproduces every published number.
3. **Separated metrics**: passage-level and source-document-level recall/MRR
   reported independently rather than merged into one denominator.
4. **Independent labels** with recorded provenance.

Gates 4, 5, and 6 are open. No retrieval-quality claim is currently supportable.
