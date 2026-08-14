# Retrieval quality evaluation

This directory measures retrieval quality, not latency. `fixture-qrels.jsonl` only tests the harness and must never be presented as product quality evidence.

**When this matters.** Adyar makes no retrieval-quality claim. Ranking is conventional BM25 with optional vectors, and what the project contributes is the evidence chain around a result rather than the result's rank. The format is usable without a number here.

A number is owed for two decisions: any comparative claim, and turning an optional retrieval mode on by default (AN-1 vectors, or the AN-7/AN-8 overlays). Both need evidence. The default lexical path does not, because it claims nothing.

A full evaluation requires:

- one real, license-compatible documentation corpus pinned to an immutable source revision;
- 50–100 human-written queries, split between `natural-language` and `technical-token`;
- one or more human relevance judgments per query, recorded as passage IDs or source paths;
- query and passage vectors generated from the exact same pinned AN-1 profile;
- published lexical, vector, and hybrid macro recall@5, hit-rate@5, and MRR@5 tied to the resulting pack root.

The evaluator accepts JSONL records shaped like:

```json
{"id":"cache-revalidation","category":"natural-language","query":"how do I revalidate cached data","relevant_source_paths":["app/api-reference/functions/revalidatePath.md"],"query_vector":[0.1,0.2]}
```

Generate a vector pack through the pinned candidate path:

```bash
target/release/adyar build docs --output target/core.adyar --name project --version VERSION
target/release/adyar export-passages target/core.adyar --output target/passages.json
npm install --prefix evals
node evals/embed.mjs --kind passages --input target/passages.json --output target/vectors.json
target/release/adyar build docs --output target/vector.adyar --name project --version VERSION --vectors target/vectors.json
node evals/embed.mjs --kind queries --input evals/project-qrels.jsonl --output target/project-qrels-vectors.jsonl
python3 evals/evaluate.py --pack target/vector.adyar --queries target/project-qrels-vectors.jsonl --vector-profile ann-1-mxbai-xsmall-v1-q8-onnx --k 5 --require-vectors --require-hybrid-not-worse --output target/retrieval-eval.json
```

`--require-hybrid-not-worse` is intentionally modest: fusion must first prove it does not reduce recall relative to the better single mode. A launch claim should publish the complete table even if hybrid loses. Relevance judgments should be reviewed separately from the retrieval implementation; generated queries without human adjudication are not an honest evaluation.

## Comparing the optional retrieval extensions

`--compare-extensions` adds two rows to the report, for AN-7 build-time query
expansion and AN-8 vocabulary expansion, both evaluated against Core lexical on
the same corpus, queries, and judgments. The overlays are pure BM25 overlays; they
run the lexical search path with a non-zero overlay weight and need no query
vector:

```bash
target/release/adyar generate expansion raw-expansion.json --output expansion.sidecar.json --threshold 0.5
target/release/adyar build docs --output target/exp.adyar --name project --version VERSION --expansion expansion.sidecar.json
python3 evals/evaluate.py --pack target/exp.adyar --queries evals/project-qrels.jsonl --k 5 \
  --compare-extensions --expansion-weight 1.0
```

**None of these methods is measured to improve retrieval, and none is enabled by
default.** The two-query fixture and the FastAPI corpus are both too easy to
differentiate methods — lexical hits the ceiling on each. A harder corpus now
exists in [`corpora/`](corpora/README.md), stratified so lexical can fail; that
is the one to evaluate an extension against. Do not report improvement numbers,
percentages, or comparisons from this harness until such a corpus and human
judgments exist. The report carries an `extensions_note` restating this.
