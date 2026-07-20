# Retrieval quality evaluation

This directory measures retrieval quality, not latency. `fixture-qrels.jsonl` only tests the harness and must never be presented as product quality evidence.

The launch evaluation requires:

- one real, license-compatible documentation corpus pinned to an immutable source revision;
- 50–100 human-written queries, split between `natural-language` and `technical-token`;
- one or more human relevance judgments per query, recorded as passage IDs or source paths;
- query and passage vectors generated from the exact same pinned ANN-1 profile;
- published lexical, vector, and hybrid macro recall@5, hit-rate@5, and MRR@5 tied to the resulting pack root.

The evaluator accepts JSONL records shaped like:

```json
{"id":"cache-revalidation","category":"natural-language","query":"how do I revalidate cached data","relevant_source_paths":["app/api-reference/functions/revalidatePath.md"],"query_vector":[0.1,0.2]}
```

Generate a vector pack through the pinned candidate path:

```bash
target/release/annpack build docs --output target/core.annpack --name project --version VERSION
target/release/annpack export-passages target/core.annpack --output target/passages.json
npm install --prefix evals
node evals/embed.mjs --kind passages --input target/passages.json --output target/vectors.json
target/release/annpack build docs --output target/vector.annpack --name project --version VERSION --vectors target/vectors.json
node evals/embed.mjs --kind queries --input evals/project-qrels.jsonl --output target/project-qrels-vectors.jsonl
python3 evals/evaluate.py --pack target/vector.annpack --queries target/project-qrels-vectors.jsonl --vector-profile ann-1-mxbai-xsmall-v1-q8-onnx --k 5 --require-vectors --require-hybrid-not-worse --output target/retrieval-eval.json
```

`--require-hybrid-not-worse` is intentionally modest: fusion must first prove it does not reduce recall relative to the better single mode. A launch claim should publish the complete table even if hybrid loses. Relevance judgments should be reviewed separately from the retrieval implementation; generated queries without human adjudication are not an honest evaluation.
