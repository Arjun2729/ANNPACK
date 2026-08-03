# Evaluation corpora

## `okf-hard-negatives` — the hard-negative evaluation

The corpus the project owed itself: one where lexical retrieval can fail, so the
comparison between retrieval modes measures something.

**Corpus.** All three OKF v0.2 bundles — `ga4`, `crypto_bitcoin`,
`stackoverflow` — from `GoogleCloudPlatform/knowledge-catalog` at
`3fcbb9f828c2f23d109c855ee403c3a4c81f3a96`, Apache-2.0. 47 documents, 153
passages, three unrelated domains so cross-domain distractors exist.

**Queries.** 63, in two strata:

- **technical-token** (28) — the query names identifiers the passage contains.
  Lexical should be strong; the question is whether vectors *cost* anything.
- **hard-negative** (35) — a paraphrase sharing **no discriminative token** with
  its target. Lexical has nothing to rank the target on; the question is whether
  vectors recover it.

"Discriminative" is defined against the corpus, not by taste: a token in ≥20% of
passages has an idf near zero and cannot rank anything, so sharing *the* or
*table* is permitted and sharing anything rarer is not.
[`okf-hard-negatives.build.py`](okf-hard-negatives.build.py) computes document
frequencies from the corpus and **rejects** any hard-negative candidate that
breaks the rule — 32 of the first 35 drafts were rejected and rewritten. Hardness
is therefore a checkable property of the data, not a claim about it.

### Results

k=5, pack root `f8c90711f7696bae…`, full report in
[`okf-hard-negatives.report.json`](okf-hard-negatives.report.json).

| mode | recall@5 | MRR@5 | hard-neg recall@5 | technical recall@5 |
|---|---|---|---|---|
| lexical | 0.397 | 0.361 | **0.029** | 0.857 |
| vector | **0.794** | **0.648** | **0.771** | 0.821 |
| hybrid | 0.556 | 0.412 | 0.286 | **0.893** |

Three things this says, in order of how actionable they are:

**1. Hybrid fusion is worse than vector alone, and fails the project's own
gate.** `--require-hybrid-not-worse` compares hybrid against the better single
mode and exits non-zero: 0.556 against 0.794. Reciprocal-rank fusion mixes in a
lexical ranking that is pure noise on the hard-negative stratum, and the noise
costs more than the signal it adds. **Hybrid must not be enabled by default**,
and the fusion needs to account for a mode having no signal for a given query
before it could be.

**2. Vectors do recover paraphrase queries** — 0.771 where lexical manages
0.029. That lexical fails there is by construction and proves nothing; that
vectors *succeed* is the finding.

**3. The corpus is not saturated.** Lexical scores 0.397 overall, nowhere near
the 1.000 that made the previous FastAPI evaluation unusable. Stratifying is what
fixed that, and it is the reusable lesson.

### What this is not

**The queries and relevance labels are machine-authored.**
[`../README.md`](../README.md) requires human-written queries and independent
human judgments before any retrieval-quality claim, and this does not meet that
bar. Nothing here supports a public claim that ANNPack retrieves well.

Further limits worth stating before anyone quotes a number:

- **63 queries on one 153-passage corpus** of technical BigQuery documentation.
  Narrow, and small enough that a handful of queries move a decimal.
- **The stratum mix is arbitrary.** 35 hard-negative against 28 technical-token
  is a choice, and it drives every "overall" figure in the table. The per-stratum
  columns are the meaningful ones; the overall column should not be quoted alone.
- **The hard-negative stratum is built to defeat lexical**, so it measures the
  ceiling of what vectors add, not what a realistic query mix would show.

The finding in (1) does not depend on any of that. Hybrid losing to vector on
the same queries, scored by the same judgments, is a comparison internal to the
run — which is why it is worth acting on even though the labels are not human.

### Reproducing

```bash
cargo build --release
./examples/okf-reproduction/reproduce.sh          # clones the pinned OKF source

mkdir -p target/okf-eval/corpus
for b in ga4 crypto_bitcoin stackoverflow; do
  for f in target/google-okf-reproduction/knowledge-catalog/okf/bundles/$b/**/*.md; do
    cp "$f" "target/okf-eval/corpus/${b}__$(basename "$f")"
  done
done

target/release/annpack build target/okf-eval/corpus \
  --output target/okf-eval/core.annpack --name okf-eval --version 0.2.0 \
  --source-revision git:3fcbb9f828c2f23d109c855ee403c3a4c81f3a96
target/release/annpack export-passages target/okf-eval/core.annpack \
  --output target/okf-eval/passages.json

# Rebuilds the query set and re-checks the hard-negative rule from scratch.
python3 evals/corpora/okf-hard-negatives.build.py \
  target/okf-eval/passages.json target/okf-eval/qrels.jsonl

npm install --prefix evals
(cd evals && node embed.mjs --kind passages \
  --input ../target/okf-eval/passages.json --output ../target/okf-eval/vectors.json)
(cd evals && node embed.mjs --kind queries \
  --input ../target/okf-eval/qrels.jsonl --output ../target/okf-eval/qrels-vec.jsonl)

target/release/annpack build target/okf-eval/corpus \
  --output target/okf-eval/vector.annpack --name okf-eval --version 0.2.0 \
  --source-revision git:3fcbb9f828c2f23d109c855ee403c3a4c81f3a96 \
  --vectors target/okf-eval/vectors.json

python3 evals/evaluate.py --binary target/release/annpack \
  --pack target/okf-eval/vector.annpack --queries target/okf-eval/qrels-vec.jsonl \
  --vector-profile ann-1-mxbai-xsmall-v1-q8-onnx --k 5 --require-vectors
```

Embeddings come from `mixedbread-ai/mxbai-embed-xsmall-v1` pinned to revision
`e6ac24e5d6efb8782b59de1647b3ececb4ece94e`, q8/ONNX on CPU — the profile in
[`default-embedding-profile.json`](../../spec/examples/default-embedding-profile.json).

---

## `fastapi-qrels.unverified.jsonl`

65 queries over the FastAPI documentation with labels recorded as passage IDs and
source paths. Shape matches [`../evaluate.py`](../evaluate.py).

**Unverified input, not evidence.** The labels' provenance was never recorded:
there is no attestation that the queries were human-written or that the judgments
were made independently of the implementation. It is also saturated — lexical
scores a perfect recall@5 across every category — so it cannot discriminate
between rankers. Kept because 65 labeled queries are expensive to recreate and it
remains a usable smoke corpus for the harness. Superseded for comparison purposes
by `okf-hard-negatives` above.
