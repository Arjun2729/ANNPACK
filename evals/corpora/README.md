# Evaluation corpora

## `okf-hard-negatives`

A retrieval evaluation corpus constructed so that lexical retrieval can fail,
permitting comparison between retrieval modes.

**Corpus.** The three OKF v0.2 bundles `ga4`, `crypto_bitcoin`, and
`stackoverflow` from `GoogleCloudPlatform/knowledge-catalog` at
`3fcbb9f828c2f23d109c855ee403c3a4c81f3a96` (Apache-2.0). 47 documents, 153
passages, three unrelated domains, providing cross-domain distractors.

**Queries.** 63, in two strata:

| Stratum | Count | Construction |
|---|---|---|
| `technical-token` | 28 | The query names identifiers the target passage contains. |
| `hard-negative` | 35 | Paraphrase sharing no discriminative token with the target. |

A token is discriminative if it occurs in fewer than 20% of passages. Tokens at
or above that threshold have an idf near zero and cannot affect ranking, so
sharing them is permitted.
[`okf-hard-negatives.build.py`](okf-hard-negatives.build.py) computes document
frequencies from the corpus and rejects any hard-negative candidate that shares
a discriminative token with its target; 32 of the first 35 drafts were rejected
and rewritten. Stratum membership is therefore a checked property of the data.

### Results

k=5, artifact root `f8c90711f7696bae…`. Full report in
[`okf-hard-negatives.report.json`](okf-hard-negatives.report.json).

| mode | recall@5 | MRR@5 | hard-negative recall@5 | technical recall@5 |
|---|---|---|---|---|
| lexical | 0.397 | 0.361 | 0.029 | 0.857 |
| vector | 0.794 | 0.648 | 0.771 | 0.821 |
| hybrid | 0.730 | 0.522 | 0.571 | 0.929 |

**Fusion defect and correction.** Reciprocal-rank fusion scored 0.556 against
vector-only at 0.794, failing `--require-hybrid-not-worse`. RRF sums per-list
ranks, making presence in both lists worth approximately twice a top position in
one: in one traced query it ranked a passage placed 47th by lexical above the
passage placed 1st by vectors, and excluded a vector-rank-1 passage from the top
8. Fusion now scores each mode on an absolute scale — BM25 over the query's
maximum achievable score, cosine unmodified — raising hybrid to 0.730 and
improving both strata (hard-negative 0.286 → 0.571, technical 0.893 → 0.929).
The table above reflects the corrected implementation.

**Hybrid remains disabled by default.** Its gain where lexical retrieval
contributes (+0.108 over 28 queries) is smaller than its loss where lexical
retrieval misleads (−0.200 over 35 queries). A weight sweep does not resolve
this: reducing lexical weight converges to vector-only rather than exceeding it,
and at weight 0.25 the technical stratum falls to 0.821, equal to vector-only,
indicating lexical has been discarded rather than balanced.

### Routing ceilings

Whether per-query mode selection could do better is bounded by what a selector
with perfect information would achieve. Counting queries whose correct passage
appears in the top 5:

| Selector | Queries | recall@5 |
|---|---|---|
| Vector-only | 50/63 | 0.7937 |
| Stratum selector — lexical for technical-token, vector for hard-negative | 51/63 | 0.8095 |
| Per-query oracle — the better mode chosen for each query individually | 54/63 | 0.8571 |

The per-query oracle is the upper bound on any routing strategy, since it
requires knowing in advance which mode succeeds. It exceeds vector-only by four
queries. The nine queries it still misses are missed by both modes — seven
hard-negative and two technical-token — and are listed in the report.

Note that the stratum selector scores *below* the per-query oracle and only
slightly above vector-only: assigning a whole stratum to one mode discards the
technical-token queries that vector retrieval answers and lexical does not.

Four queries on a 63-query machine-authored corpus does not establish that a
practical router could capture that margin, or that it would generalize. No
deployable routing signal has been demonstrated here. Establishing one would
require a corpus on which lexical retrieval decisively outperforms vector
retrieval on some identifiable class of queries; this corpus does not contain
one, since vector retrieval scores 0.821 against lexical's 0.857 even on the
stratum built to favour lexical.

**Vector retrieval recovers paraphrase queries** at 0.771 against lexical at
0.029. Lexical failure on that stratum is by construction; vector success is the
measured result.

**The corpus is not saturated.** Lexical scores 0.397 overall, against 1.000 on
the previously withdrawn FastAPI corpus. Stratification is what produces a
corpus capable of distinguishing retrieval modes.

### Scope of these results

The queries and relevance labels are machine-authored.
[`../README.md`](../README.md) requires human-written queries and independent
human judgments before any retrieval-quality claim; this corpus does not meet
that requirement and supports no claim about ANNPack's retrieval quality.

Additional constraints on interpretation:

- 63 queries over one 153-passage corpus of technical BigQuery documentation.
  Narrow, and small enough that individual queries affect the second decimal.
- The stratum mix (35 hard-negative, 28 technical-token) is a design choice and
  determines every overall figure. The per-stratum columns are the interpretable
  ones; the overall column should not be quoted in isolation.
- The hard-negative stratum is constructed to defeat lexical retrieval, so it
  measures an upper bound on vector contribution rather than a realistic query
  distribution.

The fusion findings do not depend on label provenance. The RRF ordering defect
is observable in a single query trace, and the hybrid-versus-vector comparison is
internal to one run over identical queries, judgments, and corpus.

### Reproducing

```bash
cargo build --release
evals/corpora/reproduce-okf-hard-negatives.sh
```

That script owns the whole procedure: it assembles the corpus from the pinned
vendored OKF source, builds the pack, exports passages, then asserts the
benchmark's identity — 47 documents, 153 passages, 63 queries, and every
`relevant_passage_id` present in the corpus.

Those counts check *this benchmark*, not ANNPack in general. If ingestion or
chunking changes and the corpus becomes 157 passages, the script fails rather
than quietly renumbering what every published metric refers to. A query whose
target passage has vanished is worse than a plain failure: it is unanswerable by
every mode equally, so it reads as a retrieval result.

This procedure previously lived here as shell prose and drifted from the code.
It named the pre-vendoring checkout path, so following it from a clean tree
produced zero documents and a build error — the data was reproducible, the
documented steps were not. Hence one command, and this file no longer restates
its internals.

To evaluate, embed with a pinned profile and build a vector pack:

```bash
npm install --prefix evals
(cd evals && node embed.mjs --kind passages \
  --input ../target/okf-eval/passages.json --output ../target/okf-eval/vectors.json)
(cd evals && node embed.mjs --kind queries \
  --input ../evals/corpora/okf-hard-negatives.jsonl --output ../target/okf-eval/qrels-vec.jsonl)

target/release/annpack build target/okf-eval/corpus \
  --output target/okf-eval/vector.annpack --name okf-eval --version 0.2.0 \
  --source-revision git:3fcbb9f828c2f23d109c855ee403c3a4c81f3a96 \
  --vectors target/okf-eval/vectors.json

python3 evals/evaluate.py --binary target/release/annpack \
  --pack target/okf-eval/vector.annpack --queries target/okf-eval/qrels-vec.jsonl \
  --vector-profile ann-1-mxbai-xsmall-v1-q8-onnx --k 5 --require-vectors
```

`embed.mjs` accepts `--profile FILE` to use an encoder other than the pinned
default, and the profile travels into the vectors file so a measurement always
records which encoder produced it. Alternative profiles live in
[`../profiles/`](../profiles/).

The default profile pins `mixedbread-ai/mxbai-embed-xsmall-v1` at revision
`e6ac24e5d6efb8782b59de1647b3ececb4ece94e`, q8/ONNX on CPU — see
[`default-embedding-profile.json`](../../spec/examples/default-embedding-profile.json).
It is deliberately small so the same encoder can also run in a browser under
Transformers.js. That size may cap semantic recall; whether it does has not
been established here. `--profile` exists so the question can be measured
rather than assumed, and any answer needs a corpus this one cannot yet
supply — these judgments are machine-authored and unadjudicated, and the
numbers above were produced by an implementation revision the report does
not record.


## `fastapi-qrels.unverified.jsonl`

65 queries over the FastAPI documentation with labels recorded as passage IDs and
source paths. Shape matches [`../evaluate.py`](../evaluate.py).

Unverified input, not evidence. Label provenance was not recorded: there is no
attestation that the queries were human-written or that the judgments were made
independently of the implementation. The corpus is also saturated — lexical
retrieval scores 1.000 recall@5 across every category — so it cannot distinguish
retrieval modes. It is retained as a harness smoke corpus. Superseded for
comparison purposes by `okf-hard-negatives`.
