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

k=5, 61 documents, 168 passages, corpus `8be4259d55e51e84…`. Reproduce with
[`reproduce-okf-hard-negatives.sh`](reproduce-okf-hard-negatives.sh).

| mode | recall@5 | MRR@5 | hard-negative recall@5 | technical recall@5 |
|---|---|---|---|---|
| lexical | 0.397 | 0.361 | 0.029 | 0.857 |
| vector | 0.730 | 0.591 | 0.686 | 0.786 |
| hybrid | 0.683 | 0.470 | 0.486 | 0.929 |

**These supersede an earlier table entirely.** The previous numbers — vector
0.794, hybrid 0.730 — were measured on a corpus of 47 documents that should
have held 61. Assembly flattened source paths to their basename, so every
`index.md` within a bundle collapsed onto one name and fourteen files were
silently overwritten; which fourteen depended on directory traversal order, so
two machines built measurably different corpora while both reported the same
counts. Do not compare against the old figures: they describe a corpus that was
missing a quarter of its documents. The queries are unchanged and were re-paired
to the restored passages by content.

**Hybrid remains disabled by default.** It leads on the technical stratum
(0.929 against vector's 0.786) and loses heavily where lexical retrieval
misleads (0.486 against 0.686), for a net loss overall. Earlier fusion work
replaced reciprocal-rank fusion, which ranks presence in both lists above a top
position in one, with absolute per-mode scoring; that correction stands, though
the figures quoted for it were measured on the malformed corpus.

### Routing ceilings

Whether per-query mode selection could do better is bounded by what a selector
with perfect information would achieve. Counting queries whose correct passage
appears in the top 5:

| Selector | Queries | recall@5 |
|---|---|---|
| Vector-only | 46/63 | 0.7302 |
| Stratum selector — lexical for technical-token, vector for hard-negative | 48/63 | 0.7619 |
| Per-query oracle — the better of lexical or vector, chosen per query | 52/63 | 0.8254 |
| Three-mode oracle — the best of lexical, vector or hybrid | 53/63 | 0.8413 |

The per-query oracle is the upper bound on any routing strategy, since it
requires knowing in advance which mode succeeds. **Ten queries are missed by all
three modes**, so at most seven of the seventeen vector-only failures are
reachable by selection at all; the rest need a representation none of these
provide.

Four to six queries on a 63-query machine-authored corpus does not establish
that a practical router could capture that margin, or that it would generalize.
No deployable routing signal has been demonstrated here.

**Vector retrieval recovers paraphrase queries** at 0.686 against lexical at
0.029. Lexical failure on that stratum is by construction; vector success is the
measured result.

**The corpus is not saturated.** Lexical scores 0.397 overall, against 1.000 on
the withdrawn FastAPI corpus. Stratification is what produces a corpus capable
of distinguishing retrieval modes.

### Cross-platform reproducibility

Embeddings are **not** portable across CPU architectures under any tested 8-bit
quantization. On an identical corpus — same content digest, same passage order —
macOS arm64 and Linux x64 produce different vectors:

| weights | min self-cosine | max dimension delta | portable |
|---|---|---|---|
| q8 (`model_quantized.onnx`, pinned) | 0.995439 | 1.57e-02 | no |
| int8 (`model_int8.onnx`) | 0.995439 | 1.57e-02 | no |
| uint8 (`model_uint8.onnx`) | 0.993729 | 2.00e-02 | no |
| fp32 (`model.onnx`) | 0.999999 | 1.10e-07 | yes |

`int8` produces byte-identical output to `q8`; they are the same quantization
under two filenames.

**The documented mitigation does not work.** ONNX Runtime attributes U8S8
divergence to `VPMADDUBSW` saturation on x86-64 with AVX2/AVX512 but without
VNNI — which the Linux host here reports — and recommends U8U8 as free of it.
U8U8 avoids that instruction and diverges *more*. So the cause is not one
saturating instruction; it is that architecture-specific 8-bit kernels do not
reproduce a vector across architectures.

The divergence reaches results. Across 63 queries, q8 changes 21 top-5 orderings
and uint8 changes 30. q8 changes no hit/miss verdict on this corpus and uint8
changes two — but ties at the top-5 boundary make that a property of these
queries, not a bound.

Retrieval quality on the same corpus, macOS:

| weights | vector recall@5 | MRR@5 | technical | hard-negative | hybrid recall@5 | model size |
|---|---|---|---|---|---|---|
| q8 / int8 | 46/63 | 0.591 | 22/28 | 24/35 | 43/63 | 23.3 MB |
| uint8 | 48/63 | 0.605 | 22/28 | 26/35 | 46/63 | 23.3 MB |
| fp32 | 48/63 | 0.618 | 23/28 | 25/35 | 45/63 | 91.5 MB |

The pinned profile is the weakest arm on both axes: lowest recall, lowest MRR,
and not portable. Its only demonstrated advantage is artifact size.

**The default is unchanged regardless.** fp32 is portable and scored best here,
but 63 machine-authored queries do not justify a fourfold artifact-size change,
and a browser-sized model was the reason for choosing this one. Establishing a
default needs a real-corpus evaluation. Until then, evaluation numbers should
name the platform that produced them.

Reproduce with [`../profiles/`](../profiles/) and
[`embedding-determinism.yml`](../../.github/workflows/embedding-determinism.yml).

### Adjudication status

All 63 pairs were read and compared against their targets: **no mislabels were
found**, including one that had been suspected because all three modes missed it
simultaneously. Label quality is not what limits this corpus.

What limits it is stratum design and source material. 21 of 35 hard-negative
targets are table or dataset overviews, so the stratum largely tests one skill —
recognising a table from a reworded description. And the corpus cannot support
the axes it lacks: `exception` patterns appear in 2 of 168 passages, temporal in
6. BigQuery schema reference describes what tables contain, not what is excluded
or what happens after a deadline, so negation, threshold and exception queries
cannot be written against it without inventing content the passages do not have.

See [ADJUDICATION.md](ADJUDICATION.md). The review is machine-assisted and does
not substitute for human adjudication.

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
