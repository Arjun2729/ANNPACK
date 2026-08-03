# Evaluation corpora

## `fastapi-qrels.unverified.jsonl`

65 queries over the FastAPI documentation, with relevance labels recorded as
passage IDs and source paths. Shape matches [`../evaluate.py`](../evaluate.py).

**Unverified input, not evidence.** The labels' provenance was never recorded:
there is no attestation that the queries were human-written or that the
judgments were made independently of the implementation. Establish and record
that provenance before any number derived from this file is published.

It is also saturated — lexical retrieval scores a perfect recall@5 across every
category here, so it cannot discriminate between rankers and cannot support a
comparative claim. It is kept because 65 labeled queries are expensive to
recreate and it is a usable smoke corpus for the harness itself.

What a corpus needs to actually settle the vectors-on-by-default question is
described in [`../README.md`](../README.md); the short version is
hard negatives — paraphrase-only queries with no lexical overlap with the target
span, so lexical can fail and the comparison means something.
