# ANNPack OpenTelemetry attributes

A trace records that a retrieval happened and what text it returned. It does not
record which immutable artifact that text came from, so the span outlives its own
evidence: once the corpus moves, the recorded document cannot be checked against
anything. These attributes close that gap by pinning each retrieval to an
artifact root and each returned passage to its hash.

This defines attribute names only. It introduces no exporter, no backend, and no
transport. It is not part of Core conformance, and an implementation that emits
none of these attributes is fully conformant.

## Namespace

All attributes are under `annpack.*`. The host application already sets whatever
`gen_ai.*` conventions it follows; those are still moving, and mirroring them
here would freeze a guess into a wire contract. These compose with any of them.

Values are OpenTelemetry-typed: strings, booleans, and homogeneous string
arrays. Artifact-level facts belong on the retrieval span; per-passage facts
belong on one event per passage, because OpenTelemetry attribute values cannot be
objects.

## Span attributes

| Attribute | Type | Meaning |
|---|---|---|
| `annpack.root` | string | Artifact root (BLAKE3 hex). Pinning this is what lets a later reader detect that the corpus has moved. |
| `annpack.pack` | string | `name@version` from the manifest. |
| `annpack.source_revision` | string | Source revision the artifact was compiled from. Omitted when the artifact records none. |
| `annpack.publisher.status` | string | Signature status as the reader reported it. |
| `annpack.publisher.identity_trusted` | boolean | Whether a signature was matched against a caller-supplied trusted key. |
| `annpack.passage_ids` | string[] | Returned passage IDs, in rank order. |
| `annpack.passage_hashes` | string[] | Passage evidence hashes, in the same order. |

`annpack.publisher.status` and `annpack.publisher.identity_trusted` are separate
because "this artifact carries a valid signature" and "the signer is authorised"
are different claims. A trace that merges them is misleading at the point it is
most likely to be read.

## Event attributes

One `annpack.passage` event per returned passage.

| Attribute | Type | Meaning |
|---|---|---|
| `annpack.root` | string | Repeated so an event is interpretable without its parent span. |
| `annpack.passage_id` | string | Passage ID. |
| `annpack.passage_hash` | string | Passage evidence hash, as defined in [EVIDENCE-v1](EVIDENCE-v1.md). |
| `annpack.rank` | int | 1-based rank within the response. |
| `annpack.receipt_uri` | string | Where a receipt for this passage can be fetched. Present only when the emitter was configured with a location. |

ANNPack does not define where receipts are served, so `annpack.receipt_uri` is
emitted only from a caller-supplied template containing `{passage_id}` and
optionally `{root}`. A template without `{passage_id}` is refused: it would give
every passage in the run the same URI, pointing at the wrong evidence for all but
one. Interpolated values are percent-encoded against RFC 3986's unreserved set,
so an ID from another implementation cannot escape its path segment.

## Reference tooling

```bash
annpack search <pack> <query> --otel [--otel-receipt-uri 'https://…/{root}/{passage_id}']
```

Emits both the span attribute map and the per-passage events as JSON. Output is
always JSON; the flag is mutually exclusive with `--json`, which emits the search
response instead.
