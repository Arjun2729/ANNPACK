# ANNPack Core Discovery Protocol v1

Status: candidate specification.

## Discovery

A site exposing one primary public pack SHOULD publish the immutable artifact at:

```text
GET /.well-known/knowledge.annpack
Content-Type: application/vnd.annpack.v3
```

Sites with multiple corpora, versions, access classes, or embedding variants SHOULD additionally expose the catalog:

Publishers SHOULD expose:

```text
GET /.well-known/annpack.json
Accept: application/vnd.annpack.discovery+json
```

The document lists logical corpora and immutable releases. Each release includes its version, BLAKE3 content root, artifact URL, byte length, media type, capabilities, source revision, signature key IDs, access class, and license identifier.

An `llms.txt` document MAY reference it with:

```text
Knowledge-Pack-Discovery: https://example.com/.well-known/annpack.json
```

or reference an exact artifact with:

```text
Knowledge-Pack: https://example.com/.well-known/packs/docs-4.2.1.annpack
```

These relations are candidate conventions, not registered fields.

## Immutable identity

Human-readable coordinates resolve to immutable roots:

```text
annpack://vendor.example/docs/product/4.2.1@blake3:<root>
```

Aliases such as `latest`, `4`, or `4.2` MUST resolve to a root before use. An answer-evidence record SHOULD retain the resolved root rather than only the alias.

## Capability declaration

Discovery advertises capabilities such as:

```text
content
citations
lexical-bm25
vector-flat-dot
vector-ivf-flat-dot
hybrid-absolute-scale
range-addressable-passages
section-integrity
delta-snapshot-v1
```

Core defines declaration, not HTTP capability negotiation. Publishers may expose multiple artifact variants, but every distinct artifact has its own content root. No `ANNPack-Accept-Capabilities` request header is currently specified.

## HTTP delivery

Servers MUST provide:

- `Content-Length`
- `Accept-Ranges: bytes`
- Correct `206 Partial Content`
- Exact `Content-Range`
- Stable validator, preferably a strong ETag

Clients MUST reject a `200` response to a range request unless they explicitly elected a full-artifact download. Clients MUST reject truncated ranges, incorrect `Content-Range`, unsafe integer conversion, or validator changes during a read session.

Content encoding that changes byte coordinates SHOULD be disabled for `.annpack` responses. Compression belongs inside independently addressable sections or blocks.

## Retrieval sequence

A lexical client normally performs:

1. HEAD for length and validator
2. Header range
3. Directory range
4. Manifest, documents, passage-index, and lexical-dictionary ranges
5. Query posting-list ranges
6. Selected passage ranges

The conformance query `AP-104` completes in eight range GETs after one HEAD.

## Optional extensions

OCI distribution, deltas, vector retrieval, policy descriptors, and pack dependencies are not Core protocol requirements. Their independently optional contracts live in the [extension registry](extensions/README.md). Core-only publishers and readers are conformant.

## MCP mapping

The reference MCP adapter exposes:

- `knowledge_pack_info`
- `knowledge_search`
- `knowledge_get_passage`
- `knowledge_evidence_receipt`

Search results include the [Core evidence envelope](CORE-v1.0-draft.md#evidence-envelope), identifying the immutable pack, source revision, exact passage record, and explicitly scoped publisher-verification state.

`knowledge_evidence_receipt` returns a standalone [Evidence v1](EVIDENCE-v1.md) receipt for a passage id. The receipt is verifiable without the pack, without network access, and without trusting the server that issued it — which is what allows a caller to check an agent's citation against a root it pinned independently. An adapter that omits this tool can still be Core conformant, but callers cannot verify its results offline.
