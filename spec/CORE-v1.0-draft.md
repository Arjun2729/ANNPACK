# ANNPack Core v1.0-draft

Status: frozen draft for independent implementation. Wire encoding: `ANNPACK3`.

Core is the smallest useful interoperability contract. A Core-only reader is fully conformant. It does not need to implement vectors, deltas, OCI distribution, policy commerce, or dependency traversal.

## Design constraint

A competent developer should be able to implement a read-only Core client against the golden corpus in fewer than roughly 500 source lines, excluding BLAKE3, Ed25519, DEFLATE, HTTP, and JSON libraries. If Core grows beyond that, new behavior belongs in an extension.

Core freezes these responsibilities:

1. Parse and bounds-check the 128-byte header and 80-byte section directory defined by [the v3 wire format](FORMAT-v3.md).
2. Verify the content root and every fetched stored-byte hash.
3. Read required section types 1–6: Manifest, Documents, Passage Index, Passage Data, Lexical Dictionary, and Lexical Postings.
4. Retrieve independently compressed passage blocks through exact HTTP byte ranges.
5. Rank lexical results with the specified BM25 profile and deterministic tie-breaking.
6. Emit the evidence envelope below for every returned passage.
7. Verify any present Ed25519 signature before reporting it as cryptographically valid. Publisher identity trust remains an external key-binding decision.
8. Discover a primary pack through `/.well-known/knowledge.annpack` or a release catalog through `/.well-known/annpack.json`.

Core packs declare these capabilities:

```text
citations
content
lexical-bm25
range-addressable-passages
section-integrity
```

Unknown optional sections are ignored. Unknown required sections are rejected. A partial implementation must not call itself Core conformant.

## Evidence envelope

Every retrieved passage has a machine-readable envelope:

```json
{
  "schema": "annpack-evidence-v1",
  "pack": "vendor-docs@1.0.0",
  "pack_root": "<64 lowercase hex characters>",
  "source_revision": "git:<immutable revision>",
  "passage_id": "<stable passage identity>",
  "passage_hash": "<hash of the exact passage record>",
  "canonical_url": "https://example.test/docs/page#anchor",
  "publisher": {
    "status": "cryptographically_verified",
    "key_ids": ["<key id>"],
    "asserted_identities": ["example.test"],
    "identity_trusted": false
  }
}
```

`passage_hash` is:

```text
BLAKE3(UTF8("ANNPACK3-PASSAGE-EVIDENCE\0") || deterministic_passage_json)
```

Publisher status is one of `unsigned`, `not_verified`, or `cryptographically_verified`. `identity_trusted` MUST remain false unless the caller supplied an external trust binding for one of the verified keys. An asserted identity inside a signature is not self-authenticating.

The pair `(pack_root, passage_id)` identifies the verified record in an immutable artifact. `passage_hash` lets evidence consumers compare the exact decoded record directly. `source_revision` and `canonical_url` connect that record to its publisher source.

## Core conformance artifacts

- Golden artifact: [`test-vectors/minimal-v3.annpack`](test-vectors/minimal-v3.annpack)
- Golden source: [`test-vectors/source/minimal.md`](test-vectors/source/minimal.md)
- Expected root: `b1f63b4acdbee0a89de5c3455505be279845b4eda644c0d6c931814355a9d70b`

The reference CLI reports `core_profile`, `core_conformant`, implemented extensions, and conformance issues from `inspect`, `verify`, discovery, search, and MCP pack information.

## Change rule

Until the draft marker is removed, fixes may clarify ambiguity or close security defects. New features do not enter Core. They receive a numbered extension. Removing `-draft` requires a second implementation produced from these documents and the golden corpus without using the Rust reference parser.
