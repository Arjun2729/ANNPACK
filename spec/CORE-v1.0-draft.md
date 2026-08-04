# ANNPack Core v1.0-draft

Status: frozen draft for independent implementation. Wire encoding: `ANNPACK3`.

Core is the smallest useful interoperability contract. A Core-only reader is fully conformant. It does not need to implement vectors, deltas, or OCI distribution.

## Design constraint

A read-only Core client should be implementable against the golden corpus in **under 600 source lines**, excluding BLAKE3, Ed25519, DEFLATE, HTTP, and JSON libraries.

**How to count.** Executable source lines: comments, docstrings, and blank lines excluded. Stating the method matters, because the two measurements this project has taken were not comparable until it did.

**The measurements.**

| Reader | Date | Counted | Implements |
|---|---|---|---|
| Python, clean-room | 2026-07-20 | 861 (method unrecorded) | manifest format 1–2, monolithic index |
| Python, spec-derived ([`readers/`](conformance/readers/)) | 2026-08-04 | **566** | manifest format 1–3, block-addressed lexical index and record table, Evidence v1 receipt verification |

The second reader implements strictly more than the first — including receipt verification, which the first did not attempt — at two thirds the size. That says the earlier figure counted comments and the earlier implementation was not economical, not that Core had grown. An intermediate revision of this document raised the budget to 1,000 lines on the strength of the 861 figure alone; that was wrong, and 600 replaces it.

The purpose of the number is unchanged: it is a tripwire on Core's growth, and it is only worth having if it is enforced. Whoever implements the next reader should record its line count and counting method in their conformance report.

Core freezes these responsibilities:

1. Parse and bounds-check the 128-byte header and 80-byte section directory defined by [the v3 wire format](FORMAT-v3.md).
2. Verify the content root and every fetched stored-byte hash.
3. Read required section types 1–6 — Manifest, Documents, Passage Index, Passage Data, Lexical Dictionary, Lexical Postings — plus type 16 (Lexical Terms) and type 17 (Passage Records) when the pack declares lexical index format 2 and passage index format 2 respectively.
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

## Normativity

Release and compatibility policy: [COMPATIBILITY.md](COMPATIBILITY.md).

The specification is normative. Where an implementation and the reference
implementation disagree and the specification permits the implementation's
reading, the **reference implementation** is what changes. The specification is
not retroactively edited to match reference behaviour.

## Core conformance artifacts

- Conformance packet: [`conformance/`](conformance/README.md) — artifacts,
  tokenizer and exact-score vectors, corruption corpus, and a one-command runner
- Golden artifact: [`test-vectors/minimal-v3.annpack`](test-vectors/minimal-v3.annpack)
- Golden source: [`test-vectors/source/minimal.md`](test-vectors/source/minimal.md)
- Expected root: `b1f63b4acdbee0a89de5c3455505be279845b4eda644c0d6c931814355a9d70b`

The reference CLI reports `core_profile`, `core_conformant`, implemented extensions, and conformance issues from `inspect`, `verify`, discovery, search, and MCP pack information.

## Change rule

Until the draft marker is removed, fixes may clarify ambiguity or close security defects. New features do not enter Core. They receive a numbered extension. Removing `-draft` requires a second implementation produced from these documents and the golden corpus without using the Rust reference parser.
