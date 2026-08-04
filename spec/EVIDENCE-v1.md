# ANNPack Evidence v1

Status: candidate specification. Deliberately separable from the ANNPack
container format.

A **receipt** is a self-contained document proving that a specific passage of
text existed, unmodified, inside a specific immutable artifact — verifiable
offline, with no pack, no network, and no trust in the receipt issuer.

ANNPack receipt v2 uses BLAKE3, base64, JSON, and optionally Ed25519. When a
receipt authenticates `canonical_url`, it additionally carries the artifact's
stored Documents section and the verifier must decode that section according to
its committed codec.

## What a receipt proves, and what it does not

**Proves.** The cited passage bytes and the receipt's passage identity, pack
identity, source revision, and optional canonical URL agree with bytes committed
by the named artifact root. A valid optional signature authenticates that root
to a key.

**Does not prove.** That the key belongs to the claimed publisher, that the
artifact is current rather than a valid older version, or that a model's answer
faithfully follows from the passage. Publisher identity requires an external key
binding. Currency and answer faithfulness are separate problems.

## The verification chain

```text
passage record bytes
  │ BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\0" || bytes)
  ▼
leaf ──(inclusion proof)──▶ passage_merkle_root
                                   │ appears in the manifest JSON
                                   ▼
                            manifest bytes
                                   │ BLAKE3(bytes) == directory entry hash
                                   ▼
                            section directory
                                   │ BLAKE3("ANNPACK3-CONTENT-ROOT\0" ||
                                   │        non-signature entries)
                                   ▼
                              pack_root  ◀──(Ed25519)── optional signature
```

Every arrow is recomputed from bytes carried in the receipt.

## Merkle construction

```text
leaf_i = BLAKE3(UTF8("ANNPACK3-PASSAGE-EVIDENCE\0") || passage_record_json_i)
parent = BLAKE3(UTF8("ANNPACK3-EVIDENCE-NODE\0") || left || right)
```

Leaves are in deterministic corpus order. Combine pairwise from the left. A
level with an odd node count **promotes** its final node unchanged; it MUST NOT
be duplicated. A single leaf is its own root.

Leaf and interior hashes use different domain separators. An inclusion proof is
the ordered list of siblings from leaf to root. A promoted node contributes no
step. A verifier MUST impose a finite proof-length limit before replay; the
reference verifier accepts at most 64 steps.

## Document shape

```json
{
  "schema": "annpack-receipt-v2",
  "pack": "fastapi-docs@0.115.12",
  "pack_root": "<64 hex>",
  "passage_merkle_root": "<64 hex>",
  "source_revision": "git:628c34e0",
  "passage_id": "<64 hex>",
  "passage_hash": "<64 hex>",
  "passage_ordinal": 412,
  "canonical_url": "https://fastapi.tiangolo.com/tutorial/#anchor",
  "passage_record_b64": "<exact stored passage record>",
  "inclusion_proof": [{"sibling": "<64 hex>", "sibling_is_left": false}],
  "manifest_bytes_b64": "<manifest section bytes>",
  "directory_b64": "<full section directory>",
  "manifest_section_id": 1,
  "documents_section_id": 2,
  "documents_bytes_b64": "<documents section stored bytes>",
  "signature": {
    "algorithm": "Ed25519",
    "public_key": "<64 hex>",
    "signature": "<128 hex>",
    "key_id": "<64 hex>",
    "identity": "example.test"
  }
}
```

`signature` is optional. `documents_section_id` and
`documents_bytes_b64` are required whenever `canonical_url` is present.

The receipt signature covers the artifact root, exactly as a pack signature
does. Its `identity` field is **unauthenticated metadata**, carried across from
the pack's signature envelope and bound by nothing; a verifier MUST NOT report
it as signed and MUST NOT derive trust from it. See
[FORMAT-v3 §8.1](FORMAT-v3.md).

Receipt size is **not a fixed 2–5 KB guarantee**. It consists of the compact
passage proof, manifest, and directory plus the compressed or uncompressed
Documents catalogue needed to authenticate the URL. Size therefore grows with
the pack's document metadata. Producers and consumers should measure receipt
size on their actual corpus rather than quoting a corpus-independent number.

## Required structural and resource checks

All receipt fields are attacker-controlled until checked. Before allocating or
decoding large values, a verifier MUST:

1. Reject unknown receipt schemas. A verifier that supports only v2 MUST reject,
   not reinterpret, v1, logical-only, or future schemas.
2. Impose finite limits on base64 input, decoded passage and manifest JSON,
   directory length, proof length, stored section length, and logical section
   length.
3. Require the directory to be non-empty, aligned to the directory-entry size,
   sorted by strictly increasing section ID, free of duplicate IDs, and zero in
   all reserved bytes.
4. Reject a section whose stored or logical length exceeds the implementation's
   declared section limit.
5. Check stored length and stored-byte hash before decoding a carried section.
6. Decode according to the directory entry's committed codec:
   - codec 0: use the stored bytes directly and require stored length to equal
     logical length;
   - codec 1: zlib-wrapped DEFLATE, bounded to the committed logical length;
   - any unsupported codec: fail.
7. Apply the same decompression-ratio policy used for a pack before allocation.
   The reference implementation rejects expansion above 256:1 when logical size
   exceeds 16 MiB.
8. Require the decoded byte count to equal the committed logical length.

A signature check does not replace these resource checks. Invalid and unsigned
receipts must not be able to force an unbounded allocation before rejection.

## Verification procedure

A v2 verifier MUST perform all of these and report each independently:

1. `BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\0" || passage_record) == passage_hash`.
2. Replaying `inclusion_proof` from that leaf yields
   `passage_merkle_root`.
3. The manifest JSON's `passage_merkle_root` equals the receipt's value.
4. The manifest directory entry is type Manifest (1), codec 0, has equal stored
   and logical lengths, those lengths equal the carried manifest byte length,
   and its stored hash equals `BLAKE3(manifest_bytes)`.
5. `BLAKE3("ANNPACK3-CONTENT-ROOT\0" || non-signature entries)` over the
   validated directory equals `pack_root`.
6. `passage_id` and `passage_ordinal` equal the authenticated passage record's
   `id` and `ordinal`; `source_revision` equals the manifest's value; and `pack`
   equals manifest `name@version`.
7. If `canonical_url` is present, validate and decode the carried Documents
   section using the structural and resource rules above. Find the document whose
   `id` equals the passage record's `document_id`, then require its `url`, plus the
   record's non-empty anchor as a fragment when the URL has none, to reproduce
   `canonical_url`. A URL claim without an authenticated Documents section MUST
   fail.
8. If `signature` is present, Ed25519-verify it over
   `UTF8("ANNPACK3-SIGNATURE\0") || pack_root`, and require
   `key_id == BLAKE3(public_key)`.

The receipt is **integrity verified** when steps 1–7 hold. Step 8 is a separate
authenticity claim.

## Three claims, never merged

| Claim | Established by |
|---|---|
| Integrity | steps 1–7 |
| Authenticity | step 8 |
| Identity trust | an external key binding supplied by the caller |

A cryptographically valid signature MUST NOT establish identity trust. A
self-declared `identity` string is not self-authenticating.

A verifier that reports the three claims separately MAY still expose a single
pass/fail result to a caller who supplies an external key binding. When such a
binding is supplied, the verifier MUST fail unless a valid signature from that
exact key is present; integrity alone MUST NOT satisfy it. The reference CLI
does this: `annpack verify-evidence --trusted-public-key <hex>` exits non-zero
when no signature from that key verifies, while the structured report keeps
`verified`, `signature_valid` and `identity_trusted` distinct.

## Rollback

A receipt for an older artifact stays valid forever. No implemented freshness or
revocation mechanism exists in this release. [ADR-0004](decisions/0004-freshness-and-revocation.md)
records a proposed model and is design only. Consumers enforcing freshness must
separately track accepted roots, revisions, key rotation, expiry, and revocation.

## Reference tooling

```bash
annpack receipt <pack> <passage-id> --output receipt.json
annpack verify-evidence receipt.json [--trusted-public-key <hex>]
```

`verify-evidence` opens no pack and makes no network request. It exits non-zero
when integrity verification fails or the schema is unsupported. The MCP tool
`knowledge_evidence_receipt` returns the same receipt shape.

## Non-ANNPack and logical-only receipts

A system that does not use the ANNPack container may define a distinct logical
receipt schema over a passage Merkle root. Such a schema is not
`annpack-receipt-v2`, must not include an unauthenticated canonical URL, and must
not be reported as artifact-root-bound. The reference v2 verifier rejects other
schemas rather than silently applying partial semantics.
