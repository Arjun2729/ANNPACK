# ANNPack Evidence v1

Status: candidate specification. Deliberately separable from the ANNPack
container format.

A **receipt** is a small, self-contained document proving that a specific passage
of text existed, unmodified, inside a specific immutable artifact published by a
specific key — verifiable offline, with no pack, no network, and no trust in the
issuer.

This document is independent of [FORMAT-v3](FORMAT-v3.md) on purpose. A system
that never adopts the ANNPack container can still emit and check receipts; it
needs BLAKE3, Ed25519, and base64, and nothing else. ANNPack is then simply a
convenient way to *produce* receipts cheaply and at scale.

## What a receipt proves, and what it does not

**Proves.** The cited passage bytes are exactly the bytes committed by a named
artifact root, at a named source revision, optionally signed by a named key.
Tampering with the passage, the proof, the manifest, or the directory is
detected.

**Does not prove.** That the publisher is who they claim to be (that requires an
external key binding), that the artifact is current rather than a valid older
version (see rollback below), or that a model's answer faithfully follows from
the passage. Answer faithfulness is a separate problem this does not address.

## The verification chain

```text
passage record bytes
  │ BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\0" || bytes)
  ▼
leaf ──(inclusion proof)──▶ passage_merkle_root      logical content root
                                   │
                                   │ appears verbatim in the manifest JSON
                                   ▼
                            manifest bytes
                                   │ BLAKE3(bytes) == manifest directory entry hash
                                   ▼
                            section directory
                                   │ BLAKE3("ANNPACK3-CONTENT-ROOT\0" || non-signature entries)
                                   ▼
                              pack_root  ◀──(Ed25519)── publisher signature
```

Every arrow is recomputed by the verifier from bytes carried in the receipt.

## Merkle construction

```text
leaf_i = BLAKE3(UTF8("ANNPACK3-PASSAGE-EVIDENCE\0") || passage_record_json_i)
parent = BLAKE3(UTF8("ANNPACK3-EVIDENCE-NODE\0")     || left || right)
```

Leaves are in deterministic corpus order. Combine pairwise from the left. A level
with an odd node count **promotes** its final node unchanged; it MUST NOT be
duplicated, since duplication makes an N-leaf tree collide with an (N+1)-leaf
tree whose last two leaves are equal. A single leaf is its own root.

Leaf and interior hashes use different domain separators, so a leaf can never be
reinterpreted as an interior node.

An inclusion proof is the ordered list of siblings from leaf to root. A promoted
node contributes no step. Proof length is ⌈log₂ n⌉ or less: 11 steps for 1,864
passages.

## Document shape

```json
{
  "schema": "annpack-receipt-v1",
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
  "signature": {
    "algorithm": "Ed25519",
    "public_key": "<64 hex>",
    "signature": "<128 hex>",
    "key_id": "<64 hex>",
    "identity": "example.test"
  }
}
```

`signature` is optional. Typical size is 2–5 KB; it grows only with the log of the
passage count and the size of the manifest.

## Verification procedure

A verifier MUST perform all of these and report each independently:

1. `BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\0" || passage_record) == passage_hash`
2. Replaying `inclusion_proof` from that leaf yields `passage_merkle_root`
3. The manifest JSON's `passage_merkle_root` equals the receipt's
4. `BLAKE3(manifest_bytes)` equals the stored hash in the directory entry whose
   section ID is `manifest_section_id`, and that entry's type is Manifest (1)
5. `BLAKE3("ANNPACK3-CONTENT-ROOT\0" || non-signature entries)` over
   `directory_b64` equals `pack_root`
6. If `signature` is present, Ed25519-verify it over
   `UTF8("ANNPACK3-SIGNATURE\0") || pack_root`, and check
   `key_id == BLAKE3(public_key)`

The receipt is **verified** when 1–5 hold. Step 6 is a separate claim.

### Three claims, never merged

Mirroring [SECURITY.md](SECURITY.md), a verifier MUST keep these distinct:

| Claim | Established by |
|---|---|
| Integrity | steps 1–5 |
| Authenticity | step 6 |
| Identity trust | an **external** key binding supplied by the caller |

A cryptographically valid signature MUST NOT set identity trust. A verifier
reports identity trust only when the caller supplied a trusted key and the
receipt's signature used it. A self-declared `identity` string is not
self-authenticating.

### Rollback

A receipt for an older artifact stays valid forever — that is the point of
immutability, and it is also the limitation. A consumer enforcing freshness MUST
separately track the newest accepted root, source revision, key rotation, expiry,
and revocation. Receipt validity alone does not establish currency.

## Reference tooling

```bash
annpack receipt <pack> <passage-id> --output receipt.json
annpack verify-evidence receipt.json [--trusted-public-key <hex>]
```

`verify-evidence` opens no pack and makes no network request. It exits non-zero
if any of steps 1–5 fails. The MCP tool `knowledge_evidence_receipt` returns the
same document.

## Producing receipts without ANNPack

Any system can emit `annpack-receipt-v1` if it can commit to an ordered set of
passage records and expose that commitment in a signed, hashed document. The
container-specific fields are `manifest_bytes_b64`, `directory_b64`, and
`manifest_section_id`, which bind the logical content root to an artifact root. A
non-ANNPack issuer MAY omit them and present only steps 1–3 plus a signature over
`passage_merkle_root`; such a receipt MUST declare
`"schema": "annpack-receipt-v1-logical"` so verifiers do not report an
artifact-root binding that was never checked.
