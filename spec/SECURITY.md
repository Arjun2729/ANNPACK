# ANNPack Security Model

ANNPack artifacts are untrusted binary input even when delivered by a familiar domain.

## Security invariants

A conforming parser:

- Checks every addition and multiplication used for offsets, lengths, counts, and allocations
- Rejects sections outside the source, overlapping sections, duplicate IDs, noncanonical directory order, nonzero reserved bytes, and directory overlap
- Enforces section, manifest, decompression-ratio, passage-block, result, query, and vector-dimension limits
- Bounds delta targets to 512 MiB in the reference codec and validates operation counts against both a fixed ceiling and the encoded payload before allocation
- Verifies the directory-root binding before interpreting sections
- Verifies complete sections or block/record hashes before decoding payloads
- Terminates malformed varint decoding after at most ten bytes
- Rejects non-finite vector values
- Does not cast arbitrary input into native structs
- Does not trust JavaScript `Number` for arbitrary 64-bit fields
- Never renders pack metadata as unsanitized HTML

## Integrity, authenticity, and identity

These are separate claims:

1. **Integrity:** stored bytes match hashes anchored by the content root.
2. **Authenticity:** an Ed25519 signature validates over that root.
3. **Identity trust:** an external policy binds the signing key to a publisher.

Self-declared identity strings do not establish the third claim. Suitable future bindings include domain-hosted keys, DNS records, transparency logs, registry identities, and organizational trust policies.

Retrieval output preserves that distinction in its evidence envelope. `publisher.status=cryptographically_verified` means a signature over the immutable root was checked; it does not set `identity_trusted`. The direct passage evidence hash is computed over deterministic decoded JSON and must agree across implementations.

## Rollback and expiry

A valid old pack can still be stale. Consumers enforcing freshness should track the newest accepted version/root, source revision, publisher key rotation, expiration, and revocation policy. Signature validity alone does not prevent rollback.

## Policy is not DRM

Policy metadata communicates license and access conditions. It does not cryptographically prevent copying after plaintext access. Encrypted sections, capability keys, and payment settlement are future protocol layers and should not be implied by declarative metadata.

## HTTP threats

Range clients reject servers ignoring ranges, incorrect `Content-Range`, short or oversized bodies, ETag changes during one session, and byte coordinates exceeding safe platform limits.

HTTPS authenticates transport endpoints; pack signatures authenticate content independently of mirrors.

Registry passwords MUST NOT be accepted as command-line values because process listings and shell history can expose them. The reference client reads them from an explicitly named environment variable, refuses credentials over non-HTTPS non-loopback transport, rejects insecure non-loopback Bearer realms, bounds token responses, does not forward authorization to foreign blob-upload origins, follows OCI Bearer challenges, verifies registry SHA-256 descriptors, and then verifies the independent ANNPack root before installing a pull.

## Browser rendering

The reference browser client constructs DOM nodes and assigns untrusted strings through `textContent`. It does not use `innerHTML` for titles, URLs, passages, or metadata. URLs remain subject to the embedding page's navigation and content-security policy.

## Fuzzing

The `fuzz/` workspace contains targets for arbitrary pack opening, varint decoding, and delta-envelope parsing. Corruption and property tests run in the ordinary test suite; daily CI performs short campaigns and the weekly/manual deep workflow defaults to six hours per target. Fuzzing complements rather than replaces the independent review brief.
