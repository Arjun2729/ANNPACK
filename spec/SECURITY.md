# ANNPack Security Model

ANNPack artifacts are untrusted binary input even when delivered by a familiar domain.

## Security invariants

A conforming parser:

- Checks every addition and multiplication used for offsets, lengths, counts, and allocations
- Rejects sections outside the source, overlapping sections, duplicate IDs, noncanonical directory order, nonzero reserved bytes, and directory overlap
- Enforces section, manifest, decompression-ratio, passage-block, result, query, and vector-dimension limits
- **Bounds decompression output during inflation, not only afterwards.** Checking the declared ratio from the directory and then inflating without an output limit is NOT sufficient: a pack may declare a small `logical_length` while shipping a bomb, and an unbounded inflate exhausts memory before any length comparison runs. A conforming reader MUST cap the decompressor at the declared logical length and MUST treat exceeding it as a rejection. (See "Known ambiguity" below — this requirement was previously implicit and a clean-room reader read it the weaker way.)
- Bounds delta targets to 512 MiB in the reference codec and validates operation counts against both a fixed ceiling and the encoded payload before allocation
- Verifies the directory-root binding before interpreting sections
- Verifies complete sections or block/record hashes before decoding payloads
- Terminates malformed varint decoding after at most ten bytes
- Rejects non-finite vector values
- Does not cast arbitrary input into native structs
- Does not trust JavaScript `Number` for arbitrary 64-bit fields
- Never renders pack metadata as unsanitized HTML

## Input bounds outside the container

Two surfaces read caller-supplied bytes before any container check applies, and
both are bounded before allocation:

- **MCP JSON-RPC input.** The stdio server accepts at most 8 MiB for one request
  line. A line beyond that is refused with a JSON-RPC error and skipped in
  bounded chunks, so framing survives and the next request is still served. The
  largest legitimate request is a search carrying a query vector, which stays
  well under the bound at the v3 dimension ceiling.
- **Receipt files.** `verify-evidence` checks the file size against a 64 MiB
  limit before reading it, so a hostile receipt cannot make the verifier
  allocate its whole length ahead of the per-field limits in
  [`EVIDENCE-v1.md`](EVIDENCE-v1.md).

These bound one input each. They do not by themselves make the tools immune to
memory exhaustion; concurrency, host limits, and total process footprint remain
the deployment's responsibility.

## Integrity, authenticity, and identity

These are separate claims:

1. **Integrity:** stored bytes match hashes anchored by the content root.
2. **Authenticity:** an Ed25519 signature validates over that root.
3. **Identity trust:** an external policy binds the signing key to a publisher.

Self-declared identity strings do not establish the third claim. Suitable future bindings include domain-hosted keys, DNS records, transparency logs, registry identities, and organizational trust policies.

The signature covers the artifact root only. The signature envelope's asserted
identity, expiration, transparency-log URL, revocation URL and build-attestation
fields are unauthenticated metadata: a signature section is excluded from the
root, so nothing binds them, and an attacker who can rewrite the artifact can
rewrite them while the signature still verifies. No runtime decision in the
reference implementation reads them, and none should. See
[FORMAT-v3 §8.1](FORMAT-v3.md).

Retrieval output preserves that distinction in its evidence envelope. `publisher.status=cryptographically_verified` means a signature over the immutable root was checked; it does not set `identity_trusted`. The direct passage evidence hash is computed over deterministic decoded JSON and must agree across implementations.

## Rollback and expiry

A valid old pack can still be stale, and signature validity alone does not
prevent rollback. An evidence receipt for a superseded artifact verifies
offline permanently after supersession. A pack cannot revoke itself because an
attacker can serve the older, un-revoked bytes.

Currency lives outside the artifact in a separately distributed,
publisher-signed statement scoped to a corpus and channel and carrying a
monotonic sequence. It is specified and implemented in
[RELEASE-v1](RELEASE-v1.md); the reasoning is in
[ADR-0004](decisions/0004-freshness-and-revocation.md). Nothing in the artifact
format changed to accommodate it.

What that layer does and does not provide:

- **Provides.** Role-separated publisher keys, so a key that signs artifacts
  cannot declare which artifact is current. Rollback rejection and equivocation
  detection against retained per-scope state. Four independent verdicts, with
  `unknown` never reported as `current`. A known revocation denies under every
  consumer policy. The `authorized-current-witnessed` policy verifies external
  Sigsum proofs, and cross-observation monitoring reports conflicts in supplied
  history.
- **Does not provide.** Proof that no newer statement exists; withholding is
  invisible to any offline check. Rollback resistance at first contact or after
  state loss, which is the common case for ephemeral consumers. Detection of a
  conflicting statement that no monitor observed.

Revocation is a status decision, never an integrity failure: a revoked artifact
that is genuinely authentic still reports `artifact_integrity: valid`.

## Policy is not DRM

Policy metadata communicates license and access conditions. It does not cryptographically prevent copying after plaintext access. Encrypted sections, capability keys, and payment settlement are future protocol layers and should not be implied by declarative metadata.

## HTTP threats

Range clients reject servers ignoring ranges, incorrect `Content-Range`, short or oversized bodies, ETag changes during one session, and byte coordinates exceeding safe platform limits.

HTTPS authenticates transport endpoints; pack signatures authenticate content independently of mirrors.

Registry passwords MUST NOT be accepted as command-line values because process listings and shell history can expose them. The reference client reads them from an explicitly named environment variable, refuses credentials over non-HTTPS non-loopback transport, rejects insecure non-loopback Bearer realms, bounds token responses, does not forward authorization to foreign blob-upload origins, follows OCI Bearer challenges, verifies registry SHA-256 descriptors, and then verifies the independent ANNPack root before installing a pull.

## Browser rendering

The reference browser client constructs DOM nodes and assigns untrusted strings through `textContent`. It does not use `innerHTML` for titles, URLs, passages, or metadata. URLs remain subject to the embedding page's navigation and content-security policy.

## Derived retrieval sections

The optional term overlays (ANN-7/ANN-8, section type 13) and anchor
coordinates (ANN-9, section type 15) are **derived**: their contents are
produced from passage text by an offline model and carry the derived flag (bit
one). A conforming reader treats them as untrusted, matching-only input:

- Derived sections MUST NOT be marked required, and a required-and-derived
  section is rejected at the container level.
- Overlay ordinals, posting monotonicity, weights, anchor row counts, row
  lengths, and quantization descriptors are bounds-checked before use, after the
  container's decompression-ratio and logical-length limits have already gated
  allocation. The reference reader loads them lazily, only when the feature is
  actually used, so a lexical-only client never fetches or parses them.
- A derived section MUST NOT contribute any citable text to an evidence
  envelope. Generated or expanded terms change ranking only; the evidence
  `passage_hash` is always computed over the original decoded passage record and
  is identical to the Core pack's hash for the same passage. Generation is a
  separate offline command that writes a pinned, hashed sidecar; the build
  records that digest in `manifest.derived_inputs` and runs no model itself.
- `sidecar_digest` is **recorded provenance, not a proof of derivation.** It
  attests which sidecar the builder claims to have consumed; it does not
  cryptographically bind the emitted derived section's *contents* to that
  sidecar (the build could, in principle, record one digest and write unrelated
  section bytes). It is covered by the pack root like any other manifest field,
  so it cannot be altered after signing — but a consumer who needs to verify the
  section actually came from that sidecar must re-run the deterministic
  generation and compare. Derived sections are matching-only and non-citable
  precisely so this gap cannot affect evidence integrity.

## Fuzzing

The `fuzz/` workspace contains targets for arbitrary pack opening, varint decoding, and delta-envelope parsing. Corruption and property tests run in the ordinary test suite; daily CI performs short campaigns and the weekly/manual deep workflow defaults to six hours per target. Fuzzing complements rather than replaces the independent review brief.

## Known ambiguity, carried into external review

These are open questions, not protections. They are listed here so a reviewer
finds them declared rather than discovers them, and so no reader mistakes them
for settled guarantees.

1. **Was the bounded-inflation requirement adequately stated?** Until v0.4.0 this
   document said only that a conforming parser "enforces ... decompression-ratio
   ... limits." The reference implementation caps the decompressor at the declared
   logical length. A prior clean-room Python reader read the same sentence as
   permitting a post-hoc length check and called `zlib.decompress()` with no
   output bound — while reporting that it implemented every invariant. The text
   above is now explicit. Whether it is *sufficiently* explicit for an
   implementer who has not seen this note is a question for the independent
   review, and the answer may be a further spec change.

2. **Is the RELEASE-v1 trust boundary right?** Implemented, so this is an
   attack on running code. The specific claims to attack: that revocation cannot
   live inside the artifact it revokes; that the expected scope must be
   established outside the statement, and that no path in the reference
   implementation still derives it from the document under verification; that a
   sequence plus a statement digest detects both replay and equivocation; that a
   revocation-role signature may withdraw but never promote; and that first
   contact and state loss are the residual gaps rather than hidden assumptions.

3. **Is the evidence receipt Merkle construction second-preimage resistant?**
   Leaves and interior nodes use different domain separators, and odd levels
   promote rather than duplicate. Both choices are deliberate; neither has been
   externally reviewed.
