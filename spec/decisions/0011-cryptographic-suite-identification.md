# ADR-0011: Cryptographic suite identification is explicit and additive

Status: accepted, 2026-08-10. Locks identification and backward-compatibility
requirements only; does not choose, add, or implement a second suite — see
"What this does not do."

## Context

Every hash and signature in the format today is an implicit choice, not a
named one. `format.rs` stores artifact roots and section hashes as bare
`[u8; 32]` with no algorithm tag alongside them. `passage_id` and
`document_id` are `stable_hash()`, which is `blake3::Hasher` with no other
path. `key_id` is `blake3::hash(&public_key)` in every one of the four
places that compute it (`signing.rs`, `trust.rs`, `provenance.rs`). Signature
verification carries an `algorithm` field on the wire, but it is checked in
four separate files (`evidence.rs`, `signing.rs`, `trust.rs`, `fleet.rs`) as
`if algorithm != "Ed25519" { reject }` — a field that exists syntactically
but gates exactly one value. `spec/CORE-v1.0-draft.md` names Ed25519 directly
in normative text. None of this is a defect in what exists; BLAKE3 and
Ed25519 are reasonable choices today. It is a defect in what the format
assumes about tomorrow: that today's primitives are the only ones it will
ever need to represent.

That assumption is fine for a build tool. It is not fine for a format whose
stated purpose is that evidence produced under it stays verifiable
indefinitely. ADR-0004 already established the shape this problem takes —
it calls it **historical validity**: "did these bytes exist, unmodified, in
this artifact," a claim that must remain answerable regardless of what
happens to the artifact's *current* status. ADR-0004 solved this for release
authorization (revocation does not make a past artifact stop having existed).
The same question exists one layer down, for the primitives themselves: if
BLAKE3 or Ed25519 is ever retired, does an artifact built before that
retirement stop being verifiable, or does it stay exactly as verifiable as
it was the day it was built? Today there is no mechanism to answer that
question either way, because there is nothing recorded to distinguish "this
artifact used the old algorithm on purpose" from "this artifact predates
this decision entirely."

The failure mode to avoid is not choosing the wrong algorithm now — it is
having no place to record which algorithm was chosen, discovered only after
a primitive actually needs replacing and every artifact already published
was built under an unversioned assumption. At that point there is no way to
add a suite identifier retroactively; every existing artifact is silently
"whatever suite 1 turns out to have been," indistinguishable from tampering
with no suite tag to check against. The fix has to exist before it is
needed, in the same way `SUPPORTED_MANIFEST_FORMAT_VERSIONS` and
`SUPPORTED_LEXICAL_FORMAT_VERSIONS` already let this format carry four
manifest generations and two lexical index generations side by side without
disturbing what already shipped — this is the same shape of change, applied
to the cryptographic layer instead of the container layer.

## Decision

### Every field that currently assumes an algorithm can name one explicitly; absence means today's suite

A future artifact, statement, or key MAY carry an explicit algorithm
identifier alongside a hash or signature. An object with no identifier is
suite 1: BLAKE3 for every content hash, identity hash, and root; Ed25519 for
every signature. This is not a new rule bolted onto old data — it is
today's already-normative behavior, restated as the default case of a
richer rule instead of the only case. No existing artifact's bytes, root
computation, or verification result changes because of this ADR.

### Content-identity hashing and signature algorithms are versioned independently

A passage's identity hash and a publisher's signature algorithm are not the
same kind of thing and do not need to migrate together. The hash is load-
bearing for content addressing across the entire evidence chain — passage
IDs, document IDs, evidence receipts, Merkle roots, artifact roots, key IDs
— and changing it changes identity everywhere it is used. The signature is
load-bearing for one signer's authenticity claim over one signed object and
can be rotated per-key, per-role, or per-publisher independently of what
hash function anything else in the format uses. Naming them as one combined
"suite version" would force a hash migration to drag a signature migration
along with it, or the reverse, for no reason but fewer identifiers to
track. They are named separately.

### An unrecognized suite identifier is a rejection, not a skip

A verifier that encounters a hash or signature algorithm it does not
implement MUST fail closed: report the object as unverified, the same as if
the signature had not matched. It MUST NOT silently omit that check and
report the rest of the object as verified, and MUST NOT treat an
unrecognized suite as equivalent to an absent one. This is the same
discipline this codebase already enforces for feature-gated verification —
no verification path silently downgrades to a weaker claim than the one it
appears to make.

### Retiring a suite for new signing does not revoke historical verifiability

Policy MAY forbid *newly created* artifacts or statements from using a
retired suite — a fleet policy or trust root can say "no more suite 1
signing after this date" — without that policy affecting a single byte of
what already exists. An artifact signed under suite 1 in 2026 remains a
suite-1 artifact forever; a verifier that still implements suite 1
continues to accept it under the rules suite 1 always had. This is
identical to how ADR-0004 already treats a superseded or revoked release:
the object's history does not get rewritten by a later policy decision
about what may currently be created.

## What this does not do

Add, choose, or implement a second cryptographic suite. No new hash or
signature algorithm is introduced by this ADR. Suite 1 — BLAKE3 for hashing,
Ed25519 for signatures — remains the only suite any implementation needs to
support, including whatever independent implementation eventually satisfies
`spec/CORE-v1.0-draft.md`'s `-draft`-removal condition. That condition is
scoped to today's already-normative primitives; it should stay scoped there
rather than chase a suite that does not exist yet.

Specify the wire syntax or field placement for a suite identifier — a
prefixed string on the hash value itself, a manifest-level field, a
per-hash-type field, or something else. That is normative work for whenever
a second suite is actually being added, in the same way ADR-0010 deferred
the canonical input schema until adapters existed to write it against. This
ADR locks that the identification and default-means-suite-1 behavior must
exist when that time comes; it does not design the syntax now, against no
concrete second suite to test it on.

Change any currently shipped behavior. Every artifact, receipt, statement,
and key produced today remains exactly as valid and exactly as verifiable
after this ADR as before it.

## Alternatives rejected

**Wait until a primitive actually needs replacing, then design this.**
Rejected: by that point every already-published artifact was built under an
implicit, unversioned assumption, with nothing recorded to distinguish "old
artifact, deliberately suite 1" from "old artifact, tampered, no suite
information to check." The mechanism has to exist before the need does, or
it cannot be applied to anything already published.

**One combined suite identifier covering both hashing and signatures.**
Rejected: the two have different consumers (content addressing versus
signer authenticity), different blast radii if compromised, and no reason
to share a migration timeline. Bundling them buys fewer identifiers at the
cost of forcing unrelated migrations to happen together.

**Add a second, real algorithm now — for example a post-quantum signature
scheme — as part of this decision.** Rejected: there is no measured need
for one yet, and shipping a second suite before an independent
implementation of Core exists would make that implementer's target a
moving one. The `-draft` marker's removal condition should stay aimed at
what is already normative, not at a design this ADR deliberately leaves
unfinished.
