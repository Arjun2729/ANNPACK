# ADR-0007: Transparency evidence integrates an external Sigsum log; ANNPack operates none

Status: accepted, 2026-08-09.

Wire format, verification order, and CLI contract are normative in
[RELEASE-v1](../RELEASE-v1.md) §7.1 and §8. This record holds the decision and
the reasoning.

## Context

[ADR-0004](0004-freshness-and-revocation.md) separated historical validity from
release authorization and gave release state its own signed, sequenced
statement. That closed the "is this artifact still current" gap, but left a
narrower one open: a verified statement proves who signed it, not that the
signing was public or singular. A publisher whose signing process is
compromised — or simply dishonest — can sign two different statements at the
same sequence number and show each to a different verifier. Neither verifier
sees the other's copy, and both statements verify perfectly on their own.
`policy.rs` already named this gap and left the extension point for it:
`TransparencyEvidence` and `TrustPolicy::AuthorizedCurrentWitnessed` existed
from the same phase that built the rest of the policy engine, deliberately
always denying — "a policy whose requirement cannot yet be met must refuse,
not degrade" — until something could actually produce `Verified`.

This record is that something.

## Decision

### Integrate a real external log; do not operate one

ANNPack does not run a transparency log, a witness, or a submission service.
It verifies a proof the publisher already obtained from one. This mirrors
[ADR-0006](0006-build-provenance-envelope.md)'s builder-key decision and
[PROVENANCE-v1](../PROVENANCE-v1.md) §5.3's Sigstore-verification boundary:
operating trust infrastructure and verifying its output are different
undertakings with different failure modes, and this project has consistently
chosen the second.

[Sigsum](https://www.sigsum.org/) was chosen over rolling a bespoke log or
integrating Certificate Transparency directly. It is part of the C2SP family
(tlog-tiles, tlog-checkpoint), so its wire formats are open specifications, not
one vendor's API; it is designed specifically for signing arbitrary data
(unlike CT, which is certificate-shaped) with offline-verifiable proofs, which
is exactly what a release-state statement's digest is; and a real, maintained
Rust crate exists — [`sigsum` (mullvad/sigsum-rs)](https://github.com/mullvad/sigsum-rs),
MIT-licensed, `=0.3.0` pinned as a security boundary exactly as
`sigstore-verify` is pinned in `Cargo.toml`. Its own dependency tree
(`base16ct`, `base64ct`, `ed25519-dalek`, `sha2`, `thiserror`) adds no
duplicate cryptographic implementation: `ed25519-dalek` and `sha2` are already
present in this project under the `signing` feature. Sigsum's Rust support is
younger and lower-adoption than Sigstore's — `sigsum-rs` had roughly 700
crates.io downloads and 4 GitHub stars at integration time, against
`sigstore-verify`'s considerably larger footprint — which is disclosed here
plainly rather than smoothed over: it is a real trade against maturity, made
because Sigsum's design fit is closer and no more-adopted alternative offered
offline verification of an arbitrary-data transparency proof without either a
bespoke format or a much heavier dependency.

### What gets logged: the statement's own digest, nothing new

The value bound into the Sigsum proof is
`release::statement_digest_bytes` — the exact BLAKE3 digest RELEASE-v1 §3
already defines and §5's sequence-verdict logic already uses to distinguish
`idempotent` from `equivocation`. This was a deliberate reuse, not a
convenience: inventing a second "statement identity" for transparency
purposes would raise the question of whether the two identities could ever
disagree, and answering "they can't, they're computed identically" is weaker
than "there is only one, and both features read it." A publisher submits this
digest to a Sigsum log using their release-state key; nothing about how that
submission happens is ANNPack's concern, only that a verifier can later check
the proof it produced.

### Trust configuration: operator-supplied, never fetched

An operator supplies a Sigsum policy file (the real `sigsum-go` policy
syntax — log keys, witness keys, quorum) that `transparency.rs` parses
directly via `sigsum::Policy::parse`, reusing the real ecosystem format rather
than inventing an ANNPack-specific schema, exactly as the GitHub Sigstore
trusted-root snapshot is operator-supplied JSON in `attestation.rs`. Neither
is ever fetched automatically. Updating either is a deliberate operational act
with its own review, not a side effect of verifying an artifact.

### Signer identity is checked independently, never merged with ANNPack's own signature

`verify_transparency`'s `trusted_signer_hex_keys` are the release-state role's
authorized keys from the caller's trust root
(`trust::role_public_keys(root, ROLE_RELEASE_STATE)`), resolved independently
of whichever key actually produced the statement's ANNPack-native signature.
A Sigsum leaf signature and an ANNPack channel-state signature are two
different signing operations under two different domain-separation
conventions, even when an operator reuses the same physical Ed25519 keypair
for both — `sigsum::verify`'s leaf-signing message
(`"sigsum.org/v1/tree-leaf\0" || SHA256(SHA256(digest))`) shares no bytes with
`channel_state_signing_message`'s `CHANNEL_STATE_CONTEXT || payload`. Treating
a valid Sigsum leaf signature as proof of release-state authority on its own
would be exactly the mistake ADR-0006 named for builder keys: two independent
trust decisions collapsed into one because the same key happened to satisfy
both.

### The module's own public API never leaks the third-party crate's types

`transparency.rs` takes raw strings (proof text, policy text, hex-encoded
signer keys) and returns ANNPack's own `TransparencyReport`; `sigsum::*` types
never appear in a public function signature. This matches `attestation.rs`'s
existing boundary with `sigstore-verify` and `x509-cert`: a caller of either
module never needs to depend on the underlying crate directly, and a future
change to either dependency's API surface stays contained to one module.

## What stays exactly as strict as before

`authorized-current-witnessed` still requires every `authorized-current`
check in addition to transparency evidence, never instead of them — a
witnessed proof does not substitute for a trusted clock, a channel-state
signature, or rollback protection (RELEASE-v1 §5's sequence rules are
untouched). A verified transparency proof for an old, superseded statement
verifies exactly as well as one for the current statement: **fresh inclusion
of an old release statement does not prove that statement is the latest
release.** This module answers "was this logged and witnessed," never "is
this current," and never claims otherwise.

Equivocation *detection* — comparing multiple independently observed log
entries for the same publisher/corpus/channel/sequence and flagging
disagreement — is explicitly out of scope here. This module verifies one
proof against one statement. A monitor that compares many is a separate,
stateful component with its own design questions (chiefly: where does
observation history live), deliberately not decided by this record.

## Alternatives rejected

**Operating an ANNPack-run transparency log.** Rejected for the same reason
ANNPack does not operate a certificate authority for build provenance: running
trust infrastructure and verifying its output are different responsibilities
with different failure domains, and a self-run log cannot make the "public,
independently observable" claim transparency exists to provide — a log only
the publisher controls is not meaningfully different from the publisher's own
signature.

**A bespoke ANNPack transparency-log wire format.** Consistent with the rest
of the codebase's preference for ANNPack-owned document shapes, and wrong
here for the same reason DSSE was chosen over a raw ANNPack signature for
build provenance ([ADR-0006](0006-build-provenance-envelope.md)): a
transparency proof is meant to be checkable by tooling that has never heard of
ANNPack, and Sigsum/C2SP already exist for exactly this.

**Hand-rolling Sigsum's cryptographic verification instead of depending on
`sigsum-rs`.** Rejected for the same reason `sigstore-verify` was chosen over
hand-rolling Sigstore verification ([PROVENANCE-v1](../PROVENANCE-v1.md)
§5.3): Merkle-inclusion and multi-party cosignature verification are
security-critical primitives with a real history of subtle bugs, and a real,
tested implementation exists.
