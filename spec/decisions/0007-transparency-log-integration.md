# ADR-0007: Transparency evidence integrates an external Sigsum log; Adyar operates none

Status: accepted, 2026-08-09.

Wire format, verification order, and CLI contract are normative in
[RELEASE-v1](../RELEASE-v1.md) §7.1 and §8. This record holds the decision and
the reasoning.

## Context

[ADR-0004](0004-freshness-and-revocation.md) defines signed, sequenced release
state. A verified statement proves who signed it, but not that it was publicly
visible or unique. A publisher can sign two statements at the same sequence and
show them to different verifiers. Each statement verifies independently.
`policy.rs` already named this gap and left the extension point for it:
`TransparencyEvidence` and `TrustPolicy::AuthorizedCurrentWitnessed` existed
before an implementation could produce `Verified`. Until then, the policy
denied.

## Decision

### Integrate a real external log; do not operate one

Adyar does not run a transparency log, witness, or submission service. It
verifies a proof the publisher obtained from external infrastructure.
This follows the infrastructure boundaries for builder keys
([ADR-0006](0006-build-provenance-envelope.md)) and Sigstore verification
([PROVENANCE-v1](../PROVENANCE-v1.md) §5.3).

[Sigsum](https://www.sigsum.org/) provides C2SP tlog-tiles and tlog-checkpoint
formats for arbitrary data with offline-verifiable proofs. Adyar uses the
MIT-licensed
[`sigsum` (mullvad/sigsum-rs)](https://github.com/mullvad/sigsum-rs), pinned to
`=0.3.0` as a security boundary in `Cargo.toml`, as `sigstore-verify` is. Its
dependencies are `base16ct`, `base64ct`, `ed25519-dalek`, `sha2`, and
`thiserror`; the cryptographic dependencies are already present under the
`signing` feature. At integration time, Sigsum's Rust support had lower adoption
than Sigstore's. No more widely adopted alternative provided offline
verification for arbitrary data without a bespoke format or heavier dependency.

### What gets logged: the statement's own digest, nothing new

The Sigsum proof binds `release::statement_digest_bytes`, the BLAKE3 digest
defined by RELEASE-v1 §3 and used by §5 to distinguish `idempotent` from
`equivocation`. A publisher submits this digest with its release-state key.
Submission is outside Adyar.

### Trust configuration: operator-supplied, never fetched

An operator supplies a Sigsum policy file (the `sigsum-go` policy
syntax — log keys, witness keys, quorum) that `transparency.rs` parses
directly via `sigsum::Policy::parse`. The GitHub Sigstore trusted-root snapshot
is likewise operator-supplied in `attestation.rs`. Neither is fetched
automatically. Updates require a separate operational review.

### Signer identity is independent of the Adyar signature

`verify_transparency`'s `trusted_signer_hex_keys` are the release-state role's
authorized keys from the caller's trust root
(`trust::role_public_keys(root, ROLE_RELEASE_STATE)`), resolved independently
of the key that produced the statement's Adyar-native signature.
A Sigsum leaf signature and an Adyar channel-state signature are two
different signing operations under two different domain-separation
conventions, even when an operator reuses the same physical Ed25519 keypair
for both — `sigsum::verify`'s leaf-signing message
(`"sigsum.org/v1/tree-leaf\0" || SHA256(SHA256(digest))`) shares no bytes with
`channel_state_signing_message`'s `CHANNEL_STATE_CONTEXT || payload`. A valid
Sigsum leaf signature alone does not establish release-state authority.

### Public API boundary

`transparency.rs` takes raw strings (proof text, policy text, hex-encoded signer
keys) and returns Adyar's `TransparencyReport`. `sigsum::*` types do not appear
in public function signatures. This matches the `attestation.rs` boundary for
`sigstore-verify` and `x509-cert`; dependency API changes remain contained to
the module.

## Consequences and limitations

`authorized-current-witnessed` requires every `authorized-current` check plus
transparency evidence. A witnessed proof does not substitute for a trusted
clock, channel-state signature, or rollback protection. A verified transparency
proof for an old statement remains valid but does not establish that the
statement is current.

This module verifies one proof against one statement. Equivocation detection
requires a stateful monitor to compare independently observed entries for the
same publisher, corpus, channel, and sequence.

## Alternatives rejected

**Operating an Adyar-run transparency log.** A publisher-controlled log does
not provide independent public observation. Operating trust infrastructure is
outside Adyar's scope.

**A bespoke Adyar transparency-log wire format.** Sigsum/C2SP proofs are
interoperable with tools that do not implement Adyar formats, as with DSSE for
build provenance ([ADR-0006](0006-build-provenance-envelope.md)).

**Hand-rolling Sigsum's cryptographic verification instead of depending on
`sigsum-rs`.** Merkle-inclusion and multi-party cosignature verification are
security-critical primitives with an existing tested implementation, matching
the Sigstore boundary in [PROVENANCE-v1](../PROVENANCE-v1.md) §5.3.
