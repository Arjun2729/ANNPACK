# ADR-0006: Build provenance uses DSSE/in-toto, and builder keys are a disjoint key class

Status: accepted, 2026-08-06.

Wire format, verification order, and CLI contract are normative in
[PROVENANCE-v1](../PROVENANCE-v1.md). This record holds the decision and the
reasoning.

## Context

[RELEASE-v1](../RELEASE-v1.md) answers who may publish and what is currently
authorized. Neither it nor the artifact's own signature (`FORMAT-v3` §8.1)
answers a different question: which source, which builder, and which
execution produced these specific bytes. That is what build provenance binds,
and manifest format 4 (`ADR-0005`) is what makes the binding meaningful for
the dominant input formats rather than only for OKF.

Two design questions had to be settled before writing code: what envelope
carries the statement, and whose key signs it.

## Decision

### The envelope is DSSE; the payload is an in-toto Statement

Every other ANNPack signature — artifact (`signing.rs`), trust root
(`trust.rs`), channel state (`release.rs`), evidence receipt (`evidence.rs`) —
signs a canonical re-serialization of a document ANNPack itself defines. That
was the right call in each case: the document's shape, its domain-separation
context, and its verifier are all ANNPack's to own.

A build-provenance statement is different in kind. It is meant to be produced
by CI systems and consumed by tooling that has never heard of ANNPack —
artifact registries, SLSA verifiers, supply-chain scanners. Inventing a sixth
bespoke envelope for it would mean every one of those tools has to learn
ANNPack specifically before it can check a build claim, which defeats the
purpose of publishing one. [DSSE](https://github.com/secure-systems-lab/dsse)
and the [in-toto Statement](https://in-toto.io/Statement/v1) already exist for
this, are already what SLSA and Sigstore tooling expects, and cost nothing to
adopt: a predicate type is namespaced JSON.

This changes what "sign the document" means. Every other signer here re-derives
its message from a parsed struct, because it constructed that struct and knows
its canonical form. DSSE does not offer that discipline — Pre-Authentication
Encoding signs the **exact payload bytes present in the envelope**, and a
compliant verifier has no separate notion of "the statement, canonically
serialized" to fall back to. The implementation follows that discipline
deliberately: `payload` is a base64 string, and verification recomputes PAE over
the base64-decoded bytes, never over a re-serialization of the parsed
[`Statement`]. Doing otherwise would silently reintroduce a canonicalization
step DSSE was chosen specifically to avoid.

### Builder keys are their own class, trusted only by explicit list

ANNPack already has four key roles inside a trust root: `root`, `artifact`,
`release_state`, `emergency_revocation`. A builder key is deliberately not a
fifth role inside that structure. It answers a different question — *which
process built this* — from every question a trust root answers, and putting it
in the same object would imply that publisher trust and build trust are one
decision. They are not: an organization's release process and its publishing
authority are commonly different systems with different blast radii, and a
compromised builder should not thereby be able to sign trust roots, and a
compromised artifact-signing key should not thereby be treated as an authorized
builder.

So a builder key is trusted only by appearing in a list the verifier supplies
at call time, checked independently of any `TrustRoot`. Using an
artifact-signing key to sign provenance does not make it a trusted builder;
`an_artifact_signing_key_is_not_automatically_a_trusted_builder` in
`tests/provenance.rs` asserts this directly, because it is the failure mode
most likely to be assumed away by convenience — "we already trust this key for
something" is not the same claim as "we trust this key to have built this."

### What stays carried, never verified

`repository` and `revision` are recorded in the predicate and never promoted
past `Carried` in the verification report — there is no `Verified` variant for
that claim type. A DSSE signature proves the signer wrote those strings, not
that a repository by that name contains that revision, or that the revision
ever existed. Adding a `Verified` state for them, even one that were never
actually reached, would be an invitation to read the enum's existence as a
promise the module cannot keep.

## Alternatives rejected

**A raw ANNPack-specific signature over a custom provenance document.**
Consistent with the rest of the codebase, and wrong for this document
specifically: it would need every external verifier to implement ANNPack's
canonicalization rules before it could check anything, which is the opposite
of what publishing provenance is for.

**Treating an artifact-signing or release-state key as an implicit trusted
builder.** Fewer keys to manage, and it collapses two independent trust
decisions — who may publish, and which process may build — into one, so that
compromising either compromises both.

**Signing build provenance with a fixed CI secret in this change set.** Rejected
for now, not as a permanent position: this repository has no builder signing
key provisioned, and generating one and asking it to be stored as a repository
secret is a key-custody and rotation decision for the operator to make
deliberately, not a side effect of merging code. `release.yml` instead uses
GitHub's native `actions/attest-build-provenance` — keyless, OIDC-backed, no
secret to provision — for the attestation that is actually real today, and
publishes an unsigned ANNPack-schema statement alongside it so the
ANNPack-specific bindings (artifact root, logical root, source digest) remain
inspectable in ANNPack's own shape without requiring a caller to first parse
SLSA's predicate. Local Ed25519 signing (`provenance sign`) is fully
implemented and is what a deployment with its own builder key uses.
