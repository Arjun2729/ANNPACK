# ADR-0006: Build provenance uses DSSE/in-toto, and builder keys are a disjoint key class

Status: accepted, 2026-08-06.

Wire format, verification order, and CLI contract are normative in
[PROVENANCE-v1](../PROVENANCE-v1.md). This record holds the decision and the
reasoning.

## Context

[RELEASE-v1](../RELEASE-v1.md) defines publisher and current-release authority.
It and the artifact signature (`FORMAT-v3` §8.1) do not identify the source,
builder, and execution that produced the bytes. Build provenance binds those
claims. Manifest format 4 (`ADR-0005`) authenticates the source digest for all
input formats.

Two design questions had to be settled before writing code: what envelope
carries the statement, and whose key signs it.

## Decision

### The envelope is DSSE; the payload is an in-toto Statement

Artifact (`signing.rs`), trust root (`trust.rs`), channel state (`release.rs`),
and evidence receipt (`evidence.rs`) signatures cover canonical
re-serializations of ANNPack-defined documents.

Build-provenance statements are produced and consumed by CI systems, artifact
registries, SLSA verifiers, and supply-chain scanners.
[DSSE](https://github.com/secure-systems-lab/dsse) and the
[in-toto Statement](https://in-toto.io/Statement/v1) provide the interoperable
envelope and namespaced JSON predicate expected by SLSA and Sigstore tooling.

DSSE Pre-Authentication Encoding signs the **exact payload bytes present in the
envelope**. The `payload` is a base64 string, and verification recomputes PAE over
the base64-decoded bytes, never a re-serialization of the parsed [`Statement`].

### Builder keys are their own class, trusted only by explicit list

ANNPack trust roots define `root`, `artifact`, `release_state`, and
`emergency_revocation` roles. Builder keys are not trust-root roles. Build trust
and publisher trust commonly belong to different systems. A compromised builder
cannot sign trust roots, and a compromised artifact-signing key is not an
authorized builder.

Builder keys are trusted only by appearing in a list the verifier supplies at
call time, checked independently of any `TrustRoot`. Using an
artifact-signing key to sign provenance does not make it a trusted builder;
`an_artifact_signing_key_is_not_automatically_a_trusted_builder` in
`tests/provenance.rs` covers this boundary.

### Signing key: local Ed25519, or GitHub's keyless workload identity — never a stored CI secret

Two signing environments are supported:

- **Offline or private builds** use `annpack provenance sign` with a local
  Ed25519 key, exactly as any other ANNPack signer. The verifier trusts it only
  if it appears in the caller-supplied builder list (previous section).
- **The official GitHub release workflow** signs keylessly via GitHub's OIDC
  identity and Sigstore's Fulcio, using `actions/attest` with
  `--predicate-type https://annpack.dev/attestations/build/v1` and
  `--predicate-path` pointed at the `--predicate-only` output of
  `annpack provenance create`. Fulcio issues a certificate bound to that exact
  workflow run — repository, workflow path, ref, commit — and it expires
  immediately after use. No long-lived release-process private key is stored.
  `release.yml` keeps the generic `actions/attest-build-provenance` (SLSA)
  attestation separate from the ANNPack predicate that binds the source digest
  and artifact root. Both are published; verification of one does not establish
  the other.

`annpack provenance verify-github` uses the exactly pinned
`sigstore-verify` 0.11.0 stack. The dedicated verifier establishes trusted
signing time; verifies the Fulcio chain, certificate validity, SCT, Rekor
checkpoint/inclusion/SET, DSSE signature and artifact binding; and compares
the Rekor body with the digest, signature and certificate/public key. ANNPack
does not reimplement those security-critical primitives.

Trust is explicit and offline. The operator supplies a Sigstore trusted-root
JSON snapshot containing Fulcio, Rekor, CT and applicable TSA material. The
verification command never downloads or refreshes it. Updating roots is a
separate operational TUF workflow whose resulting file and SHA-256 must be
reviewed and recorded. An old snapshot can deterministically verify historical
material but cannot establish its own present-day currency.

Only after all cryptographic checks succeed are certificate claims extracted
and builder policy evaluated. A cryptographically valid but disallowed
workflow remains untrusted; a matching string in an invalid certificate is
never considered. Overall success additionally requires the ANNPack predicate,
subject, artifact root, authenticated source digest, and certificate/predicate
repository and revision agreements.

### What stays carried, never verified

`repository` and `revision` are recorded in the predicate and never promoted
past `Carried` in the verification report. There is no `Verified` variant for
that claim type. A DSSE signature proves the signer wrote those strings, not
that the named repository contains the revision or that the revision exists.

## Alternatives rejected

**A raw ANNPack-specific signature over a custom provenance document.** External
verifiers would need to implement ANNPack's canonicalization rules.

**Treating an artifact-signing or release-state key as an implicit trusted
builder.** This combines publisher and build authority, so compromising either
compromises both.

**Signing build provenance with a fixed CI secret stored in GitHub Secrets.** A
repository secret is a long-lived private key requiring provisioning, rotation,
and protection from workflow exfiltration. Compromise affects every release
signed with that key. GitHub's OIDC-backed keyless signing binds the predicate
to the official release workflow without that key. Local Ed25519 signing
(`provenance sign`) remains available for deployments that maintain their own
builder key outside GitHub.
