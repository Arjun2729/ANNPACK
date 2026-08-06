# GitHub keyless build-verification fixture

This directory is the repository-owned offline happy path for ANNPack build
provenance. It is a GitHub/Sigstore build attestation, not ANNPack run
attestation.

## Origin

- Source commit: `9cdaf8ae36659bfa7cc825ec4aacc3e86a586df0`
- GitHub Actions run: `31104714110`
- GitHub attestation: `39266120`
- Rekor log index: `2360482196`
- Attested artifact SHA-256:
  `44174d6d3b530c4e5ea8154cba450749b3e6b53066468544ce0d6e5473945a04`

The workflow's build and `actions/attest` steps succeeded. Its initial export
step then failed because a direct, unversioned TUF target URL returned 404, so
the bundle was recovered from GitHub's public repository-attestation API by the
attested artifact digest. The deterministic artifact was rebuilt from the same
commit and matched the attested SHA-256 exactly. The trusted-root file is the
exact production snapshot shipped by the pinned `sigstore-trust-root` 0.11.0
crate; verification receives it explicitly and never falls back to an embedded
root. The manual workflow now exports that exact snapshot directly.

## Authenticated workload claims

- Issuer: `https://token.actions.githubusercontent.com`
- Repository: `https://github.com/Arjun2729/ANNPACK`
- Workflow identity:
  `https://github.com/Arjun2729/ANNPACK/.github/workflows/sigstore-verification-fixture.yml@refs/heads/codex/sigstore-verification-fixture`
- Source revision: `9cdaf8ae36659bfa7cc825ec4aacc3e86a586df0`
- Source ref: `refs/heads/codex/sigstore-verification-fixture`
- Runner environment: `github-hosted`
- Rekor log ID: `wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0=`

`sigstore-fixture.expected-report.json` is the complete successful report.
`sigstore-fixture.sha256` pins the artifact, bundle, predicate, trusted-root
snapshot, and expected report independently. Tests recompute every digest and
perform verification with all network proxies pointed at an unreachable local
endpoint.

## Pinned fixture digests

| File | SHA-256 |
| --- | --- |
| `sigstore-fixture.annpack` | `44174d6d3b530c4e5ea8154cba450749b3e6b53066468544ce0d6e5473945a04` |
| `sigstore-fixture.bundle.json` | `e5709b4a587a13dac6fc18e6d9b016847254ee7c40e60f4417364f9048711e38` |
| `sigstore-fixture.predicate.json` | `017ea7839f63b3aeb21974a1499dd41f90f084d3b4af728f58470f94cdbbabaa` |
| `sigstore-fixture.trusted-root.json` | `6494e21ea73fa7ee769f85f57d5a3e6a08725eae1e38c755fc3517c9e6bc0b66` |
| `sigstore-fixture.expected-report.json` | `59df41a93c74506fc926207d74ebe26f8d1a12d3aab917819e1ea59e1089ff12` |

## Rotation

Rotate this fixture when the supported Sigstore bundle or trusted-root schema
changes, the pinned verifier changes its accepted cryptographic evidence, the
GitHub workload-identity policy changes, a retained Fulcio/Rekor key no longer
covers the signing event, or a regression can no longer exercise the current
verification path. Review it at every explicit `sigstore-verify` dependency
upgrade even when none of those conditions appears to have changed. Rotation
means running the manual fixture workflow, reviewing the authenticated claims,
retaining the new artifact and evidence files, and replacing every digest in
both this table and `sigstore-fixture.sha256` in one change.

Trusted-root currency and historical signature validity are separate concerns.
An operator needs a reviewed, sufficiently current root snapshot for new
verification events and key rotations. This fixture pins the exact snapshot
used to verify its historical signing event; that snapshot becoming old does
not by itself invalidate the already valid Fulcio certificate, Rekor inclusion,
or artifact signature. Conversely, a currently distributed root does not make
an invalid historical signature valid. Operational root refreshes must preserve
the older trust material needed to evaluate retained historical evidence.
