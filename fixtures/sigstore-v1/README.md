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
