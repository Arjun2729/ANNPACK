# ANNPack Provenance v1

Status: implemented draft. Requires ANNPack Core v1.0-draft. Interacts with, but
does not modify, [RELEASE-v1](RELEASE-v1.md).

Defines how a builder cryptographically binds a source revision, a builder
identity, and a build execution to a distributed `.annpack` file. Provenance is
distributed as a separate signed statement, never inside the artifact.

**This layer does not touch the artifact format.** No section type, manifest
field, container byte, or content root is added or changed by anything
specified here.

The decision and its reasoning are in
[ADR-0006](decisions/0006-build-provenance-envelope.md).

## 1. Threat model

A verified statement establishes: a named builder, running a named executable,
produced the exact distributed file whose internal artifact root is `X`, from
source bytes whose digest is `Y`. For a manifest-format-4 artifact, `Y` is
independently authenticated by the artifact itself
([ADR-0005](decisions/0005-authenticated-source-digest.md)); the verifier
recomputes it from the artifact and compares, rather than trusting the
statement's copy.

A verified statement does **not** establish:

- that the source was correct, or that the named repository ever contained the
  named revision — `repository` and `revision` are signer assertions, reported
  as `carried`, never `verified`;
- that the repository was uncompromised;
- that the workflow was trustworthy merely because it had a name — trust comes
  only from a caller-supplied list of trusted builder keys;
- that the artifact is authorized for release — see §9;
- that the artifact was later retrieved or used by an agent — that is
  [EVIDENCE-v1](EVIDENCE-v1.md)'s question, not this specification's.

A DSSE signature proves who asserted a claim. It does not independently
corroborate the claim against anything outside the envelope, except where this
specification states a specific recomputation the verifier performs.

## 2. Envelope and payload

### 2.1 DSSE

```json
{
  "payload": "<base64>",
  "payloadType": "application/vnd.in-toto+json",
  "signatures": [ { "keyid": "<key id>", "sig": "<128 hex>" } ]
}
```

`payload` is the base64 encoding of the exact statement bytes. The signed
message is [Pre-Authentication Encoding](https://github.com/secure-systems-lab/dsse):

```text
PAE(type, body) = "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body
SP   = 0x20
LEN  = ASCII decimal byte length, no leading zeros
```

A verifier MUST compute PAE over the base64-**decoded** payload bytes exactly as
received, never over a re-serialization of the parsed statement. Re-serializing
would silently reintroduce a canonicalization step DSSE exists to avoid, and
would make a signature valid against a payload the signer never produced.

`keyid` is `BLAKE3(public key bytes)`, hex-encoded — the same convention as
every other key identifier in ANNPack. DSSE does not carry the public key
itself; a verifier checks each signature against each key in its trusted-builder
list (§6) and reports a match only on cryptographic success against a specific
candidate key.

### 2.2 in-toto Statement

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [ { "name": "<filename>", "digest": { "sha256": "<64 hex>" } } ],
  "predicateType": "https://annpack.dev/attestations/build/v1",
  "predicate": { "builder": {…}, "source": {…}, "build": {…}, "annpack": {…} }
}
```

`subject` MUST contain exactly one entry, naming the distributed file. Zero
entries or more than one are both refused: an ambiguous subject cannot be bound
to one distributed file, and this specification defines no rule for resolving
which of several subjects a verifier should check.

### 2.3 Predicate

```json
{
  "builder": {
    "id": "github-actions:release:1234567890",
    "annpack_version": "0.7.0-rc1",
    "annpack_binary_sha256": "<64 hex, or absent>"
  },
  "source": {
    "repository": "github.com/example/docs",
    "revision": "git:abc123",
    "tree_digest": "<64 hex>",
    "tree_digest_algorithm": "blake3",
    "format": "markdown"
  },
  "build": {
    "invocation_id": "1234567890-1",
    "started_at": "2026-08-06T00:00:00Z",
    "finished_at": "2026-08-06T00:01:00Z",
    "parameters": {},
    "environment": {},
    "platform": "x86_64-unknown-linux-gnu",
    "locked": true
  },
  "annpack": {
    "artifact_root": "<64 hex>",
    "logical_content_root": "<64 hex, or absent>",
    "manifest_format_version": 4,
    "source_binding": "authenticated"
  }
}
```

| Field | Rule |
|---|---|
| `builder.id` | Free text: workflow, workload, or operator identity. Not itself a trust decision — see §6. |
| `builder.annpack_version` | The creating process's own `CARGO_PKG_VERSION`, recorded unconditionally. |
| `builder.annpack_binary_sha256` | SHA-256 of the exact executable that performed the build, when the creator was given a path to it. Absent, never fabricated, when it was not. |
| `source.repository`, `source.revision` | Caller-supplied. Never independently corroborated by this specification. See §1. |
| `source.tree_digest` | For a format-4 artifact: read from the artifact's own authenticated `SourceDescriptor`, never accepted as an independent creation-time parameter — see §4. For a legacy artifact: a caller-supplied assertion, recorded honestly as such. |
| `source.format` | The resolved input format the digest was computed under. Absent for a legacy statement. |
| `annpack.source_binding` | `authenticated` or `absent_legacy_artifact`, mirroring the artifact's own [`SourceBinding`](RELEASE-v1.md). Carried inside the signed predicate so a verifier's independent recheck (§7) can be compared against what the creator claimed, rather than only against the artifact. |
| `build.parameters`, `build.environment` | Opt-in only. Empty unless the creator explicitly supplied entries. Nothing is captured by default — see §11. |

### 2.4 Limits

| Limit | Value |
|---|---|
| Subjects | exactly 1 |
| Signatures | 16 |
| `parameters` + `environment` entries | 64 each |
| Envelope size read by the reference CLI | 1 MiB |

## 3. Identities kept separate

Never merged, never inferred from one another:

| Identity | Meaning |
|---|---|
| Source revision | Caller-supplied context (`source.revision`). A carried claim. |
| Source-tree digest | The exact consumed bytes (`source.tree_digest`). Authenticated for format 4, carried for legacy. |
| Builder semantic version | `builder.annpack_version`. Self-reported by the creating process. |
| Builder executable digest | `builder.annpack_binary_sha256`. Independently rechecked only when a binary path is supplied to the verifier. |
| Workflow / builder identity | `builder.id`. Free text; trust comes from §6, never from this string. |
| Artifact root | `annpack.artifact_root`. Recomputed by the verifier from the artifact, always. |
| Distributed file digest | `subject[0].digest.sha256`. Recomputed by the verifier from the file on disk, always. |

## 4. Creation

Two functions, deliberately not one with a mode flag:

- **`create_build_provenance`** — the common path. Requires a manifest-format-4
  artifact. Every artifact-derived fact — file digest, artifact root, logical
  root, manifest format version, source digest — is read from the artifact or
  computed from the file; there is no parameter through which a caller can
  supply a source digest that disagrees with the artifact's own. This makes
  "the supplied digest contradicts the authenticated one" structurally
  impossible rather than merely checked for.
- **`create_legacy_build_provenance`** — requires a caller-supplied source
  digest and requires the artifact's manifest format to be below 4. The
  resulting statement's `source_binding` is honestly `absent_legacy_artifact`.

Each function refuses the artifact the other is for. Creation from either
function fails when:

- the artifact does not pass [`PackReader::verify_all`](FORMAT-v3.md) —
  signed provenance for content that is not even self-consistent asserts a
  build chain for bytes that fail their own integrity check;
- the wrong function was called for the artifact's manifest format.

## 5. Signing

Local Ed25519 signing is implemented (`sign_provenance`). The signature is over
PAE of the exact serialized statement bytes (§2.1); nothing about the signed
message is re-derived at verification time.

Keyless signing (GitHub OIDC / Sigstore, workload identity, KMS/HSM-backed
signing) is a defined but unimplemented extension point — see §12. `release.yml`
uses GitHub's native `actions/attest-build-provenance` for the attestation that
is actually verifiable today, and separately publishes an **unsigned** ANNPack
statement alongside it, so the ANNPack-specific bindings remain inspectable in
this schema without requiring a caller to first parse SLSA's predicate. An
unsigned statement establishes nothing on its own; it is data, not evidence,
until a deployment provisions a builder key and signs it.

## 6. Builder trust

A builder key is not a [RELEASE-v1](RELEASE-v1.md) trust-root role. It is
trusted only by explicit inclusion in the list a verifier supplies at call time.

Using an artifact-signing, release-state, or revocation key to sign provenance
does not make that key a trusted builder. The two trust decisions are
independent by design ([ADR-0006](decisions/0006-build-provenance-envelope.md)):
an organization's build process and its publishing authority are commonly
different systems, and a compromise of one must not silently confer the other's
authority.

## 7. Verification procedure

1. Bound and parse the envelope. Reject more than 16 signatures before
   interpreting anything.
2. Base64-decode `payload`; parse as a `Statement`.
3. Check `_type` is `https://in-toto.io/Statement/v1` and `predicateType` is
   `https://annpack.dev/attestations/build/v1`. An unsupported value is
   recorded and forces `completeness = invalid` (§8) regardless of what the
   remaining steps find; it does not skip them. The reference implementation
   still computes and reports every other binding, on the same principle
   applied throughout this specification and RELEASE-v1: a caller sees exactly
   which facts held and which did not, rather than one check suppressing the
   rest of the report.
4. Check `subject` has exactly one entry.
5. For each key in the caller's trusted-builder list, check for a valid
   signature via PAE recomputed over the decoded payload bytes (§2.1).
   `builder_identity` is `trusted` when at least one validates, `untrusted` when
   signatures exist but none validate against a trusted key, `unknown` when no
   trusted-builder list was supplied at all.
6. Open the artifact; verify its integrity (`PackReader::verify_all`).
7. Recompute the distributed file's SHA-256; compare to `subject[0].digest.sha256`.
8. Recompute the artifact root and, if present, the logical content root;
   compare to `annpack.artifact_root` / `annpack.logical_content_root`.
9. Recompute the artifact's `SourceBinding`. For `authenticated`: compare the
   artifact's own descriptor digest to `source.tree_digest`. For
   `absent_legacy_artifact`: report `source_digest_binding = absent_legacy_artifact`
   and do not treat `source.tree_digest` as anything but a carried claim.
10. If a builder-binary path was supplied: hash it and compare to
    `builder.annpack_binary_sha256`; invoke it with `--version` and compare its
    output to `builder.annpack_version`. Without a path, both report
    `unsupported` — not `missing`, since the claim may be present and simply
    unchecked.
11. Report `repository_claim` and `revision_claim` as `carried` (or `missing` if
    empty). Never promote either to a verified state.
12. Compute `completeness` and `verified` per §8.

## 8. Verdicts

Every binding is reported independently; nothing is collapsed into one boolean
before the caller sees it.

| Field | Values |
|---|---|
| `envelope_signature` | `valid` \| `invalid` \| `unsigned` |
| `builder_identity` | `trusted` \| `untrusted` \| `unknown` |
| `artifact_integrity` | `verified` \| `mismatched` |
| `distributed_file_digest`, `artifact_root_binding`, `logical_root_binding`, `builder_binary_binding`, `builder_version_binding` | `verified` \| `mismatched` \| `missing` \| `unsupported` |
| `source_digest_binding` | `authenticated` \| `absent_legacy_artifact` \| `mismatched` \| `missing` |
| `repository_claim`, `revision_claim` | `carried` \| `missing` — no `verified` variant exists for this type |
| `completeness` | `complete` \| `partial_legacy_source_binding` \| `invalid` |

`completeness` is `invalid` unless every hard binding holds: supported predicate
and statement type, exactly one subject, a signature from a trusted builder,
artifact integrity, the distributed-file digest, the artifact root, a
non-mismatched logical root, and non-mismatched builder-binary bindings when a
binary was supplied. Given those, `completeness` is `complete` when
`source_digest_binding` is `authenticated`, `partial_legacy_source_binding` when
it is `absent_legacy_artifact`, and `invalid` if it is `mismatched` or `missing`.

`verified` is true for `complete` or `partial_legacy_source_binding`, never for
`invalid`. A legacy artifact's provenance can verify; it cannot claim complete
source-to-artifact binding, and `completeness` says which case applies rather
than collapsing both into one flag.

## 9. Relationship to RELEASE-v1

Provenance answers *how* an artifact was built. [RELEASE-v1](RELEASE-v1.md)
answers *whether* it is authorized for use. The two reports are never merged:

```text
build_provenance.completeness   = complete
build_provenance.builder_identity = trusted
publisher_authority              = authorized
currency                          = current
```

A caller who wants both checked composes the two verifications and reads both
reports. Neither function reads the other's inputs; `provenance.rs` does not
read or write channel state, and `release.rs` does not read provenance
envelopes.

## 10. CLI

```bash
annpack provenance create <artifact> --output <file> \
  --repository <repo> --revision <rev> --builder-id <id> \
  [--builder-binary <path>] [--system-clock | --started-at <ts> --finished-at <ts>] \
  [--param k=v ...] [--env k=v ...] [--platform <target>] [--locked true|false] \
  [--legacy --legacy-source-digest <hex>]

annpack provenance sign <statement> --key <secret-key-file> [--output <file>]

annpack provenance verify <artifact> <envelope> \
  [--trusted-builder-key <hex> ...] [--builder-binary <path>] [--json]
```

Exit classes and the JSON failure envelope follow the contract established in
[RELEASE-v1 §8](RELEASE-v1.md#8-cli-contract): broad stable classes plus an
exact `error.kind`, and exactly one structured object on stdout in `--json`
mode on every path.

| `error.kind` | Class |
|---|---|
| `unsupported_predicate`, `malformed_input` | 3 |
| `invalid_signature`, `untrusted_builder`, `integrity_failed`, `file_digest_mismatch`, `artifact_root_mismatch`, `logical_root_mismatch`, `builder_binary_mismatch`, `builder_version_mismatch`, `source_digest_mismatch` | 5 |

`annpack provenance verify` exits non-zero whenever `verified` is false,
including the `invalid` completeness case. A `partial_legacy_source_binding`
result exits 0: the brief distinguishes "legacy artifacts correctly produce
partial binding" from "a broken statement," and only the latter is a failure.

## 11. Privacy

Nothing is captured by default. `build.parameters` and `build.environment` are
empty unless the creator explicitly names entries with `--param`/`--env`. This
specification defines no default set of environment variables or build flags
to record, and implementers MUST NOT add one without the same opt-in
discipline: a builder identity that leaks operator-specific paths, hostnames,
or credentials by default is a privacy defect, not a completeness feature.

## 12. OCI mapping

When an ANNPack artifact is distributed through OCI ([MEDIA-TYPES.md](MEDIA-TYPES.md)),
a provenance statement is published as a **separate referrer artifact**, never
embedded in the ANNPack artifact's own manifest.

- Subject: the ANNPack OCI manifest's digest.
- `artifactType`: `application/vnd.annpack.provenance.v1+json`
- Blob media type: `application/vnd.in-toto+json` (the DSSE envelope, exactly
  as produced by `provenance sign` — not re-wrapped).
- `predicateType` (recorded as an annotation, since OCI referrer discovery does
  not itself parse in-toto predicates): `https://annpack.dev/attestations/build/v1`

Recommended annotations:

```text
dev.annpack.provenance.artifact_root
dev.annpack.provenance.builder_id
org.opencontainers.image.created
```

Local-file verification (`provenance verify`) never depends on OCI. The mapping
exists so a registry-hosted artifact can carry its provenance as a discoverable
referrer, not as a requirement for verifying provenance at all.

## 13. Non-goals

Operating a transparency log for provenance statements — that is
[ADR-0004](decisions/0004-freshness-and-revocation.md)'s witnessed-profile
concern, not this specification's. Keyless signing infrastructure (Fulcio,
Rekor, KMS integration) — the signer abstraction is designed for it; nothing
here implements it. Proving repository or revision claims true. Binding
provenance to a specific agent run or retrieval — that is
[EVIDENCE-v1](EVIDENCE-v1.md) and run bundles, an entirely separate claim about
an entirely separate execution.
