# ANNPack Fleet Policy v1

Status: implemented draft.

Defines a signed, versioned document an organization issues to state what its
fleet of verifiers requires, and how a verifier checks its local
configuration against it.

This layer does not change the artifact format, trust roots, or channel-state
statements. It answers a different question than either: not who may
publish, but what a consuming organization requires.

The decision and its reasoning are in
[ADR-0009](decisions/0009-fleet-policy.md).

## 1. Scope

A [`TrustRoot`](RELEASE-v1.md) names who may publish. A `TrustPolicy`
(`policy.rs`) names what one verification call checked. Neither states what
an organization's fleet of verifiers is required to check. `FleetPolicy`
does.

ANNPack does not distribute fleet policy. This document defines the object
and its own verification. Fetching a required policy from a control plane is
later work.

## 2. Fleet policy document

```json
{
  "schema": "annpack-fleet-policy-v1",
  "domain": "acme.example",
  "revision": 3,
  "issued_at": "2026-08-09T00:00:00Z",
  "valid_until": "2027-08-09T00:00:00Z",
  "threshold": 1,
  "keys": {
    "<key-id>": { "algorithm": "Ed25519", "public_key": "<64 hex>" }
  },
  "allowed_publishers": ["example.com"],
  "allowed_scopes": [ { "corpus": "support-manual", "channel": "production" } ],
  "required_verification_policy": "authorized_current_witnessed",
  "required_transparency_policy_digest": "<64 hex, or absent>",
  "required_workload_trust_digest": "<64 hex, or absent>",
  "max_statement_validity_seconds": 3600,
  "deny_on_incident_kinds": ["equivocation", "revoked_root_advertised"],
  "signatures": [ { "key_id": "<key-id>", "signature": "<128 hex>" } ]
}
```

### 2.1 Fields

| Field | Rule |
|---|---|
| `schema` | MUST be `annpack-fleet-policy-v1`. |
| `domain` | Non-empty. Identifies the organization or security domain. A rotation MUST NOT change it. |
| `revision` | Integer ≥ 1. Strictly increases across rotations. |
| `issued_at`, `valid_until` | As `RELEASE-v1` §2.1: `YYYY-MM-DDTHH:MM:SSZ`, UTC only. |
| `threshold`, `keys` | A flat signer set, not role-separated: fleet policy authorizes a requirement, not a publication. `threshold` ≥ 1 and ≤ the number of distinct keys. |
| `allowed_publishers`, `allowed_scopes` | Which publishers and corpus/channel pairs this fleet may use. Carried; not independently enforced by this layer. |
| `required_verification_policy` | One of `policy::TrustPolicy`'s four values. |
| `required_transparency_policy_digest` | Digest of the exact Sigsum trust-policy text a verifier must use. Absent means not pinned beyond what `required_verification_policy` implies. |
| `required_workload_trust_digest` | Digest of the required workload-trust configuration for run-attestation verification. Absent means not pinned. |
| `max_statement_validity_seconds` | Absent means not constrained. |
| `deny_on_incident_kinds` | `monitor::IncidentKind` names, stored as strings so an older verifier can read a document naming a kind it does not yet recognise. |
| `signatures` | Zero or more `{key_id, signature}`. |

### 2.2 Signature

Domain separator: `ANNPACK3-FLEET-POLICY\0`

Signed message: that separator followed by the canonical serialization of the
document excluding `signatures`, over `schema`, `domain`, `revision`,
`issued_at`, `valid_until`, `threshold`, `keys`, `allowed_publishers`,
`allowed_scopes`, `required_verification_policy`,
`required_transparency_policy_digest`, `required_workload_trust_digest`,
`max_statement_validity_seconds`, `deny_on_incident_kinds`.

**Policy digest** is BLAKE3 of that same serialization. Two policies agree if
and only if their digests agree.

A threshold counts distinct authorised key ids that produced a valid
signature, never signature entries.

### 2.3 Rotation

A successor is verified only when signed by a threshold of its own keys and
a threshold of the prior policy's keys, and its revision strictly exceeds the
prior's. Both requirements exist for the same reason `TrustRoot` rotation
requires both: self-only lets anyone mint a policy and present it;
prior-only lets a compromised old key install keys nobody controls.

Rotation across a different `domain` does not verify.

First contact — no prior policy supplied — is accepted on trust and reported
as such.

### 2.4 Limits

| Limit | Value |
|---|---|
| Keys | 128 |
| Signatures | 128 |
| `allowed_publishers` entries | 4096 |
| `allowed_scopes` entries | 4096 |
| `deny_on_incident_kinds` entries | 32 |
| File size read by the reference CLI | 1 MiB |

## 3. Compliance evaluation

Given a locally configured fleet policy and the policy that should be in
effect, report whether they agree.

Both documents are independently re-verified before comparison; a caller
cannot make an unverified document count as compliant by asserting it
verified elsewhere.

| Local | Required | Result |
|---|---|---|
| missing or fails to verify | — | `unavailable` |
| — | missing or fails to verify | `unavailable` |
| verifies | verifies, same revision and digest | `compliant` |
| verifies | verifies, different revision or digest | `drifted` |

`unavailable` is never treated as compliant. Comparing policies for different
`domain` values is a caller error, reported as such, not a compliance
verdict.

## 4. CLI contract

```bash
annpack fleet policy init --output <file> --domain <domain> --revision <n> \
  --valid-until <ts> --key <public-key-file> ... --threshold <n> \
  [--allow-publisher <name> ...] [--allow-scope <corpus:channel> ...] \
  [--required-policy <policy>] [--required-transparency-policy-digest <hex>] \
  [--required-workload-trust-digest <hex>] [--max-statement-validity-seconds <n>] \
  [--deny-on-incident <kind> ...]

annpack fleet policy sign <file> --key <secret-key-file> [--output <file>]

annpack fleet policy verify <file> [--prior <file>] [--now <ts> | --system-clock] [--json]

annpack fleet policy evaluate --local <file> --required <file> \
  [--now <ts> | --system-clock] [--json]
```

`verify` and `evaluate` follow the same exit-class and single-JSON-object
contract as every other command (`RELEASE-v1` §8).

| `error.kind` | Class |
|---|---|
| `unsupported_schema`, `malformed_input` | 2–3, as `RELEASE-v1` §8.3 |
| `unauthorized_role`, `verification_failed` | 5 |
| `rollback`, `expired`, `no_trusted_clock` | 6 |
| `fleet_policy_drifted`, `fleet_policy_unavailable` | 7 |

## 5. Non-goals

Fetching a required policy from a control plane. Enforcing
`allowed_publishers`/`allowed_scopes`/`deny_on_incident_kinds` against a
specific verification result — this document defines the object and its own
verification; wiring it against `verify --policy` and `release monitor`
output is later work.
