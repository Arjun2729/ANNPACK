# Adyar Release v1

Status: implemented draft. Requires Adyar Core v1.0-draft.

Defines publisher trust roots, authorized key roles, and channel-state
statements.

This layer does not change the artifact format, container bytes, or content
roots. Trust roots and channel state are separate signed documents distributed
alongside artifacts.

The decision and its reasoning are in
[ADR-0004](decisions/0004-freshness-and-revocation.md).

## 1. Two questions, two mechanisms

| Question | Answered by | Properties |
|---|---|---|
| Did these bytes exist in this artifact at this revision? | artifact + [receipt](EVIDENCE-v1.md) | offline, permanent, unaffected by supersession |
| Is this artifact the one the publisher stands behind now? | channel-state statement | time-indexed, requires a clock, requires a release authority |

An artifact never asserts its own currency. An attacker can serve unchanged,
superseded bytes.

## 2. Trust root

Names the keys that may act for a publisher and the role each holds.

```json
{
  "schema": "annpack-trust-root-v1",
  "publisher": "example.com",
  "version": 7,
  "issued_at": "2026-08-06T00:00:00Z",
  "valid_until": "2027-08-06T00:00:00Z",
  "roles": {
    "root":                 { "threshold": 1, "keys": ["<key-id>"] },
    "artifact":             { "threshold": 1, "keys": ["<key-id>"] },
    "release_state":        { "threshold": 1, "keys": ["<key-id>"] },
    "emergency_revocation": { "threshold": 1, "keys": ["<key-id>"] }
  },
  "keys": {
    "<key-id>": { "algorithm": "Ed25519", "public_key": "<64 hex>" }
  },
  "signatures": [ { "key_id": "<key-id>", "signature": "<128 hex>" } ]
}
```

### 2.1 Fields

| Field | Rule |
|---|---|
| `schema` | MUST be `annpack-trust-root-v1`. Any other value is rejected, not partially interpreted. |
| `publisher` | Non-empty. A rotation MUST NOT change it. |
| `version` | Integer ≥ 1. Strictly increases across rotations. |
| `issued_at`, `valid_until` | `YYYY-MM-DDTHH:MM:SSZ`, UTC only. `valid_until` MUST be later than `issued_at`. |
| `roles` | MUST define all four of `root`, `artifact`, `release_state`, `emergency_revocation`. A missing role is rejected rather than treated as an empty one; the two are indistinguishable to a reader and only one is intended. |
| `roles[*].threshold` | ≥ 1 and ≤ the number of distinct keys listed for that role. A higher threshold can never be met, making the role permanently unusable. |
| `roles[*].keys` | Key ids, no duplicates, each present in `keys`. |
| `keys[*].algorithm` | MUST be `Ed25519`. |
| `keys[*].public_key` | 64 hex characters. The map key MUST equal `BLAKE3(public_key_bytes)` in hex, or a root could file an attacker's key under a trusted key's identifier. |
| `signatures` | Zero or more `{key_id, signature}`. Order is not significant. |

Timestamps are parsed strictly: exactly 20 characters, UTC with a `Z` suffix.
Offsets, fractional seconds and date-only values are rejected.

### 2.2 Signature

Domain separator: `ANNPACK3-TRUST-ROOT\0`

The signed message is that separator followed by the canonical serialization of
the document **excluding `signatures`**. Including signatures would invalidate
existing signatures when another is added.

Canonical serialization is the JSON encoding of `schema`, `publisher`, `version`,
`issued_at`, `valid_until`, `roles`, `keys` in that order, with `roles` and `keys`
ordered by key.

A threshold counts **distinct authorised key ids that produced a valid
signature**, never signature entries. One key signing twice MUST NOT satisfy a
threshold of two.

### 2.3 Who may sign

| Root | Requires |
|---|---|
| First root a consumer sees | a threshold of its **own** `root` role. Nothing further can be checked; this is first contact and MUST be reported as such. |
| Current root | a threshold of its own `root` role. |
| Successor root | a threshold of its own `root` role **and** a threshold of the **prior** root's `root` role, and `version` strictly greater. |

Prior-only authorization permits a compromised old key to install uncontrolled
keys. Self-only authorization permits an untrusted party to mint a root.

### 2.4 Rotation and rollback

A successor is rejected when its version is lower than or equal to the trusted
version. Equal is rejected as well as lower: two differently keyed roots at one
version is publisher equivocation at the trust layer.

### 2.5 Limits

| Limit | Value |
|---|---|
| Keys per root | 128 |
| Roles per root | 32 |
| Signatures per root | 128 |
| Keys per role | 128 |
| File size read by the reference CLI | 1 MiB |

## 3. Channel-state statement

Names the artifact a publisher currently stands behind for one scope.

```json
{
  "schema": "annpack-channel-state-v1",
  "publisher": "example.com",
  "corpus": "support-manual",
  "channel": "production",
  "sequence": 184,
  "issued_at": "2026-08-06T00:00:00Z",
  "valid_until": "2026-08-06T01:00:00Z",
  "current": { "version": "4.3.0", "artifact_root": "<64 hex>" },
  "superseded": [ { "artifact_root": "<64 hex>", "by": "<64 hex>", "at": "…" } ],
  "revoked":    [ { "artifact_root": "<64 hex>", "at": "…", "reason": "…" } ],
  "signatures": [ { "key_id": "<key-id>", "signature": "<128 hex>" } ]
}
```

### 3.1 Fields

| Field | Rule |
|---|---|
| `schema` | MUST be `annpack-channel-state-v1`. |
| `publisher`, `corpus`, `channel` | All non-empty. Together they are the scope. |
| `sequence` | Unsigned integer. Orders statements within one scope. |
| `issued_at`, `valid_until` | As §2.1. Validity SHOULD be measured in hours; see ADR-0004. |
| `current.artifact_root` | 64 hex characters. |
| `superseded[*]` | `artifact_root` and `by` are both 64 hex characters. |
| `revoked[*]` | `artifact_root` is 64 hex characters; `reason` is free text and is not authenticated as a claim about anything. |
| `signatures` | As §2.1. |

A statement that lists its own `current.artifact_root` among `revoked` is
malformed.

### 3.2 Signature

Domain separator: `ANNPACK3-CHANNEL-STATE\0`

Signed message and exclusion rule as §2.2, over `schema`, `publisher`, `corpus`,
`channel`, `sequence`, `issued_at`, `valid_until`, `current`, `superseded`,
`revoked`.

**Statement digest** is `BLAKE3` of that same serialization, excluding
signatures. Two statements agree if and only if their digests agree. It is the
basis of equivocation detection, so any field that can differ between two
statements MUST be inside it.

### 3.3 Role authorization

One document carries both promotion and revocation claims. Authority is
determined once, from the roles that met their thresholds, and then applied
per-claim:

| Signed by | `authority` | `current` honoured | `revoked` honoured | `superseded` honoured |
|---|---|---|---|---|
| a threshold of `release_state` | `full` | yes | yes | yes |
| only a threshold of `emergency_revocation` | `revocation_only` | **no** | yes | no |
| neither threshold met | `none` | no | no | no |

Unauthorised fields are **excluded from the authenticated decision**, not
rejected and not deleted. A `revocation_only` statement remains a valid document
and verifies; its `current` and `superseded` entries simply do not contribute,
so an artifact they name resolves to `unknown` rather than `current` or
`superseded`.

Honouring both roles equally would let a compromised revocation key declare an
artifact current.

### 3.4 Limits

| Limit | Value |
|---|---|
| `superseded` + `revoked` entries | 4096 |
| Signatures | 128 |
| File size read by the reference CLI | 4 MiB |

## 4. Verification procedure

Stages run in this order. A stage that cannot be evaluated reports so; it never
substitutes a default.

1. **Bound and parse.** Reject files above the size limit before reading them.
2. **Schema.** Reject an unrecognised `schema` outright.
3. **Structure.** Field shapes, hex lengths, thresholds, role completeness,
   key-id/public-key agreement.
4. **Trust root.** Verify per §2. A statement authorised by a trust root that did
   not itself verify MUST NOT verify.
5. **Signer role.** Determine `authority` per §3.3.
6. **Statement signatures.** Distinct authorised signers against thresholds.
7. **Scope.** Compare `publisher`, `corpus`, `channel` against the expected
   scope. **The expected scope MUST be established outside the statement** —
   publisher from the trusted root, corpus and channel from consumer
   configuration. A verifier MUST NOT default any of them from the document under
   verification.
8. **Time.** Evaluate `issued_at ≤ now < valid_until` against a caller-supplied
   clock. With no clock supplied, validity is `unknown` and the statement does
   not verify.
9. **Retained state.** Apply §5. **Skipped entirely when scope did not match**:
   retained state for the expected scope MUST NOT be read, created or modified on
   behalf of a statement scoped elsewhere.
10. **Root status.** Resolve `current` / `superseded` / `revoked` / `unknown` per
    §6, only from a statement that verified.
11. **Persist.** Update retained state atomically, and only after every preceding
    stage passed and the sequence advanced.
12. **Return** structured claims and, where a policy was requested, a decision.

## 5. Sequence rules

Retained state is keyed on `publisher + corpus + channel` **as expected**, never
as declared by the statement.

```json
{
  "publisher": "example.com", "corpus": "support-manual", "channel": "production",
  "highest_sequence": 184,
  "statement_digest": "<64 hex>",
  "artifact_root": "<64 hex>",
  "accepted_at": "2026-08-06T00:15:00Z"
}
```

| Condition | Verdict | Accepted |
|---|---|---|
| no retained state | `first_contact` | yes, with no rollback resistance |
| scope mismatch, or state belongs to another scope | `not_evaluated` | no |
| `sequence` < `highest_sequence` | `rollback` | no |
| `sequence` = `highest_sequence`, digest equal | `idempotent` | yes |
| `sequence` = `highest_sequence`, digest differs | `equivocation` | no |
| `sequence` > `highest_sequence` | `advanced` | yes, after full verification |

`not_evaluated` means no comparison occurred. `first_contact` means a comparison
found no retained state.

Persistence uses a temporary file, sync, and atomic rename. A verifier that
cannot persist MUST NOT report rollback resistance for the decision it just
made. A truncated state file reads as no state on the next start and downgrades
the next decision to `first_contact`.

## 6. Verdicts

Four independent facts, never merged into one field.

**`artifact_integrity`** — `valid` | `invalid`. Container structure, section
bounds, section hashes, content root. Unaffected by anything in this
specification. **Revocation MUST NOT change it.** A revoked artifact that is
genuinely authentic reports `valid`.

**`publisher_authority`** — `authorized` | `unauthorized` | `unknown`.
`unauthorized` is a negative answer; `unknown` is the absence of one.

**`currency`** — `current` | `superseded` | `revoked` | `unknown`, from a
verified statement only:

| Condition | Verdict |
|---|---|
| root appears in `revoked` | `revoked` |
| root appears in `superseded` | `superseded` |
| root equals `current.artifact_root` **and** authority is `full` | `current` |
| anything else, including an unverified statement | `unknown` |

Revocation is checked first and honoured under either authority. An unauthorized
promotion claim does not suppress an authorized revocation in the same
statement.

`unknown` MUST NOT be reported as `current`. A statement that does not mention an
artifact says nothing about it.

**`policy_decision`** — see §7.

## 7. Policies

| Policy | Requires |
|---|---|
| `integrity-only` | `artifact_integrity = valid` |
| `authorized-publisher` | the above, and `publisher_authority = authorized` |
| `authorized-current` | the above, and a verified statement with `currency = current` |
| `authorized-current-witnessed` | the above, and log inclusion with a witness quorum |

**Global precedence: a known revocation denies under every policy**, including
`integrity-only`. The integrity fact remains `valid`; the denial is a status
decision.

**`integrity-only` does not require a statement and does not claim currency.**
With no statement it permits, reports `currency: unknown`, and states in its
assumptions that nothing in the decision says the artifact is current. The
integrity fact remains independently observable in every case.

**No policy degrades into another.** A stronger policy whose inputs are absent
denies.

**`authorized-current-witnessed` denies unless a transparency proof is supplied
and verifies.** A policy whose requirement is unmet MUST NOT behave like the one
below it. Without `--transparency-proof` and `--transparency-policy`, the policy
denies.

### 7.1 Transparency evidence

A verified channel-state statement (§4) authenticates its signer but does not
establish public visibility or uniqueness. A publisher can sign two statements
at the same sequence and show them to different verifiers. Transparency evidence
establishes that signing was publicly checkable; it does not strengthen signer
authority or statement contents.

Adyar integrates with an external
[Sigsum](https://www.sigsum.org/) transparency log
(the C2SP tlog-tiles/tlog-checkpoint family) rather than operating one. A Sigsum
proof shows that the statement digest (§3) was logged in a public, append-only
Merkle tree at a tree state cosigned by the configured witness quorum, using a
release-state role key. §5 uses the same digest for sequence verdicts. The
publisher submits the digest to a Sigsum log outside Adyar, as with GitHub OIDC
certificate issuance for build provenance (`PROVENANCE-v1` §5.2).

**Trust configuration is operator-supplied and never fetched.** A Sigsum
policy file (the `sigsum-go` syntax: `log`/`witness`/`group`/`quorum`
lines) names which log and witness keys are trusted and what quorum is
required. Updating it is a separate operational act, as is replacing a GitHub
Sigstore trusted-root snapshot (`PROVENANCE-v1` §5.3).

**What a verified proof does not establish.** A witnessed proof for an old
statement remains valid but does not establish that the statement is current.
Currency (§6) is evaluated independently, and `authorized-current-witnessed`
requires both.
Per-statement proof verification does not detect conflicts among independently
observed entries. **Witnesses do not replace a trusted clock or durable
monotonic state**; §5 still applies.

`trusted_signers` for the Sigsum leaf-signature check are release-state role keys
from the caller's trust root. This check is independent of the statement's
Adyar signature, even when both use the same physical key. A leaf signature
from any other key does not establish statement authority.

## 8. CLI contract

### 8.1 Commands

| Command | Success predicate |
|---|---|
| `adyar trust init` | wrote an unsigned trust root |
| `adyar trust sign` | added a signature |
| `adyar trust verify` | the root verified |
| `adyar release statement` | wrote an unsigned statement |
| `adyar release sign` | added a signature |
| `adyar release verify` | the statement **authenticated** for the expected scope |
| `adyar verify --policy` | the artifact **may be used** under that policy |
| `adyar release observe` | appended one observation to a monitoring history |
| `adyar release monitor` | the observed history contains **no incident** (§10) |

`release verify` authenticates a statement. `verify --policy` decides whether an
artifact may be used. A statement that revokes another artifact can authenticate
successfully; the revoked artifact is denied.

`release verify` requires `--expect-corpus` and `--expect-channel`. The publisher
defaults to the trusted root's and may be overridden with `--expect-publisher`.
`verify --policy` requires the same two whenever `--channel-state` is supplied.

`verify --policy authorized-current-witnessed` additionally accepts
`--transparency-proof <file>` (a Sigsum proof bundle) and
`--transparency-policy <file>` (a Sigsum trust-policy file); each requires the
other, `--channel-state`, and `--trust-root`. Requires the reference CLI's
`transparency-log` build feature. A binary built without it fails usage
validation if either flag is given.

Time is never defaulted. `--now <ts>` states a time; `--system-clock` asserts the
local clock is trustworthy; supplying neither yields `unknown` validity, which
does not verify.

### 8.2 Exit classes

Broad and stable. The precise reason travels in `error.kind`, so the numeric
table does not become a brittle public API.

| Code | Meaning |
|---|---|
| 0 | the requested verification or decision succeeded |
| 2 | invalid usage or missing required configuration |
| 3 | input unavailable or malformed |
| 4 | operational or persistence failure |
| 5 | cryptographic, authority or scope verification failure |
| 6 | temporal or monotonic-state safety failure |
| 7 | status or policy denial |

### 8.3 Error kinds

| Kind | Class |
|---|---|
| `invalid_usage` | 2 |
| `malformed_input`, `input_unavailable`, `unsupported_schema` | 3 |
| `io_failure`, `state_persistence_failed` | 4 |
| `invalid_signature`, `unauthorized_role`, `scope_mismatch`, `trust_root_unavailable`, `integrity_failed` | 5 |
| `expired`, `no_trusted_clock`, `rollback`, `equivocation` | 6 |
| `revoked`, `superseded`, `currency_unknown`, `unmet_policy_requirement`, `monitor_incident` | 7 |

### 8.4 Structured output

In `--json` mode **exactly one** JSON object reaches stdout on every path,
including malformed input and missing files.

Success emits the command's report. Failure emits an envelope:

```json
{
  "ok": false,
  "permitted": false,
  "stage": "scope",
  "error": { "kind": "scope_mismatch", "message": "…" },
  "details": { }
}
```

`details` carries the full report when one was produced before the failure.
Emitting the report and then an envelope would produce two objects and is not
parseable as one.

Human-readable diagnostics go to stderr in non-JSON mode. A JSON caller MUST NOT
need to read stderr.

## 9. Non-goals

Operating a transparency log or witness network. Identity infrastructure beyond
key roles. A general update protocol. Passage-level access control. Proving that
no newer statement exists — no offline mechanism can, and none is claimed here.

## 10. Cross-observation monitoring

§7.1 verifies one proof against one statement. `adyar release monitor`
compares observations over time.

**It does not fetch.** `adyar release observe` appends an obtained statement
and observation timestamp to a JSON Lines history. The operator controls the
sources, frequency, and diversity of observations. A monitor that sees only one
side of an equivocation reports no incident.

`adyar release monitor` reads the accumulated history, groups observations
by publisher/corpus/channel, and reports six conditions:

| Condition | Meaning |
|---|---|
| `equivocation` | Same sequence, different statement digest — the publisher signed two conflicting statements. |
| `conflict` | More than one artifact root is never explicitly superseded by anything at a higher sequence. |
| `authority_violation` | A statement's signatures met no authorised role's threshold, yet it was observed as if real. |
| `sequence_gap` | A gap between consecutively observed sequence numbers — most often an incomplete view, not an attack. |
| `stale_local_state` | The history contains an authorised, higher-sequence statement than supplied retained state reflects. |
| `revoked_root_advertised` | A statement revoked a root; a later-or-equal-sequence statement still advertises that root as current. |

A logged and witnessed statement (§7.1) can conflict with another observed
statement. Monitoring compares supplied observations; transparency verifies
public visibility for one statement.

A monitor reports only on supplied observations. `sequence_gap` signals an
incomplete view and is not itself evidence of publisher misbehaviour.
