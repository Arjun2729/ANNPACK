# Adyar Run Attestation v1

Status: candidate specification. This protocol is occurrence evidence layered
above Adyar artifacts, build provenance, release state, and passage receipts.
It changes none of those objects.

## Security claim and non-goals

A verified run attestation establishes that a trusted application workload
signed a claim binding one execution, one exact receipt set, one release-state
snapshot, one query digest, model and policy identifiers, and one output digest.
It does not prove that the model used every passage, that the answer is correct,
that the workload was uncompromised, that no retrieval was omitted, or that the
claim was constructed synchronously with execution.

Publisher, release, revocation, builder, and workload authority are independent.
A key trusted in one domain MUST NOT acquire authority in another.

## Envelope and statement

The envelope is DSSE with `payloadType` `application/vnd.in-toto+json`. The
payload is an in-toto Statement v1 JSON object:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [{"name": "agent-output", "digest": {"sha256": "<hex>"}}],
  "predicateType": "https://annpack.dev/attestations/run/v1",
  "predicate": {
    "schema": "annpack-run-attestation-v1",
    "execution": {
      "run_id": "run-001",
      "trace_id": "trace-001",
      "workload_identity": "support-agent",
      "started_at": "2030-01-01T12:00:00Z",
      "completed_at": "2030-01-01T12:00:01Z"
    },
    "knowledge": {
      "run_bundle_digest": {"algorithm": "sha256", "value": "<hex>"},
      "receipts": [{
        "digest": {"algorithm": "sha256", "value": "<hex>"},
        "artifact_root": "<blake3 hex>",
        "passage_id": "<id>",
        "passage_hash": "<blake3 hex>"
      }],
      "receipt_count": 1,
      "no_passages_retrieved": false,
      "artifact_roots": ["<blake3 hex>"],
      "publisher": "example.test",
      "corpus": "support",
      "channel": "production",
      "channel_state_digest": {"algorithm": "blake3", "value": "<hex>"},
      "channel_state_sequence": 1,
      "observed_currency": "current",
      "trust_policy": "authorized_current"
    },
    "retrieval": {
      "query_digest": {"algorithm": "sha256", "value": "<hex>"},
      "retrieval_policy_revision": "retrieval-v1",
      "retrieval_mode": "lexical"
    },
    "application": {
      "application_identity": "support-agent",
      "application_version": "1.0.0",
      "model_identifier": "model-1",
      "model_provider": "example-provider",
      "prompt_policy_digest": {"algorithm": "sha256", "value": "<hex>"},
      "tool_policy_revision": "tools-v1",
      "deployment_identity": "production"
    }
  }
}
```

There MUST be exactly one subject named `agent-output`; its SHA-256 MUST be
calculated from the supplied output bytes. Optional execution, retrieval, and
application fields are omitted, never encoded as empty substitutes.

## Canonical bindings

Each receipt digest is SHA-256 over the compact JSON serialization produced by
the receipt data model. Bindings are sorted lexicographically by digest. Input
receipt order is non-semantic. Duplicate receipt digests are rejected, and no
more than 256 receipts are accepted. Each binding also records the root, passage
ID, and passage hash derived from that receipt.

The artifact-root set is the sorted unique set derived from the receipts. v1
creation permits at most one root. Empty retrieval is distinct from absence:
`receipts` is empty, `receipt_count` is zero, and
`no_passages_retrieved` is true. It requires an explicit creation option and
cannot establish publisher authority from receipts.

The run-bundle digest is SHA-256 over compact JSON containing the bundle schema,
run ID, optional metadata, query and answer fields, plus the sorted receipt
digest list. It is stable across receipt reordering. Query and prompt-policy
digests are SHA-256 over their exact UTF-8 or byte inputs. The channel-state
digest uses the BLAKE3 procedure in [RELEASE-v1](RELEASE-v1.md).

All schema structures reject unknown fields. Predicate extensions are permitted
only in the `extensions` object and every extension key MUST start with `x-`.
An unknown unprefixed extension, statement type, predicate type, schema, or
digest algorithm fails verification.

## Creation

Creation MUST independently verify every receipt, including its artifact-root
signature, verify the supplied publisher trust root and channel state, bind the
channel scope and digest, and evaluate the requested runtime trust policy. It
derives receipt, bundle, query, prompt, output, root, release, count, and currency
claims rather than accepting those digests from a caller.

Creation refuses malformed or unsigned receipts, duplicate receipts, conflicting
roots, invalid release evidence, scope disagreement, revoked or otherwise
policy-denied artifacts, contradictory run/model metadata, impossible execution
ordering, and empty retrieval unless explicitly allowed. Historical creation is
possible only under a policy that truthfully records the observed currency.

## Workload signing and trust

The local profile signs DSSE pre-authentication encoding with Ed25519. The DSSE
`keyid` is derived from the public key. Verification considers candidate keys
but trusts only a candidate explicitly configured as trusted whose external
identity equals `execution.workload_identity`.

Envelope authentication, workload identity, and signing-time evidence are
separate. An external Sigstore adapter may supply authenticated workload claims
without assigning GitHub build identity to an application. Its result is
accepted only when it names the exact in-toto payload SHA-256. Identity trust,
signer IDs, trusted signing time, and external anchoring remain separate fields.
The local profile establishes no trusted signing time.

## Verification

A verifier MUST report these stages independently: envelope signature, workload
identity, run identity, receipt-set binding, individual receipt verification,
artifact-root binding, publisher authority, channel-state binding, recorded
currency, runtime policy, query digest, model identity, prompt-policy digest,
output digest, execution time, and overall occurrence evidence. Status values
are `verified`, `carried`, `missing`, `mismatched`, `invalid`, `untrusted`,
`unknown`, and `not_evaluated`.

Overall occurrence evidence is true only when the supported predicate has a
valid trusted-workload envelope; run and trace expectations agree; the exact
receipt set and bundle digest agree; every receipt verifies; roots, publisher,
release state, currency, runtime policy, query, model, prompt policy, and output
agree; and time ordering is valid. Required output bytes that are absent fail as
`missing`. No overall boolean may replace the individual stage results.

The bound release snapshot records what the workload claims it evaluated. An
optional newer snapshot determines `currency_at_evaluation` and
`present_use_permitted`. Later supersession or revocation does not erase a valid
historical occurrence. It denies present use when the newer snapshot reports the
artifact as superseded or revoked.

## Time and occurrence strength

Execution start and completion are application claims protected by the workload
signature and reported as `carried`; completion before start is invalid. A local
Ed25519 envelope provides no cryptographically trusted time and reports signing
time as `unknown`. If a payload carries a signing-time claim it cannot precede
completion, but remains `carried` absent an external trusted-time source.
Channel validity and the verifier's trusted `now` remain release-verification
inputs and MUST NOT be inferred from execution time.

`workload_attested` means a trusted workload signed the complete claim.
`workload_attested_with_trusted_time` and `externally_anchored` are reserved for
adapters that actually establish those stronger properties. `unattested` and
`invalid` never claim occurrence. A workload signature alone is not proof of
synchronous construction.

## Privacy and retention

The predicate contains digests and operational identifiers, not plaintext query,
prompt, passage, or output. The separate run bundle still carries its query and
may carry application, model, answer, canonical URLs, document metadata, and
receipts; it must be protected according to application data policy. Receipt
format and disclosure are unchanged. Omitting optional trace, provider, mode, or
deployment identities reduces correlation but is unambiguous. Output bytes may
be withheld from a verifier, but a strict verification request then fails.

The DSSE payload limit is 16 MiB. Receipt decoding and proof limits remain those
of [EVIDENCE-v1](EVIDENCE-v1.md). Telemetry is only a locator and MUST NOT be the
sole evidence store or contain sensitive plaintext by default.

## Relationships and non-goals

- Core artifact verification establishes immutable container integrity.
- [PROVENANCE-v1](PROVENANCE-v1.md) authenticates how an artifact was built.
- [RELEASE-v1](RELEASE-v1.md) authenticates publisher channel state and currency.
- [EVIDENCE-v1](EVIDENCE-v1.md) authenticates retrieved passages.
- This protocol binds those inputs to a workload-signed occurrence and output.

Run attestation is a separate JSON object and never changes `.adyar` bytes or
artifact roots. Public transparency, witness monitoring, selective-disclosure
receipt redesign, model correctness, and a hosted control plane are out of scope.
