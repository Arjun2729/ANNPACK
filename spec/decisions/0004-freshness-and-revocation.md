# ADR-0004: Freshness and revocation are a publisher statement, not a pack field

Status: accepted (design), 2026-07-27. Implementation deferred; the model is
fixed now so that reviewers have a specified design to critique rather than an
undocumented gap.

## Context

Immutability creates a gap the format currently cannot answer.

A valid old pack stays valid forever. `SECURITY.md` says so. Evidence receipts
sharpen the problem: a receipt for a superseded artifact verifies **correctly**,
offline, for all time. Nothing in a receipt says "this is still the current
knowledge," and nothing can, because the receipt is a statement about the past.

This is correct behaviour and a genuine gap. Offline and air-gapped deployments
require a means of determining whether an artifact has been superseded or
revoked that does not depend on a hosted service being reachable.

Three distinct questions get conflated:

1. **Staleness** — a newer version exists.
2. **Expiry** — the publisher declared a shelf life.
3. **Revocation** — the publisher says *do not use this artifact*, usually
   because it was wrong or leaked.

Only the third is a security event. The first two are policy.

## Decision

**Freshness and revocation are assertions about an artifact, made by a
publisher, distributed separately from the artifact. They are never fields
inside the pack they describe.**

A pack cannot revoke itself: an attacker serving an old artifact simply serves
the old, unrevoked bytes. Any in-pack field is unauthenticated with respect to
its own supersession. So the artifact stays immutable and silent about its own
currency, and currency lives in a separate signed statement.

### The publisher statement

A small signed document, fetched from the publisher's origin, listing current
and revoked roots for a corpus:

```json
{
  "schema": "annpack-freshness-v1",
  "corpus": "vendor-docs",
  "issued_at": "2026-07-27T00:00:00Z",
  "valid_until": "2026-08-03T00:00:00Z",
  "current": {"version": "2.1.0", "root": "<64 hex>"},
  "superseded": [{"root": "<64 hex>", "superseded_at": "…", "by": "<64 hex>"}],
  "revoked": [{"root": "<64 hex>", "revoked_at": "…", "reason": "incorrect-content"}],
  "signature": { "algorithm": "Ed25519", "public_key": "…", "signature": "…" }
}
```

Signed by the same publisher key that signs packs, so trusting a publisher is
one decision, not two. `valid_until` bounds how long a cached statement may be
believed, which is what makes offline use tractable: an air-gapped deployment
ships the statement alongside the pack and knows exactly when its assurance
lapses.

Served at `/.well-known/annpack-freshness/<corpus>.json`, discoverable from the
existing catalog.

### Three verdicts, kept separate

A consumer combines a verified receipt with a freshness statement to get:

| Verdict | Meaning |
|---|---|
| `current` | The root is the current one in a statement that has not expired. |
| `superseded` | Verified and authentic; a newer root exists. **Not** a security failure. |
| `revoked` | The publisher withdrew this artifact. Treat as a security event. |
| `unknown` | No statement, or the statement expired. Report honestly; do not infer currency. |

`unknown` must never be reported as `current`. That distinction is the whole
value of the mechanism.

### What does not change

- Receipts stay verifiable with no network. Freshness is a **separate, optional**
  check that requires one, and its absence degrades to `unknown` rather than to
  failure.
- The artifact root, the logical content root, and signature semantics are
  untouched.
- `policy.expires_at` remains a declarative hint and is not a revocation channel.

## Consequences

- Rollback resistance moves from "documented limitation" to "documented
  mechanism with a stated trust boundary." It remains unimplemented.
- Air-gapped deployment becomes concrete: ship pack plus statement, and the
  deployment knows when its assurance expires.
- Reading and verifying a statement requires nothing beyond the format and an
  Ed25519 verifier. Operating a highly available statement endpoint, key
  rotation, and a transparency log are deployment concerns outside this
  specification.
- New failure mode to handle deliberately: a publisher who stops issuing
  statements silently downgrades every consumer to `unknown`. That is the correct
  behaviour and must be loud in tooling.

## Alternatives rejected

**A revocation field inside the pack.** Cannot work: revoking artifact *X*
requires a statement not carried by *X*, or an attacker just serves the
un-revoked copy.

**Short-lived signatures / expiring packs.** Forces re-signing on a timer,
breaks air-gapped deployment, and conflates authenticity with currency. A
signature answers "did the publisher make this?", not "is it current?".

**Transparency log as the primary mechanism.** Excellent for detecting publisher
equivocation, and worth building later, but it requires network access and a
witness ecosystem. It cannot be the base layer for offline and air-gapped use.
The signed statement is the base; a log strengthens it.

**Leaving the gap undocumented.** The question comes up as soon as anyone
reasons about an immutable artifact's lifetime. Recording the intended model now
makes the trust boundary reviewable and distinguishes a deliberate boundary from
an oversight.
