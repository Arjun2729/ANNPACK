# OKF team outreach — draft

Framing: technical-peer / interop feedback, **not** a product pitch. You built
something on their format that reproduces their own published bundles. Do not
overclaim: no "independently security-reviewed", no headline retrieval-quality
percentages, no "adopted standard". Keep the humility line.

Two artifacts back every version below:

1. **Self-verifying reproduction** (they run it, trust nothing):
   ```bash
   git clone https://github.com/Arjun2729/annpackv2 && cd annpackv2
   cargo build --release
   ./launch/google-okf/reproduce.sh
   ```
   Clones `GoogleCloudPlatform/knowledge-catalog` at `d44368c`, compiles the three
   bundles present at that revision, and verifies all three artifact roots against
   `launch/google-okf/expected-roots.json`.

2. **Live, zero-server browser demo** (clickable; range-fetches + verifies in-page):
   `https://arjun2729.github.io/annpackv2/?pack=./packs/google-okf-ga4.annpack&root=b6d50106c32ef2e9e944b98e589e81378948163d134ed53b26eeb5262327960b&q=what%20does%20the%20user_properties%20field%20contain`

   ✅ **Verified live 2026-07-28**: drove the real WASM `ANNPackBrowser` client against
   this exact HTTPS URL and query (not just an HTTP 200 check) — real range-fetch from
   GitHub Pages, `root_hash` matches `b6d50106…960b` exactly, top result returns the
   correct `user_properties` passage with valid `annpack-evidence-v1` evidence
   (`evidence.pack_root === header.rootHash`), `core_conformant: true`.

---

## Version A — email to maintainers (short)

> **Subject:** OKF v0.2 — two bugs in our implementation, and a question about
> the trust model
>
> Hi,
>
> I maintain ANNPack, an open format that compiles an OKF bundle into a signed,
> content-addressed artifact you can range-query straight from a CDN with no
> server. I've been reproducing your published bundles as an interop exercise.
>
> First, a bug report against myself — with one small suggestion for the spec
> that falls out of it. Building your `acme_retail` exemplar (at `3fcbb9f`), I
> found two defects in *my* OKF implementation:
>
> - I was rejecting any `log.md` with frontmatter. That was my bug — but I misread
>   the spec to get there: v0.1's "Index files contain no frontmatter" reads, at a
>   glance, as a corpus-wide rule, when it's about `index.md`; v0.2 §9 only
>   constrains the body. Your own exemplar's `log.md` carries `type: Log`, so it
>   tripped me. If §9 stated explicitly that the no-frontmatter rule is
>   `index.md`-only, the next implementer wouldn't repeat my mistake.
> - I was defaulting `okf_version` to `0.1` when absent. §12 makes it optional, so
>   absent means undeclared — I was mislabelling undeclared content as v0.1.
>
> Both fixed. `acme_retail` at `3fcbb9f` now compiles cleanly to 17 documents /
> 47 passages with `generated`, `verified`, `status`, `stale_after` and `tags`
> preserved intact.
>
> Now the one question I'd actually value your view on — I'll keep it to the
> sharpest. §2 says OKF fixes the interface, not the packaging; I'm building the
> packaging half (a Merkle inclusion proof plus a signature over an immutable
> root, so a third party can verify *the exact retrieved text* offline, with no
> server and no trust in me), and that layering seems to match your non-goals. But
> here's where I think I disagree with the spec, which is the most useful thing to
> talk about: `stale_after` lives inside the document, and I don't think freshness
> can be enforced from inside the artifact it describes — an adversary serving a
> stale copy just serves the old bytes, and a receipt for a superseded artifact
> verifies correctly forever. I put revocation in a separately distributed,
> publisher-signed statement instead. I suspect both are needed — yours is
> producer intent, mine is adversarially enforceable — but I'd rather be told I'm
> wrong. Where do you think freshness should live?
>
> (A smaller seam if you have appetite: `verified.by` is an actor,
> `human:kliu@acme`, not a key — is binding actors to signing keys in scope for
> OKF, deliberately out, or a packaging-layer concern? No need to answer both.)
>
> If it's worth five minutes: `./launch/google-okf/reproduce.sh` clones your repo
> at a pinned commit and verifies three artifact roots. A discrepancy would be
> more useful to me than a match.
>
> Apache-2.0, verifier is open and stays open. Not asking for endorsement — this
> is an independent project and I'm careful to say so.
>
> — Arjun

## Version B — GitHub issue / discussion on `knowledge-catalog` (public, warm)

**Title:** Interop: reproducing the three OKF 0.1 bundles as verifiable content-addressed artifacts

Hi — I built an open, deterministic packaging + retrieval format (ANNPack) that
consumes OKF 0.1, and wanted to share an interop result and ask for feedback.

Compiling the three bundles under `okf/bundles/` at `d44368c` produces ANNPack
artifacts whose roots are reproducible by the same builder across environments. (The artifact root commits to compression and layout, so it is an identity for one artifact, not a cross-implementation identity; the manifest also carries a layout-independent `passage_merkle_root` for that purpose.) A two-command
reproduction clones this repo at the pinned commit and verifies all three roots:

```bash
cargo build --release
./launch/google-okf/reproduce.sh
```

Expected roots (also checked in as `expected-roots.json`; these are the
**v0.4.0-rc2** roots — rc2 changed them from rc1, see `expected-roots.json` note):

| bundle | root |
|---|---|
| ga4 | `b6d50106c32ef2e9e944b98e589e81378948163d134ed53b26eeb5262327960b` |
| crypto-bitcoin | `6b0f7d6c28a807db3a715bdc449add64482063c631ccc9aa563cbe69c82e2f03` |
| stackoverflow | `3e81efeac44cfc743a6754750ef37c12e161dda827f1f0a929d41da5c545b2fe` |

What I'd love feedback on:
- Does the OKF 0.1 → verifiable-artifact mapping preserve what matters to you?
- Anything an interop layer should expose that I'm dropping?

Independent experiment; not implying any endorsement. Repro + a live browser demo
in the links.

---

## Do NOT include (until the gates close)

- Any retrieval quality percentage (Gate 4 human-label independence unconfirmed).
- "Independently security-reviewed" (Gate 2 external review outstanding).
- "Standard" / "adopted" language (needs external uptake).

---

## OKF v0.2 — read this before sending anything

**Do not send the older draft unchanged.** OKF v0.2 landed and it changes the
conversation, mostly in our favour.

v0.2 makes provenance, trust, lifecycle and attestation first-class
(`sources`, `generated`, `verified`, trust tiers, `status`, `stale_after`, and
Attested Computations in §10). Its non-goals are the important part for us:

> - Prescribing storage, serving, or query infrastructure.
> - Specifying a packaging or invocation standard for the code an executor or
>   attester points at. **OKF fixes the interface, not the packaging.**

That is close to an explicit invitation. OKF standardises *declared* trust
metadata and deliberately declines to specify packaging, serving, or runtime.
ANNPack is packaging, serving, and cryptographic enforcement. The layering is
now stated in their own document rather than asserted by us.

The distinction to hold precisely: **OKF declares, ANNPack proves.**
`verified: { by: human:kliu@acme, at: … }` is a producer's claim. Compiling the
bundle into a signed artifact whose passages carry Merkle inclusion proofs makes
the *retrieved text* independently checkable. It does not make the claim true —
and we must not imply that it does.

### What we found testing against v0.2 (lead with this)

Two defects, both **ours**, found by building their v0.2 exemplar bundle
`acme_retail`:

1. We rejected a conformant bundle. Our validator refused any `log.md` carrying
   frontmatter. No version of the spec says that — v0.1's "Index files contain no
   frontmatter" governs `index.md`, and v0.2 §9 constrains only the body. Their
   `acme_retail/log.md` carries `type: Log`, so their own exemplar was
   unbuildable by us.
2. We invented a version. `okf_version` is optional (§12); absent means
   *undeclared*, not `0.1`. We were stamping `source.version: "0.1"` on bundles
   that declare nothing — mislabelling v0.2 content.

Both fixed in v0.4.0-rc2, with regression tests. Reporting this first is the
strongest possible opener: it shows the spec is being implemented carefully
enough to find our own misreadings, and it costs them nothing to verify.

Confirmed working after the fix: `acme_retail` compiles to 17 documents /
47 passages, and `generated`, `verified`, `status`, `stale_after` and `tags` are
all preserved losslessly in document metadata.

---

## The ask, stated precisely

Not endorsement, not partnership, not a quote. Four technical questions a
maintainer can answer from the artifacts alone.

1. **Is the layering right?** Is "OKF is the authoring and interchange
   interface; a compiled, signed, range-queryable artifact is one packaging of
   it" consistent with how you intend §2's non-goals to be read? We are not
   proposing any change to OKF.

2. **How should cryptographic enforcement relate to the declared trust model?**
   v0.2 answers "how much should I trust this" with producer-declared
   frontmatter. We answer "is this the exact text the publisher signed" with a
   Merkle inclusion proof and an Ed25519 signature over an immutable root. Those
   compose, but the seam has an open question: `verified.by` is an *actor*
   (`human:kliu@acme`), not a key. Is binding actor identifiers to signing keys
   in scope for OKF, deliberately out of scope, or something you would expect a
   packaging layer to define?

3. **Freshness — where should it live?** This is our sharpest disagreement and
   the most useful thing to discuss. `stale_after` and `status` live *inside* the
   document. We concluded (ADR-0004) that freshness cannot be enforced from
   inside the artifact it describes: an adversary serving a stale copy simply
   serves the old, un-revoked bytes, and a cryptographic receipt for a superseded
   artifact verifies correctly forever. Our model puts revocation in a separately
   distributed, publisher-signed statement with a bounded validity window.
   We think both are needed — yours expresses producer intent, ours is
   adversarially enforceable — but we would rather be told we are wrong now.

4. **Would you run the reproduction?** `./launch/google-okf/reproduce.sh`
   verifies three artifact roots against a pinned commit. A discrepancy is a more
   useful result to us than a match, and either way it is a five-minute check.

### What counts as success

A public issue reply, a merged interop fixture, a reproduced root, or a written
technical objection. All four are real validation. "Google likes it" is not a
goal and should not be pursued.

### What we must never imply

Google publishes the OKF **source** bundles and the specification. We produce
ANNPack artifacts and publish the expected **ANNPack** roots of our own
reproduction. Those roots are ours. Google neither publishes ANNPack artifacts
nor endorses this project. The demo output field is named
`root_matches_expected_annpack_reproduction` precisely so a screenshot cannot be
misread as a Google-published root.

### Send only after

- [x] the v0.2 findings are merged and tagged — **DONE**: v0.4.0-rc2 merged to main
      (PR #6, commit `07e723c`) and tagged `v0.4.0-rc2`, 2026-07-28.
- [x] `reproduce.sh` passes on a fresh clone — **DONE 2026-07-28**: re-cloned
      `knowledge-catalog` at `d44368c`, all three roots verified
      (`b6d50106…`/`6b0f7d6c…`/`3e81efea…`), matching `expected-roots.json`.
- [x] the `acme_retail` claim is real — **DONE 2026-07-28**: built `acme_retail` at
      HEAD `3fcbb9f` → exactly 17 documents / 47 passages, lifecycle metadata intact.
      NOTE: `3fcbb9f` is HEAD (a moving ref); the email now pins it, but re-confirm
      the count still holds at send time in case upstream advanced the exemplar.
- [x] no retrieval-quality numbers appear anywhere in the message — confirmed.
- [x] the live CDN/Pages demo — **DONE 2026-07-28**: GitHub Pages already deployed
      (`https://arjun2729.github.io/annpackv2/`, serving `/docs` on `main`, status
      "built"). Filled the real origin into the demo link and verified it end-to-end
      with the real WASM client (see above) — not just an HTTP status check.
- [x] regenerated the Version B root table (done in-file 2026-07-28) — re-verify if
      you rebuild under any tag past rc2, since OKF roots move with root-scheme changes.

**Every item is now checked. The email is send-ready as of 2026-07-28.**
