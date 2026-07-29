# Google OKF → ANNPack interoperability fixture

This fixture compiles three public OKF bundles from Google's
`knowledge-catalog` repository into deterministic, content-addressed ANNPack
artifacts and verifies the expected roots. It is **not** an assertion that Google
publishes, endorses, or has reviewed ANNPack. Google publishes the OKF source and
specification; the ANNPack artifacts and expected ANNPack roots are produced by
this project.

## Pinned source

| | |
|---|---|
| Repository | `github.com/GoogleCloudPlatform/knowledge-catalog` |
| Revision | `d44368c` (pinned by [`reproduce.sh`](reproduce.sh)) |
| Bundles | `okf/bundles/{ga4, crypto-bitcoin, stackoverflow}` |
| Input | OKF |
| Source license | Apache-2.0 |
| Reference compiler | `annpack-reference/0.4.0-rc4` |

## Reproduce

```bash
cargo build --release
./launch/google-okf/reproduce.sh
```

The script clones the pinned revision, compiles all three bundles, and compares
the artifact roots with [`expected-roots.json`](expected-roots.json). Generated
packs and build reports are written under `target/google-okf-reproduction/`. Any
root mismatch fails the run.

An **artifact root** commits to the exact section directory, compression output,
and layout produced by this builder. It is not a cross-implementation identity.
The manifest's `passage_merkle_root` is the layout-independent commitment used
to compare the authenticated passage records produced by independent builders.

## Expected roots

Rc4 changes receipt verification only. The container, passage records, and roots
remain the rc2/rc3 values; the determinism matrix and reproduction script must
confirm these on the exact rc4 release commit.

| bundle | artifact root |
|---|---|
| ga4 | `b6d50106c32ef2e9e944b98e589e81378948163d134ed53b26eeb5262327960b` |
| crypto-bitcoin | `6b0f7d6c28a807db3a715bdc449add64482063c631ccc9aa563cbe69c82e2f03` |
| stackoverflow | `3e81efeac44cfc743a6754750ef37c12e161dda827f1f0a929d41da5c545b2fe` |

## Live browser fixture

The GA4 artifact can be opened through the zero-server range reader:

```text
https://arjun2729.github.io/annpackv2/?pack=./packs/google-okf-ga4.annpack&root=b6d50106c32ef2e9e944b98e589e81378948163d134ed53b26eeb5262327960b&q=what%20does%20the%20user_properties%20field%20contain
```

The browser fetches strict byte ranges, checks the expected artifact root, and
returns the `user_properties` passage with an evidence envelope bound to the
opened artifact. GitHub Pages currently serves the pack as
`application/octet-stream`; the client does not depend on the media type, but a
production origin should use `application/vnd.annpack.v3` with correct CORS,
immutable caching, stable ETags, and Range behavior.

A valid demo signature proves only that one key signed the root. The included
demo key is not an external identity binding and must not be described as Google
or publisher authentication.

## OKF v0.2 implementation result

Building Google's `acme_retail` v0.2 exemplar at `3fcbb9f` produced 17 documents
and 47 passages while preserving `generated`, `verified`, `status`,
`stale_after`, and `tags` in document metadata.

That exercise exposed two defects in ANNPack's OKF reader, both subsequently
fixed:

1. it incorrectly rejected frontmatter in `log.md`;
2. it treated an absent optional `okf_version` as `0.1` instead of undeclared.

The result is an implementation finding, not an endorsement or an independent
conformance certification.

## Open interoperability questions

1. **Authoring versus packaging.** Is treating OKF as the authoring/interchange
   interface and ANNPack as one compiled, signed, range-queryable packaging layer
   consistent with OKF's stated non-goals?
2. **Actor identity versus signing keys.** `verified.by` identifies an actor, not
   a cryptographic key. Is actor-to-key binding an OKF concern, deliberately out
   of scope, or a packaging-layer responsibility?
3. **Freshness and revocation.** `stale_after` and `status` express producer intent
   inside the content. An immutable old artifact can still verify indefinitely.
   Should adversarial freshness be provided by a separately distributed,
   publisher-signed revocation/current-root statement while OKF retains the
   declared lifecycle metadata?
4. **Independent reproduction.** Does a fresh run reproduce the three expected
   roots, and does an independent reader agree on the logical passage root?

## Optional deployment fixture

```bash
./launch/google-okf/deploy-gcs.sh <bucket-name> \
  target/google-okf-reproduction/ga4.annpack
```

The script uploads a content-addressed artifact and configures browser-visible
range headers and immutable caching. It does not alter bucket IAM.

## Gemini CLI demonstration

```bash
./launch/google-okf/gemini-demo.sh \
  target/google-okf-reproduction/ga4.annpack
```

The demonstration asks the client to return the ANNPack artifact root, exact
passage hash, pinned source revision, and canonical URL. That proves the retrieval
provenance exposed by ANNPack; it does not prove that a generated answer follows
from the passage or that Google endorses the integration.
