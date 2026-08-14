# ANNPack

ANNPack packages a documentation corpus into a single immutable, searchable
artifact, and gives every retrieval an offline-checkable receipt naming the
passage, the artifact, and the source revision it came from.

## The problem

An agent answers a question and cites your documentation. Later someone asks
which version of that documentation it actually read.

Many retrieval systems can record which corpus or document version they believe
they served. Proving it afterwards is the harder part: it usually rests on
application logs, a mutable index, or provenance specific to the system that
produced the answer, all held by the same party making the claim.

ANNPack changes what a retrieval returns:

```text
ordinary retrieval      mutable index ──► passage

ANNPack                 immutable artifact ──► passage
                                          ──► artifact identity
                                          ──► source revision
                                          ──► receipt anyone can check
```

A receipt is a small self-contained file. Checking one needs no artifact, no
network access, and no trust in whatever produced the citation.

## Install

ANNPack is a single binary with no runtime dependencies. There is currently no
Homebrew, apt, or winget package; use a release binary or build from source.

**Release binary** — Linux x86-64, macOS arm64, macOS x86-64:

```bash
curl -sSLO https://github.com/Arjun2729/ANNPACK/releases/download/v0.7.0/annpack-aarch64-apple-darwin.tar.gz
curl -sSLO https://github.com/Arjun2729/ANNPACK/releases/download/v0.7.0/annpack-aarch64-apple-darwin.tar.gz.sha256
shasum -a 256 -c annpack-aarch64-apple-darwin.tar.gz.sha256
tar xzf annpack-aarch64-apple-darwin.tar.gz     # extracts ./annpack
```

Substitute `x86_64-unknown-linux-gnu` or `x86_64-apple-darwin` as needed. Every
release binary carries GitHub-native build provenance, checkable with
`gh attestation verify`.

**From source** — Rust 1.88 or newer:

```bash
cargo install --git https://github.com/Arjun2729/ANNPACK --tag v0.7.0 annpack
```

## Quickstart

Compile a documentation tree, search it, and check a result without the
artifact:

```bash
annpack build docs -o knowledge.annpack \
  --name vendor-docs --version 1.0.0 \
  --source-revision git:$(git rev-parse HEAD)

annpack search knowledge.annpack "refund window" --limit 5 --json
```

A project that builds the same corpus repeatedly can put the stable fields in
[an `annpack.toml`](#project-configuration) and run `annpack build`.

Every hit carries an evidence envelope naming the artifact root, the source
revision, and the passage. Turn one into a standalone receipt and check it with
the artifact nowhere in sight:

```bash
annpack receipt knowledge.annpack <passage-id> --output receipt.json
annpack verify-evidence receipt.json
```

```text
VERIFIED: this passage was in the named artifact, unmodified.
```

That command opens no artifact and makes no network request. It recomputes the
chain from the passage bytes through the Merkle path, logical content root,
manifest, and directory to the artifact root.

## What a receipt proves

A receipt establishes that **the passage, the artifact identity, and the stated
source revision agree with bytes committed by the named artifact root** — and
that anyone can confirm it independently, offline.

The artifact commits to the revision it declares; the receipt does not witness
the build that produced it. [Build provenance](#build-provenance) answers that
separate question.

It does not establish that a model's answer follows from the passage, that the
model read the passage at all, or that the passage is true. Those are different
problems, and ANNPack does not claim them.

Two further boundaries worth stating early, because they shape where ANNPack
fits:

- **A valid signature is authenticity, not identity.** Binding a key to a
  publisher is an external trust decision this project does not make for you.
- **Freshness lives outside the artifact, by design.** A receipt for a
  superseded artifact keeps verifying, because it records what was read.
  Currency comes from separately distributed, publisher-signed release state.

Ranking is BM25 with optional vector retrieval and score fusion. ANNPack makes
no retrieval-quality claim; the contribution is the evidence chain. See
[Limitations](#limitations) for the measured detail.

## Ways to use it

| | |
|---|---|
| **CLI** | `annpack build`, `search`, `receipt`, `verify-evidence` |
| **GitHub Action** | [build and publish an artifact in CI](#github-action) |
| **MCP server** | [agent-facing tools for search and receipts](#mcp) |
| **Browser** | [lexical and vector search over HTTP ranges, no server](#browser-runtime) |
| **Python** | [`pip install annpack`](bindings/python/README.md) — wraps the CLI |
| **Node** | [`@annpack/node`](bindings/node/README.md) — wraps the CLI |
| **Static sites** | [Docusaurus, VitePress, Astro, Mintlify](integrations/README.md) |

The Python and Node packages are thin process bindings: they drive the `annpack`
binary and do not parse artifact bytes themselves, so untrusted input stays in
the Rust runtime. Both require the CLI installed as above.

Input formats: Markdown, conservative MDX, and
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundles, including OKF v0.2 as published.

Version `v0.7.0`. Core is `v1.0-draft`. Apache-2.0.

## How it works

```text
publisher content
      │
      ▼
deterministic builder ──► signed .annpack ──► CLI / MCP / browser / registry
                               │
                               └────────────► answer evidence with exact pack identity
```

The builder is deterministic: the same ANNPack version, source bytes, and build
options produce the same artifact bytes. What the artifact carries:

- Deterministic Markdown and conservative MDX ingestion
- OKF auto-detection, conformance validation, YAML metadata preservation, and
  source digests
- Structural passage chunking with stable content-derived identifiers
- Content and citations stored within a single artifact
- Technical-token-aware BM25 retrieval
- Deterministic IVF-flat vector indexing, exact vectors, and score fusion
- Random-access parsing with checked arithmetic and allocation limits
- Per-section and independently addressable block BLAKE3 verification
- Ed25519 signature sections that do not alter the content root
- Local and strict HTTP-range readers
- MCP tools for pack inspection, search, passage retrieval, and receipt issuance
- `/.well-known/annpack.json` discovery documents
- OCI Distribution push and pull with artifact manifests
- Verified bounded copy/add and snapshot delta envelopes
- Browser lexical and IVF range search, profile-checked embedding adapters, and
  WebCrypto signature verification
- Verified browser offline installation with a memory-only post-install runtime
- Rust/WASM exports for in-memory inspection and lexical search
- Standalone evidence receipts, verified with no pack and no network access
- Publisher trust roots with role-separated keys, and channel-state statements
  carrying release currency outside the artifact

### Content roots

ANNPack commits to content twice, for two distinct purposes.

**Artifact root** — BLAKE3 over the section directory. It commits to the
non-signature directory entries and, through the per-section hashes those
entries carry, to the stored section bytes they reference. Because the entries
record DEFLATE output and section layout, the root is reproducible by the same
builder across operating systems and toolchain versions (CI-enforced) but is not
a cross-implementation identity, as compression and layout are not normatively
fixed. It is not a whole-file hash: it does not authenticate unreferenced
trailing bytes or excluded signature sections.

**Logical content root** (`passage_merkle_root`) — Merkle root over per-passage
evidence hashes. Invariant under compression and layout, so two builders that
agree on ingestion and chunking produce the same value. It is the basis for
verifying an evidence receipt without the artifact.

See [FORMAT-v3 §3.1 and §4.1](spec/FORMAT-v3.md) and
[ADR-0003](spec/decisions/0003-artifact-root-and-logical-content-root.md).

### Core and extensions

The stable adoption surface is [ANNPack Core v1.0-draft](spec/CORE-v1.0-draft.md):
the sectioned container, content and passages, citations, BM25, range access,
BLAKE3 integrity, Ed25519 signatures, evidence envelopes, and well-known
discovery. A Core-only reader is fully conformant.

The size budget for a read-only client is 600 executable lines, excluding
crypto, compression, HTTP, and JSON libraries. The
[spec-derived reader](spec/conformance/readers/) that passes the current
conformance runner, including receipt verification, measures 566.

Vectors, deltas, and OCI distribution are independently optional
[numbered extensions](spec/extensions/README.md). Extension numbers are assigned
only when wire behavior exists in the reference implementation with conformance
tests, and are withdrawn when an extension no longer justifies one.

Core and extension conformance are reported independently. An artifact may be
`core_conformant: true` and `extensions_conformant: false`. In that state the
runtime serves Core lexical retrieval only and refuses every route into optional
retrieval: profile requests, vector or hybrid search with a query vector, and
any non-zero overlay weight. A malformed optional descriptor cannot influence
the default path.

### Maturity

| Tier | Components | Contract |
|---|---|---|
| **Release candidate** | Core v1.0-draft container, BM25, range access, BLAKE3 integrity, Ed25519 signatures, evidence envelopes | Normatively specified and conformance-tested. Interoperability defects are bugs. |
| **Provisional** | AN-1 vectors, AN-2 deltas, AN-3 OCI, Evidence v1 receipts | Implemented and tested. Wire contracts may change before 1.0. |
| **Experimental** | AN-7 expansion, AN-8 SPLADE, AN-10 multi-profile packs | Disabled by default. No measured retrieval benefit. Outside the conformance surface. |
| **Withdrawn** | AN-5 policy, AN-6 dependencies, AN-9 anchors | Removed in v0.5.0: no code, no sections, no contract. Section types 11, 14 and 15 are retired and will not be reused. |

### Relationship to adjacent specifications

| Specification | Scope | Relationship |
|---|---|---|
| **OKF** | Authoring and interchange of knowledge. Explicitly excludes storage, serving, and query infrastructure. | ANNPack is one packaging of an OKF bundle. It does not replace OKF or alter OKF authoring. Not an official OKF project. |
| **MCP** | Transport between agent and tool. | ANNPack ships an MCP server. |
| **llms.txt** | Crawler-facing discovery. | An artifact is the same corpus parsed, hashed, and range-queryable, and can publish an `llms.txt` bridge. |
| **C2PA / Content Credentials** | Authorship and provenance of content, extended to unstructured text as of v2.3. | Addresses a different question: which passage of which immutable artifact answered a query. |
| **Vector databases** | Retrieval. | A retrieval system can carry provenance as application-defined payload. ANNPack makes corpus identity, source revision, passage identity, and independent verification properties of a portable artifact and its receipts, rather than conventions of the application that stored them. Complementary, not a replacement. |

## GitHub Action

The action downloads a prebuilt binary, builds, optionally signs, verifies, and
reports the immutable root. No Rust toolchain is required on the runner.

> **Prerequisite.** The action resolves its binary from a GitHub release, so a
> release must exist for the referenced tag before the action can run.
>
> The reference above pins an immutable release tag. No moving major-version
> alias (`@v1`) is published: [`COMPATIBILITY.md`](spec/COMPATIBILITY.md) states
> that every published tag is immutable and never re-pointed, and a moving alias
> is the opposite convention. If one is introduced later it will be documented
> as a mutable alias, distinct from release tags.

```yaml
- uses: Arjun2729/ANNPACK@v0.7.0
  id: pack
  with:
    source: docs
    output: public/.well-known/knowledge.annpack
    base-url: https://example.com/docs
    signing-key: ${{ secrets.ANNPACK_SIGNING_KEY }}   # optional
- run: echo "published ${{ steps.pack.outputs.root }}"
```

Version defaults to the tag; the source revision is pinned to the building
commit, which is what makes citations from the artifact checkable against their
source. The action re-verifies the artifact after signing and fails if the root
changed.

## Building artifacts

Building from source requires Rust 1.88 or newer: the codebase uses `let`
chains, stabilized in 1.88, and the transitive `icu_*` crates reached through
`url` require 1.86.

### Project configuration

`--name` and `--version` are required on every build and rarely change between
them. An optional `annpack.toml` in the working directory supplies them once:

```toml
[build]
name = "vendor-docs"
version = "1.0.0"
source = "docs"                  # the input directory, the positional argument
output = "knowledge.annpack"
```

`annpack build` then takes no arguments. Explicit arguments always win, so
existing scripts and CI commands are unaffected by a file appearing beside
them, and a value read from configuration produces byte-identical output to the
same value passed as an argument.

The file is a shorthand for typing, not a source of identity. `source` is the
input directory and is unrelated to `source-revision`, which the file cannot
supply at all: a revision changes with every commit, so a checked-in one would
be stale by default. Nothing in the file is inferred either — an unrecognized key
is an error rather than a silent no-op, and a missing required field names both
ways to supply it. `description`, `base-url`, `license`, and `redistributable`
are also accepted.

Compiling an OKF bundle. Auto-detection recognizes a conformant OKF tree;
`--source-format okf` makes validation explicit:

```bash
target/release/annpack build path/to/okf-bundle \
  --source-format okf \
  --output target/knowledge.annpack \
  --name publisher-knowledge \
  --version 0.1.0 \
  --source-revision git:<immutable-commit> \
  --license Apache-2.0 \
  --redistributable true
```

The manifest records `source.format=okf`, the declared OKF version, and a
deterministic BLAKE3 digest over the sorted source tree. Unknown producer
frontmatter is preserved.

The two fixture versions used by tests and demonstrations:

```bash
target/release/annpack build fixtures/docs-v1 \
  --output target/docs-v1.annpack \
  --name vendor-docs --version 1.0.0 \
  --source-revision git:v1 \
  --base-url https://vendor.example/docs/v1

target/release/annpack build fixtures/docs-v2 \
  --output target/docs-v2.annpack \
  --name vendor-docs --version 2.0.0 \
  --source-revision git:v2 \
  --base-url https://vendor.example/docs/v2
```

## Search and verification

```bash
target/release/annpack verify target/docs-v1.annpack

target/release/annpack search target/docs-v1.annpack \
  "What does AP-104 mean?" \
  --mode lexical \
  --json
```

Each hit carries an `annpack-evidence-v1` envelope containing the pack
coordinate, immutable root, source revision, stable passage identifier, hash of
the decoded passage, canonical URL, and scoped publisher-verification state.

For a signed artifact, `--public-key publisher.pub` on `search` or `mcp` binds
the verified signature to caller-supplied publisher trust. Without that binding,
evidence reports cryptographic verification but retains
`identity_trusted=false`.

Remote artifacts use strict range semantics:

```bash
target/release/annpack search \
  https://publisher.example/.well-known/docs.annpack \
  "AP-104" \
  --mode lexical
```

A server that ignores `Range`, returns an incorrect `Content-Range`, truncates a
response, or changes `ETag` during a session is rejected.

## Evidence receipts

```bash
annpack receipt knowledge.annpack <passage-id> --output receipt.json
annpack verify-evidence receipt.json --trusted-public-key <publisher-key-hex>
```

`verify-evidence` opens no artifact and makes no network request. It recomputes
the chain — passage bytes → Merkle path → logical content root → manifest →
directory → artifact root → signature — and reports integrity, authenticity, and
identity trust as three separate claims.

Without `--trusted-public-key`, the command verifies integrity and reports
signature and identity status without asserting them. Supplying the flag asserts
that the named key signed the receipt; the command exits non-zero unless a valid
signature from that key is present. `annpack verify --public-key` applies the
same contract to an artifact.

Receipts that authenticate canonical URLs embed the stored Documents section, so
receipt size varies by corpus. The format is specified separately in
[EVIDENCE-v1](spec/EVIDENCE-v1.md) so that systems which do not adopt the
ANNPack container can still emit and check receipts.

## Run bundles

```bash
annpack bundle knowledge.annpack "rotate the signing key" \
  --limit 5 --application support-agent/2.1 --model <model-id> \
  --output run.json

annpack verify-run run.json --trusted-public-key <publisher-key-hex>
```

A run bundle collects one agent run's retrieval evidence into a single portable
file: one standalone receipt per retrieved passage, plus the metadata needed to
locate the run in an application's own logs.

The bundle defines no cryptography and no container section. Verifying it is
receipt verification applied to each receipt in turn, so it can prove nothing a
receipt could not prove alone.

The verifier separates two categories and never merges them:

- **Attested** — every receipt proved its passage existed unmodified in a named
  immutable artifact at a named source revision.
- **Carried** — the query, application, model, and answer travel with the
  receipts and are attested by nothing.

A bundle carrying no receipts is never reported as attested, signed, or trusted.
Signature aggregates are conditioned on verification, because a receipt's
signature covers the artifact root rather than the passage: a rewritten passage
record still carries a valid signature.

Bundle verification is implemented in Rust and in the browser runtime;
`web/smoke-bundle.mjs` requires both to reach the same verdict on the same file,
including tampered and emptied bundles. See [EVIDENCE-v1](spec/EVIDENCE-v1.md).

## Signed run attestation

```bash
annpack run-attestation create run.json \
  --channel-state channel.json --trust-root trust.json \
  --expect-publisher example.test --expect-corpus support \
  --expect-channel production --now 2030-01-02T00:00:00Z \
  --output-bytes answer.txt --prompt-policy prompt-policy.txt \
  --output run-statement.json --run-id run-001 --trace-id trace-001 \
  --workload-identity support-agent \
  --started-at 2030-01-01T12:00:00Z \
  --completed-at 2030-01-01T12:00:01Z \
  --retrieval-policy-revision retrieval-v1 \
  --application-identity support-agent --application-version 1.0.0 \
  --model-identifier model-1 --tool-policy-revision tools-v1

annpack run-attestation sign run-statement.json \
  --key workload.key --output run-attestation.json

annpack run-attestation verify run-attestation.json \
  --bundle run.json --channel-state channel.json --trust-root trust.json \
  --expect-publisher example.test --expect-corpus support \
  --expect-channel production --now 2030-01-02T00:00:00Z \
  --trusted-workload-key support-agent=<workload-public-key> \
  --expect-run-id run-001 --expect-trace-id trace-001 \
  --expect-model model-1 --prompt-policy prompt-policy.txt \
  --output-bytes answer.txt --require-output --json
```

This DSSE-wrapped in-toto statement turns the bundle's carried fields into a
claim by a separately trusted application workload. It binds the exact canonical
receipt set, publisher and channel-state evidence, query, model and policy
identifiers, and output SHA-256. Verification reports every stage separately;
historical occurrence evidence remains valid after supersession or revocation
while present use is denied. It neither changes `.annpack` bytes nor grants
publisher or builder keys workload authority. See
[RUN-ATTESTATION-v1](spec/RUN-ATTESTATION-v1.md).

## Trace attributes

```bash
annpack search knowledge.annpack "rotate the signing key" \
  --otel --otel-receipt-uri 'https://evidence.example/{root}/{passage_id}'
```

Emits OpenTelemetry span and event attributes that bind a retrieval to the
artifact it read: `annpack.root`, `annpack.pack`, `annpack.source_revision`, and
per passage `annpack.passage_id`, `annpack.passage_hash`, `annpack.receipt_uri`.
A span carrying them remains checkable after the corpus moves.

Attribute names only — no exporter, backend, or transport is defined, and all
names stay inside the `annpack.*` namespace so they compose with whatever
`gen_ai.*` conventions the host application already emits. Not part of Core
conformance. See [TELEMETRY](spec/TELEMETRY.md).

## Signatures

```bash
target/release/annpack keygen --output target/publisher.key

target/release/annpack sign target/docs-v1.annpack \
  --output target/docs-v1.signed.annpack \
  --key target/publisher.key \
  --identity vendor.example

target/release/annpack verify target/docs-v1.signed.annpack \
  --public-key target/publisher.pub
```

Three claims are distinguished:

1. Section and root integrity are valid.
2. A signature is cryptographically valid.
3. A key represents a trusted publisher identity.

The first two are implemented. The third requires an external trust policy,
domain binding, transparency log, or registry identity, and is never inferred
from a self-declared string.

The signature covers the artifact root only. The envelope's asserted identity,
expiration, transparency-log URL, revocation URL, and build-attestation fields
are unauthenticated metadata: nothing binds them and no runtime decision reads
them. See [FORMAT-v3 §8.1](spec/FORMAT-v3.md).

## Build provenance

```bash
target/release/annpack provenance create target/docs-v1.annpack \
  --output target/docs-v1.provenance.json \
  --repository github.com/vendor/docs --revision git:abc123 \
  --builder-id local --builder-binary target/release/annpack --system-clock

target/release/annpack provenance sign target/docs-v1.provenance.json \
  --key target/builder.key

target/release/annpack provenance verify target/docs-v1.annpack \
  target/docs-v1.provenance.json --trusted-builder-key <builder-pub-hex>
```

A DSSE-enveloped [in-toto](https://in-toto.io/Statement/v1) statement binding a
source revision, a builder identity, and a build execution to the distributed
`.annpack` file's own SHA-256, artifact root, and (for a manifest-format-4
artifact) authenticated source digest. Distinct from artifact signing above: a
provenance statement answers *how* the artifact was built, not who is
authorized to publish or use it.

A builder key is not a publisher trust-root role. Using an artifact-signing key
to sign provenance does not make it a trusted builder; trust comes only from
the key list a verifier explicitly supplies. `repository` and `revision` are
always reported as carried claims — a signature proves who wrote them, never
that they are historically true. See [PROVENANCE-v1](spec/PROVENANCE-v1.md).

Official GitHub releases sign the ANNPack predicate keylessly, via GitHub
OIDC and Sigstore's Fulcio rather than a stored repository secret (`release.yml`
uses `actions/attest` with a custom `predicate-type`; see
[ADR-0006](spec/decisions/0006-build-provenance-envelope.md)). The resulting
Sigstore bundle is published alongside each platform asset and can be
inspected offline:

```bash
target/release/annpack provenance verify-github \
  artifact.annpack \
  artifact.annpack-provenance.sigstore.json \
  --trusted-root trusted_root.json \
  --allowed-issuer https://token.actions.githubusercontent.com \
  --allowed-repository https://github.com/<owner>/annpack \
  --allowed-workflow-ref https://github.com/<owner>/annpack/.github/workflows/release.yml@refs/tags/v1.2.3 \
  --json
```

Requires the `github-attestation` build feature. Verification is fully offline:
the command reads only the artifact, exported bundle, and explicitly supplied
Sigstore trusted-root snapshot. It verifies trusted signing time, the Fulcio
chain and certificate validity, SCT evidence, Rekor checkpoint/inclusion/SET,
the DSSE signature and artifact binding, and Rekor-entry consistency before it
extracts GitHub workload claims or evaluates policy. It then checks the ANNPack
predicate against the artifact and requires its repository/revision claims to
agree with the authenticated certificate claims.

Obtain trusted-root JSON through the operator's normal Sigstore TUF update
process in a networked environment, record its SHA-256 digest, review it, then
transfer it to the offline verifier. `verify-github` never downloads a root.
Root snapshots do not remain current indefinitely: repeat that update process
to receive rotations and revocations. Historical verification against an old
snapshot is deterministic, but does not prove that snapshot is still current.

## Fleet policy

```bash
target/release/annpack fleet policy init --output fleet.json --domain acme.example \
  --revision 1 --valid-until 2027-08-09T00:00:00Z --key policy.pub --threshold 1 \
  --allow-publisher example.com --allow-scope support-manual:production \
  --required-policy authorized-current-witnessed

target/release/annpack fleet policy sign fleet.json --key policy.key

target/release/annpack fleet policy verify fleet.json --system-clock

target/release/annpack fleet policy evaluate --local fleet.json --required fleet.json \
  --system-clock --json
```

A signed, versioned document an organization issues stating what its fleet of
verifiers requires — which publishers, which scopes, which verification
policy tier, and which `annpack release monitor` incidents must deny use.
Distinct from a trust root: a publisher's trust root says who may publish;
fleet policy says what a consuming organization requires, signed by keys the
organization controls, not the publisher. `evaluate` compares a locally
configured policy against the one required and reports `compliant`,
`drifted`, or `unavailable` — never falls back to compliant when either
input is missing or fails to verify. See [FLEET-v1](spec/FLEET-v1.md).

## MCP

```bash
target/release/annpack mcp target/docs-v1.annpack
```

The stdio MCP server exposes four tools:

| Tool | Returns |
|---|---|
| `knowledge_pack_info` | Identity, roots, and declared conformance |
| `knowledge_search` | Ranked passages, each with an inline evidence envelope |
| `knowledge_get_passage` | One passage by identifier |
| `knowledge_evidence_receipt` | A standalone receipt, verifiable without the artifact, without network access, and without trusting the issuing server |

Logs are written to stderr so that stdout remains valid JSON-RPC framing.

Gemini CLI configuration. The pack is verified before the MCP server is
registered, and an existing server is not replaced unless `--force` is given:

```bash
target/release/annpack integrate gemini target/knowledge.annpack
gemini mcp list
```

The integration writes project-local `.gemini/settings.json`, so the artifact
and binary are reproducible within the workspace.

## Browser runtime

[`web/index.html`](web/index.html) is a zero-server client. It logs each HTTP
byte range, displays evidence roots and passage hashes, and can install a
complete verified artifact into a memory-only runtime.

```bash
cp target/docs-v1.annpack web/docs-v1.annpack
cd web && python3 serve.py          # http://127.0.0.1:8080
```

The client fetches and verifies the header, directory, required indexes,
matching posting lists, and result passages without downloading the complete
artifact. BLAKE3 and in-memory search exports come from the repository's
Rust/WASM core; Ed25519 verification uses WebCrypto.

The same page serves as the real-origin verification surface: pass an HTTPS
artifact URL, expected root, and query as `?pack=...&root=...&q=...`. Running it
in a browser enforces CORS and cache behavior; a Node or localhost smoke test
does not demonstrate real-CDN behavior.

An origin has to return byte ranges untransformed and, for a cross-origin
artifact, expose `Content-Range` through `Access-Control-Expose-Headers`. Hosts
that compress the artifact, or withhold that header, fail one of the two: a
range then addresses compressed bytes, or the reader cannot read the value it
validates. It refuses both rather than proceeding.

```bash
node web/smoke-range.mjs            # ranged fetch
node web/smoke-offline.mjs          # terminates the server before querying
```

For vector or hybrid browser search, pass `queryVector` directly or supply a
provider through `createEmbeddingAdapter()`. The adapter checks model, revision,
dimensions, runtime, and query-prefix behavior against the artifact's embedding
profile before the IVF runtime executes. A browser's general-purpose Prompt API
is not treated as an interoperable embedding model.

The candidate golden-path profile is the 24.1M-parameter
`mixedbread-ai/mxbai-embed-xsmall-v1`, pinned to an exact model revision and
Transformers.js 3.8.1 q8/WASM runtime in
[`default-embedding-profile.json`](spec/examples/default-embedding-profile.json).
[`annpack-transformers.js`](web/annpack-transformers.js) constructs the matching
adapter. It remains a candidate rather than a release default pending a
real-corpus evaluation of retrieval quality and cold-load behavior.

The dependency-free custom element:

```html
<script type="module" src="/annpack/annpack-widget.js"></script>
<annpack-search src="/.well-known/knowledge.annpack" limit="5"></annpack-search>
```

It renders untrusted artifact text through DOM `textContent`, exposes styling
parts and result/error events, and supports hybrid mode via its
`embeddingAdapter` property.

## Discovery, OCI distribution, and updates

```bash
target/release/annpack discovery \
  target/docs-v1.signed.annpack \
  target/docs-v2.annpack \
  --publisher vendor.example \
  --public-base-url https://vendor.example/.well-known/packs \
  --output target/annpack.json
```

Framework adapters emit the primary artifact at
`/.well-known/knowledge.annpack`; the multi-pack discovery document belongs at
`/.well-known/annpack.json`.

Push and pull implement the OCI Distribution API and verify both OCI SHA-256
digests and the ANNPack BLAKE3 root:

```bash
export ANNPACK_REGISTRY_USERNAME=publisher
export ANNPACK_REGISTRY_PASSWORD="$(security find-generic-password -w -s annpack-registry)"

target/release/annpack push \
  target/docs-v1.signed.annpack \
  ghcr.io/vendor/knowledge/docs:1.0.0

target/release/annpack pull \
  ghcr.io/vendor/knowledge/docs:1.0.0 \
  --output target/pulled-docs-v1.annpack
```

Anonymous, Basic, and OCI Bearer-challenge authentication are supported.
Passwords are read from the named environment variable, never from a
command-line argument.

Delta v1 establishes verified base-root to target-root semantics and selects the
smaller of a backward-compatible snapshot payload and a bounded copy/add
payload. The latter reuses unchanged regions, particularly independently
compressed passage blocks, and verifies the reconstructed target root before
installation.

```bash
target/release/annpack delta create \
  target/docs-v1.annpack target/docs-v2.annpack \
  --output target/v1-v2.anndelta

target/release/annpack delta apply \
  target/docs-v1.annpack target/v1-v2.anndelta \
  --output target/reconstructed-v2.annpack
```

## Reproduction

Upstream source: `GoogleCloudPlatform/knowledge-catalog` at
`3fcbb9f828c2f23d109c855ee403c3a4c81f3a96` (OKF v0.2, Apache-2.0).

```bash
cargo build --release              # Rust 1.88 or newer
./examples/okf-reproduction/reproduce.sh
```

The script clones that revision, compiles the OKF bundles it contains, and
compares the resulting artifact roots against
[`expected-roots.json`](examples/okf-reproduction/expected-roots.json):

| bundle | artifact root |
|---|---|
| `ga4` | `3b69e675699786e602ae5c1e8a83e5fdf2f11ccb27e4e7dac4ea79d9fa5fe41e` |
| `crypto-bitcoin` | `19d813bec8a3fd7136c37f737a4733dfc4349c20309d39ef632e718613783dd9` |
| `stackoverflow` | `f0ad8fb990893f28da9e193c41a97e532ff7f41599448196d440660590bb9398` |

The `ga4` artifact is the one served by the live demo. Root mismatches should be
reported as issues.

These are ANNPack roots for this reproduction, produced by this builder against
one pinned upstream revision. Google publishes the OKF source bundles and the
specification. Google does not publish ANNPack artifacts and does not endorse
this project.

## Testing and release gates

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
python3 benches/benchmark.py --binary target/release/annpack --enforce
python3 benches/crawl_vs_pack.py --binary target/release/annpack --enforce
```

Release gates run against a generated 1,000-document corpus: artifact size at
most 90% of source, build under 1.5 seconds, and process-inclusive verification
and search p95 under 25 ms on the reference machine. Verification is sampled 25
times; the report retains first-run and median timings so that a single
scheduler outlier cannot redefine the gate. Thresholds are explicit CLI flags,
so slower hardware can report the same measurements without altering the budget
implicitly.

The crawl comparison measures bytes returned by a strict Range server against an
explicit 50-page × 300 KB rendered-page model. The model is labeled as such; the
benchmark does not present synthetic HTML as observed production traffic. The
default gate requires at least 95% lower transfer and no more than twelve range
GETs.

Latency and size do not measure retrieval quality.
[`evals/evaluate.py`](evals/evaluate.py) reports lexical, vector, and hybrid
macro recall@k, hit rate, and MRR from supplied relevance judgments. The
two-query fixture tests the harness only. [`evals/corpora/`](evals/corpora/README.md)
contains the hard-negative corpus used to evaluate retrieval modes.

Three implementations run the current conformance suite on every build: the
Rust reference, the browser runtime, and a reader written from the specification
alone. All three pass the runner's 46 checks, including the two Evidence v1
receipt checks. The runner does not yet execute every Core obligation: the
range and signature vectors ship with the packet but are not consumed by it.

The conformance contract is not extended with a run-bundle verb. Bundle
verification is receipt verification applied N times, and `verify-receipt`
already holds three implementations to that. What the bundle adds — the
aggregate verdict — is gated instead by `web/smoke-bundle.mjs`, which requires
the Rust and browser verifiers to agree on the same file across intact,
tampered, wrongly-signed, and emptied cases. It found a real divergence between
the two on its first run.

Loopback HTTP tests may require permission to bind a local test server in
sandboxed environments.

Transfer efficiency is enforced as a CI gate rather than asserted.
[`web/smoke-transfer.mjs`](web/smoke-transfer.mjs) builds a corpus, searches it
over strict HTTP ranges, and fails if a query transfers more than 45% of the
artifact. It runs against a generated corpus rather than the demo pack: at 23 KB
the `ga4` artifact is smaller than its own indexes, so efficiency measurements
against it are not meaningful.

## Specifications

- [Core v1.0-draft](spec/CORE-v1.0-draft.md)
- [Binary format](spec/FORMAT-v3.md)
- [Discovery and transport protocol](spec/PROTOCOL-v1.md)
- [Evidence receipts and run bundles](spec/EVIDENCE-v1.md)
- [Trust roots and release state](spec/RELEASE-v1.md)
- [Build provenance](spec/PROVENANCE-v1.md)
- [Run attestation](spec/RUN-ATTESTATION-v1.md)
- [OpenTelemetry attributes](spec/TELEMETRY.md)
- [Security model](spec/SECURITY.md)
- [Media types and OCI mapping](spec/MEDIA-TYPES.md)
- [Compatibility boundary](spec/COMPATIBILITY.md)
- [Optional extension registry](spec/extensions/README.md)
- [Conformance suite](spec/conformance/README.md)
- [Discovery example](spec/examples/annpack.discovery.json)
- [OCI manifest example](spec/examples/oci-manifest.json)
- [`llms.txt` bridge example](spec/examples/llms.txt)
- [Independent security review brief](spec/SECURITY-REVIEW.md)
- [ADR-0001: Core and extensions](spec/decisions/0001-core-and-extensions.md)
- [ADR-0002: Browser embedding candidate](spec/decisions/0002-browser-embedding-candidate.md)
- [ADR-0004: Release authorization is time-indexed](spec/decisions/0004-freshness-and-revocation.md)
- [ADR-0005: Authenticated source digest](spec/decisions/0005-authenticated-source-digest.md)
- [ADR-0006: Build provenance envelope](spec/decisions/0006-build-provenance-envelope.md)

The terms *normative* and *conformance* describe how tightly the specification
constrains its own behavior, so that an independent implementer has an exact
contract to disagree with. They do not indicate standards-body status.

## Limitations

**Independent-reader interoperability is not established.** A reader written
from the specification alone exists —
[Python, in `spec/conformance/readers/`](spec/conformance/readers/), passing the
runner's 46 checks including exactly asserted IEEE-754 scores and offline
receipt verification. That is evidence the specification can drive a second
implementation, and that the checks the runner does execute find no divergence.
It does not establish interoperability: it was written by the same author, in
the same working session, as the reference changes it validates, so the two may
share undocumented assumptions. Core remains `v1.0-draft` until a reader written
without access to this repository's implementations passes the suite.

**No retrieval-quality claim.** Ranking is BM25 with optional vectors and score
fusion. The contribution is the evidence chain rather than the ranking. A
hard-negative evaluation on a non-saturated corpus
([`evals/corpora/`](evals/corpora/README.md)) measured reciprocal-rank fusion as
harmful — hybrid at 0.556 recall@5 against vector-only at 0.794 — which was
corrected by absolute-scale fusion (0.730, both strata improved). Hybrid remains
disabled by default: its gain where lexical retrieval contributes is smaller
than its loss where lexical retrieval misleads, and no static weighting closes
the difference. A per-query oracle — the upper bound on any routing strategy —
exceeds vector-only by four queries out of 63, which does not establish that a
practical router could capture that margin. The evaluation's queries and labels are machine-authored and
support no claim about retrieval quality.

**Signatures do not establish identity.** Cryptographic validity and publisher
identity are separate claims. The second requires an external trust policy that
this project does not supply.

**Artifact roots are builder-specific** and are not whole-file hashes. See
[Content roots](#content-roots). Use `passage_merkle_root` for
cross-implementation identity.

**Same-builder reproducibility is what is tested.** This builder produces an
identical artifact root across operating systems and toolchain versions
(CI-enforced). That is not evidence that an independent implementation would.

**Fuzz durations are short.** Structure-aware targets reach the parser:
byte-mutation targets moved 0 of 53 corpus inputs past the content-root check
after 8.1M executions, while `open_consistent_pack` moves 92.9% past it by
repairing container hashes after mutation ([`fuzz/README.md`](fuzz/README.md)).
No crashes have been found, but at 60–120 seconds per target in scheduled CI
that is a weak result. A sustained campaign is required before any
security-critical deployment.

**Freshness is not enforced by the artifact, by design.** A receipt for a
superseded artifact continues to verify, because it records what was read.
Currency comes from a separately distributed, publisher-signed channel-state
statement — see [RELEASE-v1](spec/RELEASE-v1.md). Rollback resistance requires
durable per-scope client state, so it is unavailable at first contact and after
state loss, and detecting a publisher who signs conflicting statements requires
the witnessed profile, whose transparency verification is not implemented.

**A document cannot repeat identical text under an identical heading.** Passage
identifiers are content-derived, so that case collides and the build is rejected
rather than producing an artifact no reader can open. Repeated warnings and
boilerplate encounter this; the resolution is to make one occurrence
distinguishable. See [FORMAT-v3 §5.1](spec/FORMAT-v3.md).

`spec/conformance/` and `examples/okf-reproduction/reproduce.sh` exist so that
these claims can be checked directly.
