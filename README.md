# ANNPack

ANNPack compiles a documentation tree into one signed, content-addressed file.
A browser or an agent searches that file with a handful of HTTP range requests
and no server, and every result carries a receipt identifying the exact passage
it came from.

The receipt proves something specific: that a cited passage existed, unmodified,
in a named artifact at a known revision. It says nothing about whether a model's
answer actually follows from that passage. That is a separate problem, and one
ANNPack does not try to solve.

The ranking underneath is ordinary BM25, with optional vectors and rank fusion.
There is no claim to retrieve better than anything else. The claim is that you
can prove what was retrieved.

It reads Markdown, conservative MDX, and
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundles, and can consume OKF v0.2 as published.

Status: `v0.5.0`. Core is `v1.0-draft`. See [Limits](#limits) for what is and
is not established.

## See it working

**[Live demo — Google's OKF `ga4` bundle, compiled and searched in the page](https://arjun2729.github.io/annpackv2/)**

No search server, no database, no full download. The page range-fetches the
artifact from a CDN, checks its root against a pinned value, and logs every byte
range it requested. *Install verified offline copy* downloads the remainder,
verifies each section, and removes the network reader. After that, queries issue
zero HTTP requests.

How little it actually moves is a CI gate, not a claim:
[`web/smoke-transfer.mjs`](web/smoke-transfer.mjs) builds a corpus, searches it
over strict HTTP ranges, and fails if the query transfers more than 45% of the
artifact. It runs on a generated corpus rather than the demo pack because at
23 KB the `ga4` artifact is smaller than its own indexes, where any efficiency
number is noise.

## Reproduce it yourself

**Upstream, pinned:** `GoogleCloudPlatform/knowledge-catalog` at
`3fcbb9f828c2f23d109c855ee403c3a4c81f3a96` (OKF v0.2, Apache-2.0).

```bash
cargo build --release              # Rust 1.88 or newer
./examples/okf-reproduction/reproduce.sh
```

It clones that revision, compiles the OKF bundles present in it, and checks the
resulting artifact roots against
[`expected-roots.json`](examples/okf-reproduction/expected-roots.json):

| bundle | artifact root |
|---|---|
| `ga4` | `7ae75a2da13d50fbffdbd810441c59074d4e649c06e4c547ac013dc46504b2a9` |
| `crypto-bitcoin` | `8301570579afff4f349f8b35bd7ee4af759d8e7604a97a7328f8b76984e116b4` |
| `stackoverflow` | `45aa3600f1c82284c98d26c290405c420a6525c943dad0311bfa49e0c5f405ae` |

The `ga4` artifact is the one served by the live demo. Please open an issue if
the roots do not match.

These are *ANNPack* roots for *this* reproduction, from this builder, pinned to
one upstream revision. Google publishes the OKF source bundles and the
specification; it does not publish ANNPack artifacts and does not endorse this
project.

## How it works

```text
publisher content
      │
      ▼
deterministic builder ──► signed .annpack ──► CLI / MCP / browser / registry
                               │
                               └────────────► answer evidence with exact pack identity
```

- Deterministic Markdown and conservative MDX ingestion
- OKF auto-detection, conformance validation, YAML metadata preservation, and
  source digests
- Structural passage chunking with stable content-derived IDs
- Content and citations stored inside one artifact
- Technical-token-aware BM25 retrieval
- Deterministic IVF-flat vector indexing, exact vectors, and reciprocal-rank
  hybrid fusion
- Safe random-access parsing with checked arithmetic and allocation limits
- Per-section and independently addressable passage-block BLAKE3 verification
- Ed25519 signature sections that do not change the content root
- Native local and strict HTTP-range readers
- MCP tools for pack inspection, search, and exact passage retrieval
- Candidate `/.well-known/annpack.json` discovery documents
- Native OCI Distribution push/pull plus artifact manifests
- Verified bounded copy/add or snapshot delta envelopes
- Browser lexical/IVF range search, profile-checked embedding adapters, and
  WebCrypto signature verification
- Fully verified browser offline installation with a memory-only post-install
  runtime
- Rust/WASM exports for in-memory inspection and lexical search
- Standalone evidence receipts, verified offline with no pack and no network

### Two roots

ANNPack commits to content twice, for two different jobs. Conflating them is the
most common misreading:

- **Artifact root** — BLAKE3 over the section directory. It commits to the
  non-signature directory entries and, through the per-section hashes those
  entries carry, to the stored section bytes they reference. Because the entries
  record DEFLATE output and section layout, it is reproducible by the same
  builder across operating systems and toolchains (CI-enforced) but is not a
  cross-implementation identity, since compression and layout are not
  normatively fixed. It is also not a whole-file hash: it does not authenticate
  unreferenced trailing bytes or excluded signature sections.
- **Logical content root** (`passage_merkle_root`) — Merkle root over
  per-passage evidence hashes. Invariant under compression and layout, so two
  builders that agree on ingestion and chunking agree on this value. It is what
  makes an evidence receipt verifiable without the pack.

See [FORMAT-v3 §3.1 and §4.1](spec/FORMAT-v3.md) and
[ADR-0003](spec/decisions/0003-artifact-root-and-logical-content-root.md).

### Small Core, optional extensions

The stable adoption surface is [ANNPack Core v1.0-draft](spec/CORE-v1.0-draft.md):
the sectioned container, content and passages, citations, BM25, range access,
BLAKE3 integrity, Ed25519 signatures, evidence envelopes, and well-known
discovery. A Core-only reader is fully conformant. The budget is a read-only
client in under 600 executable lines excluding standard crypto, compression,
HTTP, and JSON libraries — and the
[spec-derived reader](spec/conformance/readers/) that passes the suite measures
459.

Vectors, deltas, and OCI distribution are independently optional
[numbered extensions](spec/extensions/README.md). Unimplemented ideas do not
receive extension contracts, and ideas that stop earning theirs lose them.

Core and extension conformance are reported **independently**. A pack can be
`core_conformant: true` and `extensions_conformant: false`. In that state the
runtime serves Core lexical only, and refuses every route into optional
retrieval: a profile request, vector or hybrid search with a query vector, and
any non-zero overlay weight. A malformed optional descriptor cannot influence
the default path.

### Maturity

| Tier | Components | What it means |
|---|---|---|
| **Release candidate** | Core v1.0-draft container, BM25, range access, BLAKE3 integrity, Ed25519 signatures, evidence envelopes | Normatively specified, conformance-tested. Interoperability defects are bugs. |
| **Provisional** | ANN-1 vectors, ANN-2 deltas, ANN-3 OCI, Evidence v1 receipts | Implemented and tested; wire contracts may still change before 1.0. |
| **Experimental** | ANN-7 expansion, ANN-8 SPLADE, ANN-10 fat packs | Off by default. No measured retrieval benefit. Outside the conformance surface. Do not build on these. |
| **Withdrawn** | ANN-5 policy, ANN-6 dependencies, ANN-9 anchors | Removed entirely in v0.5.0: no code, no sections, no contract. ANN-9 held section types 14 and 15; both are retired and will not be reused. |

### How this relates to nearby work

**C2PA / Content Credentials** covers who made a piece of content and how, and
as of v2.3 that extends to unstructured text. ANNPack answers a different
question: which passage of which immutable artifact answered this query.

**OKF** defines how knowledge gets authored and interchanged, and deliberately
says nothing about packaging, serving, or query infrastructure. ANNPack is one
packaging of it. It does not replace OKF, does not change OKF authoring, is not
an official OKF project, and is not affiliated with or endorsed by Google.

**Kiso and the other OKF publishers** turn a bundle into a site people read.
ANNPack turns one into an artifact an agent queries, with a citation you can
check.

**MCP** is how an agent reaches a tool, and ANNPack ships an MCP server.

**llms.txt** tells a crawler what to read. A pack is the same corpus already
parsed, hashed, and range-queryable, and it can publish an `llms.txt` bridge.

**Vector databases** retrieve well. They do not record which revision of your
knowledge produced a given result, because the index is mutable and sits on a
server you have to trust.

## Publish from CI

The shortest path to a published pack. No Rust toolchain on the runner — the
action downloads a prebuilt binary, builds, verifies, and reports the immutable
root.

> **Not usable yet.** The action resolves its binary from a GitHub release, and
> no `v1` release exists. Tag `v0.5.0` and let
> [`release.yml`](.github/workflows/release.yml) publish the binaries first;
> until then the snippet below will fail at the download step. It is documented
> here because the workflow that produces those binaries is in this commit, not
> because the path is live.


```yaml
- uses: Arjun2729/annpackv2@v1
  id: pack
  with:
    source: docs
    output: public/.well-known/knowledge.annpack
    base-url: https://example.com/docs
    signing-key: ${{ secrets.ANNPACK_SIGNING_KEY }}   # optional
- run: echo "published ${{ steps.pack.outputs.root }}"
```

Version defaults to the tag, and the source revision is always pinned to the
building commit — that is what makes a citation from the resulting artifact
checkable against the exact source it came from. The action re-verifies the
artifact after signing and fails if the root moved.

## Build

Rust 1.88 or newer is required. The codebase uses `let` chains, stabilized in
1.88; the transitive `icu_*` crates reached through `url` additionally require
1.86.

```bash
cargo build --release
```

Compile an Open Knowledge Format bundle. Auto-detection recognizes a conformant
OKF tree; `--source-format okf` makes validation explicit:

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
frontmatter is preserved rather than discarded.

The two conflicting fixture versions used throughout the tests and demos:

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

## Search and verify

```bash
target/release/annpack verify target/docs-v1.annpack

target/release/annpack search target/docs-v1.annpack \
  "What does AP-104 mean?" \
  --mode lexical \
  --json
```

The v1 pack answers that the API key expired. The v2 pack answers that the
signature algorithm is unsupported. Every hit carries an `annpack-evidence-v1`
envelope with its pack coordinate, immutable root, source revision, stable
passage ID, direct hash of the exact decoded passage, canonical URL, and
explicitly scoped publisher-verification state.

For a signed pack, pass `--public-key publisher.pub` to `search` or `mcp` to
bind the verified signature to caller-supplied publisher trust. Without that
explicit binding, evidence can report cryptographic verification but keeps
`identity_trusted=false`.

Remote packs use strict range semantics:

```bash
target/release/annpack search \
  https://publisher.example/.well-known/docs.annpack \
  "AP-104" \
  --mode lexical
```

A server that ignores `Range`, returns an incorrect `Content-Range`, truncates a
response, or changes ETag during a session is rejected.

## Proving a citation offline

```bash
annpack receipt knowledge.annpack <passage-id> --output receipt.json
annpack verify-evidence receipt.json --trusted-public-key <publisher-key-hex>
```

`verify-evidence` opens no pack and makes no network request. It recomputes the
whole chain — passage bytes → Merkle path → logical content root → manifest →
directory → artifact root → signature — and reports integrity, authenticity, and
identity trust as three separate claims.

Without `--trusted-public-key` the command verifies integrity and reports
signature and identity status without asserting them. Supplying the flag asserts
that this exact key signed the receipt, so the command exits non-zero unless a
valid signature from that key is present. `annpack verify --public-key` has the
same contract for a pack.

Receipts that authenticate canonical URLs include the stored Documents
catalogue, so receipt size varies by corpus. The format is specified separately
in [EVIDENCE-v1](spec/EVIDENCE-v1.md) so a system that never adopts the ANNPack
container can still emit and check receipts.

## Signatures and trust

```bash
target/release/annpack keygen --output target/publisher.key

target/release/annpack sign target/docs-v1.annpack \
  --output target/docs-v1.signed.annpack \
  --key target/publisher.key \
  --identity vendor.example

target/release/annpack verify target/docs-v1.signed.annpack \
  --public-key target/publisher.pub
```

ANNPack distinguishes three claims:

1. Section and root integrity are valid.
2. A signature is cryptographically valid.
3. A key represents a trusted publisher identity.

The first two are implemented. The third requires an external trust policy,
domain binding, transparency log, or registry identity, and is never inferred
from a self-declared string.

The signature covers the artifact root only. The envelope's asserted identity,
expiration, transparency-log URL, revocation URL, and build-attestation fields
are unauthenticated metadata: nothing binds them, and no runtime decision reads
them. See [FORMAT-v3 §8.1](spec/FORMAT-v3.md).

## MCP

```bash
target/release/annpack mcp target/docs-v1.annpack
```

The stdio MCP server exposes four tools:

- `knowledge_pack_info` — identity, roots, and declared conformance
- `knowledge_search` — ranked passages, each with an inline evidence envelope
- `knowledge_get_passage` — one exact passage by ID
- `knowledge_evidence_receipt` — a standalone receipt proving a passage existed
  unmodified in this exact artifact, verifiable with `annpack verify-evidence`
  without the pack, without network access, and **without trusting the server
  that issued it**

That last tool is the point of running a pack over MCP rather than a search API:
the agent's citation can be checked against a root you pinned, by someone who
trusts neither the agent nor the server. Logs go to stderr so stdout remains
valid JSON-RPC framing.

Configure Gemini CLI without hand-editing JSON. ANNPack verifies a local pack
before adding the MCP server and refuses to replace an existing server unless
`--force` is explicit:

```bash
target/release/annpack integrate gemini target/knowledge.annpack
gemini mcp list
```

The integration writes project-local `.gemini/settings.json`, so the exact pack
and binary are reproducible within the workspace.

## Browser

[`web/index.html`](web/index.html) is a zero-server demo. It logs every HTTP
byte range, displays exact evidence roots and passage hashes, and can install a
complete verified artifact into a memory-only runtime.

```bash
cp target/docs-v1.annpack web/docs-v1.annpack
cd web && python3 serve.py          # then open http://127.0.0.1:8080
```

The client fetches and verifies the header, directory, required indexes,
matching posting lists, and result passages without downloading the complete
pack. BLAKE3 and in-memory search exports come from the repository's Rust/WASM
core; Ed25519 verification uses WebCrypto.

The same page is the real-origin verification surface: pass an HTTPS artifact
URL, expected immutable root, and query as `?pack=...&root=...&q=...`. Run it in
an actual browser so CORS and cache behavior are enforced, then preserve the
Network trace. A Node or localhost smoke does not demonstrate real-CDN behavior.

```bash
node web/smoke-range.mjs            # ranged fetch
node web/smoke-offline.mjs          # terminates the server before querying
```

For vector or hybrid browser search, pass `queryVector` directly or pass a
provider through `createEmbeddingAdapter()`. The adapter checks the model,
revision, dimensions, runtime, and query-prefix behavior against the pack's
exact embedding profile before the IVF runtime searches it. ANNPack does not
pretend that a browser's general-purpose Prompt API is an interoperable
embedding model.

The first golden-path candidate is the 24.1M-parameter
`mixedbread-ai/mxbai-embed-xsmall-v1`, pinned to an exact model revision and
Transformers.js 3.8.1 q8/WASM runtime in
[`default-embedding-profile.json`](spec/examples/default-embedding-profile.json).
[`annpack-transformers.js`](web/annpack-transformers.js) constructs the matching
browser adapter. It remains a candidate — not the blessed release default —
until a real-corpus evaluation demonstrates acceptable retrieval quality and
cold-load behavior.

The dependency-free custom element is the drop-in docs-search surface:

```html
<script type="module" src="/annpack/annpack-widget.js"></script>
<annpack-search src="/.well-known/knowledge.annpack" limit="5"></annpack-search>
```

It renders all untrusted pack text through DOM `textContent`, exposes styling
parts and result/error events, and can be upgraded to hybrid mode by setting its
`embeddingAdapter` property.

## Discovery, OCI, and updates

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
`/.well-known/annpack.json`. Push and pull speak the OCI Distribution API
directly and verify both OCI SHA-256 digests and the ANNPack BLAKE3 root:

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

Delta v1 establishes verified base-root → target-root semantics and
automatically chooses the smaller of a backward-compatible snapshot payload and
a bounded copy/add payload. The latter reuses long unchanged regions —
especially independently compressed passage blocks — then verifies the fully
reconstructed target root before installation.

```bash
target/release/annpack delta create \
  target/docs-v1.annpack target/docs-v2.annpack \
  --output target/v1-v2.anndelta

target/release/annpack delta apply \
  target/docs-v1.annpack target/v1-v2.anndelta \
  --output target/reconstructed-v2.annpack
```

## Tests

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
python3 benches/benchmark.py --binary target/release/annpack --enforce
python3 benches/crawl_vs_pack.py --binary target/release/annpack --enforce
```

The default release gates use a generated 1,000-document corpus: pack size at
most 90% of source, build under 1.5 seconds, and process-inclusive verification
and search p95 under 25 ms on the reference development machine. Verification is
sampled 25 times by default; the report also retains its first-run and median
timings so a single scheduler outlier cannot silently redefine the gate.
Thresholds are explicit CLI flags so slower CI hardware can report the same
measurements without disguising a changed budget.

The crawl comparison measures actual bytes returned by a strict Range server and
compares them with an explicit 50-page × 300 KB rendered-page model. It is
deliberately labeled as a model; the benchmark does not disguise synthetic HTML
as observed production traffic. The default gate demands at least 95% lower
transfer and no more than eight range GETs.

Latency and size say nothing about retrieval quality.
[`evals/evaluate.py`](evals/evaluate.py) exists for the one decision that will
need a number: whether vectors or the optional overlays ever get turned on by
default. It reports lexical, vector, and hybrid macro recall@k, hit rate, and
MRR from human-authored relevance judgments. The two-query fixture included here
tests the harness and nothing else. [`evals/README.md`](evals/README.md)
describes what a usable corpus takes.

Loopback HTTP tests may require permission to bind a local test server in
sandboxed environments.

## Specifications

- [Core v1.0-draft](spec/CORE-v1.0-draft.md)
- [Optional extension registry](spec/extensions/README.md)
- [Binary format](spec/FORMAT-v3.md)
- [Discovery and transport protocol](spec/PROTOCOL-v1.md)
- [Security model](spec/SECURITY.md)
- [Media types and OCI mapping](spec/MEDIA-TYPES.md)
- [Discovery example](spec/examples/annpack.discovery.json)
- [OCI manifest example](spec/examples/oci-manifest.json)
- [`llms.txt` bridge example](spec/examples/llms.txt)
- [Independent security review brief](spec/SECURITY-REVIEW.md)
- [Core layering decision](spec/decisions/0001-core-and-extensions.md)
- [Browser embedding decision](spec/decisions/0002-browser-embedding-candidate.md)

Words like *normative* and *conformance* run through the specification. They
describe how tightly the format pins its own behavior down, so a second
implementer has something exact to disagree with. They are not a claim of
standing.

## Limits

Each of these is stated once, here.

- **Independent-reader interoperability is not established.** A second reader
  now exists — [459 lines of Python](spec/conformance/readers/), written from
  the specification, passing 40/40 including exact IEEE-754 scores. That proves
  the specification is sufficient to build from and that the reference relies on
  no undocumented behaviour. It does **not** prove interoperability: it was
  written by the same author in the same session as the reference changes it
  checks, so the two share blind spots. Core stays `v1.0-draft` until a reader
  written by someone with no access to this repository's implementations passes
  the suite. That is the bar, and a conformance disagreement is still the most
  useful thing anyone can report.
- **No retrieval-quality claim.** Ranking is conventional BM25 with optional
  vectors and reciprocal-rank fusion: well-understood methods, implemented
  carefully, not improved on. The contribution is the evidence chain, not the
  ranking. A hard-negative evaluation now exists
  ([`evals/corpora/`](evals/corpora/README.md)) on a corpus that is *not*
  saturated. It found that reciprocal-rank fusion was actively harmful — hybrid
  scored 0.556 recall@5 against vector-only at 0.794 — which is now fixed
  (0.730, both strata improved). Hybrid still stays off by default, because its
  gain where lexical helps is smaller than its loss where lexical misleads, and
  no static weighting closes that. Its queries and labels are machine-authored,
  so it supports no public claim about how well ANNPack retrieves — only the
  internal comparison that stops a mode being enabled it should not be.
- **Signatures do not establish identity.** Cryptographic validity and publisher
  identity are separate claims, and the second requires an external trust policy
  this project does not supply.
- **Artifact roots are builder-specific**, and the artifact root is not a
  whole-file hash. See [Two roots](#two-roots). Use `passage_merkle_root` for
  cross-implementation identity.
- **Same-builder reproducibility is what is tested.** This builder produces an
  identical artifact root across operating systems and toolchain versions
  (CI-enforced). That is not evidence that a second implementation would.
- **Fuzz durations are short.** Structure-aware targets now reach the parser:
  byte-mutation targets got 0 of 53 corpus inputs past the content-root check
  after 8.1M executions, while `open_consistent_pack` gets 92.9% through by
  repairing the container's hashes after mutating it
  ([`fuzz/README.md`](fuzz/README.md)). No crashes found — but at 60–120s per
  target in scheduled CI that is a weak statement, and a long campaign is still
  owed before any security-critical deployment.
- **Freshness is not enforced by the artifact.** A receipt for a superseded
  artifact verifies correctly forever. Revocation needs the separately
  distributed signed statement described in
  [ADR-0004](spec/decisions/0004-freshness-and-revocation.md).
- **A document cannot repeat identical text under an identical heading.**
  Passage IDs are content-derived, so that case collides and the build is
  rejected rather than producing a pack no reader can open. Repeated warnings
  and boilerplate hit this legitimately; the fix is to make one occurrence
  distinguishable. See [FORMAT-v3 §5.1](spec/FORMAT-v3.md).

`spec/conformance/` and `examples/okf-reproduction/reproduce.sh` exist so none of this
has to be taken on trust. Run them, try to break the format, and report what
broke.

Apache-2.0 licensed.
