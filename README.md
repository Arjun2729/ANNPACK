# ANNPack

ANNPack packages reproducible, verifiable knowledge for agents and browsers. A pack is a content-addressed, range-queryable artifact, and every search result carries a tamper-evident citation to the exact immutable passage it retrieved.

It is designed to make knowledge portable and reproducible in the way software packages make executable environments portable:

```text
publisher content
      │
      ▼
deterministic builder ──► signed .annpack ──► CLI / MCP / browser / registry
                               │
                               └────────────► answer evidence with exact pack identity
```

Version-exact developer documentation is the first use case. What ANNPack guarantees, precisely: **tamper-evident provenance of the retrieved span** — cryptographic proof that a cited passage existed, unmodified, in a known immutable artifact at a known revision. A mutable hosted index cannot independently prove which knowledge revision it returned; a verified pack and evidence envelope can.

It does **not** prove that a model's answer faithfully follows from the retrieved passage — answer faithfulness is a separate problem ANNPack does not solve. The claim is auditable retrieval provenance, not hallucination-proof generation.

## What works

- Deterministic Markdown and conservative MDX ingestion
- First-class OKF 0.1 auto-detection, conformance validation, YAML metadata preservation, and source digests
- Structural passage chunking with stable content-derived IDs
- Content and citations stored inside one artifact
- Technical-token-aware BM25 retrieval
- Deterministic IVF-flat vector indexing, exact vectors, and reciprocal-rank hybrid fusion
- Safe random-access parsing with checked arithmetic and allocation limits
- Per-section and independently addressable passage-block BLAKE3 verification
- Ed25519 signature sections that do not change the content root
- Native local and strict HTTP-range readers
- MCP tools for pack inspection, search, and exact passage retrieval
- Candidate `/.well-known/annpack.json` discovery documents
- Native OCI Distribution push/pull plus artifact manifests
- Verified bounded copy/add or snapshot delta envelopes
- Browser lexical/IVF range search, profile-checked embedding adapters, and WebCrypto signature verification
- Fully verified browser offline installation with a memory-only post-install runtime
- Rust/WASM exports for in-memory inspection and lexical search
- Standalone evidence receipts: a ~2–5 KB document that proves a cited passage was in a named artifact, verified offline with no pack, no network, and no trust in the issuer
- Deterministic, corruption, property, CLI, MCP, signature, delta, hybrid, compatibility, and HTTP tests

## Maturity

Not everything here is equally settled. Treat these tiers as the actual contract:

| Tier | Components | What it means |
|---|---|---|
| **Release candidate** | Core v1.0-draft container, BM25, range access, BLAKE3 integrity, Ed25519 signatures, evidence envelopes | Normatively specified, conformance-tested. Interoperability defects are bugs. |
| **Provisional** | ANN-1 vectors, ANN-2 deltas, ANN-3 OCI, ANN-5 policy, ANN-6 dependencies, Evidence v1 receipts | Implemented and tested; wire contracts may still change before 1.0. |
| **Experimental** | ANN-7 expansion, ANN-8 SPLADE, ANN-10 fat packs | Off by default. **No measured retrieval benefit.** Outside the conformance surface. Do not build on these. |
| **Withdrawn** | ANN-9 relative-coordinate retrieval | Dominated by simpler methods. Anchor sections still ship as adapter supervision; there is no anchor retrieval path. |

**No retrieval-quality claim is currently supported.** The FastAPI evaluation was
[withdrawn](launch/evidence/withdrawn/2026-07-26-retrieval-quality/WITHDRAWN.md):
it was saturated, its vector rows were not reproducible from committed inputs,
and its ANN-7/ANN-8 rows measured a pack that contained no overlays. A
hard-negative evaluation is owed before any comparative claim is made.

## Two roots

ANNPack commits to content twice, for two different jobs. Conflating them is the
most common misreading:

- **Artifact root** — BLAKE3 over the section directory. Identity of *these exact
  bytes*, including DEFLATE output and section layout. Reproducible by the same
  builder across operating systems and toolchains (CI-enforced). It is **not** a
  cross-implementation identity, because compression and layout are not
  normatively fixed.
- **Logical content root** (`passage_merkle_root`) — Merkle root over per-passage
  evidence hashes. Invariant under compression and layout, so two builders that
  agree on ingestion and chunking agree on this value. It is what makes an
  evidence receipt verifiable without the pack.

See [FORMAT-v3 §3.1 and §4.1](spec/FORMAT-v3.md) and
[ADR-0003](spec/decisions/0003-artifact-root-and-logical-content-root.md).

## Small Core, optional extensions

The stable adoption surface is [ANNPack Core v1.0-draft](spec/CORE-v1.0-draft.md): the sectioned container, content and passages, citations, BM25, range access, BLAKE3 integrity, Ed25519 signatures, evidence envelopes, and well-known discovery. A Core-only reader is fully conformant. The design goal is a read-only client in roughly 500 lines excluding standard crypto, compression, HTTP, and JSON libraries.

Vectors, deltas, OCI, policy descriptors, and pack dependencies are independently optional [numbered extensions](spec/extensions/README.md). The reference CLI reports Core and extension conformance from inspect, verify, discovery, search, and MCP. Unimplemented ideas do not receive extension contracts.

Core and extension conformance are reported **independently**. A pack can be
`core_conformant: true` and `extensions_conformant: false`; in that state the
runtime serves Core lexical and refuses profile-enabled retrieval. A malformed
optional descriptor can never influence the default path.

## Proving a citation offline

```bash
annpack receipt knowledge.annpack <passage-id> --output receipt.json
annpack verify-evidence receipt.json --trusted-public-key <publisher-key-hex>
```

`verify-evidence` opens no pack and makes no network request. It recomputes the
whole chain — passage bytes → Merkle path → logical content root → manifest →
directory → artifact root → signature — and reports integrity, authenticity, and
identity trust as three separate claims. A cryptographically valid signature
never by itself establishes publisher identity.

The receipt format is specified separately in
[EVIDENCE-v1](spec/EVIDENCE-v1.md) so a system that never adopts the ANNPack
container can still emit and check receipts.

## Build

Rust 1.86 or newer is required. (The transitive `icu_*` crates, reached through `url` for the optional HTTP feature, declare a 1.86 minimum.)

```bash
cargo build --release
```

Build two conflicting versions of the conformance documentation:

```bash
target/release/annpack build fixtures/docs-v1 \
  --output target/docs-v1.annpack \
  --name vendor-docs \
  --version 1.0.0 \
  --source-revision git:v1 \
  --base-url https://vendor.example/docs/v1

target/release/annpack build fixtures/docs-v2 \
  --output target/docs-v2.annpack \
  --name vendor-docs \
  --version 2.0.0 \
  --source-revision git:v2 \
  --base-url https://vendor.example/docs/v2
```

Compile an Open Knowledge Format bundle directly. Auto-detection recognizes a
conformant OKF tree; `--source-format okf` makes validation explicit:

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

The resulting manifest records `source.format=okf`, the declared OKF version,
and a deterministic BLAKE3 digest over the sorted source tree. Unknown producer
frontmatter is preserved rather than discarded.

Licensed or payment-discoverable artifacts can supply the complete declarative policy with `--policy-file spec/examples/licensed-policy.json`. Policy metadata does not pretend to enforce payment or DRM after plaintext access.

## Search and verify

```bash
target/release/annpack verify target/docs-v1.annpack

target/release/annpack search target/docs-v1.annpack \
  "What does AP-104 mean?" \
  --mode lexical \
  --json
```

The v1 pack answers that the API key expired. The v2 pack answers that the signature algorithm is unsupported. Every hit carries an `annpack-evidence-v1` envelope with its pack coordinate, immutable root, source revision, stable passage ID, direct hash of the exact decoded passage, canonical URL, and explicitly scoped publisher-verification state.

For a signed pack, pass `--public-key publisher.pub` to `search` or `mcp` to bind the verified signature to caller-supplied publisher trust. Without that explicit binding, evidence can report cryptographic verification but keeps `identity_trusted=false`.

Remote packs use strict range semantics:

```bash
target/release/annpack search \
  https://publisher.example/.well-known/docs.annpack \
  "AP-104" \
  --mode lexical
```

A server that ignores `Range`, returns an incorrect `Content-Range`, truncates a response, or changes ETag during a session is rejected.

## MCP

```bash
target/release/annpack mcp target/docs-v1.annpack
```

The stdio MCP server exposes:

- `knowledge_pack_info`
- `knowledge_search`
- `knowledge_get_passage`

Logs go to stderr so stdout remains valid JSON-RPC framing.

Configure Gemini CLI without hand-editing JSON. ANNPack verifies a local pack
before adding the MCP server and refuses to replace an existing server unless
`--force` is explicit:

```bash
target/release/annpack integrate gemini target/knowledge.annpack
gemini mcp list
```

The integration writes project-local `.gemini/settings.json`, so the exact pack
and binary are reproducible within the workspace.

## Live browser proof

[`web/index.html`](web/index.html) is a zero-server artifact laboratory. It logs
every HTTP byte range, displays exact evidence roots and passage hashes, and can
install a complete verified artifact into a memory-only runtime. The offline
smoke test terminates the HTTP server before executing its query:

```bash
node web/smoke-range.mjs
node web/smoke-offline.mjs
```

The pinned Google OKF reproduction and Cloud Storage deployment scripts live in
[`launch/google-okf/`](launch/google-okf/).

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

The first two are implemented. The third requires an external trust policy, domain binding, transparency log, or registry identity and is never inferred from a self-declared string.

## Discovery and OCI

```bash
target/release/annpack discovery \
  target/docs-v1.signed.annpack \
  target/docs-v2.annpack \
  --publisher vendor.example \
  --public-base-url https://vendor.example/.well-known/packs \
  --output target/annpack.json

target/release/annpack oci-manifest \
  target/docs-v1.signed.annpack \
  --output target/oci-manifest.json
```

Framework adapters emit the primary artifact at `/.well-known/knowledge.annpack`; the multi-pack discovery document belongs at `/.well-known/annpack.json`. Push and pull speak the OCI Distribution API directly and verify both OCI SHA-256 digests and the ANNPack BLAKE3 root:

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

Anonymous, Basic, and OCI Bearer-challenge authentication are supported. Passwords are read from the named environment variable, never from a command-line argument.

## Updates

```bash
target/release/annpack delta create \
  target/docs-v1.annpack \
  target/docs-v2.annpack \
  --output target/v1-v2.anndelta

target/release/annpack delta apply \
  target/docs-v1.annpack \
  target/v1-v2.anndelta \
  --output target/reconstructed-v2.annpack
```

Delta v1 establishes verified base-root → target-root semantics and automatically chooses the smaller of a backward-compatible snapshot payload and a bounded copy/add payload. The latter reuses long unchanged regions—especially independently compressed passage blocks—then verifies the fully reconstructed target root before installation.

## Browser range demo

```bash
cp target/docs-v1.annpack web/docs-v1.annpack
cd web
python3 serve.py
```

Open `http://127.0.0.1:8080`. The browser client fetches and verifies the header, directory, required indexes, matching posting lists, and result passages without downloading the complete pack. BLAKE3 and in-memory search exports come from the repository's Rust/WASM core; Ed25519 verification uses WebCrypto.

The same page is the real-origin verification surface: pass an HTTPS artifact URL, expected immutable root, and query as `?pack=...&root=...&q=...`. Run it in an actual browser so CORS and cache behavior are enforced, then preserve the Network trace. A Node or localhost smoke does not satisfy the real-CDN launch gate.

For vector or hybrid browser search, pass `queryVector` directly or pass a provider through `createEmbeddingAdapter()`. The adapter checks the model, revision, dimensions, runtime, and query-prefix behavior against the pack's exact embedding profile before the IVF runtime searches it. ANNPack does not pretend that a browser's general-purpose Prompt API is an interoperable embedding model.

The first golden-path candidate is the 24.1M-parameter `mixedbread-ai/mxbai-embed-xsmall-v1`, pinned to an exact model revision and Transformers.js 3.8.1 q8/WASM runtime in [`default-embedding-profile.json`](spec/examples/default-embedding-profile.json). [`annpack-transformers.js`](web/annpack-transformers.js) constructs the matching browser adapter. It remains a candidate—not the blessed release default—until the real-corpus evaluation demonstrates acceptable retrieval quality and cold-load behavior.

The dependency-free custom element is the drop-in docs-search surface:

```html
<script type="module" src="/annpack/annpack-widget.js"></script>
<annpack-search src="/.well-known/knowledge.annpack" limit="5"></annpack-search>
```

It renders all untrusted pack text through DOM `textContent`, exposes styling parts and result/error events, and can be upgraded to hybrid mode by setting its `embeddingAdapter` property.

## Tests

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
python3 benches/benchmark.py --binary target/release/annpack --enforce
python3 benches/crawl_vs_pack.py --binary target/release/annpack --enforce
```

The default release gates use a generated 1,000-document corpus: pack size at most 90% of source, build under 1.5 seconds, and process-inclusive verification and search p95 under 25 ms on the reference development machine. Verification is sampled 25 times by default; the report also retains its first-run and median timings so a single scheduler outlier cannot silently redefine the gate. Thresholds are explicit CLI flags so slower CI hardware can report the same measurements without disguising a changed budget.

The crawl comparison measures actual bytes returned by a strict Range server and compares them with an explicit 50-page × 300 KB rendered-page model. It is deliberately labeled as a model; the benchmark does not disguise synthetic HTML as observed production traffic. The default gate demands at least 95% lower transfer and no more than eight range GETs.

Latency and size do not establish retrieval quality. [`evals/evaluate.py`](evals/evaluate.py) publishes lexical, vector, and hybrid macro recall@k, hit rate, and MRR from human-authored relevance judgments. The included two-query fixture tests only the harness and is not product evidence. Public quality claims require the pinned real corpus and 50–100 independently adjudicated queries described in [`evals/README.md`](evals/README.md).

Loopback HTTP tests may require permission to bind a local test server in sandboxed environments.

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
- [Public launch gates](spec/LAUNCH-GATES.md)
- [Core layering decision](spec/decisions/0001-core-and-extensions.md)
- [Browser embedding decision](spec/decisions/0002-browser-embedding-candidate.md)

## Status

This repository is an Apache-2.0-licensed candidate specification plus reference implementation. It is not an independently adopted Internet standard, has not yet passed the external launch gates, and does not count its own Python/Rust/JavaScript code as independent implementations. Removing `-draft` requires a second reader produced from the Core specification and golden corpus without importing the reference parser.
