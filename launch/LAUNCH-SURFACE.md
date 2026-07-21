# ANNPack Launch Surface

**Status:** Preparation only — do not post externally until all launch gates are closed.

---

## Show HN title and post

**Title:** ANNPack – search a signed knowledge artifact with a handful of Range GETs, no server

**Body:**

ANNPack packages documentation as a signed, content-addressed binary artifact. You
can search it from a browser using a small number of HTTP Range requests — no search
server, no database, no full download. (8 Range GETs on our 4 KB conformance pack;
12 on the 860 KB FastAPI pack — scales with corpus size, not query complexity.)

The interesting bit: every result carries a cryptographic evidence envelope. It
tells you not just "what passage matched" but exactly which immutable artifact
version and passage hash produced the answer. Results are reproducible. You can
hand someone the pack root and a passage hash and they can independently verify
both the content and that the right version was searched.

Technical summary:
- Deterministic build: same source → same bytes → same BLAKE3 root, every time
- Range-based reading: browser fetches header + directory + posting lists + passages only
- BLAKE3 integrity per section and root; Ed25519 signatures optional
- BM25 lexical + IVF-flat vector + RRF hybrid, all in the binary
- Verified offline install: fetch once, kill the server, query from memory
- MCP server for Gemini/Claude with exact provenance in every tool response

Demo: [link to CDN artifact page — pending deployment]
Source: https://github.com/[your-org]/annpack (Apache 2.0)
Spec: ANNPack Core v1.0-draft (candidate; independent implementation pending)

Happy to answer questions about the format design, retrieval quality numbers, or
the browser architecture.

---

## 30–60 second deterministic demo script

```
1. Open https://[CDN-URL]/?pack=[PACK-URL]&root=[ROOT-HEX]&q=AP-104
   (live demo URL — pack hosted on GCS with CORS, immutable headers)

2. Show the Network panel:
   - 1 HEAD + 11 GET requests (FastAPI 860 KB pack, lexical query, top-5 results)
   - Exact byte ranges visible in each response
   - Total transfer: ~460 KB cold open; ~2–5 KB per subsequent query in the same session

3. Show a result:
   - Pack root hash visible in the result card
   - Passage hash verified against stored content
   - Evidence: pack=fastapi-docs@0.115.12, passage_hash=...

4. Click "Install offline":
   - Fetches remaining sections into memory
   - Status: "Installed. Disconnect your network."
   - Disable WiFi in OS settings.

5. Search again:
   - Zero HTTP requests
   - Same result, same passage hash
   - Network panel: empty

6. Re-enable network. That's it.
```

---

## Technical launch article outline

### ANNPack: accountable retrieval from a static file

**Lead:** The question RAG systems can't answer honestly is "exactly which version of
which knowledge produced this answer?" ANNPack can.

**Section 1 — The problem with mutable indexes**
- Hosted vector databases and search APIs are opaque about which knowledge revision
  produced a retrieval result
- Reproducibility requires knowing the exact passages, not just the query
- ANNPack's answer: make the artifact the unit of knowledge, not the index

**Section 2 — What a pack is**
- Content-addressed binary: BLAKE3 root over all sections
- Sectioned container: manifest, passages, BM25 index, optional vector index
- Every section independently verified before use
- Evidence envelope: schema, pack_root, source_revision, passage_id, passage_hash

**Section 3 — Range-based retrieval**
- Header (128 bytes) → directory → posting lists → passages
- Only the matching posting lists and passages are fetched
- 12 Range GETs to open and answer the first query on the 860 KB FastAPI pack (141 docs, 1864 passages); 8 on the 4 KB conformance test pack — count scales with passage block count, not query complexity
- ETag stability enforced; range-ignoring server rejected

**Section 4 — Retrieval quality**
[Tables from eval — pending human adjudication]
- BM25 recall@5, hit-rate@5, MRR@5
- Vector recall@5 (mxbai-embed-xsmall-v1, q8/WASM)
- Hybrid recall@5

**Section 5 — Browser offline installation**
- Verified full install into memory
- WebCrypto Ed25519 signature check
- Zero HTTP requests after install

**Section 6 — MCP integration**
- knowledge_pack_info, knowledge_search, knowledge_get_passage
- Every tool response includes the full evidence envelope
- One command to configure Gemini CLI

**Section 7 — Status and roadmap**
- Apache 2.0, candidate format, reference implementation
- Independent reader pending (clear criterion stated)
- Not calling it a standard yet

---

## Architecture diagram description

```
Publisher
  |
  | annpack build docs/ → vendor-docs.annpack
  |   BLAKE3 root: abc123...
  |
  | annpack sign --key publisher.key → vendor-docs.signed.annpack
  |   Same root, Signature section added
  |
  v
CDN (GCS) → /.well-known/knowledge.annpack
              Cache-Control: public, max-age=31536000, immutable
              Content-Type: application/vnd.annpack.v3
              ETag: "abc123"

          ←→  Browser (annpack-browser.js + WASM)
              HEAD /knowledge.annpack         → Content-Length, ETag
              GET bytes=0-127                 → Header (128 B)
              GET bytes=512-1023              → Directory
              GET bytes=1024-2047             → Manifest
              GET bytes=4096-5119             → Passage index
              GET bytes=8192-8703             → Lexical dictionary
              GET bytes=9000-9200             → Posting list: "AP-104"
              GET bytes=10000-10300           → Passage block 3
              GET bytes=10500-10700           → Passage block 7
              ─────────────────────────────────────────────────────
              Total: 8–12 Range GETs depending on corpus size
              (~460 KB cold open on 860 KB/141-doc FastAPI pack; ~2–5 KB per follow-up query)

          ←→  CLI / MCP server
              Reads same artifact, same verification

          ←→  OCI registry (GHCR)
              ghcr.io/publisher/knowledge/docs:1.0.0
              Manifest digest: sha256:...
              Pack root annotation: abc123...
```

---

## FAQ

**Why not SQLite/Datasette?**
SQLite doesn't hash its content. You can't prove which version of a SQLite database
answered a query or verify that it hasn't been modified. ANNPack binds every result
to a BLAKE3 root over all content.

**Why not PMTiles?**
PMTiles is optimized for tile-based geographic data with a different access pattern.
ANNPack is designed specifically for textual knowledge retrieval with passage-level
provenance and a BM25+vector+hybrid index baked in.

**Why not a hosted vector database?**
Hosted vector DBs require a running server, ongoing costs, and have no proof of
which knowledge version answered a query. ANNPack is a static artifact — it works
from a CDN, works offline, and makes every result independently verifiable.

**Why not llms.txt?**
llms.txt describes what to crawl. ANNPack delivers the pre-processed, verified,
range-queryable form. They're complementary; ANNPack can publish an llms.txt bridge.

**Is this a web standard?**
No. ANNPack Core v1.0-draft is a candidate format with one reference implementation
(Rust). It will not be called an adopted standard until a second independent reader
passes the conformance suite. See `spec/CORE-v1.0-draft.md`.

**Does Google endorse this?**
No. ANNPack can ingest Google's Open Knowledge Format (OKF), which is publicly
available under Apache 2.0. Using OKF content does not imply Google's endorsement
of ANNPack.

---

## Status language (safe to publish)

ANNPack is an Apache 2.0-licensed candidate format and reference implementation.
The specification is ANNPack Core v1.0-draft. It is not an independently adopted
internet standard or protocol. The `-draft` designation will not be removed until
a second Core reader produced from the specification independently passes the
conformance suite. Public launch gates and their evidence are tracked in
`spec/LAUNCH-GATES.md`.

---

## Vendor outreach drafts (10)

All drafts reference actual packs of the vendor's own docs. Pack must be built and
verified before sending. Do not send until GHCR catalog and retrieval quality gates
are closed.

**Template (fill in specifics per vendor):**

Subject: Built a verified, offline-searchable pack of your docs

Hi [Name],

I built an ANNPack artifact from [Vendor]'s documentation at commit [HASH].
The result is a 860 KB signed binary your users can search from a browser with a
handful of HTTP Range requests — no server, no query API, no full download.
Open it once (~460 KB), answer every follow-up query with ~2–5 KB.

Every search result carries a cryptographic evidence envelope: pack root, passage hash,
source revision, and canonical URL. A CI job can verify any agent answer against
the exact passage that produced it.

I'd love to show you what it looks like for [product-specific query]. The pack,
build command, and root hash are public at [GHCR reference].

Happy to walk through the spec or retrieval numbers.

— [Your name]

**Target vendors (pending real pack builds):**
1. Next.js / Vercel (docs: MIT, rich API surface)
2. Tailwind CSS (MIT)
3. FastAPI (MIT) — pack already built: c7147550...
4. PostgreSQL (PostgreSQL License, permissive)
5. Astro (MIT)
6. Docusaurus (MIT)
7. React (MIT)
8. Pydantic (MIT)
9. Rust std (MIT/Apache)
10. Kubernetes (Apache 2.0)

**Note:** Verify redistribution basis for each project independently before building
public catalog entries.

---

## Security and limitation disclosure (for any public post)

- ANNPack Core v1.0-draft is a candidate format, not an adopted standard
- The reference implementation is Rust; no independent second reader yet
- Fuzz campaign: [duration] per target, [N] total executions, [N] crashes
- Retrieval quality: BM25/vector/hybrid recall@5 on FastAPI docs (see eval table)
- Browser model (mxbai-embed-xsmall-v1 q8/WASM): [cold download time, warm inference] — evaluation pending
- Transfer reduction percentages are not safe claims: 98.4% retired; 64% cold figure compares to uncompressed text (vs gzip-compressed source, pack costs more cold). Safe claim: "open once ~460 KB, ~2–5 KB per subsequent query in the same session."
- Signatures prove cryptographic validity; publisher identity trust requires an external policy
- Policy metadata does not enforce payment or DRM after plaintext access
