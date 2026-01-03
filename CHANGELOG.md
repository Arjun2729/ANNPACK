# Changelog

## 0.1.5
- Deterministic tiny-dataset centroid fallback to avoid FAISS aborts
- Registry defaults hardened with required secrets, upload caps, and rate limits
- Overflow checks and safer allocations in ann_engine
- Security tooling (CodeQL, audits) and documentation updates

## 0.1.4
- Stage-all gate script and CI integration
- Packaging exclusions for web build artifacts
- Dev extras include build/twine for release checks

## 0.1.3
- PackSet lifecycle commands + canary tooling
- Web client + UI workspaces and registry skeleton
- Structured logging helpers and extended docs

## 0.1.2
- PEP 561 typing marker and expanded API/CLI docs
- Minimal web client skeleton and WASM build notes
- CI scripts for smoke + determinism checks
- Documentation improvements and release script
