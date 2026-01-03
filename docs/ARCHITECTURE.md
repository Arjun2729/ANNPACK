# Architecture

ANNPack consists of four layers:

1. **Builder (Python)**
   - `annpack build` and `annpack.api.build_pack` create `.annpack` and `.meta.jsonl` from CSV/Parquet.
   - IVF clustering + float16 storage for vectors.

2. **Reader (Python)**
   - `annpack.reader.ANNPackIndex` memory-maps the binary file and supports search.

3. **Serving (CLI)**
   - `annpack serve <packdir>` hosts a packaged UI and mounts packs at `/pack/` with CORS enabled.

4. **PackSets (Stage 3)**
   - `annpack.packset.PackSet` merges a base pack with delta packs (append-only updates + tombstones).

5. **Registry (optional)**
   - `registry/` provides a local FastAPI service to upload and serve packs with Range support.

6. **Web client + UI**
   - `web/packages/client` provides fetch/caching utilities for packs.
   - `web/apps/ui` provides a React inspector and query UI.

## Files

A single pack directory contains:
- `pack.annpack` — binary IVF index
- `pack.meta.jsonl` — metadata rows keyed by id
- `pack.manifest.json` — shard list and schema metadata

A PackSet directory contains:
- `pack.manifest.json` (schema v3 root)
- `base/` (standard pack)
- `deltas/0001.delta/` (delta packs with tombstones)

## Format

The binary format is stable and documented in `docs/FORMAT.md`.

## Security posture

- **Signed/verified:** optional CLI signing (`annpack sign`) and verification (`annpack verify`) cover manifest integrity and pack file hashes.
- **Not signed by default:** packs are unsigned unless you explicitly generate signatures and distribute them.
- **Threat model:** a malicious pack file can cause incorrect results or resource exhaustion. Range-based serving reduces blast radius but does not guarantee safety. Treat packs as untrusted inputs and validate with `annpack verify` before use.
- **Registry defaults:** production deployments must run with `REGISTRY_DEV_MODE=0`, a strong `REGISTRY_JWT_SECRET`, and rate limits enabled.
