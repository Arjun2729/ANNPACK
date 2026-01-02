# WASM Build

This repo includes a minimal Emscripten build script for a standalone WASM demo.

## Build

```bash
bash tools/build_wasm.sh
```

Outputs are written to `web/wasm_dist/` (ignored by git).

## Hosting Packs

ANNPack files are static and can be hosted on any HTTP server that supports Range requests.

Checklist for hosting:
- Enable `Accept-Ranges` and Range responses (206)
- Serve `pack.manifest.json`, `*.annpack`, and `*.meta.jsonl` from the same origin
- If serving from a CDN (S3/Cloudflare), ensure Range requests are not stripped

A basic demo server mounts packs at `/pack/` using:

```bash
annpack serve ./out/tiny --port 8000
```

Then access:
- `http://127.0.0.1:8000/` for UI
- `http://127.0.0.1:8000/pack/pack.manifest.json` for the pack manifest

## Web UI

The React UI lives in `web/apps/ui` and uses `@annpack/client` to fetch manifests and headers.\n+The UI expects a pack mounted at `/pack/` when served by `annpack serve`.
