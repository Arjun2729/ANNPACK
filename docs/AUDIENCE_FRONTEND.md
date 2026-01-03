# Audience: Frontend Builders

## Day‑to‑day pain
- “I need search in the browser without a backend.”
- “CDN hosting is easy, but search requires a DB.”
- “WASM demos break because assets and CORS are flaky.”

## ANNPACK flow (mapped to pain)
1) Build a pack offline (tiny or medium).
2) Host it on static storage with Range enabled.
3) Load from `/pack/` in the UI or JS client.

## Copy‑paste flow
```bash
pip install annpack
ANNPACK_OFFLINE=1 annpack build --input tiny_docs.csv --text-col text --output ./out/tiny --lists 4
annpack serve ./out/tiny --port 8000
```
Open: `http://127.0.0.1:8000/`

## Deployment patterns
- **CDN**: S3/R2 + CloudFront/Cloudflare + Range + CORS.
- **Local**: `annpack serve` with `/pack/` mount.
- **PackSet**: base + deltas for app updates.

## Troubleshooting
- **CORS/Range**: verify `Accept-Ranges` and `Content-Range`.
- **Manifest 404**: ensure `/pack/pack.manifest.json` exists.
- **Cache**: hard reload when updating packs.
