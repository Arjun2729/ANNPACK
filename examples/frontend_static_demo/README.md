# Frontend Static Demo

This demo shows how to build a tiny pack and serve it from any static server, then point the UI at the manifest URL.

## Build a tiny pack (offline)

```bash
bash examples/frontend_static_demo/build_pack.sh
```

This creates `examples/frontend_static_demo/out/pack.manifest.json`.

## Serve from a static server

```bash
bash examples/frontend_static_demo/serve_static.sh
```

It prints a URL like:

```
http://127.0.0.1:9000/pack.manifest.json
```

## Point the UI at your pack

In the UI, set **Manifest URL** to the URL above and click **Boot**.

If you host on a CDN/S3, ensure:
- Range requests are supported.
- CORS allows `GET` + `Range` from your UI origin.
