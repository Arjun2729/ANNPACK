# ANNPack Registry (local)

A minimal local registry service for versioned pack bundles.

## Run

```bash
export REGISTRY_STORAGE=registry_storage
export REGISTRY_DEV_MODE=1
uvicorn registry.app:app --host 0.0.0.0 --port 8080
```

## Dev token

```bash
curl -s http://localhost:8080/auth/dev-token
```

## Upload

```bash
curl -H "Authorization: Bearer <token>" \
  -F "bundle=@pack.zip" \
  "http://localhost:8080/orgs/acme/projects/search/packs?version=v1"
```
