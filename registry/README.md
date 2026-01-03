# ANNPack Registry (local)

A minimal local registry service for versioned pack bundles.

## Run (development only)

```bash
export REGISTRY_STORAGE=registry_storage
export REGISTRY_DEV_MODE=1
export REGISTRY_JWT_SECRET=dev-only-secret-change-me
uvicorn registry.app:app --host 0.0.0.0 --port 8080
```

## Production config (example)

```bash
export REGISTRY_STORAGE=/var/lib/annpack/registry
export REGISTRY_DEV_MODE=0
export REGISTRY_JWT_SECRET="change-this-to-a-strong-secret"
export REGISTRY_JWT_AUD="annpack-registry"
export REGISTRY_JWT_ISS="annpack-registry"
export REGISTRY_RATE_LIMIT_RPS=5
export REGISTRY_RATE_LIMIT_BURST=20
export REGISTRY_MAX_UPLOAD_MB=100
uvicorn registry.app:app --host 0.0.0.0 --port 8080
```

Notes:
- `REGISTRY_DEV_MODE` defaults to off; production requires `REGISTRY_JWT_SECRET`.
- In dev mode without a secret, the registry logs a warning and uses an insecure dev secret.

## Dev token (development only)

```bash
curl -s http://localhost:8080/auth/dev-token
```

## Upload

```bash
curl -H "Authorization: Bearer <token>" \
  -F "bundle=@pack.zip" \
  "http://localhost:8080/orgs/acme/projects/search/packs?version=v1"
```
