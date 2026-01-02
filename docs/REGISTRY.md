# Registry

ANNPack includes a local FastAPI-based registry for versioned packs and Range delivery.

## Features
- Immutable pack versions per org/project
- JWT auth with admin/dev/viewer roles
- Range support for pack files
- Local filesystem storage backend

## Run

```bash
export REGISTRY_STORAGE=registry_storage
export REGISTRY_DEV_MODE=1
uvicorn registry.app:app --host 0.0.0.0 --port 8080
```

See `registry/README.md` for upload and token examples.
