# ANNPack UI

Minimal React UI for inspecting packs and running vector queries.

## Dev

```bash
cd web
npm install
npm run build -w @annpack/client
npm run dev -w annpack-ui
```

The UI expects a pack mounted at `/pack/pack.manifest.json` when served by `annpack serve`.
