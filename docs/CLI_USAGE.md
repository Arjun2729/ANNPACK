# CLI Usage

## Build

```bash
annpack build --input tiny_docs.csv --text-col text --id-col id --output ./out/pack --lists 4
```

## Serve

```bash
annpack serve ./out/pack --port 8000
```
The UI is served from packaged assets and the pack is mounted at `/pack/`.

## Smoke test

```bash
annpack smoke ./out/pack --port 8000
```

## Offline mode

```bash
export ANNPACK_OFFLINE=1
annpack build --input tiny_docs.csv --text-col text --id-col id --output ./out/pack --lists 4
```
