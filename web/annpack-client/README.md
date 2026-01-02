# @annpack/client

Minimal TypeScript helpers for fetching ANNPack manifests and range bytes.

## Install

```bash
npm install @annpack/client
```

## Usage

```ts
import { createClient } from "@annpack/client";

const client = createClient("http://127.0.0.1:8000");
const manifest = await client.fetchManifest("/pack/pack.manifest.json");
const firstShard = manifest.shards[0];
const bytes = await client.fetchRange(`/pack/${firstShard.annpack}`, 0, 71);
console.log(bytes.byteLength);
```
