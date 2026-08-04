import { readFile } from 'node:fs/promises';
import init, { inspect_pack, search_pack } from './pkg/annpack.js';

const [wasm, pack] = await Promise.all([
  readFile(new URL('./pkg/annpack_bg.wasm', import.meta.url)),
  readFile(new URL('../spec/test-vectors/minimal-v3.annpack', import.meta.url)),
]);
await init({ module_or_path: wasm });

const inspection = inspect_pack(pack);
if (inspection.manifest.name !== 'minimal-conformance') {
  throw new Error('WASM inspection returned the wrong pack');
}
const response = search_pack(pack, 'ANN-001', 3);
if (!response.results[0]?.text.includes('opened successfully')) {
  throw new Error('WASM search did not return the conformance passage');
}
console.log(JSON.stringify({
  wasm: true,
  root_hash: inspection.root_hash,
  result: response.results[0].passage_id,
}, null, 2));

