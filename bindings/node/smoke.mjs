import { Client } from './index.mjs';

const client = new Client({
  binary: process.env.ANNPACK_BINARY
    || new URL('../../target/release/adyar', import.meta.url).pathname,
});
const pack = new URL('../../spec/test-vectors/minimal-v3.annpack', import.meta.url).pathname;
const verification = client.verify(pack);
const response = client.search(pack, 'ANN-001', { mode: 'lexical' });
if (!verification.integrity_verified || !response.results[0]?.text.includes('opened successfully')) {
  throw new Error('Node binding smoke test failed');
}
console.log(JSON.stringify({ node_binding: true, root_hash: response.pack.root_hash }, null, 2));
