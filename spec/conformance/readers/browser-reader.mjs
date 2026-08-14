#!/usr/bin/env node
// Conformance adapter for the browser runtime (web/adyar-browser.js).
//
// The browser reader is a second implementation of tokenization, BM25 scoring
// and container parsing, written in a different language from rust/, and until
// this existed nothing forced the two to agree beyond a single hybrid-ranking
// check. They had already diverged once: when fusion changed in rust/, every
// browser smoke still passed while the two returned different hybrid orders.
//
// Running it through the same four-verb contract as any other implementation
// means the whole suite applies to it -- the tokenizer vectors, the exact
// IEEE-754 scores, the corruption corpus and the manifest compatibility cases.
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, '../../../web');

const { default: init, blake3_hex: blake3, inflate_zlib: inflate } = await import(`${web}/pkg/annpack.js`);
await init(await readFile(`${web}/pkg/annpack_bg.wasm`));
const { AdyarBrowser, tokenize, verifyReceipt } = await import(`${web}/adyar-browser.js`);

const [verb, ...args] = process.argv.slice(2);

async function openPack(path) {
  // `verifyAll` matters: the suite requires `open` to fail on a section-hash
  // mismatch, and section hashes are otherwise checked lazily per read.
  return AdyarBrowser.openBytes(await readFile(path), { blake3, inflate, verifyAll: true });
}

try {
  if (verb === 'tokenize') {
    console.log(JSON.stringify(tokenize(args[0])));
  } else if (verb === 'open') {
    await openPack(args[0]);
  } else if (verb === 'search') {
    const pack = await openPack(args[0]);
    const response = await pack.search(args[1], { mode: 'lexical', limit: 10 });
    console.log(JSON.stringify({
      results: response.results.map((hit) => ({
        passage_id: hit.passage_id,
        score: hit.score,
      })),
    }));
  } else if (verb === 'verify-receipt') {
    const receipt = JSON.parse(await readFile(args[0], 'utf8'));
    await verifyReceipt(receipt, { blake3, inflate });
  } else {
    process.stderr.write(`unknown verb: ${verb}\n`);
    process.exit(2);
  }
} catch (error) {
  process.stderr.write(`${error?.message ?? error}\n`);
  process.exit(1);
}
