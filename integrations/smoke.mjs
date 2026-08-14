import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { buildKnowledgePack } from './shared/build.mjs';

const directory = await mkdtemp(join(tmpdir(), 'adyar-integration-'));
try {
  const report = await buildKnowledgePack({
    binary: resolve('target/release/adyar'),
    source: resolve('fixtures/docs-v1'),
    output: join(directory, '.well-known/knowledge.adyar'),
    name: 'integration-docs',
    version: '1.0.0',
  });
  if (report.documents !== 3 || report.passages < 3) {
    throw new Error('Framework integration produced an invalid pack report');
  }
  console.log(JSON.stringify({ framework_hook: true, ...report }, null, 2));
} finally {
  await rm(directory, { recursive: true, force: true });
}

