#!/usr/bin/env node
import { resolve } from 'node:path';
import { buildKnowledgePack } from '../shared/build.mjs';

const name = process.env.ANNPACK_NAME || 'documentation';
const version = process.env.ANNPACK_VERSION || process.env.DOCS_VERSION || 'current';
const report = await buildKnowledgePack({
  source: process.env.ANNPACK_SOURCE || resolve('.'),
  output: process.env.ANNPACK_OUTPUT || resolve('.well-known/knowledge.annpack'),
  name,
  version,
  baseUrl: process.env.ANNPACK_BASE_URL || null,
  license: process.env.ANNPACK_LICENSE || null,
});
console.log(JSON.stringify(report, null, 2));

