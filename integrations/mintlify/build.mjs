#!/usr/bin/env node
import { resolve } from 'node:path';
import { buildKnowledgePack } from '../shared/build.mjs';
import { envVar } from '../shared/compat.mjs';

const name = envVar('NAME') || 'documentation';
const version = envVar('VERSION') || process.env.DOCS_VERSION || 'current';
const report = await buildKnowledgePack({
  source: envVar('SOURCE') || resolve('.'),
  output: envVar('OUTPUT') || resolve('.well-known/knowledge.adyar'),
  name,
  version,
  baseUrl: envVar('BASE_URL') || null,
  license: envVar('LICENSE') || null,
});
console.log(JSON.stringify(report, null, 2));
