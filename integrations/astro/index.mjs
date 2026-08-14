import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { buildKnowledgePack } from '../shared/build.mjs';

export default function adyarAstro(options = {}) {
  return {
    name: 'astro-adyar',
    hooks: {
      async 'astro:build:done'({ dir }) {
        await buildKnowledgePack({
          ...options,
          source: options.source || resolve('src/content/docs'),
          output: options.output || resolve(fileURLToPath(dir), '.well-known/knowledge.adyar'),
          name: options.name || 'documentation',
          version: options.version || process.env.DOCS_VERSION || 'current',
        });
      },
    },
  };
}

