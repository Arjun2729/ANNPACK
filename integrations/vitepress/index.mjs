import { resolve } from 'node:path';
import { buildKnowledgePack } from '../shared/build.mjs';

export function adyarVitePress(options = {}) {
  return {
    name: 'vitepress-adyar',
    apply: 'build',
    async closeBundle() {
      await buildKnowledgePack({
        ...options,
        source: options.source || resolve('docs'),
        output: options.output || resolve('docs/.vitepress/dist/.well-known/knowledge.adyar'),
        name: options.name || 'documentation',
        version: options.version || process.env.DOCS_VERSION || 'current',
      });
    },
  };
}

