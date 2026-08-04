import { resolve } from 'node:path';
import { buildKnowledgePack } from '../shared/build.mjs';

export function annpackVitePress(options = {}) {
  return {
    name: 'vitepress-annpack',
    apply: 'build',
    async closeBundle() {
      await buildKnowledgePack({
        ...options,
        source: options.source || resolve('docs'),
        output: options.output || resolve('docs/.vitepress/dist/.well-known/knowledge.annpack'),
        name: options.name || 'documentation',
        version: options.version || process.env.DOCS_VERSION || 'current',
      });
    },
  };
}

