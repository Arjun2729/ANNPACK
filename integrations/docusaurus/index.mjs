import { join } from 'node:path';
import { buildKnowledgePack } from '../shared/build.mjs';

export default function annpackDocusaurusPlugin(context, options = {}) {
  return {
    name: 'docusaurus-annpack',
    async postBuild({ outDir }) {
      const source = options.source || join(context.siteDir, 'docs');
      const output = options.output || join(outDir, '.well-known', 'knowledge.annpack');
      await buildKnowledgePack({
        ...options,
        source,
        output,
        name: options.name || context.siteConfig.title,
        version: options.version || process.env.DOCS_VERSION || 'current',
        baseUrl: options.baseUrl || context.siteConfig.url,
      });
    },
  };
}

