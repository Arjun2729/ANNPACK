import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { envVar } from './compat.mjs';

const execFileAsync = promisify(execFile);

export async function buildKnowledgePack({
  binary = envVar('BINARY') || 'adyar',
  source,
  output,
  name,
  version,
  baseUrl = null,
  sourceRevision = process.env.GITHUB_SHA || null,
  license = null,
  extra = [],
}) {
  if (!source || !output || !name || !version) {
    throw new TypeError('source, output, name, and version are required');
  }
  const args = [
    'build', String(source), '--output', String(output),
    '--name', name, '--version', version, '--json',
  ];
  if (baseUrl) args.push('--base-url', baseUrl);
  if (sourceRevision) args.push('--source-revision', sourceRevision);
  if (license) args.push('--license', license);
  args.push(...extra);
  const { stdout } = await execFileAsync(binary, args, { maxBuffer: 16 * 1024 * 1024 });
  return JSON.parse(stdout);
}

