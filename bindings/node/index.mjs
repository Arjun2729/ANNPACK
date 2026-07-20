import { spawn, spawnSync } from 'node:child_process';

export class ANNPackError extends Error {}

export class Client {
  constructor({ binary = process.env.ANNPACK_BINARY || 'annpack' } = {}) {
    this.binary = binary;
  }

  inspect(pack) {
    return this.#json(['inspect', String(pack), '--json']);
  }

  verify(pack, { publicKey = null } = {}) {
    const args = ['verify', String(pack), '--json'];
    if (publicKey) args.push('--public-key', String(publicKey));
    return this.#json(args);
  }

  search(pack, query, {
    limit = 10,
    mode = 'hybrid',
    queryVector = null,
    vectorProfile = null,
    vectorProbes = 4,
    debug = false,
  } = {}) {
    const args = [
      'search', String(pack), query, '--limit', String(limit), '--mode', mode, '--json',
    ];
    if (queryVector) args.push('--query-vector', String(queryVector));
    if (vectorProfile) args.push('--vector-profile', vectorProfile);
    args.push('--vector-probes', String(vectorProbes));
    if (debug) args.push('--debug');
    return this.#json(args);
  }

  build(source, output, { name, version, extra = [] }) {
    return this.#json([
      'build', String(source), '--output', String(output),
      '--name', name, '--version', version, ...extra, '--json',
    ]);
  }

  push(pack, reference, { username = null } = {}) {
    const args = ['push', String(pack), reference, '--json'];
    if (username) args.push('--username', username);
    return this.#json(args);
  }

  pull(reference, output, { username = null, force = false } = {}) {
    const args = ['pull', reference, '--output', String(output), '--json'];
    if (username) args.push('--username', username);
    if (force) args.push('--force');
    return this.#json(args);
  }

  mcp(pack, options = {}) {
    return spawn(this.binary, ['mcp', String(pack)], {
      stdio: ['pipe', 'pipe', 'inherit'],
      ...options,
    });
  }

  #json(args) {
    const result = spawnSync(this.binary, args, { encoding: 'utf8' });
    if (result.error) throw new ANNPackError(result.error.message);
    if (result.status !== 0) throw new ANNPackError((result.stderr || result.stdout).trim());
    try {
      return JSON.parse(result.stdout);
    } catch (error) {
      throw new ANNPackError(`Native runtime returned invalid JSON: ${error.message}`);
    }
  }
}
