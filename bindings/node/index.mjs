import { spawn, spawnSync } from 'node:child_process';

export class AdyarError extends Error {}

/** Canonical variable naming the CLI to drive. */
export const BINARY_ENV = 'ADYAR_BINARY';
/** The name this variable carried when the project was called ANNPack. */
export const LEGACY_BINARY_ENV = 'ANNPACK_BINARY';

const warned = new Set();

function warnOnce(message) {
  if (warned.has(message)) return;
  warned.add(message);
  console.warn(`warning: ${message}`);
}

// Node has no `which`, so resolution is a probe: ENOENT means the name is not
// on PATH. Any other outcome means it ran, including a nonzero exit.
function onPath(name) {
  const probe = spawnSync(name, ['--version'], { stdio: 'ignore' });
  return !(probe.error && probe.error.code === 'ENOENT');
}

/**
 * Locate the CLI.
 *
 * Order: ADYAR_BINARY, the legacy ANNPACK_BINARY, `adyar` on PATH, then the
 * legacy `annpack` on PATH. Either legacy hit warns once.
 */
export function discoverBinary() {
  const configured = process.env[BINARY_ENV];
  if (configured) return configured;

  const legacy = process.env[LEGACY_BINARY_ENV];
  if (legacy) {
    warnOnce(`${LEGACY_BINARY_ENV} is deprecated; use ${BINARY_ENV}`);
    return legacy;
  }

  if (onPath('adyar')) return 'adyar';

  if (onPath('annpack')) {
    warnOnce("the 'annpack' binary is deprecated; install the 'adyar' CLI");
    return 'annpack';
  }

  // Nothing found. Return the canonical name so the eventual spawn failure
  // names the binary the user is expected to install.
  return 'adyar';
}

export class Client {
  // An explicitly supplied path is used exactly as given.
  constructor({ binary = discoverBinary() } = {}) {
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
    if (result.error) throw new AdyarError(result.error.message);
    if (result.status !== 0) throw new AdyarError((result.stderr || result.stdout).trim());
    try {
      return JSON.parse(result.stdout);
    } catch (error) {
      throw new AdyarError(`Native runtime returned invalid JSON: ${error.message}`);
    }
  }
}
