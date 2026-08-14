// Compatibility shims for the ANNPack -> Adyar rename.
//
// Configuration written against the old name keeps working. This module is the
// only place that knows the legacy spelling, so closing the transition window
// is a single deletion.

const CANONICAL_PREFIX = 'ADYAR_';
const LEGACY_PREFIX = 'ANNPACK_';

const warned = new Set();

function warnOnce(message) {
  if (warned.has(message)) return;
  warned.add(message);
  console.warn(`warning: ${message}`);
}

// A variable set to the empty string counts as unset, so ADYAR_X="" cannot
// silently mask a populated ANNPACK_X mid-migration.
function nonempty(name) {
  const value = process.env[name];
  return value ? value : undefined;
}

/**
 * Read a configuration variable by suffix, e.g. `SIGNING_KEY`.
 *
 * ADYAR_* wins when both are set. A fallback to ANNPACK_* warns once, naming
 * the variable and never its value: these carry signing keys and registry
 * passwords, and a deprecation notice is not worth leaking one into CI logs.
 */
export function envVar(suffix) {
  const canonical = `${CANONICAL_PREFIX}${suffix}`;
  const preferred = nonempty(canonical);
  if (preferred) return preferred;

  const legacy = `${LEGACY_PREFIX}${suffix}`;
  const fallback = nonempty(legacy);
  if (fallback) {
    warnOnce(`${legacy} is deprecated; use ${canonical}`);
    return fallback;
  }
  return undefined;
}

/** The canonical name for a variable, for use in diagnostics. */
export function canonicalName(suffix) {
  return `${CANONICAL_PREFIX}${suffix}`;
}
