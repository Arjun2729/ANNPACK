// Compatibility shim. ANNPack was renamed to Adyar.
//
// This package exists so that `npm install @annpack/node` does not silently
// keep serving a build that stopped being maintained at the rename. It forwards
// everything to @adyar/node and warns once.
//
// The wire format did not change with the rename. Artifacts and receipts
// produced before it remain valid and verifiable; only the names moved.

import { AdyarError } from '@adyar/node';

export * from '@adyar/node';

/** Retained so existing `instanceof ANNPackError` checks still hold. */
export const ANNPackError = AdyarError;

console.warn(
  "warning: '@annpack/node' was renamed to '@adyar/node'; this shim forwards " +
    'to it and is the final release under the old name. Depend on @adyar/node ' +
    'directly.',
);
