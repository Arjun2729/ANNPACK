import {
  DEFAULT_EMBEDDING_PROFILE,
  createDefaultEmbeddingAdapter,
} from './adyar-transformers.js';

let loaded = false;
const adapter = await createDefaultEmbeddingAdapter(async (task, model, options) => {
  if (task !== 'feature-extraction'
    || model !== DEFAULT_EMBEDDING_PROFILE.model
    || options.revision !== DEFAULT_EMBEDDING_PROFILE.revision
    || options.dtype !== 'q8'
    || options.device !== 'wasm') {
    throw new Error('Default adapter did not pin its complete load descriptor');
  }
  loaded = true;
  return async (_text, inference) => {
    if (inference.pooling !== 'mean' || inference.normalize !== true) {
      throw new Error('Default adapter did not pin pooling and normalization');
    }
    return { tolist: () => [new Array(384).fill(0)] };
  };
});

const vector = await adapter.embedQuery('how do I revalidate cached data', DEFAULT_EMBEDDING_PROFILE);
if (!loaded || vector.length !== 384) throw new Error('Default embedding adapter smoke failed');
console.log(JSON.stringify({ default_embedding_adapter: true, dimensions: vector.length }));
