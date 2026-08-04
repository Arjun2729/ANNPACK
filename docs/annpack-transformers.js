import { createEmbeddingAdapter } from './annpack-browser.js';

export const DEFAULT_EMBEDDING_PROFILE = Object.freeze({
  id: 'ann-1-mxbai-xsmall-v1-q8-onnx',
  model: 'mixedbread-ai/mxbai-embed-xsmall-v1',
  revision: 'e6ac24e5d6efb8782b59de1647b3ececb4ece94e',
  dimensions: 384,
  dtype: 'float32',
  pooling: 'mean',
  normalized: true,
  query_prefix: null,
  document_prefix: null,
  runtime: Object.freeze({
    library: '@huggingface/transformers',
    library_version: '3.8.1',
    weights_dtype: 'q8',
    max_tokens: 4096,
  }),
});

export async function createDefaultEmbeddingAdapter(pipeline, { device = 'wasm' } = {}) {
  if (typeof pipeline !== 'function') {
    throw new TypeError('Pass pipeline from @huggingface/transformers 3.8.1');
  }
  const extractor = await pipeline(
    'feature-extraction',
    DEFAULT_EMBEDDING_PROFILE.model,
    {
      revision: DEFAULT_EMBEDDING_PROFILE.revision,
      dtype: DEFAULT_EMBEDDING_PROFILE.runtime.weights_dtype,
      device,
    },
  );
  return createEmbeddingAdapter(
    async (text) => extractor(text, {
      pooling: DEFAULT_EMBEDDING_PROFILE.pooling,
      normalize: DEFAULT_EMBEDDING_PROFILE.normalized,
    }),
    DEFAULT_EMBEDDING_PROFILE,
  );
}
