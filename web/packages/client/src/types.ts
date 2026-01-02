export type ManifestShard = {
  name: string;
  annpack: string;
  meta: string;
  n_vectors?: number;
};

export type ManifestV2 = {
  schema_version?: number;
  dim?: number;
  n_lists?: number;
  n_vectors?: number;
  shards: ManifestShard[];
};

export type DeltaEntry = {
  seq: number;
  annpack: string;
  meta: string;
  tombstones: string;
  base_sha256_annpack?: string;
  sha256_annpack?: string;
  sha256_meta?: string;
  sha256_tombstones?: string;
};

export type ManifestV3 = {
  schema_version: 3;
  base: {
    annpack: string;
    meta: string;
    sha256_annpack?: string;
    sha256_meta?: string;
  };
  deltas?: DeltaEntry[];
};

export type SearchRequest = {
  queryVector: number[] | Float32Array;
  topK: number;
  filters?: Record<string, string | number | boolean>;
};

export type SearchResult = {
  id: number;
  score: number;
  meta?: Record<string, unknown> | null;
  source?: string;
};

export type TelemetryEvent = {
  name: string;
  detail?: Record<string, unknown>;
};

export type TelemetryHook = (event: TelemetryEvent) => void;
