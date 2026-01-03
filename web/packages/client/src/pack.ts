import { RangeFetcher } from './http';
import { Cache } from './cache';
import {
  DeltaEntry,
  ManifestV2,
  ManifestV3,
  SearchRequest,
  SearchResult,
  TelemetryHook,
} from './types';

export type PackOptions = {
  fetcher?: RangeFetcher;
  cache?: Cache;
  telemetry?: TelemetryHook;
  verify?: boolean;
};

export class Pack {
  readonly manifest: ManifestV2;
  readonly baseUrl: string;
  private fetcher: RangeFetcher;
  private telemetry?: TelemetryHook;

  constructor(manifest: ManifestV2, baseUrl: string, opts: PackOptions = {}) {
    this.manifest = manifest;
    this.baseUrl = baseUrl;
    this.fetcher =
      opts.fetcher ?? new RangeFetcher({ cache: opts.cache, telemetry: opts.telemetry });
    this.telemetry = opts.telemetry;
  }

  async search(req: SearchRequest): Promise<SearchResult[]> {
    this.telemetry?.({ name: 'search', detail: { topK: req.topK } });
    return [];
  }

  async readHeader(): Promise<Record<string, number>> {
    const shard = this.manifest.shards[0];
    const url = new URL(shard.annpack, this.baseUrl).toString();
    const resp = await this.fetcher.fetchRange(url, 0, 71);
    const buf = resp.bytes.buffer.slice(
      resp.bytes.byteOffset,
      resp.bytes.byteOffset + resp.bytes.byteLength,
    );
    const view = new DataView(buf);
    return {
      magic: view.getUint32(0, true),
      version: view.getUint32(8, true),
      endian: view.getUint32(12, true),
      headerSize: view.getUint32(16, true),
      dim: view.getUint32(20, true),
      metric: view.getUint32(24, true),
      nLists: view.getUint32(28, true),
      nVectors: view.getUint32(32, true),
    };
  }
}

export class PackSet {
  readonly manifest: ManifestV3;
  readonly baseUrl: string;
  readonly deltas: DeltaEntry[];
  private basePack: Pack;
  private telemetry?: TelemetryHook;

  constructor(
    manifest: ManifestV3,
    baseUrl: string,
    basePack: Pack,
    deltas: DeltaEntry[],
    telemetry?: TelemetryHook,
  ) {
    this.manifest = manifest;
    this.baseUrl = baseUrl;
    this.basePack = basePack;
    this.deltas = deltas;
    this.telemetry = telemetry;
  }

  async search(req: SearchRequest): Promise<SearchResult[]> {
    this.telemetry?.({ name: 'packset_search', detail: { topK: req.topK } });
    return this.basePack.search(req);
  }

  async readHeader(): Promise<Record<string, number>> {
    return this.basePack.readHeader();
  }
}

export async function openPack(source: string | File, opts: PackOptions = {}): Promise<Pack> {
  const manifest =
    typeof source === 'string' ? await fetchJson<ManifestV2>(source) : await parseFile(source);
  if (!manifest.shards || manifest.shards.length === 0) {
    throw new Error('manifest has no shards');
  }
  const baseUrl = typeof source === 'string' ? new URL('.', source).toString() : 'file://';
  const pack = new Pack(manifest, baseUrl, opts);
  if (opts.verify) {
    await verifyManifest(manifest, baseUrl);
  }
  return pack;
}

export async function openPackSet(
  baseUrl: string,
  deltaUrls: string[] = [],
  opts: PackOptions = {},
): Promise<PackSet> {
  const maybeManifest = await fetchJson<ManifestV3 | ManifestV2>(baseUrl);
  if ((maybeManifest as ManifestV3).schema_version === 3) {
    const manifest = maybeManifest as ManifestV3;
    const baseDir = new URL('.', baseUrl).toString();
    const baseManifestUrl = new URL(
      'pack.manifest.json',
      new URL(manifest.base.annpack, baseDir),
    ).toString();
    const basePack = await openPack(baseManifestUrl, opts);
    const deltas = (manifest.deltas ?? []).slice().sort((a, b) => a.seq - b.seq);
    if (opts.verify) {
      await verifyPackSet(manifest, baseDir);
    }
    return new PackSet(manifest, baseDir, basePack, deltas, opts.telemetry);
  }

  const baseManifest = maybeManifest as ManifestV2;
  if (opts.verify) {
    await verifyManifest(baseManifest, new URL('.', baseUrl).toString());
  }
  const basePack = new Pack(baseManifest, new URL('.', baseUrl).toString(), opts);
  const synthetic: ManifestV3 = {
    schema_version: 3,
    base: {
      annpack: basePack.manifest.shards[0].annpack,
      meta: basePack.manifest.shards[0].meta,
    },
    deltas: deltaUrls.map((url, idx) => ({
      seq: idx + 1,
      annpack: url,
      meta: url,
      tombstones: url,
    })),
  };
  return new PackSet(
    synthetic,
    new URL('.', baseUrl).toString(),
    basePack,
    synthetic.deltas ?? [],
    opts.telemetry,
  );
}

async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  }
  return (await resp.json()) as T;
}

async function parseFile(file: File): Promise<ManifestV2> {
  const text = await file.text();
  return JSON.parse(text) as ManifestV2;
}

async function verifyManifest(manifest: ManifestV2, baseUrl: string): Promise<void> {
  for (const shard of manifest.shards) {
    await verifyFile(new URL(shard.annpack, baseUrl).toString());
    await verifyFile(new URL(shard.meta, baseUrl).toString());
  }
}

async function verifyPackSet(manifest: ManifestV3, baseUrl: string): Promise<void> {
  await verifyFile(
    new URL(manifest.base.annpack, baseUrl).toString(),
    manifest.base.sha256_annpack,
  );
  await verifyFile(new URL(manifest.base.meta, baseUrl).toString(), manifest.base.sha256_meta);
  for (const delta of manifest.deltas ?? []) {
    await verifyFile(new URL(delta.annpack, baseUrl).toString(), delta.sha256_annpack);
    await verifyFile(new URL(delta.meta, baseUrl).toString(), delta.sha256_meta);
    await verifyFile(new URL(delta.tombstones, baseUrl).toString(), delta.sha256_tombstones);
  }
}

async function verifyFile(url: string, expected?: string): Promise<void> {
  if (!expected) {
    return;
  }
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Integrity check failed: ${url} (${resp.status})`);
  }
  const buf = await resp.arrayBuffer();
  const hash = await sha256Hex(buf);
  if (hash !== expected) {
    throw new Error(`Hash mismatch for ${url}`);
  }
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error('WebCrypto is unavailable; integrity checks require crypto.subtle');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', buf);
  return toHex(new Uint8Array(digest));
}

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
