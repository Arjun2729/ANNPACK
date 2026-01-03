import { Cache, CacheEntry } from './cache';
import { TelemetryHook } from './types';

export type RangeResponse = {
  bytes: Uint8Array;
  etag?: string;
  status: number;
};

export type RangeFetcherOptions = {
  retries?: number;
  backoffMs?: number;
  cache?: Cache;
  telemetry?: TelemetryHook;
};

export class RangeFetcher {
  private retries: number;
  private backoffMs: number;
  private cache?: Cache;
  private telemetry?: TelemetryHook;

  constructor(opts: RangeFetcherOptions = {}) {
    this.retries = opts.retries ?? 3;
    this.backoffMs = opts.backoffMs ?? 200;
    this.cache = opts.cache;
    this.telemetry = opts.telemetry;
  }

  async fetchRange(url: string, start: number, end: number): Promise<RangeResponse> {
    const key = `${url}:${start}-${end}`;
    const cached = this.cache ? await this.cache.get(key) : undefined;
    const etag = cached?.etag;

    for (let attempt = 0; attempt <= this.retries; attempt += 1) {
      try {
        const headers: Record<string, string> = {
          Range: `bytes=${start}-${end}`,
        };
        if (etag) {
          headers['If-None-Match'] = etag;
        }

        this.telemetry?.({ name: 'range_request', detail: { url, start, end, attempt } });
        const resp = await fetch(url, { headers });
        if (resp.status === 304 && cached) {
          return { bytes: cached.bytes, etag: cached.etag, status: resp.status };
        }
        if (!(resp.status === 206 || resp.status === 200)) {
          throw new Error(`Range fetch failed: ${resp.status}`);
        }
        const buf = new Uint8Array(await resp.arrayBuffer());
        const respEtag = resp.headers.get('ETag') ?? undefined;
        const entry: CacheEntry = { key, bytes: buf, etag: respEtag };
        if (this.cache) {
          await this.cache.set(entry);
        }
        return { bytes: buf, etag: respEtag, status: resp.status };
      } catch (err) {
        this.telemetry?.({ name: 'range_error', detail: { url, attempt, message: String(err) } });
        if (attempt >= this.retries) {
          throw err;
        }
        await sleep(this.backoffMs * (attempt + 1));
      }
    }
    throw new Error('Range fetch failed after retries');
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
