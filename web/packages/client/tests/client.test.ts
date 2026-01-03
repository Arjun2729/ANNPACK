import { describe, expect, it, vi } from 'vitest';
import { MemoryCache } from '../src/cache';
import { RangeFetcher } from '../src/http';
import { openPack } from '../src/pack';

const makeResponse = (status: number, body: Uint8Array, headers?: Record<string, string>) => {
  const map = headers ?? {};
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (key: string) => map[key] ?? null },
    arrayBuffer: async () => body.buffer,
  } as Response;
};

describe('MemoryCache', () => {
  it('stores and retrieves entries', async () => {
    const cache = new MemoryCache();
    await cache.set({ key: 'k', bytes: new Uint8Array([1, 2]), etag: 'x' });
    const entry = await cache.get('k');
    expect(entry?.etag).toBe('x');
    expect(entry?.bytes[0]).toBe(1);
  });
});

describe('RangeFetcher', () => {
  it('uses If-None-Match and cache on 304', async () => {
    const cache = new MemoryCache();
    await cache.set({ key: 'u:0-1', bytes: new Uint8Array([1, 2]), etag: 'etag1' });

    const fetchMock = vi.fn().mockResolvedValue(makeResponse(304, new Uint8Array(), {}));
    // @ts-expect-error override global fetch
    globalThis.fetch = fetchMock;

    const rf = new RangeFetcher({ cache });
    const resp = await rf.fetchRange('u', 0, 1);
    expect(resp.bytes[0]).toBe(1);
    expect(fetchMock.mock.calls[0][1]?.headers?.['If-None-Match']).toBe('etag1');
  });
});

describe('openPack', () => {
  it('parses manifest', async () => {
    const manifest = { schema_version: 2, shards: [{ name: 's', annpack: 'a', meta: 'm' }] };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => manifest,
    });
    // @ts-expect-error override global fetch
    globalThis.fetch = fetchMock;

    const pack = await openPack('http://example/pack.manifest.json');
    expect(pack.manifest.shards.length).toBe(1);
  });
});
