export interface ManifestShard {
  name: string;
  annpack: string;
  meta: string;
  n_vectors?: number;
}

export interface ManifestV2 {
  schema_version?: number;
  dim?: number;
  n_lists?: number;
  n_vectors?: number;
  shards: ManifestShard[];
}

function joinUrl(baseUrl: string, path: string): string {
  if (!baseUrl.endsWith("/")) {
    baseUrl += "/";
  }
  if (path.startsWith("/")) {
    path = path.slice(1);
  }
  return baseUrl + path;
}

export async function fetchManifest(
  baseUrl: string,
  manifestPath = "/pack/pack.manifest.json"
): Promise<ManifestV2> {
  const url = joinUrl(baseUrl, manifestPath);
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`manifest fetch failed (${resp.status}): ${url}`);
  }
  return (await resp.json()) as ManifestV2;
}

export async function fetchRange(url: string, start: number, end: number): Promise<ArrayBuffer> {
  const resp = await fetch(url, {
    headers: {
      Range: `bytes=${start}-${end}`
    }
  });
  if (!resp.ok && resp.status !== 206) {
    throw new Error(`range fetch failed (${resp.status}): ${url}`);
  }
  return await resp.arrayBuffer();
}

export function createClient(baseUrl: string) {
  return {
    fetchManifest: (manifestPath?: string) => fetchManifest(baseUrl, manifestPath),
    fetchRange: (path: string, start: number, end: number) =>
      fetchRange(joinUrl(baseUrl, path), start, end)
  };
}
