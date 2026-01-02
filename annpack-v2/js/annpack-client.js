export function annResultSize(Module) {
  return Module.ccall('ann_result_size_bytes', 'number', [], []);
}

export async function search(Module, ctx, queryF32, k) {
  if (!ctx) return [];
  const resultSize = annResultSize(Module);
  const kVal = Number.isFinite(k) && k > 0 ? Math.min(Math.floor(k), 100) : 10;
  const queryBytes = queryF32.length * 4;
  const queryPtr = Module._malloc(queryBytes);
  Module.HEAPF32.set(queryF32, queryPtr >> 2);

  const outBytes = kVal * resultSize;
  const outPtr = Module._malloc(outBytes);

  const count = await Module.ccall(
    'ann_search',
    'number',
    ['number', 'number', 'number', 'number'],
    [ctx, queryPtr, outPtr, kVal],
    { async: true }
  );

  const view = new DataView(Module.HEAPU8.buffer, outPtr, Math.max(0, count) * resultSize);
  const results = [];
  for (let i = 0; i < count; i++) {
    const base = i * resultSize;
    const id = Number(view.getBigUint64(base, true));
    const score = view.getFloat32(base + 8, true);
    results.push({ id, score });
  }
  Module._free(queryPtr);
  Module._free(outPtr);
  return results;
}
