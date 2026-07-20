export function annResultSize(Module) {
  return Module.ccall('ann_result_size_bytes', 'number', [], []);
}

export async function wasmSearch(Module, queryF32, k) {
  const resultSize = annResultSize(Module);
  const queryBytes = queryF32.length * 4;
  const queryPtr = Module._malloc(queryBytes);
  Module.HEAPF32.set(queryF32, queryPtr >> 2);

  const outBytes = k * resultSize;
  const outPtr = Module._malloc(outBytes);

  const count = await Module.ccall(
    'ann_search',
    'number',
    ['number', 'number', 'number', 'number'],
    [0, queryPtr, outPtr, k],
    { async: true }
  );

  const results = [];
  const view = new DataView(Module.HEAPU8.buffer, outPtr, outBytes);
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
