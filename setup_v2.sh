#!/usr/bin/env bash
set -euo pipefail

# Create dirs
mkdir -p annpack-v2/{tools,src,include,js,docs,examples}

########################################
# docs/FORMAT.md
########################################
cat > annpack-v2/docs/FORMAT.md <<'EOF'
# ANNPack File Format
Static IVF index for L2-normalized vectors.

Header (72 bytes, little-endian):
- uint64 magic           @0   must be 0x00000000504E4E41 ("ANNP")
- uint32 version         @8   1
- uint32 endian          @12  1 (little)
- uint32 header_size     @16  72
- uint32 dim             @20  vector dimension
- uint32 metric          @24  1 = dot-product/cosine
- uint32 n_lists         @28  IVF lists
- uint32 n_vectors       @32  total vectors
- uint64 offset_table    @36  absolute offset of list offset table
- padding/reserved       @44..71 zero

Centroids:
- start @ header_size
- n_lists * dim float32 (little), row-major.

Lists (for each i in 0..n_lists-1, at offset_table[i].offset):
- uint32 count
- uint64 ids[count]
- float16 vecs[count][dim] (little), row-major.
- total length stored in offset_table[i].length

Offset table:
- at offset_table_pos
- n_lists entries of:
  struct { uint64_t offset; uint64_t length; }

Semantics:
- Vectors are L2-normalized before write.
- Search: dot-product coarse on centroids (top PROBE), fine scan within those lists, keep top-K.
- Endianness is little-endian; header_size/version allow future evolution.
EOF

########################################
# include/ann_format.h
########################################
cat > annpack-v2/include/ann_format.h <<'EOF'
#ifndef ANN_FORMAT_H
#define ANN_FORMAT_H

#include <stdint.h>

#define ANN_MAGIC 0x504E4E41ULL
#define ANN_VERSION 1
#define ANN_ENDIAN_LITTLE 1
#define ANN_METRIC_DOT 1
#define ANN_HEADER_SIZE 72

#pragma pack(push, 1)
typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t endian;
    uint32_t header_size;
    uint32_t dim;
    uint32_t metric;
    uint32_t n_lists;
    uint32_t n_vectors;
    uint64_t offset_table_pos;
    uint8_t reserved[28];
} ann_header_t;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint64_t offset;
    uint64_t length;
} ann_list_meta_t;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint64_t id;   // 8
    float score;   // 4
} ann_result_t;    // packed: 12 bytes
#pragma pack(pop)

#endif
EOF

########################################
# include/ann_io.h
########################################
cat > annpack-v2/include/ann_io.h <<'EOF'
#ifndef ANN_IO_H
#define ANN_IO_H

#include <stdint.h>

typedef struct io_req {
    uint64_t offset;
    uint64_t len;
    void *dst;
    uint64_t result_len;
} io_req_t;

typedef struct io_reader {
    void *ctx;
    void (*read_batch)(void *ctx, io_req_t *reqs, int n_reqs);
    void (*destroy)(void *ctx);
} io_reader_t;

io_reader_t *io_reader_http(const char *url);

#endif
EOF

########################################
# src/ann_engine.c
########################################
cat > annpack-v2/src/ann_engine.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ann_format.h"
#include "ann_io.h"

#define PROBE 8

typedef struct {
    ann_header_t header;
    float *centroids;
    uint64_t *list_offsets;
    uint64_t *list_lengths;
    io_reader_t *reader;
} ann_index_t;

static ann_index_t *G_IDX = NULL;

static inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 0x0001;
    uint32_t e = (h >> 10) & 0x001f;
    uint32_t m = h & 0x03ff;
    if (e == 0) return (m == 0) ? (s ? -0.0f : 0.0f) : (s ? -1.0f : 1.0f) * powf(2.0f, -14.0f) * ((float)m / 1024.0f);
    if (e == 31) return (m == 0) ? (s ? -INFINITY : INFINITY) : NAN;
    return (s ? -1.0f : 1.0f) * powf(2.0f, (float)e - 15.0f) * (1.0f + ((float)m / 1024.0f));
}

static void insert_topk(ann_result_t *res, int *count, int k, uint64_t id, float score) {
    if (*count < k || score > res[*count - 1].score) {
        int pos = (*count < k) ? *count : k - 1;
        while (pos > 0 && score > res[pos - 1].score) pos--;
        if (*count < k) (*count)++;
        for (int m = (*count) - 1; m > pos; m--) res[m] = res[m - 1];
        res[pos].id = id;
        res[pos].score = score;
    }
}

int ann_load_index(const char *url) {
    if (G_IDX) return 1;
    io_reader_t *r = io_reader_http(url);
    if (!r) return 0;

    uint8_t head_buf[256];
    io_req_t req = { .offset = 0, .len = sizeof(head_buf), .dst = head_buf, .result_len = 0 };
    r->read_batch(r->ctx, &req, 1);
    if (req.result_len < sizeof(ann_header_t)) return 0;

    ann_header_t *h = (ann_header_t *)head_buf;
    if (h->magic != ANN_MAGIC || h->version != ANN_VERSION || h->endian != ANN_ENDIAN_LITTLE || h->header_size != ANN_HEADER_SIZE) return 0;

    ann_index_t *idx = (ann_index_t *)calloc(1, sizeof(ann_index_t));
    idx->reader = r;
    memcpy(&idx->header, h, sizeof(ann_header_t));

    size_t cent_sz = (size_t)h->n_lists * h->dim * sizeof(float);
    idx->centroids = (float *)malloc(cent_sz);
    size_t table_sz = (size_t)h->n_lists * sizeof(ann_list_meta_t);
    ann_list_meta_t *table = (ann_list_meta_t *)malloc(table_sz);

    io_req_t reqs[2] = {
        { .offset = h->header_size, .len = cent_sz, .dst = idx->centroids, .result_len = 0 },
        { .offset = h->offset_table_pos, .len = table_sz, .dst = table, .result_len = 0 }
    };
    r->read_batch(r->ctx, reqs, 2);

    idx->list_offsets = (uint64_t *)malloc(h->n_lists * sizeof(uint64_t));
    idx->list_lengths = (uint64_t *)malloc(h->n_lists * sizeof(uint64_t));
    for (uint32_t i = 0; i < h->n_lists; i++) {
        idx->list_offsets[i] = table[i].offset;
        idx->list_lengths[i] = table[i].length;
    }
    free(table);
    G_IDX = idx;
    return 1;
}

int ann_search(void *ctx_unused, const float *query, ann_result_t *out_results, int k) {
    if (!G_IDX || !query || !out_results || k <= 0) return 0;
    ann_index_t *idx = G_IDX;
    uint32_t dim = idx->header.dim;

    float best_scores[PROBE];
    int best_ids[PROBE];
    for (int i = 0; i < PROBE; i++) { best_scores[i] = -1e9f; best_ids[i] = -1; }
    for (uint32_t c = 0; c < idx->header.n_lists; c++) {
        float dot = 0.0f;
        const float *cent = idx->centroids + c * dim;
        for (uint32_t j = 0; j < dim; j++) dot += query[j] * cent[j];
        if (dot > best_scores[PROBE - 1]) {
            int pos = PROBE - 1;
            while (pos > 0 && dot > best_scores[pos - 1]) pos--;
            for (int m = PROBE - 1; m > pos; m--) { best_scores[m] = best_scores[m - 1]; best_ids[m] = best_ids[m - 1]; }
            best_scores[pos] = dot; best_ids[pos] = (int)c;
        }
    }

    int top_count = 0;
    for (int bi = 0; bi < PROBE; bi++) {
        int list_id = best_ids[bi];
        if (list_id < 0) continue;
        uint64_t off = idx->list_offsets[list_id];
        uint64_t len = idx->list_lengths[list_id];
        if (len < 4) continue;

        uint8_t *buf = (uint8_t *)malloc(len);
        io_req_t req = { .offset = off, .len = len, .dst = buf, .result_len = 0 };
        idx->reader->read_batch(idx->reader->ctx, &req, 1);
        uint32_t count = *(uint32_t *)buf;
        size_t needed = 4 + (size_t)count * 8 + (size_t)count * dim * 2;
        if (needed > len || count == 0) { free(buf); continue; }

        uint64_t *ids = (uint64_t *)(buf + 4);
        uint16_t *vecs = (uint16_t *)(buf + 4 + (size_t)count * 8);

        for (uint32_t i = 0; i < count; i++) {
            float dot = 0.0f;
            uint16_t *v = vecs + i * dim;
            for (uint32_t j = 0; j < dim; j++) dot += query[j] * half_to_float(v[j]);
            insert_topk(out_results, &top_count, k, ids[i], dot);
        }
        free(buf);
    }
    return top_count;
}
EOF

########################################
# js/annpack-client.js
########################################
cat > annpack-v2/js/annpack-client.js <<'EOF'
export function annResultSize(Module) {
  return Module.ccall('ann_result_size_bytes', 'number', [], []);
}

export async function search(Module, queryF32, k) {
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

  const view = new DataView(Module.HEAPU8.buffer, outPtr, count * resultSize);
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
EOF

########################################
# tools/builder.py
########################################
cat > annpack-v2/tools/builder.py <<'EOF'
import argparse
import json
import os
import struct
import numpy as np
import polars as pl
import faiss
from sentence_transformers import SentenceTransformer

def parse_args():
    p = argparse.ArgumentParser(description="ANNPack builder (generic CSV/Parquet/JSON)")
    p.add_argument("--input", required=True, help="Input file (.parquet/.csv/.json)")
    p.add_argument("--text-col", required=True, help="Column to embed")
    p.add_argument("--id-col", help="ID column (int64). Auto-generate if missing.")
    p.add_argument("--meta-cols", nargs="*", default=[], help="Columns to include in metadata (JSONL)")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--output", default="generic_index", help="Output prefix")
    p.add_argument("--lists", type=int, default=4096, help="IVF clusters")
    p.add_argument("--batch-size", type=int, default=512)
    return p.parse_args()

def load_df(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".parq"):
        return pl.read_parquet(path)
    if ext == ".csv":
        return pl.read_csv(path)
    if ext == ".json":
        return pl.read_json(path)
    raise ValueError(f"Unsupported extension: {ext}")

def embed_texts(texts, model_name, batch_size):
    model = SentenceTransformer(model_name, device="mps" if SentenceTransformer._flavor == "pytorch" else None)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors

def train_ivf(vectors, n_lists):
    dim = vectors.shape[1]
    kmeans = faiss.Kmeans(dim, n_lists, niter=20, verbose=True)
    kmeans.train(vectors)
    _, list_ids = kmeans.index.search(vectors, 1)
    return kmeans.centroids.astype(np.float32), list_ids.flatten()

def write_annpack(prefix, dim, n_lists, vectors, ids, centroids, list_ids):
    fn = f"{prefix}.annpack"
    print(f"[write] {fn}")
    with open(fn, "wb") as f:
        magic = 0x504E4E41
        header_size = 72
        version = 1
        endian = 1
        metric = 1
        f.write(struct.pack("<QIIIIIIIQ", magic, version, endian, header_size, dim, metric, n_lists, vectors.shape[0], 0))
        f.write(b"\x00" * (header_size - f.tell()))
        f.write(centroids.tobytes())

        order = np.argsort(list_ids)
        vecs_sorted = vectors[order]
        ids_sorted = ids[order]
        counts = np.bincount(list_ids, minlength=n_lists)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))

        list_offsets = []
        list_lengths = []

        for i in range(n_lists):
            count = int(counts[i])
            start = int(starts[i])
            offset = f.tell()
            f.write(struct.pack("<I", count))
            if count > 0:
                sl = slice(start, start + count)
                f.write(ids_sorted[sl].tobytes())
                f.write(vecs_sorted[sl].astype(np.float16).tobytes())
            list_offsets.append(offset)
            list_lengths.append(f.tell() - offset)

        table_pos = f.tell()
        for off, length in zip(list_offsets, list_lengths):
            f.write(struct.pack("<QQ", off, length))

        f.seek(36)
        f.write(struct.pack("<Q", table_pos))

    print(f"[done] {fn} ({os.path.getsize(fn)/(1024*1024):.2f} MB)")

def write_meta(prefix, df, ids, text_col, meta_cols):
    fn = f"{prefix}.meta.jsonl"
    print(f"[meta] {fn}")
    cols = meta_cols + [text_col]
    with open(fn, "w", encoding="utf-8") as w:
        for i, row in zip(ids.tolist(), df.select(cols).to_dicts()):
            row["id"] = int(i)
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    df = load_df(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"text col '{args.text_col}' not found. cols={df.columns}")
    ids = df[args.id_col].to_numpy().astype(np.int64, copy=False) if args.id_col else np.arange(df.height, dtype=np.int64)
    texts = [str(t) for t in df[args.text_col].to_list()]

    print("[embed] generating embeddings...")
    vectors = embed_texts(texts, args.model, args.batch_size)
    if vectors.shape[0] != len(ids):
        raise SystemExit("ids length mismatch vectors")
    centroids, list_ids = train_ivf(vectors, args.lists)
    dim = vectors.shape[1]
    write_annpack(args.output, dim, args.lists, vectors, ids, centroids, list_ids)
    write_meta(args.output, df, ids, args.text_col, args.meta_cols)
    print("[done] ANNPack build complete")

if __name__ == "__main__":
    main()
EOF

########################################
# index.html
########################################
cat > annpack-v2/index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ANNPack v2 – Semantic Search</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }
    .card { background: #1c1c1c; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #2a2a2a; }
    button { background: #2563eb; color: #fff; border: none; padding: 0.65rem 1.2rem; border-radius: 6px; cursor: pointer; }
    button:disabled { background: #444; cursor: not-allowed; }
    input[type="text"] { width: 100%; padding: 0.75rem; border-radius: 6px; border: 1px solid #333; background: #0f0f0f; color: #eee; }
    #log { font-family: monospace; color: #4ade80; white-space: pre-wrap; max-height: 220px; overflow-y: auto; background: #0f0f0f; padding: 0.75rem; border-radius: 6px; border: 1px solid #222; }
  </style>
  <script type="module">
    import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0';
    env.allowLocalModels = false;
    window._annpack_pipeline = pipeline;
  </script>
</head>
<body>
  <h1>ANNPack v2 – Semantic Search</h1>
  <p>Generic ANNPack index served via HTTP Range; embeddings run in-browser with MiniLM.</p>

  <div class="card">
    <h3>1) Boot</h3>
    <p>Status: <span id="status" style="color:yellow">Waiting for WASM...</span></p>
    <button id="btnBoot" disabled onclick="doBoot()">Boot Engine</button>
  </div>

  <div class="card">
    <h3>2) Search</h3>
    <input id="searchInput" type="text" placeholder="Type a query" disabled />
    <div style="margin-top:0.75rem;">
      <button id="btnSearch" disabled onclick="doSearch()">Search</button>
    </div>
    <div id="result" style="margin-top: 1rem; font-family: monospace; line-height: 1.5;"></div>
  </div>

  <div class="card">
    <h3>Logs</h3>
    <div id="log"></div>
  </div>

  <script type="module">
    import { search as wasmSearch, annResultSize } from './js/annpack-client.js';

    const INDEX_FILE = "generic_index.annpack";
    const META_FILE  = "generic_index.meta.jsonl";
    const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
    const DEFAULT_K = 10;

    let RESULT_SIZE = 0;
    let embedder = null;
    let metadata = new Map();
    let Module = {
      print: (t) => log(t),
      printErr: (t) => log("[WASM Error] " + t),
      onRuntimeInitialized: () => {
        log("[JS] WASM Runtime Initialized.");
        document.getElementById('status').innerText = "WASM Ready. Click 'Boot'.";
        document.getElementById('btnBoot').disabled = false;
      }
    };
    window.Module = Module;
  </script>
  <script src="annpack.js"></script>
  <script type="module">
    import { search as wasmSearch, annResultSize } from './js/annpack-client.js';

    function log(text) {
      const el = document.getElementById('log');
      el.innerText += text + "\n";
      el.scrollTop = el.scrollHeight;
    }

    async function doBoot() {
      const btn = document.getElementById('btnBoot');
      btn.disabled = true;
      document.getElementById('status').innerText = "Booting...";

      log(`[JS] Loading Transformers.js pipeline (${MODEL_NAME})...`);
      embedder = await window._annpack_pipeline('feature-extraction', MODEL_NAME);
      log("[JS] AI Model Ready.");

      log(`[JS] Loading index: ${INDEX_FILE} ...`);
      const ret = await Module.ccall('ann_load_index', 'number', ['string'], [INDEX_FILE], { async: true });
      if (ret !== 1) { log("[JS] ann_load_index failed."); document.getElementById('status').innerText = "Load failed"; return; }

      RESULT_SIZE = annResultSize(Module);
      log(`[JS] ann_result_size_bytes = ${RESULT_SIZE}`);

      log(`[JS] Loading metadata from ${META_FILE} ...`);
      const resp = await fetch(META_FILE);
      if (resp.ok) {
        const text = await resp.text();
        text.split(/\r?\n/).forEach(line => {
          if (!line.trim()) return;
          try {
            const obj = JSON.parse(line);
            metadata.set(Number(obj.id), obj);
          } catch (e) { console.error("Bad meta line", line, e); }
        });
        log(`[JS] Metadata entries: ${metadata.size}`);
      } else {
        log(`[JS] Failed to fetch metadata: ${resp.status}`);
      }

      document.getElementById('status').innerText = "System Ready.";
      document.getElementById('searchInput').disabled = false;
      document.getElementById('btnSearch').disabled = false;
    }

    async function doSearch() {
      if (!embedder || !RESULT_SIZE) { log("[JS] Boot first."); return; }
      const qtext = document.getElementById('searchInput').value.trim();
      if (!qtext) return;
      const resDiv = document.getElementById('result');
      resDiv.innerText = "Searching...";

      const t0 = performance.now();
      const out = await embedder(qtext, { pooling: 'mean', normalize: true });
      const vec = out.data;
      const t1 = performance.now();

      const results = await wasmSearch(Module, vec, DEFAULT_K);
      const t2 = performance.now();

      const html = results.map((r, idx) => {
        const meta = metadata.get(r.id) || {};
        const title = meta.title || meta.text || `(id=${r.id})`;
        const url = meta.url || null;
        const snippet = (meta.text || "").slice(0, 240).replace(/\s+/g, " ");
        const titleHtml = url ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>` : title;
        return `<div style="margin-bottom:0.75rem;"><span style="color:#facc15;">#${idx+1}</span> <span style="color:#4ade80;">[score=${r.score.toFixed(4)}]</span><br><strong>${titleHtml}</strong><br><small>${snippet}...</small></div>`;
      }).join("");

      resDiv.innerHTML = `<div>Found ${results.length} results | Embed: ${(t1 - t0).toFixed(2)} ms | ANN: ${(t2 - t1).toFixed(2)} ms | Total: ${(t2 - t0).toFixed(2)} ms</div><br>${html}`;
      log(`[JS] Search done. Total ${(t2 - t0).toFixed(2)} ms.`);
    }

    window.doBoot = doBoot;
    window.doSearch = doSearch;
  </script>
</body>
</html>
EOF

########################################
# server.py
########################################
cat > annpack-v2/server.py <<'EOF'
import http.server, socketserver, os

class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None
        fs = os.fstat(f.fileno())
        size = fs.st_size
        range_header = self.headers.get("Range")
        if range_header:
            try:
                _, rng = range_header.split("=")
                start_s, end_s = rng.split("-")
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
                if start >= size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return None
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", end - start + 1)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", ctype)
                self.end_headers()
                f.seek(start)
                self.copyfile(f, self.wfile, length=end - start + 1)
                f.close()
                return None
            except Exception:
                pass
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return f

    def copyfile(self, source, outputfile, length=None):
        if length is None:
            return super().copyfile(source, outputfile)
        bufsize = 64 * 1024
        remaining = length
        while remaining > 0:
            chunk = source.read(min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

if __name__ == "__main__":
    with socketserver.TCPServer(("", 8080), RangeHandler) as httpd:
        print("Serving on http://localhost:8080")
        httpd.serve_forever()
EOF

########################################
# build.sh
########################################
cat > annpack-v2/build.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EMCC=${EMCC:-emcc}
EMCACHE=${EMCACHE:-.emcache}
mkdir -p "$EMCACHE"
EM_CACHE="$EMCACHE" $EMCC -O3 src/ann_engine.c io_wasm.c -Iinclude -o annpack.js \
  -s ASYNCIFY \
  -s FETCH=1 \
  -s "EXPORTED_FUNCTIONS=['_malloc','_free','_ann_load_index','_ann_result_size_bytes','_ann_search']" \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32','HEAPU8']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s NO_EXIT_RUNTIME=1
EOF
chmod +x annpack-v2/build.sh

echo "ANNPack v2 scaffold created under annpack-v2/"
