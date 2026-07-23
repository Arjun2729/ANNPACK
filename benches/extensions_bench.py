#!/usr/bin/env python3
"""Benchmark the optional retrieval extensions (ANN-7/8/9/10).

Reports, per profile: build time, pack-size delta versus Core, and query p95.
Then serves a fat pack over HTTP byte ranges and counts how many range requests
touch an unused profile's bytes during a lexical-only search — demonstrating the
ANN-10 "unused profiles are never fetched" property under range serving.

Nothing here is a retrieval-quality measurement. No extension is enabled by
default, and no method is claimed to improve retrieval.
"""

import argparse
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading

from benchmark import generate_corpus, percentile_95, run
from crawl_vs_pack import RangeHandler


def build(binary, corpus, out, *extra):
    _, ms = run(
        [binary, "build", str(corpus), "--output", str(out),
         "--name", "ext-bench", "--version", "3.0.0",
         "--base-url", "https://benchmark.example", "--json", *extra]
    )
    return ms, out.stat().st_size


def craft_sidecars(binary, core_pack, directory):
    passages = json.loads(
        subprocess.run(
            [binary, "export-passages", str(core_pack)],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    ids = [p["id"] for p in passages]
    # Deterministic synthetic generation: every passage gets a handful of
    # generated question tokens and vocabulary weights. This measures format and
    # index cost, not model quality.
    raw_exp = {
        "generator": "bench", "model": "bench", "revision": "r1",
        "passages": [
            {"passage_id": i, "candidates": [
                {"text": "how do I recover this component", "score": 0.9},
                {"text": "what error does this component raise", "score": 0.8},
            ]} for i in ids
        ],
    }
    raw_splade = {
        "generator": "bench", "model": "bench", "revision": "r1",
        "vocabulary": {"id": "bert-base-uncased-wordpiece", "size": 30522,
                       "quantization": "linear-u16", "scale": 0.001},
        "passages": [
            {"passage_id": i, "weights": {"component": 0.9, "error": 0.7, "recover": 0.5}}
            for i in ids
        ],
    }
    exp_raw = directory / "raw-exp.json"
    splade_raw = directory / "raw-splade.json"
    exp_raw.write_text(json.dumps(raw_exp))
    splade_raw.write_text(json.dumps(raw_splade))
    exp_side = directory / "exp.sidecar.json"
    splade_side = directory / "splade.sidecar.json"
    subprocess.run([binary, "generate", "expansion", str(exp_raw),
                    "--output", str(exp_side), "--threshold", "0.5"],
                   check=True, capture_output=True)
    subprocess.run([binary, "generate", "splade", str(splade_raw),
                    "--output", str(splade_side)], check=True, capture_output=True)
    return exp_side, splade_side


def profile_ranges(binary, pack):
    info = json.loads(
        subprocess.run([binary, "inspect", str(pack)],
                       check=True, capture_output=True, text=True).stdout
    )
    profile_types = {"term_overlay", "anchor_set", "anchor_coordinates",
                     "vector_profile", "vector_data", "vector_index"}
    return [
        (s["offset"], s["offset"] + s["stored_length"])
        for s in info["sections"] if s["type"] in profile_types
    ]


def query_p95(binary, pack, documents, queries, *extra):
    latencies = []
    for index in range(queries):
        code = f"AP-{100 + (index % documents):04d}"
        _, ms = run([binary, "search", str(pack), code,
                     "--mode", "lexical", "--limit", "5", "--json", *extra])
        latencies.append(ms)
    return statistics.median(latencies), percentile_95(latencies)


def count_unused_fetches(binary, fat_pack, ranges):
    pack_bytes = fat_pack.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    server.pack_bytes = pack_bytes
    server.range_requests = 0
    server.transferred_bytes = 0
    server.counter_lock = threading.Lock()
    server.fetched_ranges = []
    # Wrap the handler to also record the exact fetched ranges.
    original = RangeHandler.do_GET

    def recording_get(self):
        value = self.headers.get("Range", "")
        if value.startswith("bytes="):
            start, end = value.removeprefix("bytes=").split("-", 1)
            self.server.fetched_ranges.append((int(start), int(end) + 1))
        original(self)

    RangeHandler.do_GET = recording_get
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/pack.annpack"
        subprocess.run([binary, "search", url, "recoverComponent",
                        "--mode", "lexical", "--limit", "5", "--json"],
                       check=True, capture_output=True)
    finally:
        RangeHandler.do_GET = original
        server.shutdown()
        thread.join()
    unused = 0
    for read_start, read_end in server.fetched_ranges:
        for section_start, section_end in ranges:
            if read_start < section_end and read_end > section_start:
                unused += 1
    return unused, len(server.fetched_ranges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/annpack")
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--output")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())

    with tempfile.TemporaryDirectory(prefix="annpack-ext-bench-") as directory:
        root = Path(directory)
        corpus = root / "docs"
        corpus.mkdir()
        generate_corpus(corpus, args.documents)

        core = root / "core.annpack"
        core_ms, core_bytes = build(binary, corpus, core)
        exp_side, splade_side = craft_sidecars(binary, core, root)

        exp = root / "exp.annpack"
        exp_ms, exp_bytes = build(binary, corpus, exp, "--expansion", str(exp_side))
        splade = root / "splade.annpack"
        splade_ms, splade_bytes = build(binary, corpus, splade, "--splade", str(splade_side))
        fat = root / "fat.annpack"
        fat_ms, fat_bytes = build(binary, corpus, fat,
                                  "--expansion", str(exp_side), "--splade", str(splade_side))

        lex_p50, lex_p95 = query_p95(binary, core, args.documents, args.queries)
        exp_p50, exp_p95 = query_p95(binary, exp, args.documents, args.queries,
                                     "--expansion-weight", "1.0")
        splade_p50, splade_p95 = query_p95(binary, splade, args.documents, args.queries,
                                           "--splade-weight", "1.0")

        unused, total = count_unused_fetches(binary, fat, profile_ranges(binary, fat))

        report = {
            "documents": args.documents,
            "profiles": {
                "core_lexical": {
                    "build_ms": core_ms, "pack_bytes": core_bytes,
                    "size_delta_vs_core": 0,
                    "query_p50_ms": lex_p50, "query_p95_ms": lex_p95,
                },
                "expansion": {
                    "build_ms": exp_ms, "pack_bytes": exp_bytes,
                    "size_delta_vs_core": exp_bytes - core_bytes,
                    "query_p50_ms": exp_p50, "query_p95_ms": exp_p95,
                },
                "splade": {
                    "build_ms": splade_ms, "pack_bytes": splade_bytes,
                    "size_delta_vs_core": splade_bytes - core_bytes,
                    "query_p50_ms": splade_p50, "query_p95_ms": splade_p95,
                },
                "fat_pack": {
                    "build_ms": fat_ms, "pack_bytes": fat_bytes,
                    "size_delta_vs_core": fat_bytes - core_bytes,
                },
            },
            "range_serving": {
                "total_range_requests_lexical_search": total,
                "range_requests_touching_unused_profiles": unused,
            },
            "gates": {
                "unused_profiles_never_fetched": {
                    "actual": unused, "limit": 0, "pass": unused == 0,
                },
            },
            "note": (
                "Latency includes CLI process startup. Nothing here measures "
                "retrieval quality; no extension is enabled by default."
            ),
        }

    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.enforce:
        failures = [name for name, gate in report["gates"].items() if not gate["pass"]]
        if failures:
            raise SystemExit(f"extension benchmark gates failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
