#!/usr/bin/env python3
"""Reproducible ANNPack build/search benchmark using generated technical docs."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


def run(command):
    started = time.perf_counter()
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result, (time.perf_counter() - started) * 1000


def percentile_95(samples):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def generate_corpus(root: Path, documents: int):
    source_bytes = 0
    for index in range(documents):
        group = index % 32
        error = f"AP-{100 + index:04d}"
        text = f"""---
title: API component {index}
url: https://benchmark.example/v3/components/{index}
---
# Component {index}

Component {index} belongs to service group {group}. It processes deterministic benchmark requests.

## Error {error}

`{error}` means component {index} rejected a request in service group {group}. Inspect `component_{index}.log` and call `recoverComponent({index})`.

## Configuration

Use `ComponentOptions` with `serviceGroup: {group}` and `retryLimit: {index % 7}`. This passage contains stable filler text so compression and indexing measurements are comparable between runs.
"""
        path = root / f"component-{index:05d}.md"
        path.write_text(text, encoding="utf-8")
        source_bytes += len(text.encode())
    return source_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/adyar")
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--verify-runs", type=int, default=25)
    parser.add_argument("--output")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--max-pack-ratio", type=float, default=0.90)
    parser.add_argument("--max-build-ms", type=float, default=1500.0)
    parser.add_argument("--max-verify-ms", type=float, default=25.0)
    parser.add_argument("--max-query-p95-ms", type=float, default=25.0)
    args = parser.parse_args()
    if args.documents < 1:
        parser.error("--documents must be at least 1")
    if args.queries < 1:
        parser.error("--queries must be at least 1")
    if args.verify_runs < 2:
        parser.error("--verify-runs must be at least 2")
    binary = str(Path(args.binary).resolve())

    with tempfile.TemporaryDirectory(prefix="annpack-bench-") as directory:
        root = Path(directory)
        corpus = root / "docs"
        corpus.mkdir()
        source_bytes = generate_corpus(corpus, args.documents)
        pack = root / "benchmark.annpack"
        build, build_ms = run([
            binary, "build", str(corpus), "--output", str(pack),
            "--name", "benchmark-docs", "--version", "3.0.0", "--json",
        ])
        build_report = json.loads(build.stdout)

        verify_latencies = []
        for _ in range(args.verify_runs):
            _, elapsed = run([binary, "verify", str(pack), "--json"])
            verify_latencies.append(elapsed)
        verify_p95 = percentile_95(verify_latencies)
        latencies = []
        first_response = None
        for index in range(args.queries):
            code = f"AP-{100 + (index % args.documents):04d}"
            result, elapsed = run([
                binary, "search", str(pack), code,
                "--mode", "lexical", "--limit", "5", "--json",
            ])
            latencies.append(elapsed)
            if first_response is None:
                first_response = json.loads(result.stdout)

        p95 = percentile_95(latencies)
        pack_ratio = pack.stat().st_size / source_bytes
        gates = {
            "pack_ratio": {
                "actual": pack_ratio,
                "limit": args.max_pack_ratio,
                "pass": pack_ratio <= args.max_pack_ratio,
            },
            "build_ms": {
                "actual": build_ms,
                "limit": args.max_build_ms,
                "pass": build_ms <= args.max_build_ms,
            },
            "verify_p95_ms_process_inclusive": {
                "actual": verify_p95,
                "limit": args.max_verify_ms,
                "pass": verify_p95 <= args.max_verify_ms,
            },
            "query_p95_ms_process_inclusive": {
                "actual": p95,
                "limit": args.max_query_p95_ms,
                "pass": p95 <= args.max_query_p95_ms,
            },
            "retrieval_correctness": {
                "actual": "rejected a request" in first_response["results"][0]["text"],
                "limit": True,
                "pass": "rejected a request" in first_response["results"][0]["text"],
            },
        }
        report = {
            "documents": args.documents,
            "passages": build_report["passages"],
            "terms": build_report["terms"],
            "source_bytes": source_bytes,
            "pack_bytes": pack.stat().st_size,
            "pack_to_source_ratio": pack_ratio,
            "build_ms": build_ms,
            "verify_runs": args.verify_runs,
            "verify_first_ms_process_inclusive": verify_latencies[0],
            "verify_p50_ms_process_inclusive": statistics.median(verify_latencies),
            "verify_p95_ms_process_inclusive": verify_p95,
            "query_count": args.queries,
            "query_p50_ms_process_inclusive": statistics.median(latencies),
            "query_p95_ms_process_inclusive": p95,
            "first_result_correct": "rejected a request" in first_response["results"][0]["text"],
            "binary": binary,
            "note": "Verify and query latency include CLI process startup and pack open; p95 gates reduce single-sample scheduler noise without hiding sustained regressions.",
            "gates": gates,
        }

    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.enforce:
        failures = [name for name, gate in report["gates"].items() if not gate["pass"]]
        if failures:
            raise SystemExit(f"benchmark gates failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
