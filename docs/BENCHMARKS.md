# Benchmarks

ANNPACK benchmarks are designed to be reproducible and offline-friendly.

Run the benchmark harness:

```bash
ANNPACK_OFFLINE=1 python benchmarks/run_bench.py --rows 2000 --lists 64
```

This writes a report to `docs/benchmarks/bench_report.md`.

Notes:
- Offline mode uses deterministic dummy embeddings, so results are stable.
- If embedding deps are installed and `ANNPACK_OFFLINE` is not set, the build cost will be higher.
