# Benchmarks

Run the lightweight benchmark harness (offline, synthetic data):

```bash
ANNPACK_OFFLINE=1 python benchmarks/run_bench.py --rows 2000 --lists 64
```

This writes a report to `docs/benchmarks/bench_report.md`.
