# Benchmarks

Run the lightweight benchmark harness (offline, synthetic data):

```bash
ANNPACK_OFFLINE=1 python tools/bench/bench.py --rows 2000 --dim 16 --lists 64
```

This writes a report to `docs/benchmarks/bench_report.md`.
