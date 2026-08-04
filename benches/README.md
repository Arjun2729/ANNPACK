# Benchmarks

The release source of truth is the benchmark gate executed by GitHub Actions on each pull request. There is intentionally no tracked `latest.json`: wall-clock CLI timings are machine- and scheduler-specific, and a copied local run quickly becomes misleading.

Run `python3 benches/benchmark.py --enforce` for the release gate, or add `--output <path>` to preserve a dated report. `benches/history/` contains explicitly labelled historical runs, including failures; they are evidence, not current guarantees.
