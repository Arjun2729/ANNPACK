# ANNPack v2 Milestones

## 2025-12-17 — Fidelity Gate Exact Pass (FiQA)
- Command: `python tools/fidelity_gate.py --manifest data/fiqa/fiqa.manifest.json --queries fiqa --queries-path data/fiqa/beir_cache/fiqa --sample 200 --k 10 --seed 42 --mode exact --gt annpack`
- Result: overlap@10=1.000, min overlap@10=1.000, jaccard@10=1.000, tau=1.000, PASS
