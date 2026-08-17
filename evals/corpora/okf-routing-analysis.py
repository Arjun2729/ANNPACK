#!/usr/bin/env python3
"""Re-analyse the committed OKF hard-negative report for routing headroom.

This reads `okf-hard-negatives.report.json` and recomputes, per policy, which
queries land their target in the top k. It runs no retrieval and rebuilds no
pack: `missed_query_ids` is already recorded per mode and per stratum in the
report, so every number below is a deterministic function of committed bytes.
That is the point — the routing question was answerable from the run we already
had, and nobody had asked it in the one direction that mattered.

WHAT THIS FOUND
---------------
`README.md` evaluated one routing rule, lexical-for-technical + vector-for-
hard-negative, which scores 51/63 against vector-only at 50/63 — one query, and
the natural conclusion was that routing is not worth building. But the rule
under test routes between *lexical and vector*. Routing between *hybrid and
vector* scores 53/63, capturing three of the four queries available from the
lexical/vector oracle:

    hybrid is the strongest mode on technical-token   26/28
    vector is the strongest mode on hard-negative     27/35

So the recoverable headroom is reached by varying how much lexical evidence
contributes to a ranking, not by selecting which engine answers the query. That
is a fusion-weight question, and it is ANN-10's open question rather than a new
one.

WHAT THIS IS NOT
----------------
**The policy was selected on the same 63 queries it is scored on.** 53/63 is
training performance. It is a hypothesis about where headroom lives, not
evidence that a fusion policy generalizes, and it must not be quoted as a
retrieval-quality result. Establishing generalization needs a corpus large
enough to hold out a split whose granularity is finer than the effect: at n=63
a single query moves recall by 0.016, and the whole effect is three queries.

The per-query oracle is not a policy either. It is the ceiling any router is
bounded by, since it requires knowing in advance which mode succeeds.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPORT = HERE / "okf-hard-negatives.report.json"
QRELS = HERE / "okf-hard-negatives.jsonl"

#: Published in README.md. Recomputed here so a re-run cannot drift from prose.
PUBLISHED = {
    "lexical": 25,
    "vector": 50,
    "hybrid": 46,
    "lexical/T + vector/H": 51,
    "hybrid/T + vector/H": 53,
    "oracle (lexical, vector)": 54,
    "oracle (lexical, vector, hybrid)": 55,
}


def load() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    report = json.loads(REPORT.read_text())
    categories: dict[str, set[str]] = {}
    every: set[str] = set()
    for line in QRELS.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        every.add(record["id"])
        categories.setdefault(record["category"], set()).add(record["id"])

    hits = {}
    for mode, body in report["modes"].items():
        missed = set(body["overall"]["missed_query_ids"])
        unknown = missed - every
        if unknown:
            raise SystemExit(f"{mode} reports queries absent from the query set: {sorted(unknown)}")
        hits[mode] = every - missed
    return hits, categories


def main() -> int:
    hits, categories = load()
    technical = categories["technical-token"]
    hard = categories["hard-negative"]
    total = len(technical | hard)

    rows: list[tuple[str, set[str]]] = []
    for mode in ("lexical", "vector", "hybrid"):
        rows.append((mode, hits[mode]))
    for technical_mode, hard_mode in (("lexical", "vector"), ("hybrid", "vector")):
        label = f"{technical_mode}/T + {hard_mode}/H"
        rows.append((label, (hits[technical_mode] & technical) | (hits[hard_mode] & hard)))
    rows.append(("oracle (lexical, vector)", hits["lexical"] | hits["vector"]))
    rows.append(("oracle (lexical, vector, hybrid)", hits["lexical"] | hits["vector"] | hits["hybrid"]))

    width = max(len(label) for label, _ in rows)
    print(f"| {'policy'.ljust(width)} | hits | recall@5 | technical | hard-negative |")
    print(f"|{'-' * (width + 2)}|------|----------|-----------|---------------|")
    drifted = []
    for label, selected in rows:
        print(
            f"| {label.ljust(width)} | {len(selected):2d}/{total} | "
            f"{len(selected) / total:.4f}   | {len(selected & technical):2d}/{len(technical)}     | "
            f"{len(selected & hard):2d}/{len(hard)}         |"
        )
        expected = PUBLISHED.get(label)
        if expected is not None and expected != len(selected):
            drifted.append(f"{label}: README says {expected}, report yields {len(selected)}")

    unrecoverable = sorted(
        (technical | hard) - hits["lexical"] - hits["vector"] - hits["hybrid"]
    )
    print()
    print(f"Missed by every existing mode: {len(unrecoverable)}")
    for query_id in unrecoverable:
        print(f"  {query_id}")
    print()
    print(
        "No routing policy can recover these; they bound what any fusion or selection\n"
        "rule can achieve over the representations in this pack. There are more of them\n"
        f"({len(unrecoverable)}) than the queries routing can win "
        f"({len(hits['lexical'] | hits['vector']) - len(hits['vector'])})."
    )

    if drifted:
        print("\nDrift against README.md:", file=sys.stderr)
        for line in drifted:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
