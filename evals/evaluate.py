#!/usr/bin/env python3
"""Evaluate ANNPack retrieval against human-authored query relevance judgments."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def load_qrels(path: Path):
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        query_id = record.get("id")
        query = record.get("query")
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError(f"line {line_number}: query id must be non-empty and unique")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"line {line_number}: query must be non-empty")
        passage_ids = record.get("relevant_passage_ids", [])
        source_paths = record.get("relevant_source_paths", [])
        if not passage_ids and not source_paths:
            raise ValueError(f"line {line_number}: at least one relevance judgment is required")
        if not all(isinstance(value, str) and value for value in passage_ids + source_paths):
            raise ValueError(f"line {line_number}: relevance judgments must be strings")
        vector = record.get("query_vector")
        if vector is not None and (
            not isinstance(vector, list)
            or not vector
            or not all(isinstance(value, (int, float)) for value in vector)
        ):
            raise ValueError(f"line {line_number}: query_vector must be a non-empty number array")
        seen.add(query_id)
        records.append(record)
    if not records:
        raise ValueError("query file contains no judgments")
    return records


def labels_for_record(record):
    return {
        *(f"passage:{value}" for value in record.get("relevant_passage_ids", [])),
        *(f"source:{value}" for value in record.get("relevant_source_paths", [])),
    }


def labels_for_hit(hit):
    return {f"passage:{hit['passage_id']}", f"source:{hit['source_path']}"}


def run_search(binary, pack, record, mode, limit, vector_profile, directory):
    command = [
        binary,
        "search",
        pack,
        record["query"],
        "--mode",
        mode,
        "--limit",
        str(limit),
        "--json",
    ]
    if mode != "lexical":
        vector_path = directory / f"{record['id']}.json"
        vector_path.write_text(json.dumps(record["query_vector"]), encoding="utf-8")
        command.extend(["--query-vector", str(vector_path)])
        if vector_profile:
            command.extend(["--vector-profile", vector_profile])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def summarize(outcomes, limit):
    recalls = []
    reciprocal_ranks = []
    hits = 0
    misses = []
    for outcome in outcomes:
        relevant = labels_for_record(outcome["record"])
        retrieved = [labels_for_hit(hit) for hit in outcome["response"]["results"][:limit]]
        matched = set().union(*(labels & relevant for labels in retrieved)) if retrieved else set()
        recalls.append(len(matched) / len(relevant))
        first_rank = next(
            (rank for rank, labels in enumerate(retrieved, 1) if labels & relevant),
            None,
        )
        if first_rank is None:
            reciprocal_ranks.append(0.0)
            misses.append(outcome["record"]["id"])
        else:
            hits += 1
            reciprocal_ranks.append(1.0 / first_rank)
    count = len(outcomes)
    return {
        "queries": count,
        f"macro_recall_at_{limit}": sum(recalls) / count,
        f"hit_rate_at_{limit}": hits / count,
        f"mrr_at_{limit}": sum(reciprocal_ranks) / count,
        "missed_query_ids": misses,
    }


def category_breakdown(outcomes, limit):
    categories = sorted({outcome["record"].get("category", "uncategorized") for outcome in outcomes})
    return {
        category: summarize(
            [
                outcome
                for outcome in outcomes
                if outcome["record"].get("category", "uncategorized") == category
            ],
            limit,
        )
        for category in categories
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/annpack")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--vector-profile")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--require-vectors", action="store_true")
    parser.add_argument("--min-hybrid-recall", type=float)
    parser.add_argument("--require-hybrid-not-worse", action="store_true")
    args = parser.parse_args()
    if args.k < 1 or args.k > 1000:
        parser.error("--k must be between 1 and 1000")

    binary = str(Path(args.binary).resolve())
    pack = str(Path(args.pack).resolve())
    records = load_qrels(Path(args.queries))
    vectors_complete = all(record.get("query_vector") is not None for record in records)
    if args.require_vectors and not vectors_complete:
        raise SystemExit("vector evaluation requested but at least one query has no query_vector")
    modes = ["lexical", "vector", "hybrid"] if vectors_complete else ["lexical"]
    results = {}
    pack_identity = None
    with tempfile.TemporaryDirectory(prefix="annpack-eval-") as temp:
        directory = Path(temp)
        for mode in modes:
            outcomes = []
            for record in records:
                response = run_search(
                    binary,
                    pack,
                    record,
                    mode,
                    args.k,
                    args.vector_profile,
                    directory,
                )
                if pack_identity is None:
                    pack_identity = response["pack"]
                elif response["pack"]["root_hash"] != pack_identity["root_hash"]:
                    raise RuntimeError("pack root changed during evaluation")
                outcomes.append({"record": record, "response": response})
            results[mode] = {
                "overall": summarize(outcomes, args.k),
                "categories": category_breakdown(outcomes, args.k),
            }

    metric = f"macro_recall_at_{args.k}"
    gates = {}
    if args.min_hybrid_recall is not None:
        if "hybrid" not in results:
            gates["minimum_hybrid_recall"] = {
                "pass": False,
                "reason": "hybrid evaluation is unavailable without query vectors",
            }
        else:
            actual = results["hybrid"]["overall"][metric]
            gates["minimum_hybrid_recall"] = {
                "actual": actual,
                "limit": args.min_hybrid_recall,
                "pass": actual >= args.min_hybrid_recall,
            }
    if args.require_hybrid_not_worse:
        if "hybrid" not in results:
            gates["hybrid_not_worse"] = {
                "pass": False,
                "reason": "hybrid evaluation is unavailable without query vectors",
            }
        else:
            hybrid = results["hybrid"]["overall"][metric]
            alternatives = [results[mode]["overall"][metric] for mode in ("lexical", "vector")]
            gates["hybrid_not_worse"] = {
                "actual": hybrid,
                "best_single_mode": max(alternatives),
                "pass": hybrid >= max(alternatives),
            }

    report = {
        "schema": "annpack-retrieval-eval-v1",
        "dataset": str(Path(args.queries)),
        "judgment_count": len(records),
        "k": args.k,
        "pack": pack_identity,
        "vectors_complete": vectors_complete,
        "modes": results,
        "gates": gates,
        "claim_scope": "Metrics are valid only for the pinned corpus, pack root, queries, and relevance judgments in this report.",
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.enforce:
        failures = [name for name, gate in gates.items() if not gate["pass"]]
        if failures:
            raise SystemExit(f"retrieval evaluation gates failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
