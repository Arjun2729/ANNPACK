#!/usr/bin/env python3
"""
Adjudication helper for ANNPack retrieval evaluation.

For each query in a candidate qrels JSONL file, retrieves top-k passages
from the pack and writes a CSV the user can annotate with relevance judgments.

Usage:
  python3 evals/adjudicate.py \
    --pack target/fastapi-eval/fastapi.annpack \
    --queries evals/corpora/fastapi-qrels.unverified.jsonl \
    --binary target/release/annpack \
    --output target/adjudication.csv \
    --k 5

Output CSV columns:
  query_id, category, query, rank, passage_id, source_path, heading, text_snippet, relevant (0/1/blank)

The user fills in the "relevant" column (1=relevant, 0=not relevant) then
saves the CSV. Re-run evaluate.py using the adjudicated file.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def search(binary: str, pack: str, query: str, mode: str = "lexical", k: int = 5) -> list[dict]:
    try:
        result = subprocess.run(
            [binary, "search", pack, query, "--mode", mode, "--json", "--limit", str(k)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return data.get("results", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"  search error for '{query}': {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate adjudication CSV for retrieval eval")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--binary", default="target/release/annpack")
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", default="lexical", choices=["lexical", "vector", "hybrid"])
    args = parser.parse_args()

    queries = []
    with open(args.queries) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in queries:
        qid = q["id"]
        category = q["category"]
        query_text = q["query"]
        expected_paths = q.get("relevant_source_paths", [])
        is_not_present = category == "not-present"

        print(f"  [{qid}] {query_text[:60]}...", file=sys.stderr)
        results = search(args.binary, args.pack, query_text, mode=args.mode, k=args.k)

        if not results:
            rows.append({
                "query_id": qid,
                "category": category,
                "query": query_text,
                "rank": "",
                "passage_id": "",
                "source_path": "",
                "heading": "",
                "text_snippet": "(no results returned)",
                "expected_source_paths": "|".join(expected_paths),
                "pre_labeled_relevant": "0" if is_not_present else "",
                "relevant": "",
            })
            continue

        for rank, r in enumerate(results, 1):
            evidence = r.get("evidence", {})
            source_url = evidence.get("canonical_url", "")
            passage_id = evidence.get("passage_id", r.get("id", ""))
            heading = r.get("heading_path", [""])[0] if r.get("heading_path") else ""
            text = r.get("text", r.get("passage", ""))
            snippet = text[:300].replace("\n", " ") if text else ""

            # Derive source path from canonical_url or heading
            source_path = source_url.replace("https://fastapi.tiangolo.com/", "").rstrip("/")

            # Pre-label if source path matches an expected path
            pre_label = ""
            if is_not_present:
                pre_label = "0"
            elif expected_paths:
                for ep in expected_paths:
                    ep_norm = ep.replace(".md", "").replace("tutorial/", "").replace("advanced/", "")
                    if ep_norm and ep_norm in source_path:
                        pre_label = "1"
                        break

            rows.append({
                "query_id": qid,
                "category": category,
                "query": query_text,
                "rank": rank,
                "passage_id": passage_id,
                "source_path": source_path,
                "heading": heading,
                "text_snippet": snippet,
                "expected_source_paths": "|".join(expected_paths),
                "pre_labeled_relevant": pre_label,
                "relevant": pre_label,  # user overrides this column
            })

    fieldnames = [
        "query_id", "category", "query", "rank", "passage_id",
        "source_path", "heading", "text_snippet",
        "expected_source_paths", "pre_labeled_relevant", "relevant"
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out}", file=sys.stderr)
    print(f"Queries: {len(queries)}", file=sys.stderr)
    print(f"\nInstructions:", file=sys.stderr)
    print(f"  Open {out} in a spreadsheet.", file=sys.stderr)
    print(f"  Review each row. The 'text_snippet' shows what the passage contains.", file=sys.stderr)
    print(f"  Set 'relevant' to 1 if the passage answers the query, 0 if not.", file=sys.stderr)
    print(f"  'pre_labeled_relevant' is auto-filled where source paths matched — verify these.", file=sys.stderr)
    print(f"  For 'not-present' queries, all should be 0.", file=sys.stderr)
    print(f"  Save and pass to evaluate.py.", file=sys.stderr)


if __name__ == "__main__":
    main()
