#!/usr/bin/env python3
"""ANNPack Core conformance runner.

One command. Runs every vector in this packet against an implementation and
writes a machine-readable report.

    ./run.py --adapter ./my-reader-adapter [--output report.json]

The adapter is any executable implementing the contract in README.md. To check
the reference implementation:

    ./run.py --adapter ../../scripts/reference-adapter.sh

Exit status is 0 only when every check passes.
"""

import argparse
import json
import pathlib
import struct
import subprocess
import sys

PACKET = pathlib.Path(__file__).resolve().parent
VECTORS = PACKET / "vectors"
ARTIFACTS = PACKET / "artifacts"


def load(name):
    return json.loads((VECTORS / name).read_text(encoding="utf-8"))


class Adapter:
    def __init__(self, command):
        self.command = command

    def _run(self, *args, expect_json=True):
        result = subprocess.run(
            [self.command, *args], capture_output=True, text=True, timeout=120
        )
        if expect_json:
            if result.returncode != 0:
                raise RuntimeError(f"adapter failed: {result.stderr.strip()[:300]}")
            return json.loads(result.stdout)
        return result.returncode

    def tokenize(self, text):
        return self._run("tokenize", text)

    def search(self, pack, query):
        return self._run("search", str(pack), query)

    def open_pack(self, pack):
        """Return 0 if the implementation accepts and can serve the artifact."""
        return self._run("open", str(pack), expect_json=False)

    def verify_receipt(self, path):
        return self._run("verify-receipt", str(path), expect_json=False)


def check(results, name, passed, detail=""):
    results.append({"check": name, "pass": bool(passed), "detail": detail})
    status = "ok  " if passed else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail and not passed else ''}")
    return passed


def run_tokenizer(adapter, results):
    print("tokenizer (FORMAT-v3 §6.1)")
    vectors = load("tokenizer.json")
    for case in vectors["cases"]:
        try:
            actual = adapter.tokenize(case["input"])
        except Exception as error:  # noqa: BLE001 - report, do not abort the suite
            check(results, f"tokenize {case['input']!r}", False, str(error)[:200])
            continue
        check(
            results,
            f"tokenize {case['input']!r}",
            actual == case["expected"],
            f"expected {case['expected']} got {actual}",
        )


def run_scoring(adapter, results):
    print("scoring (FORMAT-v3 §6.2) — exact IEEE-754 comparison")
    vectors = load("scoring.json")
    pack = ARTIFACTS / "conformance-v2.annpack"
    for entry in vectors["queries"]:
        query = entry["query"]
        try:
            response = adapter.search(pack, query)
        except Exception as error:  # noqa: BLE001
            check(results, f"search {query!r}", False, str(error)[:200])
            continue
        hits = response.get("results", [])
        if not check(
            results,
            f"search {query!r} hit count",
            len(hits) == entry["result_count"],
            f"expected {entry['result_count']} got {len(hits)} — a different hit "
            f"count is how a divergent tokenizer shows up",
        ):
            continue
        for got, want in zip(hits, entry["results"]):
            check(
                results,
                f"search {query!r} rank {want['rank']} passage",
                got.get("passage_id") == want["passage_id"],
                f"expected {want['passage_id'][:16]} got {str(got.get('passage_id'))[:16]}",
            )
            got_bits = struct.pack(">d", float(got.get("score", float("nan")))).hex()
            check(
                results,
                f"search {query!r} rank {want['rank']} score",
                got_bits == want["score_bits"],
                f"expected bits {want['score_bits']} ({want['score']}) got {got_bits}",
            )


def run_compatibility(adapter, results):
    print("manifest compatibility (FORMAT-v3 §4.2)")
    vectors = load("compatibility.json")
    for key in ("manifest_v1_legacy", "manifest_v2_current"):
        entry = vectors[key]
        path = PACKET / entry["artifact"]
        try:
            code = adapter.open_pack(path)
        except Exception as error:  # noqa: BLE001
            check(results, f"open {key}", False, str(error)[:200])
            continue
        check(results, f"open {key}", code == 0, f"exit {code}, expected 0")


def run_multiplicity(adapter, results):
    print("section id/type namespaces (FORMAT-v3 §2)")
    vectors = load("multiplicity.json")
    for key in ("core_only", "both_overlays"):
        entry = vectors[key]
        path = PACKET / entry["artifact"]
        try:
            code = adapter.open_pack(path)
        except Exception as error:  # noqa: BLE001
            check(results, f"open {key}", False, str(error)[:200])
            continue
        check(results, f"open {key}", code == 0, f"exit {code}, expected 0")


def run_corruption(adapter, results):
    print("corruption corpus — every artifact must be rejected")
    vectors = load("corruption.json")
    for name, reason in vectors["artifacts"].items():
        path = ARTIFACTS / "corruption" / name
        try:
            code = adapter.open_pack(path)
        except Exception:  # noqa: BLE001 - a thrown error is a valid rejection
            code = 1
        check(results, f"reject {name}", code != 0, f"accepted; expected rejection ({reason})")


def run_evidence(adapter, results, tmp):
    print("evidence receipt (EVIDENCE-v1) — offline, no pack")
    vectors = load("evidence.json")
    receipt_path = tmp / "receipt.json"
    receipt_path.write_text(json.dumps(vectors["receipt"]), encoding="utf-8")
    try:
        code = adapter.verify_receipt(receipt_path)
        check(results, "verify published receipt", code == 0, f"exit {code}")
    except Exception as error:  # noqa: BLE001
        check(results, "verify published receipt", False, str(error)[:200])

    tampered = json.loads(json.dumps(vectors["receipt"]))
    import base64

    record = bytearray(base64.b64decode(tampered["passage_record_b64"]))
    record[len(record) // 2] ^= 0x01
    tampered["passage_record_b64"] = base64.b64encode(bytes(record)).decode()
    tampered_path = tmp / "receipt-tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        code = adapter.verify_receipt(tampered_path)
    except Exception:  # noqa: BLE001
        code = 1
    check(results, "reject tampered receipt", code != 0, "accepted a tampered receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="executable implementing the adapter contract")
    parser.add_argument("--implementation", default="", help="name recorded in the report")
    parser.add_argument("--output", help="write the JSON report here")
    parser.add_argument("--skip-evidence", action="store_true", help="receipts are an optional stretch goal")
    args = parser.parse_args()

    adapter = Adapter(args.adapter)
    results = []
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        tmp = pathlib.Path(temp)
        run_tokenizer(adapter, results)
        run_scoring(adapter, results)
        run_compatibility(adapter, results)
        run_multiplicity(adapter, results)
        run_corruption(adapter, results)
        if not args.skip_evidence:
            run_evidence(adapter, results, tmp)

    passed = sum(1 for r in results if r["pass"])
    failed = len(results) - passed
    report = {
        "schema": "annpack-conformance-report-v1",
        "implementation": args.implementation or args.adapter,
        "packet_pack_root": load("scoring.json")["pack_root"],
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "conformant": failed == 0,
        "results": results,
    }
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{passed}/{len(results)} checks passed")
    if failed:
        print("NOT CONFORMANT")
        print("\nEvery failure above is a specification finding, not just a bug in your")
        print("reader. Please record it in the ambiguity log.")
    else:
        print("CONFORMANT")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
