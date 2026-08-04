"""The Python Core reader (binding) must open, verify, and lexically search
extension-bearing packs with results identical to a Core-only pack.

This exercises invariant 4 (graceful degradation) across the binding boundary:
the binding understands nothing about ANN-7/8/9/10, yet extension packs behave
exactly like Core packs for verification and default lexical search.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bindings/python"))

from annpack import Client  # noqa: E402

BINARY = Path(os.environ.get("ANNPACK_BINARY", ROOT / "target/release/annpack"))


def cli(*args, capture=True):
    return subprocess.run(
        [str(BINARY), *args], check=True, capture_output=capture, text=True
    )


@unittest.skipUnless(BINARY.exists(), "annpack binary not built")
class ExtensionReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        d = Path(self.temp.name)
        self.core = d / "core.annpack"
        self.exp = d / "exp.annpack"
        corpus = ROOT / "fixtures/docs-v1"
        cli("build", str(corpus), "--output", str(self.core),
            "--name", "binding-demo", "--version", "1.0.0",
            "--base-url", "https://example.test")
        passages = json.loads(cli("export-passages", str(self.core)).stdout)
        ids = [p["id"] for p in passages]
        raw = {"generator": "t", "model": "t", "revision": "r",
               "passages": [{"passage_id": ids[0],
                             "candidates": [{"text": "chartreuse marker", "score": 0.9}]}]}
        raw_path = d / "raw.json"
        raw_path.write_text(json.dumps(raw))
        side = d / "exp.sidecar.json"
        cli("generate", "expansion", str(raw_path), "--output", str(side), "--threshold", "0.5")
        cli("build", str(corpus), "--output", str(self.exp),
            "--name", "binding-demo", "--version", "1.0.0",
            "--base-url", "https://example.test", "--expansion", str(side))
        self.client = Client(BINARY)

    def tearDown(self):
        self.temp.cleanup()

    def test_verify_extension_pack(self):
        report = self.client.verify(self.exp)
        self.assertTrue(report["integrity_verified"])
        self.assertTrue(report["conformance"]["core_conformant"])
        self.assertIn("ANN-7", report["conformance"]["extensions"])

    def test_default_lexical_search_matches_core(self):
        query = "cache"
        core = self.client.search(self.core, query, mode="lexical")
        exp = self.client.search(self.exp, query, mode="lexical")
        core_ids = [(h["passage_id"], round(h["score"], 6)) for h in core["results"]]
        exp_ids = [(h["passage_id"], round(h["score"], 6)) for h in exp["results"]]
        self.assertEqual(core_ids, exp_ids)

    def test_generated_term_not_citable(self):
        # The generated marker never appears in any lexical result's text.
        result = self.client.search(self.exp, "cache", mode="lexical")
        for hit in result["results"]:
            self.assertNotIn("chartreuse", hit["text"].lower())


if __name__ == "__main__":
    unittest.main()
