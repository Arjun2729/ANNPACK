import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bindings/python"))

from annpack import Client  # noqa: E402


def client() -> Client:
    return Client(Path(os.environ.get("ANNPACK_BINARY", ROOT / "target/release/annpack")))


PACK = ROOT / "spec/test-vectors/minimal-v3.annpack"


class RunBundleTest(unittest.TestCase):
    def test_bundle_attests_and_names_its_artifact(self):
        api = client()
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "run.json"
            bundle = api.bundle(PACK, "ANN-001", path, limit=2, application="test/1.0")
            self.assertEqual(bundle["schema"], "annpack-run-bundle-v1")
            self.assertTrue(bundle["receipts"])

            report = api.verify_run(path)
            self.assertTrue(report["attested"])
            self.assertEqual(report["receipts_verified"], report["receipts_total"])
            self.assertEqual(len(report["pack_roots"]), 1)

    def test_a_tampered_bundle_reports_rather_than_raises(self):
        # The report is the useful artifact on failure: it names which receipt
        # failed. A caller must check `attested`, not rely on an exception.
        api = client()
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "run.json"
            bundle = api.bundle(PACK, "ANN-001", path, limit=2)
            bundle["receipts"][0]["passage_hash"] = "00" * 32
            tampered = Path(work) / "tampered.json"
            tampered.write_text(json.dumps(bundle), encoding="utf-8")

            report = api.verify_run(tampered)
            self.assertFalse(report["attested"])
            self.assertFalse(report["receipts"][0]["verification"]["verified"])


class TelemetryTest(unittest.TestCase):
    def test_attributes_bind_passages_to_the_artifact(self):
        api = client()
        attributes = api.telemetry(
            PACK,
            "ANN-001",
            limit=2,
            receipt_uri_template="https://evidence.test/{root}/{passage_id}",
        )
        root = attributes["span"]["annpack.root"]
        self.assertEqual(len(root), 64)
        self.assertTrue(attributes["events"])
        for event in attributes["events"]:
            self.assertEqual(event["annpack.root"], root)
            self.assertTrue(
                event["annpack.receipt_uri"].endswith(event["annpack.passage_id"])
            )


if __name__ == "__main__":
    unittest.main()
