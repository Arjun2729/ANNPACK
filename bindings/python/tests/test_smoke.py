import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bindings/python"))

from annpack import Client  # noqa: E402


class BindingSmokeTest(unittest.TestCase):
    def test_verified_search(self):
        binary = Path(os.environ.get("ANNPACK_BINARY", ROOT / "target/release/annpack"))
        client = Client(binary)
        pack = ROOT / "spec/test-vectors/minimal-v3.annpack"
        self.assertTrue(client.verify(pack)["integrity_verified"])
        result = client.search(pack, "ANN-001", mode="lexical")
        self.assertIn("opened successfully", result["results"][0]["text"])


if __name__ == "__main__":
    unittest.main()

