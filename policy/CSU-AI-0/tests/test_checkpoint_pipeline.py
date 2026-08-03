from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


POLICY_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))
from verify_checkpoints import build_report  # noqa: E402


class CheckpointVerificationTest(unittest.TestCase):
    def test_complete_manifest_and_partial_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "bundle"
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            sources = {}
            for expert in ("xiaomi", "spatial_forcing", "hunyuan"):
                content = f"{expert}-weight".encode()
                relative = f"{expert}/checkpoint"
                target = root / relative / "weight.bin"
                target.parent.mkdir(parents=True)
                target.write_bytes(content)
                sources[expert] = {
                    "provider": "test",
                    "repo_id": expert,
                    "revision": "fixed",
                    "destination_relpath": relative,
                    "files": [
                        {
                            "path": "weight.bin",
                            "source_path": "weight.bin",
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    ],
                }
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "bundle": {"name": "test", "root": "unused"},
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(root, manifest_path)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["checked_files"], 3)

            partial = root / "hunyuan/checkpoint/model.safetensors.partial"
            partial.write_bytes(b"partial")
            report = build_report(root, manifest_path)
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("unexpected file" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
