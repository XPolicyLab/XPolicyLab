"""Small offline acquisition and containment tests; no model download or GPU required."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import download_wan as download


class WanAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.source = self.root / "source"
        self.source.mkdir()
        pinned = download.load_manifest()
        names = [item["path"] for item in pinned["files"]]
        shards = [name for name in names if name.endswith(".safetensors")]
        self.content = {name: f"fixture:{name}".encode() for name in names}
        self.content[download.INDEX_NAME] = json.dumps({
            "weight_map": {f"tensor_{i}": name for i, name in enumerate(shards)}
        }).encode()
        self.manifest = {"files": []}
        for name, content in self.content.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.manifest["files"].append({"path": name, "size_bytes": len(content),
                                           "sha256": hashlib.sha256(content).hexdigest()})
        self.output = self.work / "assets/Wan"

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_local_copy_and_idempotent_verification(self):
        result = download.acquire_assets(self.output, self.work, self.manifest, self.source)
        self.assertEqual(result, self.output)
        for name, data in self.content.items():
            self.assertEqual((result / name).read_bytes(), data)
        self.assertFalse(self.output.with_name("Wan.partial").exists())
        self.assertEqual(download.acquire_assets(self.output, self.work, self.manifest), result)

    def test_corrupt_source_is_rejected_before_copy(self):
        (self.source / "config.json").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            download.acquire_assets(self.output, self.work, self.manifest, self.source)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("Wan.download.lock").exists())

    def test_same_size_corruption_is_rejected_by_sha256(self):
        path = self.source / "config.json"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            download.verify_assets(self.source, self.manifest)

    def test_existing_bad_destination_is_not_overwritten(self):
        self.output.mkdir(parents=True)
        marker = self.output / "do-not-replace"
        marker.write_text("existing")
        with self.assertRaises(FileNotFoundError):
            download.acquire_assets(self.output, self.work, self.manifest, self.source)
        self.assertEqual(marker.read_text(), "existing")

    def test_destination_outside_workspace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            download.acquire_assets(self.root / "outside", self.work, self.manifest, self.source)
        (self.work / "escape").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            download.acquire_assets(self.work / "escape/outside", self.work, self.manifest, self.source)

    def test_source_symlink_escape_is_rejected(self):
        path = self.source / "config.json"
        data = path.read_bytes()
        path.unlink()
        outside = self.root / "outside-config.json"
        outside.write_bytes(data)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes its directory"):
            download.verify_assets(self.source, self.manifest)

    def test_hf_requests_are_pinned_and_complete(self):
        calls = []

        def fetch(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.content[kwargs["filename"]])
            return str(target)

        hub = types.SimpleNamespace(hf_hub_download=fetch)
        cache = str(self.work / "cache/hf/hub")
        with patch.dict("sys.modules", {"huggingface_hub": hub}), patch.dict("os.environ", {"HF_HUB_CACHE": cache}):
            download.acquire_assets(self.output, self.work, self.manifest)
        self.assertEqual({call["filename"] for call in calls}, set(self.content))
        self.assertEqual(len(calls), 11)
        for call in calls:
            self.assertEqual((call["repo_id"], call["revision"]), (download.REPO_ID, download.REVISION))
            self.assertEqual(call["cache_dir"], cache)
            self.assertTrue(Path(call["local_dir"]).is_relative_to(self.work))

    def test_incomplete_index_is_rejected(self):
        index_path = self.source / download.INDEX_NAME
        index = json.loads(index_path.read_text())
        del index["weight_map"]["tensor_0"]
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "exactly the pinned shards"):
            download.verify_index(self.source, self.manifest)

    def test_download_metadata_symlink_escape_is_rejected(self):
        stage = self.output.with_name("Wan.partial")
        stage.mkdir(parents=True)
        (stage / ".cache").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            download.acquire_assets(self.output, self.work, self.manifest, self.source)


if __name__ == "__main__":
    unittest.main()
