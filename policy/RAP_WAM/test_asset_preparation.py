"""Offline checks for release selection, exact-file copying and shared-disk boundaries."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import asset_storage, download_weights, export_weights


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.original_tempdir = tempfile.tempdir
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.work / "assets/bundle"
        self.content = {name: f"fixture:{name}".encode() for name in download_weights.BUNDLE_FILES}
        for name, content in self.content.items():
            (self.source / name).write_bytes(content)
        self.env_patch = patch.dict(os.environ, {"RAP_WAM_WORK_DIR": str(self.work)})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        tempfile.tempdir = self.original_tempdir
        self.temporary.cleanup()

    def validate_fixture(self, root):
        for name, content in self.content.items():
            if (Path(root) / name).read_bytes() != content:
                raise ValueError(f"Corrupt fixture: {name}")

    def test_copy_contains_exactly_six_files_and_does_not_follow_extras(self):
        (self.source / "old-training-data").symlink_to(self.root, target_is_directory=True)
        (self.source / "README.md").write_text("not a model asset")
        result = download_weights.acquire_bundle(self.output, self.work, self.source, None, None, self.validate_fixture)
        self.assertEqual(result, self.output)
        self.assertEqual({p.name for p in result.iterdir()}, set(download_weights.BUNDLE_FILES))
        self.validate_fixture(result)

    def test_snapshot_links_are_copied_as_verified_bytes(self):
        blob = self.root / "blob"
        blob.write_bytes(self.content["model.pt"])
        (self.source / "model.pt").unlink()
        (self.source / "model.pt").symlink_to(blob)
        download_weights.acquire_bundle(self.output, self.work, self.source, None, None, self.validate_fixture)
        self.assertFalse((self.output / "model.pt").is_symlink())
        self.validate_fixture(self.output)

    def test_validation_failure_does_not_publish_or_leave_staging(self):
        calls = []

        def reject_staging(root):
            calls.append(root)
            self.validate_fixture(root)
            if Path(root) != self.source:
                raise ValueError("Rejected staging")

        with self.assertRaisesRegex(ValueError, "Rejected staging"):
            download_weights.acquire_bundle(self.output, self.work, self.source, None, None, reject_staging)
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])

    def test_existing_output_is_never_replaced(self):
        self.output.mkdir(parents=True)
        marker = self.output / "keep"
        marker.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            download_weights.acquire_bundle(self.output, self.work, self.source, None, None, self.validate_fixture)
        self.assertEqual(marker.read_text(), "unchanged")

    def test_output_and_nested_symlink_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            download_weights.acquire_bundle(self.root / "outside", self.work, self.source, None, None, self.validate_fixture)
        (self.work / "escape").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            download_weights.acquire_bundle(self.work / "escape/outside", self.work, self.source, None, None, self.validate_fixture)

    def test_workspace_replaces_inherited_home_and_temporary_caches(self):
        with patch.dict(os.environ, {"HF_HOME": "/root/.cache/huggingface", "HF_XET_CACHE": "/tmp/xet", "TMPDIR": "/tmp"}):
            tempfile.tempdir = "/tmp"
            work = asset_storage.configure_workspace()
            for name in ("HF_HOME", "HF_XET_CACHE", "HF_HUB_CACHE", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
                self.assertTrue(Path(os.environ[name]).resolve().is_relative_to(work))
            self.assertEqual(Path(tempfile.gettempdir()), work / "tmp")
            self.assertTrue(sys.dont_write_bytecode)

    def test_cache_symlink_escape_is_rejected_before_writes(self):
        (self.work / "cache").mkdir()
        (self.work / "cache/hf").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            asset_storage.configure_workspace()
        self.assertFalse((self.work / "tmp").exists())

    def test_default_release_selects_immutable_pin(self):
        release = self.root / "release.json"
        release.write_text(json.dumps({"repo_id": "example/policy", "revision": "a" * 40}))
        with patch.object(download_weights, "RELEASE_PATH", release):
            self.assertEqual(download_weights.select_source(), (None, "example/policy", "a" * 40))
        release.write_text(json.dumps({"repo_id": "example/policy", "revision": "main"}))
        with patch.object(download_weights, "RELEASE_PATH", release), self.assertRaisesRegex(ValueError, "40-character"):
            download_weights.select_source()

    def test_missing_release_and_custom_repo_without_pin_fail(self):
        with patch.object(download_weights, "RELEASE_PATH", self.root / "missing"), self.assertRaisesRegex(ValueError, "No public release"):
            download_weights.select_source()
        with self.assertRaisesRegex(ValueError, "40-character"):
            download_weights.select_source(repo_id="example/policy")
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            download_weights.select_source(revision="a" * 40)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            download_weights.select_source(source=self.source, revision="a" * 40)

    def test_default_cli_uses_release_and_shared_output(self):
        release = self.root / "release.json"
        release.write_text(json.dumps({"repo_id": "example/policy", "revision": "b" * 40}))
        calls = []

        def snapshot(repo_id, **kwargs):
            calls.append((repo_id, kwargs))
            self.assertTrue(Path(os.environ["HF_XET_CACHE"]).is_relative_to(self.work))
            self.assertTrue(Path(os.environ["TMPDIR"]).is_relative_to(self.work))
            return str(self.source)

        assets_name = download_weights.__package__ + ".assets"
        modules = {"huggingface_hub": types.SimpleNamespace(snapshot_download=snapshot),
                   assets_name: types.SimpleNamespace(validate_bundle=self.validate_fixture)}
        with patch.dict(sys.modules, modules), patch.object(download_weights, "RELEASE_PATH", release), patch.object(sys, "argv", ["download_weights"]):
            download_weights.main()
        output = self.work / "assets/RAP-WAM-s742500"
        self.validate_fixture(output)
        self.assertEqual(calls, [("example/policy", {"revision": "b" * 40,
            "cache_dir": str(self.work / "cache/hf/hub"), "allow_patterns": list(download_weights.BUNDLE_FILES)})])

    def test_exporter_rejects_system_output_before_loading_checkpoint(self):
        args = ["export_weights", "--checkpoint", "missing.ckpt", "--task-context", "missing.pt",
                "--output", str(self.root / "outside")]
        with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "inside RAP_WAM_WORK_DIR"):
            export_weights.main()
        self.assertFalse((self.root / "outside").exists())

    def test_exporter_rejects_existing_output_before_loading_checkpoint(self):
        self.output.mkdir(parents=True)
        args = ["export_weights", "--checkpoint", "missing.ckpt", "--task-context", "missing.pt",
                "--output", str(self.output)]
        with patch.object(sys, "argv", args), self.assertRaises(FileExistsError):
            export_weights.main()


if __name__ == "__main__":
    unittest.main()
