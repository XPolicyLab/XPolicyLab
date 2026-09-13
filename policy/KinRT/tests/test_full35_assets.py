"""Portable artifact-verification tests; no Hub access or model weights required."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "kinrt_full35_assets", Path(__file__).resolve().parents[1] / "full35_assets.py"
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Full35AssetsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="kinrt-full35-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "artifact"
        self.delivery = self.root / "delivery"
        self.delivery.mkdir(parents=True)
        self.dataset = Path(temporary.name) / "dataset"
        self.openpi = Path(temporary.name) / "openpi"

        labels = self.delivery / assets.LABELS_NAME
        np.save(labels, np.array([0, 0, 1, 2, 3, 3], dtype=np.int32))
        norm = {
            "norm_stats": {
                group: {"mean": [0.0] * 14, "std": [1.0] * 14, "q01": [-1.0] * 14, "q99": [1.0] * 14}
                for group in ("state", "actions")
            }
        }
        (self.delivery / "norm_stats.json").write_text(json.dumps(norm), encoding="utf-8")
        self.checkpoint_norm = self.root / assets.CHECKPOINT_NORM
        self.checkpoint_norm.parent.mkdir(parents=True)
        self.checkpoint_norm.write_text(json.dumps(norm, indent=2), encoding="utf-8")
        metadata = self.delivery / "dataset_meta"
        metadata.mkdir()
        (metadata / "info.json").write_text(
            json.dumps(
                {
                    "total_tasks": 35,
                    "total_episodes": 3500,
                    "total_frames": 6,
                    "codebase_version": "v3.0",
                }
            ),
            encoding="utf-8",
        )
        (metadata / "tasks.parquet").write_bytes(b"fixture-task-metadata")
        (metadata / "stats.json").write_text("{}", encoding="utf-8")

        checkpoint_paths = [
            "assets/" + assets.DATASET_ID + "/norm_stats.json",
            "_CHECKPOINT_METADATA",
            "commit_success.txt",
            "train_state/optimizer/checkpoint",
            *(f"params/ocdbt.process_0/d/{index:08x}" for index in range(1053)),
        ]
        self.expected_checkpoint_entries = {
            assets.CHECKPOINT_PREFIX + name: digest(self.checkpoint_norm)
            if name.startswith("assets/")
            else hashlib.sha256(name.encode()).hexdigest()
            for name in checkpoint_paths
        }
        manifest = self.delivery / assets.CHECKPOINT_MANIFEST
        manifest.write_text(
            "".join(
                f"{checksum}  {assets.SOURCE_PREFIX}{name[len(assets.CHECKPOINT_PREFIX):]}\n"
                for name, checksum in self.expected_checkpoint_entries.items()
            ),
            encoding="utf-8",
        )
        (self.delivery / "DELIVERY.md").write_text("Fixture delivery.\n", encoding="utf-8")
        delivery_files = [labels, self.delivery / "norm_stats.json", manifest, *sorted(metadata.iterdir())]
        (self.delivery / "delivery_files.sha256").write_text(
            "".join(f"{digest(path)}  {path.relative_to(self.delivery).as_posix()}\n" for path in delivery_files),
            encoding="utf-8",
        )

        trusted = mock.patch.multiple(
            assets,
            LABELS_SHA256=digest(labels),
            NORM_SHA256=digest(self.delivery / "norm_stats.json"),
            CHECKPOINT_NORM_SHA256=digest(self.checkpoint_norm),
            TOTAL_FRAMES=6,
            COUNTS=[2, 1, 1, 2],
            MANIFEST_DIGESTS={
                name: digest(self.root / name)
                for name in (
                    "delivery/delivery_files.sha256",
                    "delivery/" + assets.CHECKPOINT_MANIFEST,
                    "delivery/DELIVERY.md",
                )
            },
            DATASET_DIGESTS={path.name: digest(path) for path in metadata.iterdir()},
        )
        trusted.start()
        self.addCleanup(trusted.stop)

    def prepare_dataset(self, *, content: bool = True):
        shutil.copytree(self.delivery / "dataset_meta", self.dataset / "meta")
        if content:
            for relative in (
                "data/chunk-000/file-000.parquet",
                "videos/cam_high/chunk-000/file-000.mp4",
                "meta/episodes/chunk-000/file-000.parquet",
            ):
                path = self.dataset / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture-dataset-content")

    def test_checkpoint_manifest_maps_every_final_step_path(self):
        self.assertEqual(assets.checkpoint_entries(self.root), self.expected_checkpoint_entries)

    def test_checkpoint_selection_excludes_optimizer_state_by_default(self):
        entries = assets.checkpoint_entries(self.root)
        default = {
            name
            for name in entries
            if assets.selected_checkpoint_file(name, assets_only=False, include_training_state=False)
        }
        resume = {
            name
            for name in entries
            if assets.selected_checkpoint_file(name, assets_only=False, include_training_state=True)
        }
        metadata = {
            name
            for name in entries
            if assets.selected_checkpoint_file(name, assets_only=True, include_training_state=False)
        }
        self.assertEqual(resume - default, {assets.CHECKPOINT_PREFIX + "train_state/optimizer/checkpoint"})
        self.assertEqual(metadata, {assets.CHECKPOINT_NORM})
        self.assertIn(assets.CHECKPOINT_PREFIX + "params/ocdbt.process_0/d/00000000", default)

    def test_assets_only_verification_checks_real_files_without_requiring_weights(self):
        result = assets.verify(self.root, assets_only=True)
        self.assertEqual(result["checkpoint_files_verified"], 1)
        self.assertEqual(result["router_labels"]["counts"], [2, 1, 1, 2])
        self.assertFalse(result["inference_executed"])
        with self.assertRaises(FileNotFoundError):
            assets.verify(self.root)

    def test_corrupt_delivery_file_is_rejected(self):
        labels = self.delivery / assets.LABELS_NAME
        labels.write_bytes(labels.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            assets.verify(self.root, assets_only=True)

    def test_untrusted_manifest_change_is_rejected_before_mapping(self):
        manifest = self.delivery / assets.CHECKPOINT_MANIFEST
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(assets.SOURCE_PREFIX, "other-run/60000/", 1), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            assets.checkpoint_entries(self.root)

    def test_manifest_rejects_unsafe_and_duplicate_paths(self):
        manifest = self.root / "untrusted.sha256"
        checksum = "0" * 64
        for name in ("../outside", "/outside", "folder/../../outside", "C:/outside", "folder\\outside"):
            with self.subTest(path=name):
                manifest.write_text(f"{checksum}  {name}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
                    assets.read_manifest(manifest)
        manifest.write_text(f"{checksum}  params/chunk\n{checksum} *params/chunk\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate checksum path"):
            assets.read_manifest(manifest)

    def test_matching_metadata_without_training_content_is_rejected(self):
        self.prepare_dataset(content=False)
        with self.assertRaisesRegex(FileNotFoundError, "dataset content is missing"):
            assets.prepare_training(self.root, self.dataset, self.openpi)
        self.assertFalse((self.dataset / "meta/router_labels_k4_full35/router_labels.npy").exists())
        self.assertFalse(self.openpi.exists())

    def test_preparation_installs_and_validates_assets_idempotently(self):
        self.prepare_dataset()
        first = assets.prepare_training(self.root, self.dataset, self.openpi)
        second = assets.prepare_training(self.root, self.dataset, self.openpi)
        self.assertEqual(first, second)
        self.assertEqual(Path(first["labels_path"]).read_bytes(), (self.delivery / assets.LABELS_NAME).read_bytes())
        self.assertTrue(first["metadata_matches_delivery"])
        self.assertFalse(first["frame_order_verified"])

    def test_preparation_refuses_to_replace_modified_labels(self):
        self.prepare_dataset()
        labels = self.dataset / "meta/router_labels_k4_full35/router_labels.npy"
        labels.parent.mkdir()
        labels.write_bytes(b"custom-frame-order-labels")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            assets.prepare_training(self.root, self.dataset, self.openpi)
        self.assertEqual(labels.read_bytes(), b"custom-frame-order-labels")

    def test_preparation_refuses_to_replace_modified_normalization(self):
        self.prepare_dataset()
        norms = self.openpi / "assets" / assets.CONFIG_NAME / assets.DATASET_ID / "norm_stats.json"
        norms.parent.mkdir(parents=True)
        norms.write_bytes(b"custom-normalization")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            assets.prepare_training(self.root, self.dataset, self.openpi)
        self.assertEqual(norms.read_bytes(), b"custom-normalization")

    def test_preparation_accepts_alternate_approved_normalization_bytes(self):
        self.prepare_dataset()
        norms = self.openpi / "assets" / assets.CONFIG_NAME / assets.DATASET_ID / "norm_stats.json"
        norms.parent.mkdir(parents=True)
        shutil.copyfile(self.checkpoint_norm, norms)
        self.assertNotEqual(norms.read_bytes(), (self.delivery / "norm_stats.json").read_bytes())
        self.assertEqual(assets.validate_norm(norms), assets.validate_norm(self.delivery / "norm_stats.json"))
        assets.prepare_training(self.root, self.dataset, self.openpi)
        self.assertEqual(norms.read_bytes(), self.checkpoint_norm.read_bytes())

    def test_downloader_patterns_select_only_requested_payload(self):
        candidate_files = (
            "delivery/router_labels_k4_full35.npy",
            "delivery/dataset_meta/info.json",
            assets.CHECKPOINT_NORM,
            "checkpoints/60000/params/ocdbt.process_0/d/chunk",
            "checkpoints/60000/train_state/ocdbt.process_0/d/chunk",
            "checkpoints/60000/_CHECKPOINT_METADATA",
            "checkpoints/60000/commit_success.txt",
            "checkpoints/50000/params/ocdbt.process_0/d/chunk",
            "unrelated-model.bin",
        )
        for assets_only, include_training_state in ((True, False), (False, False), (False, True)):
            with self.subTest(assets_only=assets_only, include_training_state=include_training_state):
                hub = types.ModuleType("huggingface_hub")
                hub.snapshot_download = mock.Mock()
                args = argparse.Namespace(
                    destination=self.root,
                    repo_id="fixture/repo",
                    revision="fixed-revision",
                    assets_only=assets_only,
                    include_training_state=include_training_state,
                )
                with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), mock.patch.object(
                    assets, "verify", return_value={"verified": True}
                ) as verify:
                    self.assertEqual(assets.download(args), {"verified": True})
                kwargs = hub.snapshot_download.call_args.kwargs
                selected = {
                    name
                    for name in candidate_files
                    if any(fnmatch.fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"])
                }
                self.assertNotIn("checkpoints/50000/params/ocdbt.process_0/d/chunk", selected)
                self.assertNotIn("unrelated-model.bin", selected)
                self.assertIn("delivery/dataset_meta/info.json", selected)
                self.assertIn(assets.CHECKPOINT_NORM, selected)
                self.assertEqual("checkpoints/60000/params/ocdbt.process_0/d/chunk" in selected, not assets_only)
                self.assertEqual(
                    "checkpoints/60000/train_state/ocdbt.process_0/d/chunk" in selected, include_training_state
                )
                if not assets_only:
                    self.assertIn("checkpoints/60000/_CHECKPOINT_METADATA", selected)
                    self.assertIn("checkpoints/60000/commit_success.txt", selected)
                self.assertEqual(kwargs["repo_id"], "fixture/repo")
                self.assertEqual(kwargs["revision"], "fixed-revision")
                verify.assert_called_once_with(
                    self.root, assets_only=assets_only, include_training_state=include_training_state
                )


if __name__ == "__main__":
    unittest.main()
