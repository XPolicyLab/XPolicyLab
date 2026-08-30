from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from XPolicyLab.policy.MolmoAct2.model import (
    _DEFAULT_REPO_ID,
    _DEFAULT_REVISION,
    _as_rgb_pil,
    _parse_hf_reference,
    _resolve_instruction,
    resolve_checkpoint_source,
)


class MolmoAct2HelperTest(unittest.TestCase):
    def test_configured_repo_id_is_pinned(self) -> None:
        parsed = _parse_hf_reference(
            _DEFAULT_REPO_ID,
            configured_repo_id=_DEFAULT_REPO_ID,
            default_revision=_DEFAULT_REVISION,
        )
        self.assertEqual(parsed, (_DEFAULT_REPO_ID, _DEFAULT_REVISION))

    def test_hf_uri_revision_overrides_default(self) -> None:
        revision = "1" * 40
        parsed = _parse_hf_reference(
            f"hf://{_DEFAULT_REPO_ID}@{revision}",
            configured_repo_id=_DEFAULT_REPO_ID,
            default_revision=_DEFAULT_REVISION,
        )
        self.assertEqual(parsed, (_DEFAULT_REPO_ID, revision))

    def test_model_page_url_uses_pinned_default(self) -> None:
        parsed = _parse_hf_reference(
            f"https://huggingface.co/{_DEFAULT_REPO_ID}",
            configured_repo_id=_DEFAULT_REPO_ID,
            default_revision=_DEFAULT_REVISION,
        )
        self.assertEqual(parsed, (_DEFAULT_REPO_ID, _DEFAULT_REVISION))

    def test_mutable_revision_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable"):
            _parse_hf_reference(
                f"hf://{_DEFAULT_REPO_ID}@main",
                configured_repo_id=_DEFAULT_REPO_ID,
                default_revision=_DEFAULT_REVISION,
            )

    def test_local_snapshot_uses_shared_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            for filename in (
                "config.json",
                "processor_config.json",
                "norm_stats.json",
                "model.safetensors.index.json",
            ):
                (checkpoint_dir / filename).touch()
            source = resolve_checkpoint_source({"checkpoint_path": str(checkpoint_dir)})
            self.assertTrue(source.is_local)
            self.assertEqual(Path(source.pretrained_name_or_path), checkpoint_dir)

    def test_rgb_values_are_not_channel_swapped(self) -> None:
        rgb = np.array([[[11, 22, 33]]], dtype=np.uint8)
        converted = np.asarray(_as_rgb_pil(rgb))
        np.testing.assert_array_equal(converted, rgb)

    def test_dynamic_instruction_precedes_fallback(self) -> None:
        observation = {"instruction": "Pick up the red block."}
        self.assertEqual(
            _resolve_instruction(observation, "fallback"),
            "Pick up the red block.",
        )


if __name__ == "__main__":
    unittest.main()
