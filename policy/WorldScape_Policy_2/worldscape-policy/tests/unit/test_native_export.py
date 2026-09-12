from __future__ import annotations

import torch
from torch import nn

from tests.helpers.fixture_model_config import fixture_native_model_config
from worldscape_policy.checkpoint import load_native_checkpoint
from worldscape_policy.training.native_export import (
    build_identity_transform_bundle,
    export_training_checkpoint,
)
from tools.checkpoint.export_hf_checkpoint import export_checkpoint


def _model_config() -> dict:
    return fixture_native_model_config()


def _generation_config() -> dict:
    return {
        "schema_version": "1",
        "num_dit_steps": 8,
        "num_inference_steps": 16,
        "dynamic_cache_schedule": False,
        "kv_cache_fifo": False,
        "cfg_scale": 5.0,
        "sigma_shift": 5.0,
    }


def _bundle():
    return build_identity_transform_bundle(
        image_input_range="zero_one",
        embodiments={
            "fixture": {
                "embodiment_id": 0,
                "max_state_dim": 3,
                "max_action_dim": 3,
                "state_fields": [["state.value", 3]],
                "action_fields": [["action.value", 3]],
            }
        },
        provenance={"source": "unit-test"},
    )


def test_relative_eef_bundle_marks_only_position_and_rotation_relative() -> None:
    bundle = build_identity_transform_bundle(
        image_input_range="uint8",
        action_mode="eef",
        relative_action=True,
        embodiments={
            "agilex": {
                "embodiment_id": 0,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "state_fields": [["state.left_pos", 3]],
                "action_fields": [
                    ["action.left_pos", 3],
                    ["action.left_rot6d", 6],
                    ["action.left_gripper", 1],
                    ["action.right_pos", 3],
                    ["action.right_rot6d", 6],
                    ["action.right_gripper", 1],
                ],
            }
        },
        provenance={"source": "unit-test"},
    )

    fields = bundle.embodiments["agilex"].action_fields
    assert {field.key for field in fields if not field.absolute} == {
        "action.left_pos",
        "action.left_rot6d",
        "action.right_pos",
        "action.right_rot6d",
    }
    assert {field.key for field in fields if field.absolute} == {
        "action.left_gripper",
        "action.right_gripper",
    }


def _export(source, destination, *, max_shard_size="5GB"):
    return export_training_checkpoint(
        source,
        destination,
        model_variant="fixture",
        model_config=_model_config(),
        generation_config=_generation_config(),
        normalization={
            "schema_version": "1",
            "visual": {
                "input_range": "zero_one",
                "model_range": "minus_one_one",
            },
        },
        transform_bundle=_bundle(),
        tokenizer_source="org/test-tokenizer",
        provenance={"source": {"format": "native-training"}},
        max_shard_size=max_shard_size,
    )


def test_exports_file_trainer_checkpoint_as_native_bundle(tmp_path):
    source_model = nn.Linear(2, 3)
    source = tmp_path / "final.pt"
    torch.save({"model": source_model.state_dict()}, source)

    destination = _export(source, tmp_path / "final")

    target = nn.Linear(2, 3)
    load_native_checkpoint(target, destination)
    torch.testing.assert_close(target.weight, source_model.weight)
    assert (destination / "config.json").is_file()
    assert (destination / "transform_bundle.json").is_file()


def test_exports_hugging_face_shards_with_manifest_inventory(tmp_path):
    source_model = nn.Linear(8, 8)
    source = tmp_path / "final.pt"
    torch.save({"model": source_model.state_dict()}, source)

    destination = _export(source, tmp_path / "final", max_shard_size=128)

    assert (destination / "model.safetensors.index.json").is_file()
    assert not (destination / "model.safetensors").exists()
    assert len(list(destination.glob("model-*-of-*.safetensors"))) == 2
    target = nn.Linear(8, 8)
    load_native_checkpoint(target, destination)
    torch.testing.assert_close(target.weight, source_model.weight)

    hf_destination = tmp_path / "hf"
    export_checkpoint(destination, hf_destination)
    assert (hf_destination / "model.safetensors.index.json").is_file()
    assert len(list(hf_destination.glob("model-*-of-*.safetensors"))) == 2


def test_merges_legacy_policy_into_deepspeed_resume_directory(tmp_path):
    source_model = nn.Linear(2, 3)
    destination = tmp_path / "final"
    destination.mkdir()
    torch.save(source_model.state_dict(), destination / "policy.pt")
    (destination / ".complete").write_text("deepspeed-checkpoint-v2\n")

    _export(destination, destination)

    assert (destination / ".complete").is_file()
    assert (destination / "checkpoint_manifest.json").is_file()
    target = nn.Linear(2, 3)
    load_native_checkpoint(target, destination)
    torch.testing.assert_close(target.bias, source_model.bias)


def test_enriches_canonical_safetensors_checkpoint_in_place(tmp_path):
    from safetensors.torch import save_file

    source_model = nn.Linear(2, 3)
    destination = tmp_path / "checkpoint-20"
    destination.mkdir()
    save_file(source_model.state_dict(), destination / "model.safetensors")
    (destination / ".complete").write_text("native-training-checkpoint-v2\n")

    _export(destination, destination)

    assert (destination / ".complete").is_file()
    assert (destination / "checkpoint_manifest.json").is_file()
    target = nn.Linear(2, 3)
    load_native_checkpoint(target, destination)
    torch.testing.assert_close(target.weight, source_model.weight)


def test_replaces_stale_final_without_preserving_resume_state(tmp_path):
    source_model = nn.Linear(2, 3)
    source = tmp_path / "checkpoint.pt"
    torch.save({"model": source_model.state_dict()}, source)
    destination = tmp_path / "final"
    destination.mkdir()
    (destination / "trainer_state.pt").write_bytes(b"stale")
    (destination / ".complete").write_text("deepspeed-checkpoint-v1\n")

    _export(source, destination)

    assert not (destination / "trainer_state.pt").exists()
    assert not (destination / ".complete").exists()
    target = nn.Linear(2, 3)
    load_native_checkpoint(target, destination)
    torch.testing.assert_close(target.bias, source_model.bias)
