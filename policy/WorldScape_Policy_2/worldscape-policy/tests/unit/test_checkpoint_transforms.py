from __future__ import annotations

import json

import pytest
import torch

from evals.common.checkpoint_runtime import (
    load_checkpoint_transform_bundle,
    without_state_action_normalization,
)
from worldscape_policy.checkpoint import (
    TRANSFORM_BUNDLE_FILENAME,
    CheckpointTransformArtifact,
    EmbodimentTransform,
    TransformField,
)


def _fixture_artifact(*, image_input_range: str = "uint8") -> CheckpointTransformArtifact:
    state_stats = {
        "max": [2.0, 4.0],
        "min": [0.0, 0.0],
        "mean": [1.0, 2.0],
        "std": [1.0, 2.0],
        "q01": [0.0, 0.0],
        "q99": [2.0, 4.0],
    }
    action_stats = {
        "max": [[2.0, 4.0], [4.0, 8.0]],
        "min": [[0.0, 0.0], [0.0, 0.0]],
        "mean": [[1.0, 2.0], [2.0, 4.0]],
        "std": [[1.0, 2.0], [2.0, 4.0]],
        "q01": [[0.0, 0.0], [0.0, 0.0]],
        "q99": [[2.0, 4.0], [4.0, 8.0]],
    }
    return CheckpointTransformArtifact(
        image_input_range=image_input_range,
        embodiments={
            "robot": EmbodimentTransform(
                embodiment_id=7,
                max_state_dim=4,
                max_action_dim=4,
                state_fields=(
                    TransformField(
                        key="state.joint",
                        size=2,
                        normalization="q99",
                        statistics=state_stats,
                        per_horizon_statistics=None,
                        absolute=True,
                    ),
                ),
                action_fields=(
                    TransformField(
                        key="action.joint",
                        size=2,
                        normalization="q99",
                        statistics={"q01": [0.0, 0.0], "q99": [2.0, 4.0]},
                        per_horizon_statistics=action_stats,
                        absolute=False,
                    ),
                ),
            )
        },
        provenance={"source_format": "fixture"},
    )


def test_transform_bundle_load_and_unapply_parity(tmp_path):
    artifact = _fixture_artifact(image_input_range="minus_one_one")
    artifact.write(tmp_path / TRANSFORM_BUNDLE_FILENAME)

    loaded = load_checkpoint_transform_bundle(tmp_path, "robot")

    assert loaded.embodiment_id == 7
    torch.testing.assert_close(
        loaded.transform.apply_state({"state.joint": [1.0, 2.0]}),
        torch.tensor([0.0, 0.0]),
    )
    unpacked = loaded.transform.unapply(
        {"action": torch.tensor([[0.0, 0.0], [0.0, 0.0]])}
    )
    torch.testing.assert_close(
        unpacked["action.joint"],
        torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
    )
    flattened = loaded.transform.unapply({"action": torch.zeros(4, 2)})
    torch.testing.assert_close(
        flattened["action.joint"],
        torch.tensor([[1.0, 2.0], [2.0, 4.0]]).repeat(2, 1),
    )
    assert loaded.transform.relative_action_keys == {"action.joint"}
    image = loaded.transform.apply_image(torch.tensor([0, 255], dtype=torch.uint8))
    torch.testing.assert_close(image, torch.tensor([-1.0, 1.0]))


def test_transform_artifact_rejects_tampering(tmp_path):
    artifact = _fixture_artifact()
    path = tmp_path / TRANSFORM_BUNDLE_FILENAME
    artifact.write(path)
    value = json.loads(path.read_text())
    value["embodiments"]["robot"]["embodiment_id"] = 8
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_checkpoint_transform_bundle(tmp_path, "robot")


def test_eval_can_force_identity_without_rewriting_checkpoint(tmp_path):
    artifact = _fixture_artifact()
    artifact.write(tmp_path / TRANSFORM_BUNDLE_FILENAME)
    bundle = without_state_action_normalization(
        load_checkpoint_transform_bundle(tmp_path, "robot")
    )

    torch.testing.assert_close(
        bundle.transform.apply_state({"state.joint": [1.0, 2.0]}),
        torch.tensor([1.0, 2.0]),
    )
    unpacked = bundle.transform.unapply({"action": torch.tensor([[1.0, 2.0]])})
    torch.testing.assert_close(
        unpacked["action.joint"],
        torch.tensor([[1.0, 2.0]]),
    )


def test_native_eval_requires_transform_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="transform_bundle.json"):
        load_checkpoint_transform_bundle(tmp_path, "robot")


def test_transform_artifact_round_trip_is_deterministic(tmp_path):
    artifact = _fixture_artifact()
    path = tmp_path / TRANSFORM_BUNDLE_FILENAME
    artifact.write(path)
    loaded = CheckpointTransformArtifact.read(path)
    assert loaded.to_dict() == artifact.to_dict()
