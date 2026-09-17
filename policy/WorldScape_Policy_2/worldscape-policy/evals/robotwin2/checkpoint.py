from __future__ import annotations

from pathlib import Path

import yaml

from worldscape_policy.checkpoint.transforms import (
    TRANSFORM_BUNDLE_FILENAME,
    CheckpointTransformArtifact,
    EmbodimentTransform,
    NativeCheckpointTransform,
)


def load_robotwin2_checkpoint_transform(
    checkpoint: str | Path,
) -> NativeCheckpointTransform:
    """Load and strictly validate the native RoboTwin2 joint transform."""

    checkpoint = Path(checkpoint)
    artifact = CheckpointTransformArtifact.read(
        checkpoint / TRANSFORM_BUNDLE_FILENAME
    )
    try:
        embodiment = artifact.embodiments["robotwin2"]
    except KeyError as exc:
        raise ValueError("checkpoint transform bundle has no robotwin2 embodiment") from exc
    _validate_robotwin2_transform(artifact, embodiment)
    _validate_robotwin2_model_dimensions(checkpoint, embodiment)
    return NativeCheckpointTransform(
        image_input_range=artifact.image_input_range,
        embodiment=embodiment,
    )


def _validate_robotwin2_transform(
    artifact: CheckpointTransformArtifact,
    embodiment: EmbodimentTransform,
) -> None:
    if artifact.image_input_range != "zero_one":
        raise ValueError("RoboTwin2 checkpoint image input range must be zero_one")
    provenance = artifact.provenance
    expected = {
        "normalization": "global_zscore",
        "action_mode": "joint",
        "relative_action": False,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "RoboTwin2 checkpoint provenance is incompatible with native joint "
            f"evaluation: {mismatches}"
        )
    if len(embodiment.state_fields) != 1 or len(embodiment.action_fields) != 1:
        raise ValueError("RoboTwin2 checkpoint requires one state and one action vector")
    state, action = embodiment.state_fields[0], embodiment.action_fields[0]
    if (state.key, state.size) != ("state.vector", 14):
        raise ValueError("RoboTwin2 checkpoint state must be state.vector with width 14")
    if (action.key, action.size, action.absolute) != ("action.vector", 14, True):
        raise ValueError(
            "RoboTwin2 checkpoint action must be absolute action.vector with width 14"
        )
    for field in (state, action):
        if field.normalization != "mean_std":
            raise ValueError("RoboTwin2 checkpoint must use mean_std normalization")
        if field.statistics.get("clip_min") != [-5.0] * 14:
            raise ValueError("RoboTwin2 checkpoint clip_min must be -5 for all dimensions")
        if field.statistics.get("clip_max") != [5.0] * 14:
            raise ValueError("RoboTwin2 checkpoint clip_max must be 5 for all dimensions")


def _validate_robotwin2_model_dimensions(
    checkpoint: Path,
    embodiment: EmbodimentTransform,
) -> None:
    config_path = checkpoint / "model_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "RoboTwin2 checkpoint is missing model_config.yaml required for "
            "dimension validation"
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        shape = payload["model"]["shape"]
        core = payload["model"]["wam"]["core"]["parameters"]
        dimensions = {
            "shape.action_horizon": int(shape["action_horizon"]),
            "shape.action_dim": int(shape["action_dim"]),
            "shape.max_state_dim": int(shape["max_state_dim"]),
            "core.action_dim": int(core["action_dim"]),
            "core.max_state_dim": int(core["max_state_dim"]),
            "core.max_num_embodiments": int(core["max_num_embodiments"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "RoboTwin2 checkpoint model_config.yaml has an invalid dimension contract"
        ) from exc
    if dimensions["shape.action_horizon"] not in {24, 48}:
        raise ValueError(
            "RoboTwin2 checkpoint action_horizon must be 24 or 48, got "
            f"{dimensions['shape.action_horizon']}"
        )
    expected = {
        "shape.action_dim": embodiment.max_action_dim,
        "shape.max_state_dim": embodiment.max_state_dim,
        "core.action_dim": embodiment.max_action_dim,
        "core.max_state_dim": embodiment.max_state_dim,
    }
    mismatches = {
        name: (dimensions[name], value)
        for name, value in expected.items()
        if dimensions[name] != value
    }
    if embodiment.embodiment_id >= dimensions["core.max_num_embodiments"]:
        mismatches["embodiment_id"] = (
            embodiment.embodiment_id,
            dimensions["core.max_num_embodiments"],
        )
    if mismatches:
        raise ValueError(
            "RoboTwin2 checkpoint model/transform dimensions are incompatible: "
            f"{mismatches}"
        )


__all__ = ["load_robotwin2_checkpoint_transform"]
