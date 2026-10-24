"""RoboDojo ARX-X5 joint-space normalization and checkpoint contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from worldscape_policy.checkpoint.transforms import (
    CheckpointTransformArtifact,
    EmbodimentTransform,
    TransformField,
)
from worldscape_policy.robodojo_velocity import (
    VELOCITY_FIELDS,
    VELOCITY_FIRST_FRAME,
    VELOCITY_METHOD,
    load_velocity_statistics,
)


ROBODOJO_JOINT_TEXT_DATASET = "worldscape_robodojo_joint_hdf5_text"
ROBODOJO_JOINT_INITIAL_DATASET = "worldscape_robodojo_joint_hdf5_initial"
ROBODOJO_JOINT_GOAL_DATASET = "worldscape_robodojo_joint_hdf5_goal"
ROBODOJO_JOINT_DEMO_DATASET = "worldscape_robodojo_joint_hdf5_demo"

JOINT_DIM = 14
LEFT_JOINTS = slice(0, 6)
LEFT_GRIPPER = slice(6, 7)
RIGHT_JOINTS = slice(7, 13)
RIGHT_GRIPPER = slice(13, 14)
GRIPPER_INDICES = (6, 13)
JOINT_LAYOUT = (
    "left_arm_joint_states[0:6]",
    "left_gripper",
    "right_arm_joint_states[0:6]",
    "right_gripper",
)


def load_joint_statistics(
    path: str | Path, *, normalization: str | None = None,
) -> dict[str, Any]:
    """Validate explicitly tagged z-score or legacy q01/q99 statistics."""

    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"RoboDojo joint statistics not found: {stats_path}. "
            "Provide the existing statistics artifact via ROBODOJO_JOINT_STATS."
        )
    value = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RoboDojo joint statistics must contain a JSON object")
    if value.get("schema_version") != "1":
        raise ValueError("RoboDojo joint statistics require schema_version '1'")
    if tuple(value.get("layout", ())) != JOINT_LAYOUT:
        raise ValueError("RoboDojo joint statistics use an unexpected field layout")
    if not isinstance(value.get("relative_joint_action"), bool):
        raise ValueError("relative_joint_action must be a boolean")
    mode = value.get("normalization", "q99")
    if mode not in {"q99", "zscore"}:
        raise ValueError(f"Unsupported joint normalization: {mode!r}")
    if normalization is not None and normalization != mode:
        raise ValueError(
            f"Requested normalization {normalization!r} does not match statistics {mode!r}"
        )
    if mode == "zscore" and value.get("clip_range") != [-5.0, 5.0]:
        raise ValueError("RoboDojo zscore requires clip_range [-5, 5]")
    for group_name in ("state", "action"):
        group = value.get(group_name)
        if not isinstance(group, Mapping):
            raise ValueError(
                f"RoboDojo joint statistics are missing {group_name!r}"
            )
        names = ("mean", "std") if mode == "zscore" else ("q01", "q99")
        for statistic_name in names:
            array = np.asarray(group.get(statistic_name), dtype=np.float32)
            if array.shape != (JOINT_DIM,) or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{group_name}.{statistic_name} must contain "
                    f"{JOINT_DIM} finite values"
                )
        if mode == "zscore":
            if np.any(np.asarray(group["std"], dtype=np.float32) <= 0):
                raise ValueError(f"{group_name}.std must be positive")
        else:
            low = np.asarray(group["q01"], dtype=np.float32)
            high = np.asarray(group["q99"], dtype=np.float32)
            if np.any(high < low):
                raise ValueError(f"{group_name}.q99 must be greater than or equal to q01")
    return value


def normalize_q99(value: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    """Normalize one joint array to [-1, 1] using q01/q99 bounds."""

    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != JOINT_DIM:
        raise ValueError(f"joint arrays must end in dimension {JOINT_DIM}")
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    scale = high - low
    safe = np.where(scale == 0, 1.0, scale)
    normalized = 2.0 * (array - low) / safe - 1.0
    normalized = np.where(scale == 0, array, normalized)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def normalize_joint(
    value: np.ndarray, artifact: Mapping[str, Any], group: str,
) -> np.ndarray:
    """Apply the artifact's normalization to all 14 position/gripper fields."""
    if artifact.get("normalization", "q99") == "q99":
        return normalize_q99(value, artifact[group])
    if artifact.get("normalization") != "zscore":
        raise ValueError("Unsupported joint normalization")
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != JOINT_DIM:
        raise ValueError(f"joint arrays must end in dimension {JOINT_DIM}")
    mean = np.asarray(artifact[group]["mean"], dtype=np.float32)
    std = np.asarray(artifact[group]["std"], dtype=np.float32)
    return np.clip((array - mean) / std, *artifact["clip_range"]).astype(
        np.float32, copy=False,
    )


def joint_actions_relative_to_chunk_anchor(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    action_horizon: int,
) -> np.ndarray:
    """Subtract each chunk anchor from arm joints and preserve grippers."""

    action = np.asarray(actions, dtype=np.float32)
    state = np.asarray(states, dtype=np.float32)
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if action.ndim != 2 or action.shape[-1] != JOINT_DIM:
        raise ValueError(f"joint actions must have shape [T,{JOINT_DIM}]")
    if state.ndim != 2 or state.shape[-1] != JOINT_DIM:
        raise ValueError(f"joint states must have shape [T,{JOINT_DIM}]")
    if not len(action) or not len(state):
        raise ValueError("relative joint conversion requires non-empty arrays")
    if len(action) == len(state):
        anchors = np.arange(len(action)) // action_horizon * action_horizon
        reference = state[np.minimum(anchors, len(state) - 1)]
    else:
        if len(action) % len(state):
            raise ValueError("packed joint action/state lengths are incompatible")
        steps_per_anchor = len(action) // len(state)
        if steps_per_anchor != action_horizon:
            raise ValueError(
                "packed joint action/state ratio must equal action_horizon"
            )
        reference = np.repeat(state, steps_per_anchor, axis=0)
    result = action.copy()
    result[:, LEFT_JOINTS] -= reference[:, LEFT_JOINTS]
    result[:, RIGHT_JOINTS] -= reference[:, RIGHT_JOINTS]
    return result


def build_robodojo_joint_transform_bundle(
    *,
    stats_path: str | Path,
    normalization: str | None = None,
    image_input_range: str = "uint8",
    embodiment_id: int = 0,
    max_state_dim: int = 64,
    max_action_dim: int = 32,
    include_joint_velocity: bool = False,
    velocity_stats_path: str | Path | None = None,
) -> CheckpointTransformArtifact:
    """Export exactly the joint normalization used by the training dataset."""

    stats = load_joint_statistics(stats_path, normalization=normalization)
    mode = stats.get("normalization", "q99")
    relative = bool(stats["relative_joint_action"])
    field_specs = (
        ("left_joint", LEFT_JOINTS, not relative),
        ("left_gripper", LEFT_GRIPPER, True),
        ("right_joint", RIGHT_JOINTS, not relative),
        ("right_gripper", RIGHT_GRIPPER, True),
    )

    def fields(prefix: str) -> tuple[TransformField, ...]:
        group = stats[prefix]
        def field_statistics(section: slice) -> dict[str, list[float]]:
            if mode == "q99":
                return {name: list(group[name][section]) for name in ("q01", "q99")}
            width = section.stop - section.start
            return {
                "mean": list(group["mean"][section]),
                "std": list(group["std"][section]),
                "clip_min": [stats["clip_range"][0]] * width,
                "clip_max": [stats["clip_range"][1]] * width,
            }

        return tuple(
            TransformField(
                key=f"{prefix}.{name}",
                size=section.stop - section.start,
                normalization="mean_std" if mode == "zscore" else "q99",
                statistics=field_statistics(section),
                per_horizon_statistics=None,
                absolute=True if prefix == "state" else absolute,
            )
            for name, section, absolute in field_specs
        )

    state_fields = fields("state")
    velocity_provenance = {}
    if include_joint_velocity:
        velocity_stats = load_velocity_statistics(velocity_stats_path)["velocity"]
        state_fields += tuple(
            TransformField(
                key=key,
                size=6,
                normalization="q99",
                statistics={
                    name: list(velocity_stats[name][offset : offset + 6])
                    for name in ("q01", "q99")
                },
                per_horizon_statistics=None,
                absolute=True,
            )
            for key, offset in zip(VELOCITY_FIELDS, (0, 6), strict=True)
        )
        velocity_provenance = {
            "include_joint_velocity": True,
            "velocity_method": VELOCITY_METHOD,
            "velocity_first_frame": VELOCITY_FIRST_FRAME,
            "velocity_stats_path": str(velocity_stats_path),
        }

    return CheckpointTransformArtifact(
        image_input_range=image_input_range,
        embodiments={
            "arx_x5_joint": EmbodimentTransform(
                embodiment_id=embodiment_id,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                state_fields=state_fields,
                action_fields=fields("action"),
            )
        },
        provenance={
            "source": "robodojo-joint-posttraining",
            "normalization": mode,
            "action_space": "robodojo_joint_14d",
            "relative_joint_action": relative,
            "stats_path": str(Path(stats_path)),
            **velocity_provenance,
        },
    )


__all__ = [
    "GRIPPER_INDICES",
    "JOINT_DIM",
    "JOINT_LAYOUT",
    "LEFT_GRIPPER",
    "LEFT_JOINTS",
    "RIGHT_GRIPPER",
    "RIGHT_JOINTS",
    "ROBODOJO_JOINT_DEMO_DATASET",
    "ROBODOJO_JOINT_GOAL_DATASET",
    "ROBODOJO_JOINT_INITIAL_DATASET",
    "ROBODOJO_JOINT_TEXT_DATASET",
    "build_robodojo_joint_transform_bundle",
    "joint_actions_relative_to_chunk_anchor",
    "load_joint_statistics",
    "normalize_q99",
    "normalize_joint",
]
