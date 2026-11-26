"""Optional causal arm-joint velocity features for RoboDojo states."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


VELOCITY_DIM = 12
VELOCITY_LAYOUT = ("left_arm_joint_velocity[0:6]", "right_arm_joint_velocity[0:6]")
VELOCITY_METHOD = "backward_difference_per_second"
VELOCITY_FIRST_FRAME = "zero"
VELOCITY_FIELDS = ("state.left_joint_velocity", "state.right_joint_velocity")


def validate_frequency(frequency: float) -> float:
    value = np.asarray(frequency)
    if value.ndim != 0:
        raise ValueError("joint velocity frequency must be a positive finite scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("joint velocity frequency must be a positive finite scalar")
    return result


def arm_joint_velocity(states: np.ndarray, frequency: float) -> np.ndarray:
    """Difference raw consecutive episode frames; the first velocity is zero.

    Call before temporal packing/subsampling. Gripper channels are excluded.
    Units are the source joint-position unit per second; no angle wrapping or
    smoothing is applied.
    """

    frequency = validate_frequency(frequency)
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[-1] != 14 or not len(states):
        raise ValueError("joint velocity requires non-empty raw states [T,14]")
    if not np.all(np.isfinite(states)):
        raise ValueError("joint velocity requires finite raw states")
    joints = np.concatenate((states[:, :6], states[:, 7:13]), axis=-1)
    velocity = np.zeros_like(joints)
    velocity[1:] = np.diff(joints, axis=0) * frequency
    if not np.all(np.isfinite(velocity)):
        raise ValueError("computed joint velocity must be finite")
    return velocity


def load_velocity_statistics(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError(
            "include_joint_velocity requires velocity_stats_path; run "
            "tools/data/compute_robodojo_velocity_stats.py first"
        )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "1":
        raise ValueError("joint velocity statistics require schema_version '1'")
    if tuple(value.get("layout", ())) != VELOCITY_LAYOUT:
        raise ValueError("joint velocity statistics use an unexpected layout")
    if (
        value.get("method") != VELOCITY_METHOD
        or value.get("first_frame") != VELOCITY_FIRST_FRAME
    ):
        raise ValueError(
            "joint velocity statistics use an incompatible difference method"
        )
    group = value.get("velocity")
    if not isinstance(group, Mapping):
        raise ValueError("joint velocity statistics are missing 'velocity'")
    bounds = []
    for name in ("q01", "q99"):
        bound = np.asarray(group.get(name), dtype=np.float32)
        if bound.shape != (VELOCITY_DIM,) or not np.all(np.isfinite(bound)):
            raise ValueError(f"velocity.{name} must contain 12 finite values")
        bounds.append(bound)
    if np.any(bounds[1] < bounds[0]):
        raise ValueError("velocity.q99 must be greater than or equal to q01")
    return value


def normalize_velocity(value: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != VELOCITY_DIM:
        raise ValueError("joint velocity arrays must end in dimension 12")
    low = np.asarray(statistics["q01"], dtype=np.float32)
    scale = np.asarray(statistics["q99"], dtype=np.float32) - low
    safe = np.where(scale == 0, 1.0, scale)
    normalized = np.where(scale == 0, array, 2.0 * (array - low) / safe - 1.0)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)
