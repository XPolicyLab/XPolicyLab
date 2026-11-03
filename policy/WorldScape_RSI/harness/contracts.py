"""Validate the official RGB/proprioception and joint-action boundaries."""
from __future__ import annotations

import numpy as np


def action_dimensions(env_cfg_type: str) -> dict[str, int]:
    from XPolicyLab.utils.process_data import get_robot_action_dim_info
    info = get_robot_action_dim_info(env_cfg_type)
    arm, ee = info["arm_dim"], info["ee_dim"]
    if len(arm) != 2 or len(ee) != 2:
        raise ValueError("This Harness requires the documented two-arm interface")
    return {side + suffix: int(size)
            for i, side in enumerate(("left", "right"))
            for suffix, size in (("_arm_joint_state", arm[i]), ("_ee_joint_state", ee[i]))}


def observation(raw: dict, dims: dict[str, int]) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("state"), dict):
        raise ValueError("Expected an official observation dictionary")
    instruction = raw.get("instruction")
    if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
        instruction = instruction[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Public instruction is missing")
    state = {}
    shapes = {**dims, "left_ee_pose": 7, "right_ee_pose": 7}
    for key, size in shapes.items():
        array = np.asarray(raw["state"].get(key), dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError("Invalid public robot channel: " + key)
        state[key] = array.copy()
    images = {}
    for camera, value in raw.get("vision", {}).items():
        if not isinstance(value, dict) or "color" not in value:
            continue
        array = np.asarray(value["color"])
        if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                or array.shape[0] < 1 or array.shape[1] < 1):
            raise ValueError("Policy server must provide decoded RGB arrays")
        images[camera] = {"color": array.copy()}
    if not images:
        raise ValueError("At least one RGB view is required")
    result = {"instruction": instruction, "state": state, "vision": images,
              "data_format_version": "v1.0"}
    additional = raw.get("additional_info")
    if isinstance(additional, dict) and "frequency" in additional:
        frequency = additional["frequency"]
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Invalid public control frequency")
        result["additional_info"] = {"frequency": frequency}
    return result


def action(value: dict, dims: dict[str, int]) -> dict:
    if not isinstance(value, dict) or set(value) != set(dims):
        raise ValueError("Unexpected joint-action channels")
    result = {}
    for key, size in dims.items():
        array = np.asarray(value[key], dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError("Invalid joint-action value: " + key)
        if key.endswith("_ee_joint_state") and (np.any(array < 0) or np.any(array > 1)):
            raise ValueError("ARX X5 gripper command must be in [0, 1]")
        result[key] = array.copy()
    return result
