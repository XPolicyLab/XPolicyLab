"""Controlled live-embodiment probes for validating ARX X5 operating notes."""

from __future__ import annotations

from typing import Any

import numpy as np


def _vector(state: dict[str, Any], key: str, width: int) -> np.ndarray:
    value = np.asarray(state[key], dtype=np.float64).reshape(-1)
    if value.shape != (width,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"probe expected finite state[{key!r}] with shape {(width,)}")
    return value


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _rotation_vector_world(before_wxyz: np.ndarray, after_wxyz: np.ndarray) -> np.ndarray:
    """Return the world-frame axis-angle vector taking before to after."""
    before = before_wxyz / np.linalg.norm(before_wxyz)
    after = after_wxyz / np.linalg.norm(after_wxyz)
    relative = _quaternion_multiply(
        after, np.asarray([before[0], -before[1], -before[2], -before[3]])
    )
    if relative[0] < 0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(relative[0]))
    return relative[1:] / vector_norm * angle


def _action(
    left: np.ndarray,
    left_gripper: np.ndarray,
    right: np.ndarray,
    right_gripper: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "left_arm_joint_state": left.astype(np.float32, copy=True),
        "left_ee_joint_state": left_gripper.astype(np.float32, copy=True),
        "right_arm_joint_state": right.astype(np.float32, copy=True),
        "right_ee_joint_state": right_gripper.astype(np.float32, copy=True),
    }


def probe_joint_directions(task_env: Any, *, delta_rad: float = 0.05) -> dict[str, Any]:
    """Perturb each arm joint in both directions and restore the reset state."""
    if not np.isfinite(delta_rad) or delta_rad <= 0:
        raise ValueError("delta_rad must be finite and > 0")
    initial_state = dict(task_env.get_obs().get("state") or {})
    baseline = {
        "left_arm": _vector(initial_state, "left_arm_joint_state", 6),
        "left_gripper": _vector(initial_state, "left_ee_joint_state", 1),
        "right_arm": _vector(initial_state, "right_arm_joint_state", 6),
        "right_gripper": _vector(initial_state, "right_ee_joint_state", 1),
    }
    baseline_action = _action(
        baseline["left_arm"],
        baseline["left_gripper"],
        baseline["right_arm"],
        baseline["right_gripper"],
    )
    probes: list[dict[str, Any]] = []
    for side in ("left", "right"):
        pose_key = f"{side}_ee_pose"
        for joint_index in range(6):
            for signed_delta in (delta_rad, -delta_rad):
                if task_env.is_episode_end():
                    raise RuntimeError("RoboDojo episode ended before the probe completed")
                before_state = dict(task_env.get_obs().get("state") or {})
                before_pose = _vector(before_state, pose_key, 7)
                left = baseline["left_arm"].copy()
                right = baseline["right_arm"].copy()
                arm = left if side == "left" else right
                arm[joint_index] += signed_delta
                try:
                    task_env.take_action(
                        _action(
                            left,
                            baseline["left_gripper"],
                            right,
                            baseline["right_gripper"],
                        )
                    )
                    after_state = dict(task_env.get_obs().get("state") or {})
                    measured_joint = _vector(
                        after_state, f"{side}_arm_joint_state", 6
                    )
                    after_pose = _vector(after_state, pose_key, 7)
                    probes.append(
                        {
                            "dimension": f"{side}_joint{joint_index + 1}",
                            "delta_rad": float(signed_delta),
                            "commanded_joint": float(arm[joint_index]),
                            "measured_joint": float(measured_joint[joint_index]),
                            "before_pose_wxyz": before_pose.tolist(),
                            "after_pose_wxyz": after_pose.tolist(),
                            "position_delta_world_m": (
                                after_pose[:3] - before_pose[:3]
                            ).tolist(),
                            "rotation_vector_world_rad": _rotation_vector_world(
                                before_pose[3:], after_pose[3:]
                            ).tolist(),
                        }
                    )
                finally:
                    task_env.take_action(baseline_action)
                    task_env.get_obs()
    return {
        "schema_version": "arx-x5-joint-probe/v1",
        "delta_rad": float(delta_rad),
        "baseline": {name: value.tolist() for name, value in baseline.items()},
        "probes": probes,
    }
