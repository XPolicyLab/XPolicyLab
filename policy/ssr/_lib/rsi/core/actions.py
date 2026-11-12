from __future__ import annotations

import numpy as np

OPEN = 1.0


def ee_action(left_pose, right_pose, left_grip: float, right_grip: float) -> dict:
    return {
        "left_ee_pose": np.asarray(left_pose, dtype=np.float32).reshape(7),
        "right_ee_pose": np.asarray(right_pose, dtype=np.float32).reshape(7),
        "left_ee_joint_state": np.array([float(np.clip(left_grip, 0.0, 1.0))], dtype=np.float32),
        "right_ee_joint_state": np.array([float(np.clip(right_grip, 0.0, 1.0))], dtype=np.float32),
    }


def joint_action(left_q, right_q, left_grip: float, right_grip: float) -> dict:
    return {
        "left_arm_joint_state": np.asarray(left_q, dtype=np.float32).reshape(6),
        "right_arm_joint_state": np.asarray(right_q, dtype=np.float32).reshape(6),
        "left_ee_joint_state": np.array([float(np.clip(left_grip, 0.0, 1.0))], dtype=np.float32),
        "right_ee_joint_state": np.array([float(np.clip(right_grip, 0.0, 1.0))], dtype=np.float32),
    }


def hold(obs, n: int = 1, left_grip: float | None = None, right_grip: float | None = None) -> list[dict]:
    lg = float(obs.state["left_ee_joint_state"][0]) if left_grip is None else left_grip
    rg = float(obs.state["right_ee_joint_state"][0]) if right_grip is None else right_grip
    a = ee_action(obs.left_ee_pose, obs.right_ee_pose, lg, rg)
    return [a] * n
