"""DM0 32D state/action layout helpers (aligned with transform_dm0_dexdata_format.py)."""

from __future__ import annotations

import numpy as np

STATE_DIM = 32
ACTION_CHUNK_SIZE = 50
GRIPPER_NON_DELTA_INDICES = (6, 20)


def quat_wxyz_to_rotm(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotm2aa(rotm: np.ndarray) -> np.ndarray:
    rotm = np.asarray(rotm, dtype=np.float32).reshape(3, 3)
    theta = np.arccos(np.clip((np.trace(rotm) - 1.0) / 2.0, -1.0, 1.0))
    if theta <= 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rotm[2, 1] - rotm[1, 2],
            rotm[0, 2] - rotm[2, 0],
            rotm[1, 0] - rotm[0, 1],
        ],
        dtype=np.float32,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    return axis / axis_norm * theta


def aa_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q_xyzw = Rotation.from_rotvec(np.asarray(axis_angle, dtype=np.float64).reshape(3)).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def pack_dm0_state(observation: dict) -> np.ndarray:
    """Pack XPolicyLab observation state into DM0 32D proprio vector."""
    state_dict = observation["state"]
    vec = np.zeros(STATE_DIM, dtype=np.float32)

    for side, base in (("left", 0), ("right", 14)):
        ee_pose = np.asarray(state_dict[f"{side}_ee_pose"], dtype=np.float32).reshape(-1)
        arm_joint = np.asarray(state_dict[f"{side}_arm_joint_state"], dtype=np.float32).reshape(-1)[:6]
        gripper = np.asarray(state_dict[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)[:1]

        ee_pos = ee_pose[:3]
        ee_aa = rotm2aa(quat_wxyz_to_rotm(ee_pose[3:7]))

        vec[base : base + 6] = arm_joint
        vec[base + 6] = gripper[0]
        vec[base + 7 : base + 10] = ee_pos
        vec[base + 10 : base + 13] = ee_aa

    return vec


def unpack_dm0_action_step(
    action_vec: np.ndarray,
    action_type: str,
    robot_action_dim_info: dict,
) -> dict:
    """Convert one 32D DM0 action vector into XPolicyLab action dict."""
    vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] < STATE_DIM:
        raise ValueError(f"DM0 action dim expected >= {STATE_DIM}, got {vec.shape[0]}")

    action_dict: dict = {}
    num_arms = len(robot_action_dim_info["arm_dim"])

    for side, base, arm_idx in (("left", 0, 0), ("right", 14, 1)):
        if arm_idx >= num_arms:
            break

        arm_dim = robot_action_dim_info["arm_dim"][arm_idx]
        ee_dim = robot_action_dim_info["ee_dim"][arm_idx]

        arm_joint = vec[base : base + 6][:arm_dim]
        gripper = vec[base + 6 : base + 7][:ee_dim]
        ee_pos = vec[base + 7 : base + 10]
        ee_aa = vec[base + 10 : base + 13]
        ee_pose = np.concatenate([ee_pos, aa_to_quat_wxyz(ee_aa)], axis=0)

        if action_type == "joint":
            action_dict[f"{side}_arm_joint_state"] = arm_joint.astype(np.float32)
        else:
            action_dict[f"{side}_ee_pose"] = ee_pose.astype(np.float32)

        action_dict[f"{side}_ee_joint_state"] = gripper.astype(np.float32)

    return action_dict
