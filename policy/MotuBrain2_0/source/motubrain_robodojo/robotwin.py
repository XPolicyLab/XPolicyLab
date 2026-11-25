"""RoboTwin end-effector convention at the XPolicyLab boundary."""

import numpy as np
from morph.convs.geometry import mat_to_rot, rot_to_mat


# Columns are the optical tool axes expressed in the simulator link frame.
_TOOL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)


def convert_tool_poses(packed, *, to_model):
    """Convert both absolute tool poses, preserving XYZ and gripper values."""
    result = packed.copy()
    arms = result.reshape(-1, 8)
    if to_model:
        rotation = rot_to_mat(arms[:, 3:7], "wxyz") @ _TOOL
        arms[:, 3:7] = mat_to_rot(rotation, "xyzw")
    else:
        rotation = rot_to_mat(arms[:, 3:7], "xyzw") @ _TOOL.T
        arms[:, 3:7] = mat_to_rot(rotation, "wxyz")
    return result


def camera_pose(obs):
    """Convert the head camera's OpenCV world-to-camera matrix to its pose."""
    extrinsic = np.asarray(obs["vision"]["cam_head"]["extrinsics_matrix"], dtype=np.float32)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, np.array([0, 0, 0, 1], dtype=np.float32)))
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("Expected a finite head-camera extrinsic matrix")
    return np.linalg.inv(extrinsic)
