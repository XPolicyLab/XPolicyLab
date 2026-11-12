"""RoboTwin EEF representations at the XPolicyLab boundary (no frame change)."""

import importlib.util
from pathlib import Path

import numpy as np

# Load the upstream pure-numpy helper without importing benchmarks.utils,
# whose package initializer imports websocket clients using global names.
_helper = Path(__file__).parent / "OpenWAM/benchmarks/utils/action_conversion.py"
_spec = importlib.util.spec_from_file_location("_openwam_action_conversion", _helper)
conversion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conversion)


def pose_order(pose, source, target):
    pose = np.asarray(pose, dtype=np.float32).copy()
    if source not in ("xyzw", "wxyz") or target not in ("xyzw", "wxyz"):
        raise ValueError("Quaternion order must be xyzw or wxyz.")
    if pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected finite xyz + quaternion poses.")
    norm = np.linalg.norm(pose[..., 3:], axis=-1)
    if not np.allclose(norm, 1, atol=1e-3):
        raise ValueError(
            "Expected unit quaternions; verify state/action representation."
        )
    if source != target:
        order = [4, 5, 6, 3] if source == "wxyz" else [6, 3, 4, 5]
        pose[..., 3:] = pose[..., order]
    return pose


def state_to_eef20(state):
    # Boundary poses are wxyz, matching transforms3d.mat2quat in this runtime.
    # OpenWAM's RoboTwin dataloader and eval client feed those raw components
    # to quat_xyzw_to_rot6d without swapping. Reordering here would change the
    # rot6d the released checkpoint was trained on.
    left = pose_order(state["left_ee_pose"], "wxyz", "wxyz")
    right = pose_order(state["right_ee_pose"], "wxyz", "wxyz")
    return conversion.robotwin_endpose_to_eef20d(
        left, right, state["left_ee_joint_state"], state["right_ee_joint_state"]
    )


def eef20_to_steps(chunk):
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[1] != 20 or not np.isfinite(chunk).all():
        raise ValueError("Expected finite [H,20] EEF actions.")
    steps = []
    for row in chunk:
        ee = conversion.eef20d_to_ee16d(row)
        steps.append(
            {
                "left_ee_pose": pose_order(ee[:7], "wxyz", "wxyz"),
                "left_ee_joint_state": np.clip(ee[7:8], 0, 1),
                "right_ee_pose": pose_order(ee[8:15], "wxyz", "wxyz"),
                "right_ee_joint_state": np.clip(ee[15:16], 0, 1),
            }
        )
    return steps
