"""Explicit information boundary, applied before transport AND in the model.

Camera matrices are not taken from an episode: the head calibration is a static
prior. No seed, reward, object identity/pose, depth, or layout is sent to policy.
"""
from copy import deepcopy
import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ARMS = ("left", "right")
STATE_KEYS = tuple(f"{s}_{k}" for s in ARMS for k in
                   ("arm_joint_state", "ee_pose", "ee_joint_state"))
TASKS = ("play_Xylophone", "plug_in_charger", "pour_balls_into_vase",
         "pour_liquid_into_cup", "press_by_number", "push_T", "put_bottles_into_dustbin")


def restrict_observation(raw):
    """Select allowed fields, never forward an arbitrary observation mapping."""
    result = {"vision": {}, "state": {},
              "instruction": str(raw.get("instruction", raw.get("instructions", "")))}
    for camera in CAMERAS:
        frame = raw.get("vision", {}).get(camera, {})
        if "color" in frame:
            result["vision"][camera] = {"color": frame["color"]}
            if "shape" in frame:
                result["vision"][camera]["shape"] = deepcopy(frame["shape"])
    for key in STATE_KEYS:
        if key in raw.get("state", {}):
            result["state"][key] = np.asarray(raw["state"][key], dtype=float).copy()
        elif key.endswith("ee_joint_state") and key in raw.get("action", {}):
            result["state"][key] = np.asarray(raw["action"][key], dtype=float).copy()
    return result


def validate_observation(obs):
    if set(obs) != {"vision", "state", "instruction"}:
        raise ValueError("Policy observation must be explicitly filtered")
    for side in ARMS:
        pose = np.asarray(obs["state"].get(f"{side}_ee_pose", []))
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"Missing/invalid {side} end-effector pose")
    frame = obs["vision"].get("cam_head", {}).get("color")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Decoded HWC RGB cam_head required; shared XPolicyLab server decodes")


def validate_action(action):
    expected = {f"{s}_{k}" for s in ARMS for k in ("ee_pose", "ee_joint_state")}
    if set(action) != expected:
        raise ValueError("Action keys do not match dual-arm end-effector control")
    for side in ARMS:
        p = np.asarray(action[f"{side}_ee_pose"], dtype=float)
        g = np.asarray(action[f"{side}_ee_joint_state"], dtype=float)
        if p.shape != (7,) or g.shape != (1,) or not np.isfinite(p).all() or not np.isfinite(g).all():
            raise ValueError("Nonfinite or malformed action")
        if not 0 <= g[0] <= 1 or abs(np.linalg.norm(p[3:]) - 1) > 1e-3:
            raise ValueError("Invalid gripper/quaternion")
