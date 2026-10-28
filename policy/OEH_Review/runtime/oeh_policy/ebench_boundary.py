"""EBench's native boundary: GenManip's ``ee_pose`` action dict and observation state, checked and copied.

GenManip 1958e37 (EBench's simulator) drives the lift2 robot: a mobile base, a lift held at 0.46 m and two six-joint
R5a arms with two-finger grippers (``genmanip/extensions/robots/default/r5a/lift2.py``). One native action is one
dict (``genmanip/core/evaluator/env.py`` ``IsaacEvalEnvRay.step``, line 442; it reads the dict at lines 480-497)::

    {"action": [(left_xyz, left_quat_wxyz, [finger, finger]), (right_xyz, right_quat_wxyz, [finger, finger])],
     "control_type": "ee_pose", "is_rel": False, "base_motion": [dx_m, dy_m, dyaw_deg], "base_is_rel": True}

Each arm's pose is an absolute target in that arm's own base frame (the cuRobo ``base_link`` of its planner, which
moves with the mobile base): GenManip solves IK per arm on the server (``genmanip/core/evaluator/utils.py``
``parse_embodiment_action``, lines 260-324) and, when the solve fails, holds that arm's measured joints for the step
(lines 316-318).
The finger targets go to the finger joints unchanged, 0 closed to 0.044 m open per finger. ``base_motion`` with
``base_is_rel`` adds a displacement to the base's target pose, clipped to +-0.015 m and +-1 deg per step
(``genmanip/core/robot/dualarm_manip.py`` ``delta_move_to``, lines 113-126). This is exactly what OpenWAM's released
EBench bridge sends (XPolicyLab fa431ec ``benchmarks/utils/action_conversion.py`` ``raw23_to_ebench_action``): the
positions and quaternions are plain Python lists on purpose, because GenManip concatenates ``position +
orientation`` before IK.

The observation state (``env.py`` ``get_obs``, the dict at lines 340-352) has ``state.ee_pose`` (per arm
``[xyz, quat_wxyz]`` from the same planner's forward kinematics, so in the same per-arm frames), ``state.gripper``
(the four measured finger positions ``[L, L, R, R]``, metres), ``state.base`` (the base joints
``[x_m, y_m, yaw_rad]``) and ``state.joints``.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

ARMS = ("left", "right")
GRIPPER_OPEN = 0.044  # lift2.py: gripper_open [0.044, 0.044], gripper_close [0.0, 0.0] (metres per finger)
CONTROL_TYPE = "ee_pose"
ACTION_KEYS = ("action", "control_type", "is_rel", "base_motion", "base_is_rel")
# OpenWAM's bridge refuses a quaternion whose norm is more than this from 1 (openwam2ebench_interface.py act()).
QUATERNION_TOLERANCE = 0.05
# A correction holds the base: a zero displacement keeps the base's target pose (delta_move_to adds it).
HOLD_BASE = (0.0, 0.0, 0.0)


def _floats(value, size: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of {size} numbers")
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "fiu" or not np.isfinite(array.astype(float)).all():
        raise ValueError(f"{name} must be {size} finite numbers")
    return [float(v) for v in array.astype(float)]


def native_arm(value, arm: str) -> tuple[list[float], list[float], list[float]]:
    """One arm's ``(position, quaternion_wxyz, fingers)`` triple, checked: three, four and two finite numbers, a
    quaternion OpenWAM's bridge would accept and fingers in GenManip's range."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"the {arm} arm's action is (position, quaternion_wxyz, fingers)")
    position = _floats(value[0], 3, f"{arm} position")
    quaternion = _floats(value[1], 4, f"{arm} quaternion_wxyz")
    fingers = _floats(value[2], 2, f"{arm} fingers")
    norm = math.sqrt(sum(v * v for v in quaternion))
    if not abs(norm - 1.0) <= QUATERNION_TOLERANCE:
        raise ValueError(f"the {arm} quaternion is not unit (norm {norm:.4f})")
    if not all(0.0 <= v <= GRIPPER_OPEN for v in fingers):
        raise ValueError(f"the {arm} finger targets must lie in [0, {GRIPPER_OPEN}] m")
    return position, quaternion, fingers


def native_action(action) -> dict:
    """A checked copy of one native action, every number a plain float and the containers OpenWAM's bridge uses: a
    tuple per arm holding three lists (GenManip concatenates the position and quaternion lists before IK)."""
    if not isinstance(action, Mapping) or set(action) != set(ACTION_KEYS):
        raise ValueError(f"an EBench action has exactly {list(ACTION_KEYS)}")
    if action["control_type"] != CONTROL_TYPE or action["is_rel"] is not False or action["base_is_rel"] is not True:
        raise ValueError("EBench actions are absolute ee_pose arm targets with a relative base_motion")
    arms = action["action"]
    if not isinstance(arms, (list, tuple)) or len(arms) != len(ARMS):
        raise ValueError("an EBench action holds one entry per arm, left then right")
    return {"action": [native_arm(value, arm) for value, arm in zip(arms, ARMS)],
            "control_type": CONTROL_TYPE, "is_rel": False,
            "base_motion": _floats(action["base_motion"], 3, "base_motion"), "base_is_rel": True}


def opening(fingers) -> float:
    """The opening (1 open, 0 closed) of a pair of finger positions or targets, clipped into [0, 1]."""
    value = float(np.mean(np.asarray(fingers, dtype=float)))
    return float(np.clip(value / GRIPPER_OPEN, 0.0, 1.0))


def fingers(opening_value: float) -> list[float]:
    """Both finger targets for an opening, as GenManip takes them (the bridge commands both fingers alike)."""
    width = float(np.clip(float(opening_value), 0.0, 1.0)) * GRIPPER_OPEN
    return [width, width]
