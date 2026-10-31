"""Opt-in Q1 absolute EE targets, without a planner or a new execution path.

Coordinates are the adapter's native EE reference point in env_relative_world,
metres and unit wxyz quaternions. This is not a grasp-point conversion. A target
may exceed the executable prefix: returning rows never asserts physical arrival.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

import numpy as np

ARMS = ("left", "right")
POSITION_PER_ROW_M = .005
ROTATION_PER_ROW_RAD = .025
FINISH_CAPABILITY = "oeh_finish"


def finish_schema() -> dict:
    return {"type": "object", "additionalProperties": False,
            "description": "End this episode without another motion command. This is the agent's declaration, "
                           "not benchmark success. Use only with the current observation_id.",
            "required": ["observation_id", "status", "reason"],
            "properties": {
                "observation_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "status": {"type": "string", "enum": ["completed", "unable_to_continue"]},
                "reason": {"type": "string", "minLength": 1, "maxLength": 512}}}


def validate_finish(params: dict) -> dict:
    if not isinstance(params, dict) or set(params) != {"observation_id", "status", "reason"}:
        raise ValueError("finish requires exactly observation_id, status and reason")
    if params["status"] not in ("completed", "unable_to_continue"):
        raise ValueError("finish status must be completed or unable_to_continue")
    for key, limit in (("observation_id", 128), ("reason", 512)):
        if not isinstance(params[key], str) or not params[key].strip() or len(params[key]) > limit:
            raise ValueError(f"invalid finish {key}")
    return copy.deepcopy(params)


def targets_schema() -> dict:
    target = {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {
            "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                         "description": "Absolute native EE position, env_relative_world, metres; not fingertip position."},
            "quaternion_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                                "description": "Absolute native EE orientation, unit quaternion [w,x,y,z]."},
            "opening": {"type": "number", "minimum": 0, "maximum": 1,
                        "description": "Native gripper command, not measured object width."}}}
    return {"type": "object", "additionalProperties": False, "minProperties": 1,
            "description": "One or both arms. Omitted pose components hold their observed values; omitted opening "
                           "holds the last acknowledged command. execute_steps is only a bounded motion prefix, "
                           "not a guarantee of reaching the requested target.",
            "properties": {arm: copy.deepcopy(target) for arm in ARMS}}


def _vector(value, size, name):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1) or len(value) != size):
        raise ValueError(f"{name} must have {size} finite numeric values")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value):
        raise ValueError(f"{name} must have {size} finite numeric values")
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {size} finite numeric values")
    return result


def _quaternion(value, name):
    result = _vector(value, 4, name)
    norm = float(np.linalg.norm(result))
    if abs(norm - 1) > 1e-3:
        raise ValueError(f"{name} must be a unit wxyz quaternion")
    # Only unit-quaternion floating-point drift is corrected, never arbitrary input.
    return result / norm


def validate_targets(targets: dict) -> dict:
    if not isinstance(targets, dict) or not targets or set(targets) - set(ARMS):
        raise ValueError("pose targets must name one or both of left/right")
    result = {}
    for arm, value in targets.items():
        if not isinstance(value, dict) or not value or set(value) - {"position", "quaternion_wxyz", "opening"}:
            raise ValueError(f"pose target {arm} needs position, quaternion_wxyz and/or opening only")
        result[arm] = {}
        if "position" in value:
            result[arm]["position"] = _vector(value["position"], 3, f"{arm}.position")
        if "quaternion_wxyz" in value:
            result[arm]["quaternion_wxyz"] = _quaternion(value["quaternion_wxyz"], f"{arm}.quaternion_wxyz")
        if "opening" in value:
            opening = value["opening"]
            if isinstance(opening, (bool, np.bool_)) or not isinstance(opening, Real) or not math.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError(f"{arm}.opening must be finite and in [0,1]")
            result[arm]["opening"] = float(opening)
    return result


def pose_goal(current: dict, targets: dict) -> dict:
    goal = copy.deepcopy(current)
    for arm, target in validate_targets(targets).items():
        if "position" in target:
            goal[arm + "_ee_pose"][:3] = target["position"]
        if "quaternion_wxyz" in target:
            goal[arm + "_ee_pose"][3:] = target["quaternion_wxyz"]
        if "opening" in target:
            goal[arm + "_ee_joint_state"] = np.array([target["opening"]], dtype=float)
    return goal


def orientation_error_rad(actual, requested) -> float:
    a = _quaternion(actual, "actual orientation")
    b = _quaternion(requested, "requested orientation")
    return 2 * math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1)))


def _slerp(a, b, fraction):
    a, b = _quaternion(a, "current orientation"), _quaternion(b, "goal orientation")
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, 0, 1))
    if dot > .999999:
        out = (1 - fraction) * a + fraction * b
    else:
        theta = math.acos(dot)
        out = (math.sin((1 - fraction) * theta) * a + math.sin(fraction * theta) * b) / math.sin(theta)
    return out / np.linalg.norm(out)


def pose_prefix(current: dict, goal: dict, count: int) -> list[dict]:
    """Synchronous arm prefixes; caller validates workspace and admits every row.

    Both arms use one horizon so no arm silently completes a long motion beyond
    the requested prefix. These are proposed commands, not measured trajectories.
    """
    if type(count) is not int or count < 1:
        raise ValueError("pose prefix count must be a positive integer")
    horizon = count
    for arm in ARMS:
        a, b = current[arm + "_ee_pose"], goal[arm + "_ee_pose"]
        horizon = max(horizon, math.ceil(float(np.linalg.norm(b[:3] - a[:3])) / POSITION_PER_ROW_M),
                      math.ceil(orientation_error_rad(a[3:], b[3:]) / ROTATION_PER_ROW_RAD))
    rows = []
    for step in range(1, count + 1):
        row = {}
        for arm in ARMS:
            key = arm + "_ee_pose"
            a, b = current[key], goal[key]
            row[key] = np.concatenate((a[:3] + step / horizon * (b[:3] - a[:3]),
                                       _slerp(a[3:], b[3:], step / horizon))).astype(np.float32)
            grip = arm + "_ee_joint_state"
            row[grip] = (current[grip] + step / count * (goal[grip] - current[grip])).astype(np.float32)
        rows.append(row)
    return rows


def pose_feedback(targets: dict, observed_state: dict) -> dict:
    """Errors to requested components from a NEW observation, without success labels.

    The caller owns observation freshness and time association. Gripper state can
    be a previous command rather than a measurement, so no grasp/width inference
    or opening-error calculation is made here.
    """
    result = {}
    for arm, target in validate_targets(targets).items():
        measured = _vector(observed_state[arm + "_ee_pose"], 7, f"{arm} observed EE pose")
        entry = {}
        if "position" in target:
            entry["position_error_m"] = float(np.linalg.norm(measured[:3] - target["position"]))
        if "quaternion_wxyz" in target:
            entry["orientation_error_rad"] = orientation_error_rad(measured[3:], target["quaternion_wxyz"])
        if "opening" in target:
            entry["requested_opening_command"] = target["opening"]
        result[arm] = entry
    return result
