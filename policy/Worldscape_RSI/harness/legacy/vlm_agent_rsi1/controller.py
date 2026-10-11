"""Arm state and geometry shared by the policy server and the simulation client.

RoboDojo reports each arm as an absolute 7-D wrist pose ``[x, y, z, qw, qx, qy, qz]``
in the environment-local frame plus a normalised gripper opening. This module
reads that state, holds the few execution knobs the discrete commands need
(:class:`ControllerConfig`) and provides the quaternion helpers
:mod:`main_route` resolves commands with. Only numpy is imported: the module
is loaded by both the policy server and the Isaac Sim client.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# RoboDojo names the two target arms "left"/"right"; the observation and
# action keys are built from those prefixes.
DUAL_ARM_PREFIXES = ("left", "right")
MAX_ACTION_STEPS = 10000

GRIPPER_OPEN = 1.0


class ControllerError(ValueError):
    """Raised when an observation or configuration cannot be interpreted."""


class ArmState:
    """Current end-effector pose and gripper opening of one arm.

    ``gripper`` is the commanded opening RoboDojo reports back; ``gripper_real``
    is the fingers' measured position when the simulation client supplied it,
    and ``None`` otherwise. ``gripper_closed_floor`` is its physical lower
    opening limit in the same command scale. Neither confirms a target grasp.
    """

    __slots__ = ("gripper", "gripper_real", "gripper_closed_floor", "pos", "prefix", "quat")

    def __init__(self, prefix: str, pos: np.ndarray, quat: np.ndarray, gripper: float,
                 gripper_real: float | None = None, gripper_closed_floor: float | None = None):
        self.prefix = prefix
        self.pos = np.asarray(pos, dtype=float).reshape(3)
        self.quat = _normalize_quat(np.asarray(quat, dtype=float).reshape(4))
        self.gripper = float(gripper)
        self.gripper_real = None if gripper_real is None else float(gripper_real)
        self.gripper_closed_floor = None if gripper_closed_floor is None else float(gripper_closed_floor)
        if not np.all(np.isfinite(self.pos)) or not math.isfinite(self.gripper):
            raise ControllerError(f"{prefix} arm state must contain only finite values")
        if self.gripper_real is not None and not math.isfinite(self.gripper_real):
            self.gripper_real = None
        if self.gripper_closed_floor is not None and (
            not math.isfinite(self.gripper_closed_floor) or not 0 <= self.gripper_closed_floor <= 1
        ):
            self.gripper_closed_floor = None

    @property
    def yaw_deg(self) -> float:
        return math.degrees(quat_to_yaw(self.quat))

    def pose7(self) -> list[float]:
        return [*self.pos.tolist(), *self.quat.tolist()]


def read_arm_states(obs: dict) -> dict[str, ArmState]:
    """Extract per-arm state from a RoboDojo observation.

    Raises:
        ControllerError: the observation does not describe a dual-arm robot,
            which is the only layout RoboDojo's ``ee`` action mode supports.
    """
    state = (obs or {}).get("state") or {}
    # RoboDojo derives `<side>_ee_joint_state` from the last applied gripper
    # command, not from the fingers, so it reaches 0 whether or not an object
    # stopped them. `deploy._gripper_real` reads the joint itself and puts it
    # here; it is absent when that probe is off or unavailable.
    measured = {str(key): value for key, value in ((obs or {}).get("gripper_real") or {}).items()}
    closed_floors = (obs or {}).get("gripper_closed_floor") or {}
    states: dict[str, ArmState] = {}
    for prefix in DUAL_ARM_PREFIXES:
        pose = state.get(f"{prefix}_ee_pose")
        if pose is None:
            raise ControllerError(
                f"observation has no '{prefix}_ee_pose'; the VLM policy targets the "
                f"dual-arm ee action mode (available state keys: {sorted(state)})"
            )
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.shape[0] != 7:
            raise ControllerError(f"'{prefix}_ee_pose' must be 7-D, got shape {pose.shape}")
        gripper = state.get(f"{prefix}_ee_joint_state", [GRIPPER_OPEN])
        gripper = float(np.asarray(gripper, dtype=float).reshape(-1)[0])
        states[prefix] = ArmState(prefix, pose[:3], pose[3:], gripper, measured.get(prefix), closed_floors.get(prefix))
    return states


def action_dict(states: dict[str, ArmState]) -> dict[str, list[float]]:
    """Build the env action dict that commands every arm to the given state.

    ``take_action_batch`` reads all four keys unconditionally, so an action
    that moves one arm still has to name the other one's hold pose.
    """
    action: dict[str, list[float]] = {}
    for prefix, arm in states.items():
        action[f"{prefix}_ee_pose"] = arm.pose7()
        action[f"{prefix}_ee_joint_state"] = [float(np.clip(arm.gripper, 0.0, 1.0))]
    return action


class ControllerConfig:
    """Execution knobs of the discrete commands (the ``controller`` block of deploy.yml)."""

    KEYS = ("gripper_steps", "hold_steps", "table_z", "control_hz")

    def __init__(self, **kwargs: Any):
        unknown = sorted(set(kwargs) - set(self.KEYS))
        if unknown:
            raise ControllerError(f"unknown controller option(s): {unknown}")
        # Steps spent on a pure gripper command, so the fingers finish moving
        # and the contact settles before the next observation is taken.
        self.gripper_steps: int = _positive_integer(kwargs.get("gripper_steps", 6), "gripper_steps")
        # Steps spent on a "wait", keeping a degenerate "do nothing" answer
        # from burning one inference call per env step.
        self.hold_steps: int = _positive_integer(kwargs.get("hold_steps", 3), "hold_steps")
        # Height of the table top in the env-local frame; the prompt and the
        # grid overlay are rendered from it.
        self.table_z: float = float(kwargs.get("table_z", 0.765))
        # Environment control rate. RoboDojo's arx_x5 config collects at 25 Hz;
        # the episode-start report overrides it with the measured period.
        self.control_hz: float = float(kwargs.get("control_hz", 25.0))
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ControllerError("control_hz must be finite and positive")
        if not math.isfinite(self.table_z):
            raise ControllerError("table_z must be finite")
        for key in ("gripper_steps", "hold_steps"):
            if getattr(self, key) > MAX_ACTION_STEPS:
                raise ControllerError(f"{key} must be at most {MAX_ACTION_STEPS}")

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.KEYS}


def forward_axis(quat: np.ndarray) -> np.ndarray:
    """Direction the fingers point: link6's local +X, in the pose's frame.

    ``grounding.tool_forward`` computes the same axis for the image overlay;
    this copy keeps the module importable without Pillow, which the
    simulation client does not have.
    """
    return _rotate_vector(_normalize_quat(np.asarray(quat, dtype=float).reshape(4)),
                          np.array([1.0, 0.0, 0.0]))


def approach_quat(quat: np.ndarray, approach: Any, wrist_deg: float | None = None) -> np.ndarray:
    """Resolve a finger direction and an optional absolute wrist roll.

    Omitted roll keeps the shortest rotation from the current wrist. Explicit
    roll uses a fixed reference: jaw +Y points along world -X projected onto
    the plane normal to the fingers, or world +Y when those axes are parallel.
    This matches the initial forward pose at zero roll and makes repeated
    absolute commands independent of the current wrist orientation.
    """
    quat = _normalize_quat(np.asarray(quat, dtype=float).reshape(4))
    target = forward_axis(quat) if approach is None else np.asarray(approach, dtype=float).reshape(3)
    norm = float(np.linalg.norm(target))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ControllerError("approach direction must be a finite nonzero vector")
    target = target / norm
    current = forward_axis(quat)
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    if dot > 1 - 1e-12:
        turn = np.array([1.0, 0.0, 0.0, 0.0])
    elif dot < -1 + 1e-12:
        # Opposite directions have no unique shortest arc; any perpendicular
        # axis is a half turn, so pick the most stable one.
        axis = np.cross(current, np.array([0.0, 0.0, 1.0]))
        if float(np.linalg.norm(axis)) < 1e-9:
            axis = np.cross(current, np.array([1.0, 0.0, 0.0]))
        turn = _axis_angle_quat(axis, math.pi)
    else:
        turn = _axis_angle_quat(np.cross(current, target), math.acos(dot))
    result = quat_mul(turn, quat)
    if wrist_deg is not None:
        reference = np.array([-1.0, 0.0, 0.0])
        reference -= float(np.dot(reference, target)) * target
        if float(np.linalg.norm(reference)) < 1e-9:
            reference = np.array([0.0, 1.0, 0.0])
            reference -= float(np.dot(reference, target)) * target
        reference /= np.linalg.norm(reference)
        jaw = _rotate_vector(result, np.array([0.0, 1.0, 0.0]))
        reference_angle = math.atan2(float(np.dot(target, np.cross(jaw, reference))),
                                    float(np.dot(jaw, reference)))
        result = quat_mul(_axis_angle_quat(target, reference_angle + math.radians(float(wrist_deg))), result)
    return _normalize_quat(result)


def _axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = float(angle_rad) / 2
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def _rotate_vector(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector + 2 * np.cross(quat[1:], np.cross(quat[1:], vector) + quat[0] * vector)


def _positive_integer(value: Any, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControllerError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ControllerError(f"{name} must be a positive integer")
    return int(number)


# --- quaternion helpers (w, x, y, z), matching RoboDojo's pose convention ---


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(quat)):
        raise ControllerError("quaternion must contain only finite values")
    scale = float(np.max(np.abs(quat)))
    if scale < 1e-12:
        raise ControllerError("quaternion must be nonzero")
    scaled = quat / scale
    return scaled / math.hypot(*scaled)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b``; applying ``a`` after ``b``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_to_yaw(quat: np.ndarray) -> float:
    """Yaw (rotation about world +Z) of a quaternion, in radians."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
