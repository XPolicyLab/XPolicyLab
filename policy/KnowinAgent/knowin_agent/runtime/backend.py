"""Robot backend contract shared by simulators and physical robots.

A backend advances exactly one control tick per `step` call and always receives
a command for every arm, so the runtime executor can drive both arms in the same
tick. Evaluator state (score, success) is host-only and never reaches an agent.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from knowin_agent.core.geometry import Pose, rot_y, rot_z, rot_x, wrap180

ARMS = ("left", "right")


def other_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return "right" if arm == "left" else "left"


@dataclass
class CameraFrame:
    """One camera image with optional calibration and metric depth.

    extrinsics: 4x4 camera-to-world of the optical frame (+X right, +Y down, +Z forward).
    depth: HxW metres along the optical axis (distance to image plane), if the backend has it.
    """

    name: str
    rgb: np.ndarray
    intrinsics: Optional[np.ndarray] = None
    extrinsics: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

    @property
    def calibrated(self) -> bool:
        return self.intrinsics is not None and self.extrinsics is not None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h


@dataclass
class ArmState:
    """gripper_command: the last command sent. gripper_opening: the measured finger opening, normalised
    (1 fully open, about 0 closed on nothing); None when the robot does not report it."""

    ee_pose: Pose
    joints: np.ndarray
    gripper_command: float
    gripper_opening: Optional[float] = None


@dataclass
class Observation:
    tick: int
    arms: dict[str, ArmState]
    cameras: dict[str, CameraFrame]
    instruction: str = ""
    obs_id: str = ""

    def __post_init__(self):
        if not self.obs_id:
            self.obs_id = observation_id(self)

    def camera(self, name: str) -> CameraFrame:
        if name not in self.cameras:
            raise KeyError(f"camera {name!r} not in observation ({sorted(self.cameras)})")
        return self.cameras[name]


def observation_id(obs: Observation) -> str:
    digest = hashlib.sha256(str(obs.tick).encode())
    for arm in sorted(obs.arms):
        digest.update(np.asarray(obs.arms[arm].ee_pose.array(), dtype=np.float32).tobytes())
    for name in sorted(obs.cameras):
        rgb = obs.cameras[name].rgb
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(rgb[::8, ::8]).tobytes())
    return digest.hexdigest()[:12]


@dataclass
class ArmCommand:
    """One tick's command for one arm: a flange pose and/or joint targets, plus the gripper (0 closed, 1 open).

    When every arm's command carries joints the backend sends a joint action; otherwise an end-effector
    action (joint-only commands are converted with forward kinematics)."""

    ee_pose: Optional[Pose]
    gripper: float
    joints: Optional[np.ndarray] = None

    def __post_init__(self):
        if not (0.0 <= float(self.gripper) <= 1.0) or not math.isfinite(float(self.gripper)):
            raise ValueError("gripper command must be in [0, 1] (0 closed, 1 open)")
        if self.ee_pose is None and self.joints is None:
            raise ValueError("an arm command needs a flange pose or joint targets")
        if self.joints is not None:
            self.joints = np.asarray(self.joints, dtype=float).reshape(-1)
            if not np.isfinite(self.joints).all():
                raise ValueError("joint targets must be finite")


@dataclass
class StepOutcome:
    observation: Observation
    episode_over: bool = False
    reason: Optional[str] = None


@dataclass
class Budget:
    ticks_used: int
    ticks_limit: Optional[int]

    @property
    def remaining(self) -> Optional[int]:
        return None if self.ticks_limit is None else max(0, self.ticks_limit - self.ticks_used)

    def as_dict(self) -> dict:
        return {"ticks_used": self.ticks_used, "ticks_limit": self.ticks_limit, "ticks_remaining": self.remaining}


@dataclass
class Reach:
    """Kinematic evidence for a flange pose. reachable=None means the solver could not decide."""

    reachable: Optional[bool]
    reason: str
    joint_solution: Optional[list] = None
    joint_travel_deg: Optional[float] = None

    def as_dict(self) -> dict:
        out = {"reachable": self.reachable if self.reachable is not None else "unknown", "reason": self.reason}
        if self.joint_travel_deg is not None:
            out["joint_travel_deg"] = round(self.joint_travel_deg, 1)
        return out


@runtime_checkable
class Kinematics(Protocol):
    """Required: reach, fk. Optional `plan(arm, q0, flange, q_goal, *, step_m, rot_step, precision, path,
    straight) -> (waypoints | None, how_or_why)`: when present, motions are sent as planned joint targets."""

    def reach(self, arm: str, flange: Pose, joints: np.ndarray) -> Reach: ...

    def fk(self, arm: str, joints: np.ndarray) -> Pose: ...


def default_orientation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Flange rotation whose tool axis (+x) points down at pitch 90 and toward yaw at pitch 0."""
    return rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)


@dataclass
class RobotProfile:
    """Embodiment facts the runtime needs; nothing task specific.

    grasp_offset: flange-frame point between the finger pads (the grasp point).
    tool_axis: flange-frame approach direction; closing_axis: direction the fingers close along.
    orientation_fn(yaw, pitch, roll) -> flange rotation, the embodiment's gripper convention.
    """

    name: str
    cameras: Mapping[str, str]
    wrist_camera: Mapping[str, str]
    grasp_offset: tuple = (0.0, 0.0, 0.0)
    tool_axis: tuple = (1.0, 0.0, 0.0)
    closing_axis: tuple = (0.0, 1.0, 0.0)
    fingertip_beyond_grasp_m: float = 0.0
    table_z: Optional[float] = None
    control_hz: float = 25.0
    arms: tuple = ARMS
    workspace_min: tuple = (-1.0, -1.0, 0.0)
    workspace_max: tuple = (1.0, 1.0, 2.0)
    kinematics: Optional[Kinematics] = None
    orientation_fn: Callable[[float, float, float], np.ndarray] = field(default=default_orientation)
    notes: str = ""

    def orientation(self, yaw: float = 0.0, pitch: float = 90.0, roll: float = 0.0) -> np.ndarray:
        return self.orientation_fn(float(yaw), float(pitch), float(roll))

    def closing_heading(self, rotation: np.ndarray) -> float:
        """World heading (deg) of the finger closing axis projected on the table plane."""
        axis = np.asarray(rotation) @ np.asarray(self.closing_axis, dtype=float)
        return wrap180(math.degrees(math.atan2(axis[1], axis[0])))

    def yaw_for_heading(self, closing_heading_deg: float, pitch: float = 90.0) -> float:
        """The `yaw` argument whose finger closing axis points along this world heading."""
        reference = self.closing_heading(self.orientation(0.0, pitch, 0.0))
        return wrap180(float(closing_heading_deg) - reference)

    def yaw_of(self, rotation: np.ndarray, pitch: float = 90.0) -> float:
        """The `yaw` argument that reproduces this rotation's closing heading at the given pitch."""
        return self.yaw_for_heading(self.closing_heading(rotation), pitch)

    def grasp_point(self, flange: Pose) -> np.ndarray:
        return flange.transform(self.grasp_offset)

    def in_workspace(self, xyz: Sequence[float]) -> bool:
        p = np.asarray(xyz, dtype=float)
        return bool(np.all(p >= np.asarray(self.workspace_min)) and np.all(p <= np.asarray(self.workspace_max)))


@runtime_checkable
class RobotBackend(Protocol):
    profile: RobotProfile

    def observe(self) -> Observation: ...

    def step(self, commands: Mapping[str, ArmCommand]) -> StepOutcome: ...

    def budget(self) -> Budget: ...

    def instruction(self) -> str: ...

    def evaluation(self) -> dict: ...

    def close(self) -> None: ...
