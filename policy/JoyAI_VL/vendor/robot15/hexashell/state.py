"""One observation from RoboDojo, parsed into what the agent's tools use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import pose_from_xyzwxyz
from .world import ARMS

#: RoboDojo camera keys -> the names the agent uses.
CAMERA_KEYS = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
CAMERAS = tuple(CAMERA_KEYS.values())
#: Key under obs["additional_info"] where our client-side loop puts the episode status.
STATUS_KEY = "hexashell"


@dataclass
class ArmReading:
    joints: np.ndarray  # 6 joint angles (rad)
    flange: np.ndarray  # 4x4 world pose of link6 (RoboDojo's <arm>_ee_pose)
    gripper: float  # reported gripper value (0 closed .. 1 open): the command RoboDojo applied


@dataclass
class Snapshot:
    """The robot and its cameras at one observation."""

    seq: int
    arms: dict[str, ArmReading]
    images: dict[str, np.ndarray]  # camera name -> RGB uint8 (H, W, 3)
    instruction: str
    step: int | None = None
    step_limit: int | None = None
    pictures_behind: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    #: arm readings the pictures show (the previous observation's when pictures run behind)
    pictured_arms: dict[str, ArmReading] | None = None

    @property
    def arms_in_pictures(self) -> dict[str, ArmReading]:
        return self.pictured_arms or self.arms

    @property
    def steps_left(self) -> int | None:
        if self.step is None or self.step_limit is None:
            return None
        return max(0, self.step_limit - self.step)


def parse_obs(obs: dict[str, Any], seq: int, previous: Snapshot | None = None) -> Snapshot:
    state = obs.get("state") or {}
    arms = {}
    for arm in ARMS:
        joints = np.asarray(state.get(f"{arm}_arm_joint_state", [0.0] * 6), float).reshape(-1)[:6]
        ee = state.get(f"{arm}_ee_pose")
        flange = pose_from_xyzwxyz(ee) if ee is not None and len(ee) >= 7 else np.eye(4)
        grip = state.get(f"{arm}_ee_joint_state", [1.0])
        try:
            g = float(np.asarray(grip, float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            g = 1.0
        arms[arm] = ArmReading(joints=joints, flange=flange, gripper=g)
    images = {}
    for key, name in CAMERA_KEYS.items():
        try:
            color = obs["vision"][key]["color"]
        except (KeyError, TypeError):
            continue
        arr = np.asarray(color)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            images[name] = np.ascontiguousarray(arr[:, :, :3]).astype(np.uint8, copy=False)
    info = dict(((obs.get("additional_info") or {}).get(STATUS_KEY)) or {})
    instruction = obs.get("instruction") or ""
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else ""
    behind = int(info.get("pictures_behind", 0) or 0)
    snap = Snapshot(
        seq=seq,
        arms=arms,
        images=images,
        instruction=str(instruction),
        step=_int_or_none(info.get("step")),
        step_limit=_int_or_none(info.get("step_limit")),
        pictures_behind=behind,
        info=info,
    )
    if behind and previous is not None:
        snap.pictured_arms = previous.arms
    return snap


def _int_or_none(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
