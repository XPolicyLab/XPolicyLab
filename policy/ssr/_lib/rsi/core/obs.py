from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CAMERAS = "cam_head", "cam_left_wrist", "cam_right_wrist"
STATE_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_pose",
    "right_ee_pose",
    "left_ee_joint_state",
    "right_ee_joint_state",
)


@dataclass
class Observation:
    instruction: str
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    step: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def left_ee_pose(self) -> np.ndarray:
        return self.state["left_ee_pose"]

    @property
    def right_ee_pose(self) -> np.ndarray:
        return self.state["right_ee_pose"]


def parse(raw: dict, step: int) -> Observation:
    vision = raw.get("vision") or {}
    images = {}
    for cam in CAMERAS:
        color = (vision.get(cam) or {}).get("color")
        if color is not None:
            images[cam] = np.asarray(color)
    st = raw.get("state") or {}
    state = {k: np.asarray(st[k], dtype=np.float64).reshape(-1) for k in STATE_KEYS if k in st}
    return Observation(instruction=str(raw.get("instruction") or ""), images=images, state=state, step=step)
