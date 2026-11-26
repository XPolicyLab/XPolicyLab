"""Step 2d: Independent HarnessRobot facade for HarnessPolicyIncremental.

Owns the ``robo`` API seen by all manipulation skills:
- ``task``, ``image(camera)``, ``segment(text_prompt, camera, min_score)`` run directly
  on our own EpisodeEnv + HttpSam3 client.
- ``state``, ``camera_meta``, ``locate``, ``back_project``, ``move_to``, ``set_gripper``,
  ``home`` delegate to local sight, geometry, and motion helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bridge import CAMERAS, EpisodeEnv
from . import geometry as geometry_mod
from . import sight as sight_mod
from .motion import MotionSession


@dataclass(frozen=True)
class SegmentResult:
    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str


def _normalize_camera(camera: str) -> str:
    cam = str(camera or "cam_head")
    if cam not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}, got {cam!r}")
    return cam


class HarnessRobot:
    """Our own robot interface exposed as ``robo`` to skill programs."""

    # CLOSE=1 closes, OPEN=-1 opens, KEEP=0 holds last command.
    CLOSE: int = 1
    OPEN: int = -1
    KEEP: int = 0
    CAMERAS = CAMERAS

    def __init__(
        self,
        env: EpisodeEnv,
        sam3_client: Any,
        *,
        toolkit: Any,
    ) -> None:
        self._env = env
        self._sam3 = sam3_client
        self._toolkit = toolkit
        self._motion = MotionSession(toolkit)

    # -- Own observation & SAM3 segmentation ----------------------------------

    @property
    def task(self) -> str:
        return self._env.get_obs()["instruction"]

    def image(self, camera: str = "cam_head", resolution: str = "high", step: int = -1) -> np.ndarray:
        del resolution  # RoboDojo default client delivers native RGB frames
        del step  # current observation is always the live frame on the EpisodeEnv bridge
        cam = _normalize_camera(camera)
        obs = self._env.get_obs()
        vision = obs.get("vision") or {}
        entry = vision.get(cam) or {}
        color = entry.get("color")
        if color is None:
            color = entry.get("rgb")
        if color is None:
            raise RuntimeError(f"no colour image for camera {cam!r} in observation")
        arr = np.asarray(color)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
        raise RuntimeError(f"unexpected image shape for camera {cam!r}: {arr.shape}")

    def segment(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
    ) -> SegmentResult:
        cam = _normalize_camera(camera)
        prompt = str(text_prompt or "").strip()
        if not prompt:
            raise ValueError("text_prompt must not be empty")
        img = self.image(cam)
        res = self._sam3.segment(img, text_prompt=prompt, min_score=float(min_score))
        if not res.found or res.mask is None:
            return SegmentResult(False, None, 0, None, None, cam)
        rows, cols = np.nonzero(res.mask)
        if len(rows) == 0:
            return SegmentResult(False, None, 0, None, None, cam)
        centroid = (int(rows.mean()), int(cols.mean()))
        return SegmentResult(
            found=True,
            centroid=centroid,
            area_px=int(len(rows)),
            score=float(res.score) if res.score is not None else None,
            box=res.box,
            camera=cam,
        )

    # -- Own camera calibration (Step 2e) --------------------------------------

    def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict[str, Any]:
        del step  # live observation only on the EpisodeEnv bridge
        cam = _normalize_camera(camera)
        if cam == "cam_head":
            return sight_mod.camera_meta(cam)
        arm = "left" if cam == "cam_left_wrist" else "right"
        st = self.state().arm(arm)
        ee = [*st.eef_pos, *st.eef_quat]
        return sight_mod.camera_meta(cam, ee_pose=ee)

    # -- Own locate / back_project (Step 2e2) ----------------------------------

    def locate(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
    ) -> Any:
        cam = _normalize_camera(camera)
        return geometry_mod.locate(self._toolkit, text_prompt, camera=cam, min_score=min_score)

    def back_project(self, row: int, col: int, camera: str = "cam_head") -> Any:
        cam = _normalize_camera(camera)
        return geometry_mod.back_project(self._toolkit, row, col, camera=cam)

    # -- Own state & motion (Step 2f) ------------------------------------------

    def state(self, step: int = -1) -> Any:
        return self._motion.state(step=step)

    def move_to(self, *args: Any, **kwargs: Any) -> Any:
        return self._motion.move_to(*args, **kwargs)

    def set_gripper(self, *args: Any, **kwargs: Any) -> Any:
        return self._motion.set_gripper(*args, **kwargs)

    def home(self, *args: Any, **kwargs: Any) -> Any:
        return self._motion.home(*args, **kwargs)
