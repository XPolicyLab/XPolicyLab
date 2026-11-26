"""Camera calibration from rig constants, for the official observation (which carries no camera calibration).

The RoboDojo ARX-X5 rig has a fixed head camera (``env_cfg/camera/camera_config.yml``: ``cam_head`` at a set
position and tilt) and a camera rigidly mounted on each wrist. Its constants were measured once in simulation
the way a real rig is calibrated (``rig_constants.json``: head intrinsics and camera-to-world pose; wrist
intrinsics and the flange-to-camera transform) and are read here without touching the simulator. A wrist
camera's pose is the arm's flange pose from the observation times its flange-to-camera transform.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Mapping

import numpy as np

from knowin_agent.backends.robodojo.calibration import OpticalCalibration, validate_intrinsics, validate_rigid
from knowin_agent.core.geometry import Pose

# Native camera name per mount; the wrist cameras follow the arm with the same name.
HEAD_CAMERA = "cam_head"
WRIST_CAMERAS = {"left": "cam_left_wrist", "right": "cam_right_wrist"}


@lru_cache(maxsize=1)
def constants() -> dict:
    text = resources.files("knowin_agent.backends.robodojo").joinpath("rig_constants.json").read_text(encoding="utf-8")
    return json.loads(text)


def _intrinsics(entry: dict, name: str) -> tuple[np.ndarray, tuple[int, int]]:
    size = (int(entry["size"][0]), int(entry["size"][1]))
    return validate_intrinsics(np.asarray(entry["intrinsics"], dtype=np.float64), size, name), size


def rig_calibration(ee_poses: Mapping[str, Pose], rgb_by_name: Mapping[str, np.ndarray]) -> dict[str, OpticalCalibration]:
    """Calibration of every native camera in ``rgb_by_name`` from the rig constants and the flange poses."""
    data = constants()
    out = {}
    for name, rgb in rgb_by_name.items():
        if name == HEAD_CAMERA:
            entry = data["head"]
            k, size = _intrinsics(entry, name)
            extrinsics = np.asarray(entry["camera_to_world"], dtype=np.float64)
        else:
            arm = next((a for a, cam in WRIST_CAMERAS.items() if cam == name), None)
            if arm is None or arm not in ee_poses:
                raise KeyError(f"no rig constant for camera {name!r}")
            entry = data["wrist"][arm]
            k, size = _intrinsics(entry, name)
            extrinsics = ee_poses[arm].matrix() @ np.asarray(entry["camera_to_flange"], dtype=np.float64)
        height, width = np.shape(rgb)[:2]
        if (width, height) != size:
            raise ValueError(f"{name}: rig constant is for {size[0]}x{size[1]}, image is {width}x{height}")
        out[name] = OpticalCalibration(k, validate_rigid(extrinsics, name), size)
    return out
