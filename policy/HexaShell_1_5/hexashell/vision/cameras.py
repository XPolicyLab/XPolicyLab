"""The three cameras as pinhole models in the world frame.

Pixels are (x, y) = (column, row) indices: pixel (0, 0) is the top-left one and its centre is at
(0, 0), so the optical axis passes between the four middle pixels, at (319.5, 239.5).  (The renderer
samples each pixel at its centre; the two table edges, which lie symmetric about the head camera,
are centred on column 319.56 in the pictures.)  A camera looks along its own -z axis with +y up the
picture (Isaac's convention); ``pose`` is camera-to-world.
The head camera is fixed; a wrist camera rides on its arm's link6 (the pose RoboDojo reports as
``<arm>_ee_pose``), so its model is built from the reading the picture was taken with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import Refused
from ..robot.results import CameraInfo
from ..state import CAMERAS, Snapshot

WIDTH, HEIGHT = 640, 480
HEAD_FOCAL = 288.1325
HEAD_POSITION = (0.0, -0.41, 1.308)
HEAD_TILT_DEG = 60.0  # below the horizontal
WRIST_FOCAL = 397.0413
#: The wrist camera in its arm's link6 frame (camera-to-link6).
WRIST_MOUNT = np.array([
    [0.0, 0.5, -0.8660254, 0.084842],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.8660254, 0.5, 0.05094],
    [0.0, 0.0, 0.0, 1.0],
])


def _head_pose() -> np.ndarray:
    t = math.radians(HEAD_TILT_DEG)
    pose = np.eye(4)
    pose[:3, 0] = (1.0, 0.0, 0.0)
    pose[:3, 1] = (0.0, math.sin(t), math.cos(t))
    pose[:3, 2] = (0.0, -math.cos(t), math.sin(t))
    pose[:3, 3] = HEAD_POSITION
    return pose


HEAD_POSE = _head_pose()


@dataclass
class Camera:
    name: str
    focal: float
    pose: np.ndarray  # 4x4 camera-to-world
    width: int = WIDTH
    height: int = HEIGHT

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def axis(self) -> np.ndarray:
        """Unit vector the camera looks along (world)."""
        return -self.pose[:3, 2]

    def to_camera(self, points) -> np.ndarray:
        p = np.asarray(points, float).reshape(-1, 3)
        return (p - self.pose[:3, 3]) @ self.pose[:3, :3]

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """(pixels (N, 2) as (x, y), depth (N,)) of world points; depth <= 0 is behind the camera."""
        local = self.to_camera(points)
        depth = -local[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        x = self.focal * local[:, 0] / safe + self.cx
        y = -self.focal * local[:, 1] / safe + self.cy
        return np.stack([x, y], axis=1), depth

    def pixel(self, point) -> tuple[float, float] | None:
        """Pixel of one world point, or None when it is behind the camera or outside the picture."""
        px, depth = self.project(point)
        if depth[0] <= 1e-6:
            return None
        x, y = float(px[0, 0]), float(px[0, 1])
        if not (-0.5 <= x <= self.width - 0.5 and -0.5 <= y <= self.height - 0.5):
            return None
        return (x, y)

    def rays(self, pixels) -> np.ndarray:
        """Unit world directions of the rays through pixels (N, 2) as (x, y)."""
        px = np.asarray(pixels, float).reshape(-1, 2)
        local = np.stack([(px[:, 0] - self.cx) / self.focal, -(px[:, 1] - self.cy) / self.focal,
                          -np.ones(len(px))], axis=1)
        world = local @ self.pose[:3, :3].T
        return world / np.linalg.norm(world, axis=1, keepdims=True)

    def ray(self, pixel) -> tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), self.rays([pixel])[0]

    def on_plane(self, pixels, z: float) -> np.ndarray:
        """Where the rays through pixels meet the horizontal plane at height z (NaN if they do not)."""
        d = self.rays(pixels)
        o = self.position
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - o[2]) / d[:, 2]
        t = np.where((t > 0) & np.isfinite(t), t, np.nan)
        return o + d * t[:, None]

    def info(self) -> CameraInfo:
        return CameraInfo(name=self.name, width=self.width, height=self.height, focal_px=self.focal,
                          center_px=(self.cx, self.cy), position=tuple(float(v) for v in self.position),
                          rotation=[[float(v) for v in row] for row in self.pose[:3, :3]])


def wrist_pose(flange: np.ndarray) -> np.ndarray:
    return np.asarray(flange, float) @ WRIST_MOUNT


def cameras_for(snap: Snapshot) -> dict[str, Camera]:
    """The cameras as they were when the snapshot's pictures were taken."""
    arms = snap.arms_in_pictures
    cams = {"head": Camera("head", HEAD_FOCAL, HEAD_POSE.copy())}
    for arm in ("left", "right"):
        cams[f"{arm}_wrist"] = Camera(f"{arm}_wrist", WRIST_FOCAL, wrist_pose(arms[arm].flange))
    return cams


def check_cam(cam) -> str:
    aliases = {"left": "left_wrist", "right": "right_wrist", "cam_head": "head",
               "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
    name = aliases.get(cam, cam)
    if name not in CAMERAS:
        raise Refused(f"camera must be one of {', '.join(CAMERAS)}, not {cam!r}")
    return name


def triangulate(o1, d1, o2, d2) -> tuple[np.ndarray, float, float]:
    """Midpoint of the closest approach of two rays, the gap between them (m) and the angle
    between them (degrees)."""
    o1, d1, o2, d2 = (np.asarray(v, float) for v in (o1, d1, o2, d2))
    w = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-12:
        t1, t2 = 0.0, e / c
    else:
        t1 = (b * e - c * d) / den
        t2 = (a * e - b * d) / den
    p1, p2 = o1 + t1 * d1, o2 + t2 * d2
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(b / math.sqrt(a * c))))))
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), angle
