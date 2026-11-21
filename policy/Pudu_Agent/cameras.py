"""Fixed-camera geometry for the general ``locate`` skill.

The head camera of the ARX X5 RoboDojo embodiment is mounted at a fixed pose
that RoboDojo publishes in ``env_cfg/camera/camera_config.yml`` (position
``[0.0, -0.41, 1.308]`` m, Euler ``[30, 0, 0]`` deg) with the Gemini 345Lg
optics in ``env_cfg/camera/template.py`` (640x480, focal length 10 mm,
horizontal aperture 22.212 mm). RoboDojo's own ``get_camera_intrinsics`` builds
``fx = width * focal / horizontal_aperture`` with square pixels and the
principal point at the image centre; this module does the same.

Nothing here depends on the task. It is embodiment geometry, the same for every
episode, and it only answers "which world point lies under this pixel at this
height", or, given a second view, "where do these two pixel rays meet".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """A pinhole camera in USD convention: looks along local -Z, +Y is up."""

    name: str
    position: tuple[float, float, float]
    euler_deg: tuple[float, float, float]
    width: int
    height: int
    focal_length_mm: float
    horizontal_aperture_mm: float

    def rotation(self) -> np.ndarray:
        """Camera-to-world rotation from extrinsic XYZ Euler angles.

        Isaac Sim's ``euler_angles_to_quat(..., extrinsic=True)`` composes the
        rotation as ``Rz @ Ry @ Rx``.
        """
        rx, ry, rz = (math.radians(a) for a in self.euler_deg)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def focal_px(self, image_width: int) -> float:
        return image_width * self.focal_length_mm / self.horizontal_aperture_mm

    def ray(self, u: float, v: float, image_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World-frame origin and unit direction of the ray through pixel (u, v).

        ``image_size`` is ``(width, height)`` of the image the pixel was read
        from; it defaults to the native resolution.
        """
        width, height = image_size or (self.width, self.height)
        f = self.focal_px(width)
        local = np.array([(u - width / 2.0) / f, -(v - height / 2.0) / f, -1.0])
        direction = self.rotation() @ local
        direction /= np.linalg.norm(direction)
        return np.asarray(self.position, dtype=float), direction

    def project(self, point: np.ndarray, image_size: tuple[int, int] | None = None) -> tuple[float, float] | None:
        """Pixel of a world point, or None when it is behind the camera."""
        width, height = image_size or (self.width, self.height)
        f = self.focal_px(width)
        local = self.rotation().T @ (np.asarray(point, dtype=float) - np.asarray(self.position))
        if local[2] >= -1e-9:
            return None
        u = width / 2.0 + f * local[0] / -local[2]
        v = height / 2.0 - f * local[1] / -local[2]
        return float(u), float(v)


HEAD_CAMERA = PinholeCamera(
    name="cam_head",
    position=(0.0, -0.41, 1.308),
    euler_deg=(30.0, 0.0, 0.0),
    width=640,
    height=480,
    focal_length_mm=10.0,
    horizontal_aperture_mm=22.212,
)

FIXED_CAMERAS: dict[str, PinholeCamera] = {HEAD_CAMERA.name: HEAD_CAMERA}


def intersect_plane(origin: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray | None:
    """Point where the ray meets the horizontal plane at height ``z``."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (z - origin[2]) / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def triangulate(
    o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Midpoint of the closest approach of two rays and the gap between them."""
    w0 = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    if s <= 0 or t <= 0:
        return None
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2))


#: Wrist camera pose in the flange frame of its own arm (4x4, camera-to-flange),
#: identical for both arms. Derived from the official RoboDojo demonstration
#: data (data/demo/arx_x5, which records per-frame camera extrinsics next to the
#: ``<arm>_ee_poses`` flange poses, wxyz quaternions): inv(T_world_flange) @
#: T_world_camera is constant to 1e-5 over the episode. USD camera convention,
#: like the head camera.
WRIST_MOUNT = np.array(
    [
        [0.0, 0.5, -0.8660254, 0.0848],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.8660254, 0.5, 0.0509],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
#: Wrist camera optics: the demonstration intrinsics, 640x480, fx = fy = 397.04.
WRIST_FOCAL_PX_AT_640 = 397.04


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def wrist_camera_pose(flange_pose: np.ndarray) -> np.ndarray:
    """World camera-to-world matrix of a wrist camera from its flange pose."""
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(flange_pose[3:7])
    T[:3, 3] = flange_pose[:3]
    return T @ WRIST_MOUNT


def ray_from_pose(
    T_world_cam: np.ndarray, focal_px_at_640: float, u: float, v: float, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    f = focal_px_at_640 * width / 640.0
    local = np.array([(u - width / 2.0) / f, -(v - height / 2.0) / f, -1.0])
    direction = T_world_cam[:3, :3] @ local
    return T_world_cam[:3, 3].copy(), direction / np.linalg.norm(direction)
