"""Explicit calibrated pinhole geometry for RGB-derived depth/height estimates.

Camera coordinates use USD axes: +X right, +Y up, -Z forward. Pixel coordinates
are u right, v down. Depth is positive forward-axis distance, not ray length.
No depth, object pose, or benchmark layout is fetched from the simulator.
"""

import numpy as np

from .planner_bridge import quaternion
from .primitives import vector


class PinholeCamera:
    def __init__(self, *, width, height, fx, fy, cx, cy, world_pose):
        if any(type(n) is not int or n < 1 for n in (width, height)):
            raise ValueError("Positive integer image dimensions required")
        self.width, self.height = width, height
        self.fx, self.fy, self.cx, self.cy = vector(
            [fx, fy, cx, cy], 4, "camera intrinsics"
        )
        if fx <= 0 or fy <= 0 or not 0 <= cx < width or not 0 <= cy < height:
            raise ValueError("Invalid pinhole intrinsics")
        pose = vector(world_pose, 7, "world camera pose")
        w, x, y, z = quaternion(pose[3:])
        self._origin = np.array(pose[:3], dtype=float)
        self._rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def _values(value, columns, name):
        values = np.asarray(value)
        if (
            values.ndim != 2
            or values.shape[1] != columns
            or not len(values)
            or values.dtype.kind not in "ifu"
            or not np.isfinite(values).all()
        ):
            raise ValueError(name + " must be a nonempty finite real matrix")
        return values.astype(float)

    def rays(self, pixels):
        pixels = self._values(pixels, 2, "pixels")
        if (
            np.any(pixels < 0)
            or np.any(pixels[:, 0] >= self.width)
            or np.any(pixels[:, 1] >= self.height)
        ):
            raise ValueError("Pixel outside calibrated image")
        camera = np.column_stack(
            (
                (pixels[:, 0] - self.cx) / self.fx,
                -(pixels[:, 1] - self.cy) / self.fy,
                -np.ones(len(pixels)),
            )
        )
        return camera @ self._rotation.T

    def unproject(self, pixels, axial_depth_m):
        rays = self.rays(pixels)
        depth = np.asarray(axial_depth_m)
        if (
            depth.shape != (len(rays),)
            or depth.dtype.kind not in "ifu"
            or not np.isfinite(depth).all()
            or np.any(depth <= 0)
        ):
            raise ValueError("Positive finite axial depth per pixel required")
        return self._origin + rays * depth[:, None]

    def plane_depth(self, pixels, *, world_z):
        height = vector([world_z], 1, "world plane height")[0]
        rays = self.rays(pixels)
        if np.any(np.abs(rays[:, 2]) < 1e-9):
            raise ValueError("Camera ray is parallel to height plane")
        depth = (height - self._origin[2]) / rays[:, 2]
        if np.any(depth <= 0) or not np.isfinite(depth).all():
            raise ValueError("Height plane is behind camera")
        return depth

    def project(self, world_points):
        points = self._values(world_points, 3, "world points")
        camera = (points - self._origin) @ self._rotation
        depth = -camera[:, 2]
        if np.any(depth <= 1e-9):
            raise ValueError("World point is behind or on camera plane")
        pixels = np.column_stack(
            (
                self.cx + self.fx * camera[:, 0] / depth,
                self.cy - self.fy * camera[:, 1] / depth,
            )
        )
        return pixels, depth
