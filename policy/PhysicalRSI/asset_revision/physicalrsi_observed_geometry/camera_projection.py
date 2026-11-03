"""Projection using the current observation's calibrated camera."""
import numpy as np
from . import GeometryUnavailable


def pixel_to_world_on_height(pixel_xy, *, world_z, camera):
    uv = np.asarray(pixel_xy, dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or not np.isfinite(world_z):
        raise GeometryUnavailable('Finite pixel and measured world height required')
    origin = camera.transform[:3, 3]
    ray = camera.unproject([uv], [1.])[0] - origin
    if abs(ray[2]) < 1e-9:
        raise GeometryUnavailable('Pixel ray parallel to height plane')
    distance = (float(world_z) - origin[2]) / ray[2]
    if distance <= 0:
        raise GeometryUnavailable('Measured height plane behind camera')
    return tuple(float(v) for v in (origin + distance*ray)[:2])


def world_to_pixel_on_height(world_xy, *, world_z, camera):
    xy = np.asarray(world_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(world_z):
        raise GeometryUnavailable('Finite world point required')
    pixels, _ = camera.project([[xy[0], xy[1], float(world_z)]])
    return tuple(float(value) for value in pixels[0])
