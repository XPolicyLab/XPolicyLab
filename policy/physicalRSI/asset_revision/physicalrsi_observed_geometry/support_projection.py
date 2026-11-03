"""Calibrated pixel intersection with a measured support plus vertical offset."""
import numpy as np
from . import GeometryUnavailable
from .upright_contour import support_height


def support_point(pixel, offset, *, camera, support_plane):
    coefficients = np.asarray(support_plane, float)
    uv = np.asarray(pixel, float)
    if (camera is None or coefficients.shape != (3,) or uv.shape != (2,)
            or not np.isfinite(coefficients).all() or not np.isfinite(uv).all()
            or not np.isfinite(offset)):
        raise GeometryUnavailable('Current camera, measured support and finite offset required')
    normal = np.r_[-coefficients[:2], 1.]
    ray = np.linalg.solve(camera.k, np.r_[uv, 1.])
    ray = camera.transform[:3, :3] @ (ray * [1., -1., -1.])
    origin = camera.transform[:3, 3]
    denominator = ray @ normal
    if abs(denominator) < 1e-9:
        raise GeometryUnavailable('Pixel ray parallel to measured support')
    distance = (coefficients[2] + offset - origin @ normal) / denominator
    if distance <= 0:
        raise GeometryUnavailable('Measured support intersection behind camera')
    return origin + distance * ray
