import cv2
import numpy as np

from skills.shared.perception.public_camera_projection.implementation import points_on_plane


def run(api, camera, region, height):
    pixels = np.asarray(region['pixels'], dtype=np.float32)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not 3 <= len(pixels) <= 307200 or not np.isfinite(pixels).all():
        raise ValueError('A circular surface requires bounded finite current image pixels')
    hull = cv2.convexHull(pixels).reshape(-1, 2)
    if len(hull) > 1024:
        hull = hull[np.linspace(0, len(hull) - 1, 1024).astype(int)]
    points = points_on_plane(api, camera, hull, float(height))
    if points is None or not np.isfinite(points).all():
        return None
    center, radius = cv2.minEnclosingCircle(points[:, :2].astype(np.float32))
    if not .005 <= radius <= .12:
        return None
    return {'point': np.asarray([*center, float(height)]), 'radius_m': float(radius),
            'face_area_pixels': int(region['area'])}
