import numpy as np

from skills.shared.perception.rgb_surface_geometry.implementation import pixel_rays


MAXIMUM_POINTS = 1024


def project_points(api, camera, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= MAXIMUM_POINTS or not np.isfinite(points).all():
        raise ValueError('Project only a bounded finite current public point set')
    height, width = api.image(camera).shape[:2]
    center = np.asarray([width / 2, height / 2], dtype=float)
    probes = np.asarray([center, center + [1, 0], center + [0, 1]])
    origin, rays = pixel_rays(api, camera, probes)
    normal = rays[0]
    horizontal = rays[1] / float(rays[1] @ normal) - normal
    vertical = rays[2] / float(rays[2] @ normal) - normal
    displacement = points - origin
    depth = displacement @ normal
    valid = depth > 1e-8
    projected = np.full((len(points), 2), np.nan, dtype=float)
    coordinates = displacement[valid] / depth[valid, None] - normal
    projected[valid] = coordinates @ np.linalg.pinv(np.vstack([horizontal, vertical])) + center
    return projected


def points_on_plane(api, camera, pixels, height):
    pixels = np.asarray(pixels, dtype=float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not 1 <= len(pixels) <= MAXIMUM_POINTS
            or not np.isfinite(pixels).all() or not np.isfinite(height)):
        raise ValueError('Use a bounded current-image sample and a candidate measured surface')
    origin, rays = pixel_rays(api, camera, pixels)
    valid = np.abs(rays[:, 2]) > 1e-8
    distance = np.full(len(pixels), np.nan)
    distance[valid] = (height - origin[2]) / rays[valid, 2]
    valid &= distance > 0
    points = np.full((len(pixels), 3), np.nan)
    points[valid] = origin + rays[valid] * distance[valid, None]
    return points
