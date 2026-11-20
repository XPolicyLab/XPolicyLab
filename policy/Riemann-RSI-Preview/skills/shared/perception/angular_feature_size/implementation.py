import numpy as np

from skills.shared.perception.rgb_surface_geometry.implementation import pixel_rays


def run(api, camera, region, point):
    point = np.asarray(point, dtype=float)
    centre = np.asarray(region['centroid'], dtype=float)
    width, height = region['bbox'][2:]
    if point.shape != (3,) or centre.shape != (2,) or not np.isfinite(point).all() or min(width, height) <= 0:
        raise ValueError('Angular size requires a finite current feature and public ray intersection')
    offset = np.asarray([max(width, height) / 2, 0.]) if width >= height else np.asarray([0., max(width, height) / 2])
    origin, rays = pixel_rays(api, camera, np.stack([centre, centre - offset, centre + offset]))
    angle = np.arccos(np.clip(float(rays[1] @ rays[2]), -1, 1)) / 2
    return float(np.linalg.norm(point - origin) * np.sin(angle))
