import numpy as np


def run(candidate_points, public_references, maximum_distances, units):
    points = np.asarray(candidate_points, dtype=float)
    references = np.asarray(public_references, dtype=float)
    radii = np.asarray(maximum_distances, dtype=float)
    dimensions = 2 if units == 'native_pixels' else 3 if units == 'metres' else None
    if (dimensions is None or points.ndim != 2 or points.shape[1] != dimensions or len(points) > 64
            or references.ndim != 2 or references.shape[1] != dimensions or not 1 <= len(references) <= 8
            or radii.shape != (len(references),) or not np.isfinite(references).all()
            or not np.isfinite(radii).all() or np.any(radii <= 0)
            or np.any(radii > (32. if units == 'native_pixels' else .2))
            or np.max(np.abs(references)) > (8192 if units == 'native_pixels' else 3)):
        raise ValueError('Correspondence uses only bounded current public features and explicit public references')
    valid = np.isfinite(points).all(axis=1)
    if units == 'native_pixels':
        valid &= (points >= 0).all(axis=1) & (points[:, 0] < 640) & (points[:, 1] < 480)
    else:
        valid &= (np.abs(points) <= 3).all(axis=1)
    errors = np.linalg.norm(points[:, None] - references[None, :], axis=2)
    matched = np.flatnonzero(valid & np.any(errors <= radii[None, :], axis=1))
    return int(matched[0]) if len(matched) == 1 else None
