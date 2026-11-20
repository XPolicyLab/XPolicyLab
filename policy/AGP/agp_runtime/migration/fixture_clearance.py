"""Sampled sphere clearance against known installation geometry, not task objects."""
import numpy as np


def sphere_box_clearance(spheres, center, dimensions):
    """Signed distance to an axis-aligned solid box, minus sphere radius (metres)."""
    spheres = np.asarray(spheres, dtype=float)
    center = np.asarray(center, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    if (spheres.ndim < 2 or spheres.shape[-1] != 4 or spheres.size == 0
            or center.shape != (3,) or dimensions.shape != (3,)):
        raise ValueError('Expected nonempty xyzr spheres and xyz box center/dimensions')
    if (not np.isfinite(spheres).all() or not np.isfinite(center).all()
            or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)
            or np.any(spheres[..., 3] < 0)):
        raise ValueError('Invalid fixture geometry')
    delta = np.abs(spheres[..., :3] - center) - dimensions / 2
    signed = (np.linalg.norm(np.maximum(delta, 0), axis=-1)
              + np.minimum(np.max(delta, axis=-1), 0))
    return signed - spheres[..., 3]


def first_fixture_violation(spheres, start_spheres, moving_mask, center, dimensions,
                            numerical_epsilon_m=1e-6):
    """Allow non-deepening escape from existing nominal overlap, per sphere.

    Below zero, clearance must maintain/improve the best previous clearance.
    Once clear, reentry is rejected. The epsilon is numerical tolerance, not a
    physical margin. Only sampled configurations are checked, not swept meshes.
    """
    gaps = sphere_box_clearance(spheres, center, dimensions)
    start = sphere_box_clearance(start_spheres, center, dimensions)
    moving = np.asarray(moving_mask)
    if (gaps.ndim != 2 or start.shape != (gaps.shape[1],)
            or moving.shape != start.shape or moving.dtype != np.dtype(bool)
            or not moving.any()):
        raise ValueError('Invalid sample/sphere mask')
    if (not np.isscalar(numerical_epsilon_m) or not np.isfinite(numerical_epsilon_m)
            or numerical_epsilon_m < 0):
        raise ValueError('Invalid numerical tolerance')
    floor = np.minimum(start, 0)
    for i, gap in enumerate(gaps):
        bad = moving & (gap < floor - numerical_epsilon_m)
        if bad.any():
            j = int(np.flatnonzero(bad)[0])
            return {'sample_index': i, 'sphere_index': j, 'clearance_m': float(gap[j]),
                    'minimum_allowed_m': float(floor[j])}
        floor = np.maximum(floor, np.minimum(gap, 0))
    return None
