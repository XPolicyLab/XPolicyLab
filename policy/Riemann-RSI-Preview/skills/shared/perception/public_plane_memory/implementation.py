from copy import deepcopy
import math

import numpy as np


FIELDS = {'support_plane_m', 'edge_m', 'camera_transform', 'focal_pixels',
          'support_pixels', 'support_sizes_px', 'cap_top_m'}


def validate(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('A public support hypothesis cannot contain layout, object state or score fields')
    for name in ('support_plane_m', 'edge_m', 'focal_pixels'):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]):
            raise ValueError('Public plane evidence requires finite measured scalars')
    if not 0 < value['edge_m'] <= .2 or value['focal_pixels'] <= 0:
        raise ValueError('Public plane evidence must retain a bounded observed edge and valid camera')
    if value['cap_top_m'] is not None and (type(value['cap_top_m']) not in (int, float)
                                         or not math.isfinite(value['cap_top_m'])):
        raise ValueError('A cap hypothesis must be an earlier public surface estimate or absent')
    transform = np.asarray(value['camera_transform'], dtype=float)
    pixels = np.asarray(value['support_pixels'], dtype=float)
    sizes = np.asarray(value['support_sizes_px'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not 1 <= len(pixels) <= 8 or pixels.shape != (len(pixels), 2)
            or sizes.shape != (len(pixels),) or not np.isfinite(pixels).all()
            or not np.isfinite(sizes).all() or np.any(sizes <= 0) or np.any(sizes > 640)
            or np.any(pixels < 0) or np.any(pixels[:, 0] >= 640) or np.any(pixels[:, 1] >= 480)):
        raise ValueError('A support hypothesis contains only bounded public camera and image measurements')
    return deepcopy(value)


def compatible(value, transform, focal, pixels, sizes):
    value = validate(value)
    pixels, sizes = np.asarray(pixels, dtype=float), np.asarray(sizes, dtype=float)
    old_pixels, old_sizes = np.asarray(value['support_pixels']), np.asarray(value['support_sizes_px'])
    return (pixels.shape == old_pixels.shape and sizes.shape == old_sizes.shape
            and np.allclose(transform, value['camera_transform'], atol=1e-5, rtol=0)
            and math.isclose(focal, value['focal_pixels'], abs_tol=1e-5, rel_tol=0)
            and np.all(np.linalg.norm(pixels - old_pixels, axis=1) <= np.maximum(2., old_sizes * .1))
            and np.all(np.abs(sizes - old_sizes) <= old_sizes * .15))
