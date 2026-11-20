import numpy as np

from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix


DOWNWARD_QUATERNION = np.asarray([.5, -.5, .5, .5])


def run(center, height, forward=.155, quaternion=None):
    point = np.asarray(center, dtype=float).reshape(-1)
    orientation = np.asarray(DOWNWARD_QUATERNION if quaternion is None else quaternion, dtype=float).reshape(-1)
    if (point.size != 3 or orientation.size != 4 or not np.isfinite(point).all()
            or not np.isfinite(orientation).all() or not .99 <= np.linalg.norm(orientation) <= 1.01
            or not np.isfinite(height) or not np.isfinite(forward) or not 0 <= forward <= .3):
        raise ValueError('A wrist target requires a finite public contact and declared own tool geometry')
    point = point.copy()
    point[2] = float(height)
    return np.r_[point - rotation_matrix(orientation)[:, 0] * forward, orientation]
