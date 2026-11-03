"""Original X5 finger envelope measured relative to observed support."""
import numpy as np
from . import GeometryUnavailable


def observed_finger_clearance(transform, support):
    translation = np.asarray(transform.translation, float)
    rotation = np.asarray(transform.rotation, float)
    if translation.shape != (3,) or rotation.shape != (3,3) or not (
            np.isfinite(translation).all() and np.isfinite(rotation).all()):
        raise GeometryUnavailable('Finite public robot transform required')
    tip = translation + rotation[:,0] * .15757000232458113
    # Preserve the original open-finger lateral envelope. Evaluate both sides
    # against the measured slope rather than subtracting an absolute world Z.
    points = [tip + sign*.04*rotation[:,1] for sign in (-1,1)]
    return float(min(point[2]-support.height(point[0],point[1]) for point in points))
