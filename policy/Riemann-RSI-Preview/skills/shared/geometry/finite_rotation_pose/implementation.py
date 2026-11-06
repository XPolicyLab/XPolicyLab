from __future__ import annotations
import math
import numpy as np

def _array(value, shape, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite with shape {shape}') from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result

def _scalar(value, name, *, positive=False, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ' and nonnegative' if nonnegative else ''))
    return result

def normalize_quat_wxyz(quaternion):
    """Normalize a scalar-first quaternion; reject zero/NaN/Inf, never repair.

    The first numerically significant component is positive, so q and -q
    share a deterministic representative, including exact half turns.
    Scaling before the norm avoids overflow for finite non-unit inputs.
    """
    quaternion = _array(quaternion, (4,), 'quaternion')
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0:
        raise ValueError('zero quaternion has no rotation')
    quaternion /= scale
    quaternion /= np.linalg.norm(quaternion)
    first = np.flatnonzero(np.abs(quaternion) > 1e-15)[0]
    if quaternion[first] < 0:
        quaternion *= -1
    return quaternion

def validate_pose(pose):
    """Copy/normalize a finite 7D XYZ/wxyz pose; reject malformed rotations."""
    result = _array(pose, (7,), 'pose')
    result[3:] = normalize_quat_wxyz(result[3:])
    return result
