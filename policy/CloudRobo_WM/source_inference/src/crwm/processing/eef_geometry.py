"""SO(3) inference routines from the pinned EEF training snapshot."""
from __future__ import annotations

import numpy as np

def quaternion_to_matrix(q, order="wxyz"):
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4 or order not in {"wxyz", "xyzw"}:
        raise ValueError("Quaternion must end in 4 and specify wxyz/xyzw")
    if order == "xyzw":
        q = q[..., [3, 0, 1, 2]]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-12):
        raise ValueError("Nonfinite or zero quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))

def matrix_to_quaternion(matrix):
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape[-2:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Expected finite rotation matrices")
    if not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-5) or np.any(np.linalg.det(r) < 0):
        raise ValueError("Rotation is not in SO(3)")
    a, b, c = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    sq = np.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), -1)
    v = np.stack((
        np.stack((sq[..., 0], r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), -1),
        np.stack((r[..., 2, 1]-r[..., 1, 2], sq[..., 1], r[..., 0, 1]+r[..., 1, 0], r[..., 0, 2]+r[..., 2, 0]), -1),
        np.stack((r[..., 0, 2]-r[..., 2, 0], r[..., 0, 1]+r[..., 1, 0], sq[..., 2], r[..., 1, 2]+r[..., 2, 1]), -1),
        np.stack((r[..., 1, 0]-r[..., 0, 1], r[..., 0, 2]+r[..., 2, 0], r[..., 1, 2]+r[..., 2, 1], sq[..., 3]), -1),
    ), -2)
    idx = np.argmax(sq, axis=-1)
    q = np.take_along_axis(v, np.broadcast_to(idx[..., None, None], idx.shape+(1, 4)), axis=-2)[..., 0, :]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    pivot = np.take_along_axis(q[..., 1:], np.argmax(abs(q[..., 1:]), -1)[..., None], -1)[..., 0]
    flip = (q[..., 0] < -1e-12) | ((abs(q[..., 0]) <= 1e-12) & (pivot < 0))
    return np.where(flip[..., None], -q, q)

def rotvec_to_matrix(vector):
    v = np.asarray(vector, dtype=np.float64)
    if v.shape[-1] != 3 or not np.isfinite(v).all():
        raise ValueError("Expected finite rotvec")
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    q = np.concatenate((np.cos(angle/2), v * (0.5*np.sinc(angle/(2*np.pi)))), -1)
    return quaternion_to_matrix(q)

def rotation_6d(matrix):
    r = np.asarray(matrix)
    return np.concatenate((r[..., :, 0], r[..., :, 1]), -1)
