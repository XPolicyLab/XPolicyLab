from __future__ import annotations
import numpy as np

def quaternion_to_matrix(q, order='wxyz'):
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4 or order not in {'wxyz', 'xyzw'}:
        raise ValueError('Quaternion must end in 4 and specify wxyz/xyzw')
    if order == 'xyzw':
        q = q[..., [3, 0, 1, 2]]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-12):
        raise ValueError('Nonfinite or zero quaternion')
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(q.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix):
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape[-2:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError('Expected finite rotation matrices')
    if not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-05) or np.any(np.linalg.det(r) < 0):
        raise ValueError('Rotation is not in SO(3)')
    a, b, c = (r[..., 0, 0], r[..., 1, 1], r[..., 2, 2])
    sq = np.stack((1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), -1)
    v = np.stack((np.stack((sq[..., 0], r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1), np.stack((r[..., 2, 1] - r[..., 1, 2], sq[..., 1], r[..., 0, 1] + r[..., 1, 0], r[..., 0, 2] + r[..., 2, 0]), -1), np.stack((r[..., 0, 2] - r[..., 2, 0], r[..., 0, 1] + r[..., 1, 0], sq[..., 2], r[..., 1, 2] + r[..., 2, 1]), -1), np.stack((r[..., 1, 0] - r[..., 0, 1], r[..., 0, 2] + r[..., 2, 0], r[..., 1, 2] + r[..., 2, 1], sq[..., 3]), -1)), -2)
    idx = np.argmax(sq, axis=-1)
    q = np.take_along_axis(v, np.broadcast_to(idx[..., None, None], idx.shape + (1, 4)), axis=-2)[..., 0, :]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    pivot = np.take_along_axis(q[..., 1:], np.argmax(abs(q[..., 1:]), -1)[..., None], -1)[..., 0]
    flip = (q[..., 0] < -1e-12) | (abs(q[..., 0]) <= 1e-12) & (pivot < 0)
    return np.where(flip[..., None], -q, q)

def rotvec_to_matrix(vector):
    v = np.asarray(vector, dtype=np.float64)
    if v.shape[-1] != 3 or not np.isfinite(v).all():
        raise ValueError('Expected finite rotvec')
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    q = np.concatenate((np.cos(angle / 2), v * (0.5 * np.sinc(angle / (2 * np.pi)))), -1)
    return quaternion_to_matrix(q)

def matrix_to_rotvec(matrix):
    q = matrix_to_quaternion(matrix)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, np.maximum(q[..., :1], 0))
    return q[..., 1:] * np.divide(angle, s, out=np.full_like(s, 2.0), where=s > 1e-12)

def euler_xyz_to_matrix(euler):
    r, p, y = np.moveaxis(np.asarray(euler, dtype=np.float64), -1, 0)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y))
    return np.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr), -1).reshape(r.shape + (3, 3))

def continuous_rotvec(matrices, valid=None):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError('Continuous lift expects [T,3,3]')
    valid = np.ones(len(matrices), bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != (len(matrices),):
        raise ValueError('Rotation validity must have shape [T]')
    result = np.zeros((len(matrices), 3), np.float64)
    previous = np.zeros(3)
    for t in range(len(matrices)):
        if not valid[t]:
            previous = np.zeros(3)
            continue
        principal = matrix_to_rotvec(matrices[t])
        theta = np.linalg.norm(principal)
        if theta < 1e-10:
            length = np.linalg.norm(previous)
            chosen = np.zeros(3)
            if length > 1e-10:
                n = int(np.floor(length / (2 * np.pi) + 0.5))
                chosen = previous / length * (2 * np.pi * n)
        else:
            axis = principal / theta
            center = int(np.rint((np.dot(previous, axis) - theta) / (2 * np.pi)))
            windings = sorted(set([0, center - 1, center, center + 1]), key=lambda n: (abs(n), n))
            candidates = [(theta + 2 * np.pi * n) * axis for n in windings]
            distances = [np.linalg.norm(c - previous) for c in candidates]
            best = min(distances)
            chosen = candidates[next((i for i, d in enumerate(distances) if d <= best + 1e-12))]
        result[t] = chosen
        previous = chosen
    return result

def rotation_6d(matrix):
    r = np.asarray(matrix)
    return np.concatenate((r[..., :, 0], r[..., :, 1]), -1)

def continuous_rotvec_batch(matrices, valid):
    matrices = np.asarray(matrices, dtype=np.float64)
    valid = np.asarray(valid, bool)
    if matrices.ndim != 4 or matrices.shape[-2:] != (3, 3) or valid.shape != matrices.shape[:2]:
        raise ValueError('Expected matrices [B,T,3,3] and validity [B,T]')
    principal = np.zeros((*valid.shape, 3))
    if valid.any():
        principal[valid] = matrix_to_rotvec(matrices[valid])
    result = np.zeros_like(principal)
    previous = np.zeros((len(matrices), 3))
    for t in range(matrices.shape[1]):
        vector = principal[:, t]
        theta = np.linalg.norm(vector, axis=-1)
        axis = vector / np.maximum(theta[:, None], 1e-300)
        center = np.rint(((previous * axis).sum(-1) - theta) / (2 * np.pi)).astype(np.int64)
        windings = np.stack((np.zeros_like(center), center - 1, center, center + 1), -1)
        order = np.lexsort((windings, abs(windings)), axis=-1)
        windings = np.take_along_axis(windings, order, axis=-1)
        candidates = (theta[:, None] + 2 * np.pi * windings)[..., None] * axis[:, None]
        distances = np.linalg.norm(candidates - previous[:, None], axis=-1)
        selected = np.argmax(distances <= distances.min(-1, keepdims=True) + 1e-12, axis=-1)
        chosen = candidates[np.arange(len(matrices)), selected]
        length = np.linalg.norm(previous, axis=-1)
        identity = previous / np.maximum(length[:, None], 1e-300)
        identity *= (2 * np.pi * np.floor(length / (2 * np.pi) + 0.5))[:, None]
        chosen = np.where((theta < 1e-10)[:, None], identity, chosen)
        previous = np.where(valid[:, t, None], chosen, 0)
        result[:, t] = previous
    return result
