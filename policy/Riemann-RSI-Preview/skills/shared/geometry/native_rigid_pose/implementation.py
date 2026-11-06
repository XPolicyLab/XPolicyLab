"""implementation capabilities."""
from __future__ import annotations
from typing import Any
from typing import Iterable
from skills.shared.geometry.orientation_quaternion.implementation import mat_to_quat_wxyz
import math
import numpy as np

def as_numpy(value: Any) -> np.ndarray:
    """Convert torch/numpy/list values to a detached float numpy array."""
    if value is None:
        return np.asarray([], dtype=np.float32)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'cpu') and hasattr(value, 'numpy'):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)

def quat_to_mat_wxyz(quat: Iterable[float]) -> np.ndarray:
    q = as_numpy(list(quat)).reshape(4).astype(np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)

def quat_mul_wxyz(q1: Iterable[float], q2: Iterable[float]) -> np.ndarray:
    w1, x1, y1, z1 = as_numpy(list(q1)).reshape(4)
    w2, x2, y2, z2 = as_numpy(list(q2)).reshape(4)
    return np.asarray([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=np.float32)

def quaternion_angle_deg(q_a: Iterable[float], q_b: Iterable[float]) -> float:
    """Shortest scalar-first quaternion angle in degrees."""
    a = as_numpy(list(q_a)).reshape(4).astype(np.float64)
    b = as_numpy(list(q_b)).reshape(4).astype(np.float64)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))

def _finite_array(value, shape, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite with shape {shape}') from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result

def normalize_quat_wxyz(quaternion):
    """Normalize a scalar-first quaternion; reject zero/NaN/Inf, never repair.

    The first numerically significant component is positive, so q and -q
    share a deterministic representative, including exact half turns.
    Scaling before the norm avoids overflow for finite non-unit inputs.
    """
    quaternion = _finite_array(quaternion, (4,), 'quaternion')
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
    result = _finite_array(pose, (7,), 'pose')
    result[3:] = normalize_quat_wxyz(result[3:])
    return result

def validated_quat_mul_wxyz(left, right):
    """Compose two rotation quaternions, returning a unit quaternion."""
    w, x, y, z = normalize_quat_wxyz(left)
    a, b, c, d = normalize_quat_wxyz(right)
    return normalize_quat_wxyz([w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c, w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a])

def validated_quat_inverse_wxyz(quaternion):
    result = normalize_quat_wxyz(quaternion)
    result[1:] *= -1
    return normalize_quat_wxyz(result)

def validated_quat_to_mat_wxyz(quaternion):
    w, x, y, z = normalize_quat_wxyz(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)

def quat_inverse_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm_sq = float(quat @ quat)
    if norm_sq < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64) / norm_sq).astype(np.float32)

def rotate_vec_by_quat(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate a three vector with a scalar first quaternion."""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w = float(quat[0])
    xyz = quat[1:]
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    cross = 2.0 * np.cross(xyz, vector)
    return (vector + w * cross + np.cross(xyz, cross)).astype(np.float32)

def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """T_world_a * T_a_b, using scalar-first quaternions."""
    a, b = (np.asarray(a), np.asarray(b))
    return np.r_[a[:3] + quat_to_mat_wxyz(a[3:]) @ b[:3], quat_mul_wxyz(a[3:], b[3:])].astype(np.float32)

def inverse_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    rotation = quat_to_mat_wxyz(pose[3:]).T
    return np.r_[-rotation @ pose[:3], mat_to_quat_wxyz(rotation)].astype(np.float32)

def top_down_quat(closing_axis: np.ndarray, flip: bool=False) -> np.ndarray:
    x = np.array([0.0, 0.0, -1.0])
    y = np.asarray(closing_axis, dtype=float).copy()
    y[2] = 0
    if np.linalg.norm(y) < 1e-06:
        raise ValueError('grasp closing axis has no horizontal component')
    y /= np.linalg.norm(y)
    if flip:
        y = -y
    return mat_to_quat_wxyz(np.column_stack((x, y, np.cross(x, y))))

def transformed_vertices(pose: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices) @ quat_to_mat_wxyz(pose[3:]).T + pose[:3]

def interpolate_pose(start: np.ndarray, goal: np.ndarray, fraction: float) -> np.ndarray:
    """Linear translation and shortest-arc quaternion interpolation."""
    a, b = (np.asarray(start[3:], dtype=float), np.asarray(goal[3:], dtype=float))
    a, b = (a / np.linalg.norm(a), b / np.linalg.norm(b))
    dot = float(a @ b)
    if dot < 0:
        b, dot = (-b, -dot)
    if dot > 0.9995:
        q = a + fraction * (b - a)
    else:
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        q = (math.sin((1.0 - fraction) * angle) * a + math.sin(fraction * angle) * b) / math.sin(angle)
    q /= np.linalg.norm(q)
    return np.r_[start[:3] + fraction * (goal[:3] - start[:3]), q].astype(np.float32)

def _same_quaternion_rotation(left, right):
    """Quaternion sign/normalization representation only; no angular policy gate."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or (not np.isfinite(np.r_[a, b]).all()):
        return False
    scales = (np.max(np.abs(a)), np.max(np.abs(b)))
    if min(scales) <= 0.0:
        return False
    a = a / scales[0]
    b = b / scales[1]
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if a @ b < 0.0:
        b = -b
    return bool(np.max(np.abs(a - b)) <= 8 * np.finfo(float).eps)
