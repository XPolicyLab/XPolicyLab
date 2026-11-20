"""Pure pose and event-value helpers from the frozen standard toast chain."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np




from skills.shared.geometry.native_rigid_pose.implementation import as_numpy, interpolate_pose

def quat_to_mat_wxyz(quat: Iterable[float]) -> np.ndarray:
    q = as_numpy(list(quat)).reshape(4).astype(np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Stable 3x3 rotation matrix to (w,x,y,z)."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([(s / 4), (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4])
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:
        q = -q
    return q.astype(np.float32)


def quat_mul_wxyz(q1: Iterable[float], q2: Iterable[float]) -> np.ndarray:
    w1, x1, y1, z1 = as_numpy(list(q1)).reshape(4)
    w2, x2, y2, z2 = as_numpy(list(q2)).reshape(4)
    return np.asarray(
        [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2],
        dtype=np.float32,
    )


def quaternion_angle_deg(q_a: Iterable[float], q_b: Iterable[float]) -> float:
    """Shortest scalar-first quaternion angle in degrees."""
    a = as_numpy(list(q_a)).reshape(4).astype(np.float64)
    b = as_numpy(list(q_b)).reshape(4).astype(np.float64)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """T_world_a * T_a_b, using scalar-first quaternions."""
    a, b = np.asarray(a), np.asarray(b)
    return np.r_[a[:3] + quat_to_mat_wxyz(a[3:]) @ b[:3],
                 quat_mul_wxyz(a[3:], b[3:])].astype(np.float32)


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    rotation = quat_to_mat_wxyz(pose[3:]).T
    return np.r_[-rotation @ pose[:3], mat_to_quat_wxyz(rotation)].astype(np.float32)


def grasp_pose(key_pose: np.ndarray, local_point: np.ndarray,
               tool_quat: np.ndarray, finger_offset: float) -> np.ndarray:
    contact = key_pose[:3] + quat_to_mat_wxyz(key_pose[3:]) @ local_point
    wrist = contact - quat_to_mat_wxyz(tool_quat)[:, 0] * finger_offset
    return np.r_[wrist, tool_quat].astype(np.float32)




__all__ = ('as_numpy', 'quat_to_mat_wxyz', 'mat_to_quat_wxyz', 'quat_mul_wxyz', 'quaternion_angle_deg', 'jsonable', 'compose_pose', 'inverse_pose', 'grasp_pose', 'interpolate_pose')
