"""ARX X5 robot geometry; no motion skills or simulator imports."""
import math
import numpy as np
from typing import Any

ARMS = ("left", "right")
# X5A joint8 opens along -link6.Y; joint7 opens along +link6.Y. These
# identities travel with the wrist, independent of camera/world left and right.
POSE_FINGER_LINKS = ("link8", "link7")
#: ``gripper_bias`` in Assets/Robots/x5/robot_config.yml.
GRASP_POINT_OFFSET_M = 0.145
#: The fingertips reach this far past the grasp point along the tool axis.
FINGERTIP_REACH_M = 0.1576 - GRASP_POINT_OFFSET_M
#: ``gripper_scale`` in robot_config.yml: finger joint metres at gripper 0 and 1.
FINGER_JOINT_RANGE = (-0.01, 0.044)
FINGER_FACE_OFFSET_M = 0.0004
# Benchmark scene/default.yml: world centre z=0.740, thickness=0.050.
TABLE_SURFACE_Z = 0.765
TABLE_THICKNESS_M = 0.050
# Geometric lower bound; any future tracking/contact margin is separate.
TOOL_TABLE_GUARD_Z = TABLE_SURFACE_Z


def table_collision_cuboid(surface_z: float) -> dict:
    """Conservative XY table; surface_z is its top in the planner's base frame."""
    return {'dims': [3.0, 3.0, TABLE_THICKNESS_M],
            'pose': [0.0, 0.0, float(surface_z) - TABLE_THICKNESS_M / 2,
                     1.0, 0.0, 0.0, 0.0]}


def jaw_gap_mm(gripper: float) -> float:
    lo, hi = FINGER_JOINT_RANGE
    finger = lo + float(np.clip(gripper, 0.0, 1.0)) * (hi - lo)
    return max(0.0, 2.0 * (finger + FINGER_FACE_OFFSET_M) * 1000.0)


def gripper_for_gap(gap_mm: float) -> float:
    lo, hi = FINGER_JOINT_RANGE
    return float(np.clip((gap_mm / 2000.0 - FINGER_FACE_OFFSET_M - lo) / (hi - lo), 0.0, 1.0))


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(m: np.ndarray, reference_quaternion=None) -> np.ndarray:
    """Convert a rotation, choosing its quaternion sign near an optional reference.

    q and -q are the SAME pose. Hemisphere selection must never swap finger points
    or rotate the physical jaw by 180 degrees. Without a reference, use w >= 0.
    """
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q) / np.linalg.norm(q)
    if reference_quaternion is None:
        return q * (1.0 if q[0] >= 0 else -1.0)
    reference = np.asarray(reference_quaternion, dtype=float)
    if reference.shape != (4,) or not np.isfinite(reference).all() or np.linalg.norm(reference) < 1e-12:
        raise ValueError('reference quaternion must be a finite nonzero wxyz vector')
    return -q if float(q @ reference) < 0 else q


def grasp_frame(flange_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grasp point and link6 rotation from a flange pose [x, y, z, qw, qx, qy, qz]."""
    pose = np.asarray(flange_pose, dtype=float).reshape(-1)
    rotation = quat_wxyz_to_matrix(pose[3:7])
    return pose[:3] + GRASP_POINT_OFFSET_M * rotation[:, 0], rotation


def flange_pose(grasp_point: np.ndarray, rotation: np.ndarray, reference_quaternion=None) -> np.ndarray:
    """Flange pose [x, y, z, qw, qx, qy, qz] that puts the grasp point at ``grasp_point``."""
    position = np.asarray(grasp_point, dtype=float) - GRASP_POINT_OFFSET_M * rotation[:, 0]
    return np.concatenate([position, matrix_to_quat_wxyz(rotation, reference_quaternion)])


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
