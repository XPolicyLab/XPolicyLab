"""Pose helpers. Poses are [x, y, z, qw, qx, qy, qz]; rotations are 3x3 matrices."""

import numpy as np
import transforms3d as t3d


def pose_to_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = t3d.quaternions.quat2mat(np.asarray(pose[3:7], dtype=float))
    matrix[:3, 3] = np.asarray(pose[:3], dtype=float)
    return matrix


def matrix_to_pose(matrix):
    quat = t3d.quaternions.mat2quat(matrix[:3, :3])
    if quat[0] < 0:
        quat = -quat
    return list(map(float, matrix[:3, 3])) + list(map(float, quat))


def rpy_deg(rotation):
    """Extrinsic XYZ (roll about x, then pitch about y, then yaw about z), degrees."""
    return [float(np.degrees(a)) for a in t3d.euler.mat2euler(rotation, axes="sxyz")]


def rotation_from_rpy_deg(roll, pitch, yaw):
    return t3d.euler.euler2mat(np.radians(roll), np.radians(pitch), np.radians(yaw), axes="sxyz")


def angle_between_deg(rotation_a, rotation_b):
    cos = (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def slerp(rotation_a, rotation_b, fraction):
    qa = t3d.quaternions.mat2quat(rotation_a)
    qb = t3d.quaternions.mat2quat(rotation_b)
    if np.dot(qa, qb) < 0:
        qb = -qb
    dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if dot > 0.9995:
        q = qa + fraction * (qb - qa)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - fraction) * theta) * qa + np.sin(fraction * theta) * qb) / np.sin(theta)
    return t3d.quaternions.quat2mat(q / np.linalg.norm(q))
