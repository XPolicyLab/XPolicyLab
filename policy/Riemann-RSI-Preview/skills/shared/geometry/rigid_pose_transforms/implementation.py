"""Canonical stateless helpers; caller-owned state and clocks remain local."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose):
    translation_x, translation_y, translation_z, qw, qx, qy, qz = pose
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    rot_matrix = rotation.as_matrix()
    matrix = np.eye(4)
    matrix[0:3, 0:3] = rot_matrix
    matrix[0:3, 3] = [translation_x, translation_y, translation_z]
    return matrix

def matrix_to_pose(matrix):
    position = matrix[0:3, 3]
    rot_matrix = matrix[0:3, 0:3]
    rotation = Rotation.from_matrix(rot_matrix)
    quaternion = rotation.as_quat()
    qx, qy, qz, qw = quaternion
    quaternion_wxyz = [qw, qx, qy, qz]
    pose = np.concatenate([position, quaternion_wxyz])
    return pose

def calculate_target_pose(real_pose, set_pose, relative_real_pose):
    T_A = pose_to_matrix(real_pose)
    T_B = pose_to_matrix(set_pose)
    T_C = pose_to_matrix(relative_real_pose)
    R_A = T_A[0:3, 0:3]
    t_A = T_A[0:3, 3]
    inv_A = np.eye(4)
    inv_A[0:3, 0:3] = R_A.T
    inv_A[0:3, 3] = -R_A.T @ t_A
    T_relative = inv_A @ T_C
    T_D = T_B @ T_relative
    relative_set_pose = matrix_to_pose(T_D)
    return relative_set_pose
