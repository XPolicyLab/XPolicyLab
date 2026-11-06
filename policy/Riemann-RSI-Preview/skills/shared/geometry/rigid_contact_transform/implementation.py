from __future__ import annotations
import math
import numpy as np
from skills.shared.geometry.finite_rotation_pose.implementation import normalize_quat_wxyz, validate_pose, _array

def quat_mul_wxyz(left, right):
    """Compose two rotation quaternions, returning a unit quaternion."""
    scalar_first, vector_first, vector_second, vector_third = normalize_quat_wxyz(left)
    scalar_second, other_first, other_second, other_third = normalize_quat_wxyz(right)
    return normalize_quat_wxyz([scalar_first * scalar_second - vector_first * other_first - vector_second * other_second - vector_third * other_third, scalar_first * other_first + vector_first * scalar_second + vector_second * other_third - vector_third * other_second, scalar_first * other_second - vector_first * other_third + vector_second * scalar_second + vector_third * other_first, scalar_first * other_third + vector_first * other_second - vector_second * other_first + vector_third * scalar_second])

def quat_inverse_wxyz(quaternion):
    result = normalize_quat_wxyz(quaternion)
    result[1:] *= -1
    return normalize_quat_wxyz(result)

def quat_to_mat_wxyz(quaternion):
    scalar_first, vector_first, vector_second, vector_third = normalize_quat_wxyz(quaternion)
    return np.array([[1 - 2 * (vector_second * vector_second + vector_third * vector_third), 2 * (vector_first * vector_second - vector_third * scalar_first), 2 * (vector_first * vector_third + vector_second * scalar_first)], [2 * (vector_first * vector_second + vector_third * scalar_first), 1 - 2 * (vector_first * vector_first + vector_third * vector_third), 2 * (vector_second * vector_third - vector_first * scalar_first)], [2 * (vector_first * vector_third - vector_second * scalar_first), 2 * (vector_second * vector_third + vector_first * scalar_first), 1 - 2 * (vector_first * vector_first + vector_second * vector_second)]], dtype=np.float64)

def rotate_vec_by_quat(vector, quaternion):
    return quat_to_mat_wxyz(quaternion) @ _array(vector, (3,), 'vector')

def compose_pose(parent, child):
    """SE(3) product: a parent/world pose times a child/parent pose."""
    parent, child = (validate_pose(parent), validate_pose(child))
    return np.r_[parent[:3] + rotate_vec_by_quat(child[:3], parent[3:]), quat_mul_wxyz(parent[3:], child[3:])]

def inverse_pose(pose):
    pose = validate_pose(pose)
    inverse = quat_inverse_wxyz(pose[3:])
    return np.r_[rotate_vec_by_quat(-pose[:3], inverse), inverse]

def capture_contact_transform(object_pose, tool_pose):
    """Record object-in-tool from the *measured* live world poses.

    ``relative = inv(T_world_tool) @ T_world_object``. Capture after contact
    settles, rather than using the commanded tool pose or a guessed offset.
    """
    return compose_pose(inverse_pose(tool_pose), object_pose)

def tool_pose_for_object(desired_object_pose, object_in_tool):
    """Solve the rigid-contact target: T_world_object_goal @ inv(relative)."""
    return compose_pose(desired_object_pose, inverse_pose(object_in_tool))
