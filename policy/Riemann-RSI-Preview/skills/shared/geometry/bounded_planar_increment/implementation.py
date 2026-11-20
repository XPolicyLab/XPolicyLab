from __future__ import annotations
import math
import numpy as np
from skills.shared.geometry.finite_rotation_pose.implementation import validate_pose, normalize_quat_wxyz, _array, _scalar
from skills.shared.geometry.rigid_contact_transform.implementation import quat_mul_wxyz, quat_to_mat_wxyz

def quat_axis_angle_wxyz(axis, angle):
    """Active right-handed rotation by ``angle`` radians around ``axis``."""
    axis = _array(axis, (3,), 'rotation axis')
    scale = float(np.max(np.abs(axis)))
    if scale == 0:
        raise ValueError('rotation axis must be nonzero')
    axis /= scale
    axis /= np.linalg.norm(axis)
    half = 0.5 * _scalar(angle, 'angle')
    return normalize_quat_wxyz(np.r_[math.cos(half), axis * math.sin(half)])

def yaw_quat(angle):
    return quat_axis_angle_wxyz([0, 0, 1], angle)

def wrap_yaw(angle):
    """Canonical radians in [-pi, pi); an exact pi tie turns clockwise."""
    angle = _scalar(angle, 'yaw')
    return (angle + math.pi) % (2 * math.pi) - math.pi

def yaw_of_quaternion(quaternion):
    """Heading of body-local X projected on world XY (ZYX yaw).

    A vertical local X has no planar heading. Fail explicitly instead of
    inventing yaw zero; full quaternion errors remain valid in that case.
    """
    rotation = quat_to_mat_wxyz(quaternion)
    if np.linalg.norm(rotation[:2, 0]) < 1e-10:
        raise ValueError('yaw is undefined when body-local X is vertical')
    return wrap_yaw(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))

def planar_step(current_pose, goal_pose, initial_pose, *, max_translation=0.02, max_yaw=math.radians(8)):
    """One bounded XY/yaw command on the initial object's planar manifold.

    XY displacement is bounded in Euclidean norm; yaw uses the shortest
    wrapped world-Z direction. Z is always initial Z, even if the observed
    object or requested goal has lifted. The initial full roll/pitch is
    retained by ``q_next = q_world_z(yaw_next-yaw_initial) * q_initial``.
    This deliberately cannot repair a tilted goal: use full-pose errors to
    detect a goal outside the planar manifold and live lift/slip guards to
    decide whether another physical push is appropriate.
    """
    current, goal, initial = map(validate_pose, (current_pose, goal_pose, initial_pose))
    distance = _scalar(max_translation, 'max_translation', positive=True)
    angle = _scalar(max_yaw, 'max_yaw', positive=True)
    if angle > math.pi:
        raise ValueError('max_yaw must not exceed pi radians')
    delta = goal[:2] - current[:2]
    norm = float(np.linalg.norm(delta))
    if not math.isfinite(norm):
        raise ValueError('XY displacement overflow')
    next_xy = current[:2] + delta * min(1.0, distance / norm) if norm else current[:2].copy()
    current_yaw = yaw_of_quaternion(current[3:])
    goal_yaw = yaw_of_quaternion(goal[3:])
    initial_yaw = yaw_of_quaternion(initial[3:])
    step = float(np.clip(wrap_yaw(goal_yaw - current_yaw), -angle, angle))
    quaternion = quat_mul_wxyz(yaw_quat(wrap_yaw(current_yaw + step - initial_yaw)), initial[3:])
    return np.r_[next_xy, initial[2], quaternion]
