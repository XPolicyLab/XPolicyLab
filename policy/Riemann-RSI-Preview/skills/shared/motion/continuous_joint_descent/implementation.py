import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def finite(values, size):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError('Own continuous planning requires finite declared robot vectors')
    return array.copy()


def local_goal(pose, own_base, frame_bias):
    target, base = finite(pose, 7), finite(own_base, 7)
    if any(not .99 <= np.linalg.norm(value[3:]) <= 1.01 for value in (target, base)):
        raise ValueError('Own target and root orientations must be unit quaternions')
    base_rotation = Rotation.from_quat(base[[4, 5, 6, 3]])
    target_rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
    return (base_rotation.inv().apply(target[:3] - base[:3]) + finite(frame_bias, 3),
            base_rotation.inv() * target_rotation)


def unwrap_path(positions, current):
    path = np.asarray(positions, dtype=float)
    previous = finite(current, 6)
    if path.ndim != 2 or path.shape[1] != 6 or not 1 <= len(path) <= 20000 or not np.isfinite(path).all():
        raise ValueError('Unwrap only a bounded existing own-arm path')
    result = path.copy()
    for destination in result:
        nearest = previous[:5] + (destination[:5] - previous[:5] + np.pi) % (2 * np.pi) - np.pi
        destination[:5] = np.where(np.abs(nearest) <= 3.14, nearest,
                                   (destination[:5] + np.pi) % (2 * np.pi) - np.pi)
        previous = destination.copy()
    return result.astype(np.float32)


def plan_local_descent(current, pose, own_base, frame_bias, forward_kinematics, joint_planner):
    current = finite(current, 6)
    if not callable(forward_kinematics) or not callable(joint_planner):
        raise ValueError('Use only the worker-owned model and joint planner callbacks')
    position, rotation = local_goal(pose, own_base, frame_bias)

    def residual(joints):
        measured_position, measured_quaternion = forward_kinematics(joints)
        measured_position = finite(measured_position, 3)
        measured_quaternion = finite(measured_quaternion, 4)
        orientation = Rotation.from_quat(measured_quaternion[[1, 2, 3, 0]])
        return np.r_[measured_position - position, .12 * (rotation.inv() * orientation).as_rotvec()]

    solution = least_squares(residual, current,
        bounds=(np.maximum(current - .8, -3.14), np.minimum(current + .8, 3.14)),
        diff_step=.001, max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    error = residual(solution.x)
    if np.linalg.norm(error[:3]) >= .003 or np.linalg.norm(error[3:]) / .12 >= .035:
        raise RuntimeError('No continuous local descent IK path')
    planned = joint_planner(current.copy(), solution.x.copy())
    if planned.get('status') == 'Success':
        positions = np.asarray(planned['position'], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 6 or not 1 <= len(positions) <= 20000 or not np.isfinite(positions).all():
            raise ValueError('The own joint planner returned an unbounded path')
        return positions
    count = max(30, int(math.ceil(np.max(np.abs(solution.x - current)) / .01)))
    return np.linspace(current, solution.x, count + 1)[1:].astype(np.float32)
