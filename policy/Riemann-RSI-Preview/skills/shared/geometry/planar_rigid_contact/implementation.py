import math

import numpy as np
from scipy.spatial.transform import Rotation

from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.geometry.bounded_planar_increment.implementation import planar_step as original_planar_step, yaw_of_quaternion
from skills.shared.geometry.contact_pose_error.implementation import pose_errors
from skills.shared.geometry.rigid_contact_transform.implementation import capture_contact_transform, tool_pose_for_object


def checked_pose(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (7,) or not np.isfinite(result).all()
            or np.max(np.abs(result[:3])) > 3 or not .99 <= np.linalg.norm(result[3:]) <= 1.01):
        raise ValueError('Contact geometry contains only a finite bounded public pose hypothesis')
    result = result.copy()
    result[3:] /= np.linalg.norm(result[3:])
    return result


def yaw(value):
    return yaw_of_quaternion(checked_pose(value)[3:])


def wrapped(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def errors(actual, destination):
    actual, destination = checked_pose(actual), checked_pose(destination)
    return pose_errors(actual, destination)


def planar_step(current, destination, initial, translation=.018, angle=math.radians(6)):
    current, destination, initial = map(checked_pose, (current, destination, initial))
    if (type(translation) not in (int, float) or not math.isfinite(translation) or not 0 < translation <= 2
            or type(angle) not in (int, float) or not math.isfinite(angle) or not 0 < angle <= math.pi):
        raise ValueError('A planar increment has a declared finite translation and yaw bound')
    return original_planar_step(current, destination, initial,
        max_translation=translation, max_yaw=angle)


def capture(public_shape_pose, observed_own_wrist):
    return capture_contact_transform(checked_pose(public_shape_pose), checked_pose(observed_own_wrist))


def own_wrist_target(destination, shape_in_wrist):
    return tool_pose_for_object(checked_pose(destination), checked_pose(shape_in_wrist))


def contact_wrist(shape_pose, turn, forward_offset=.154):
    shape_pose = checked_pose(shape_pose)
    if turn not in (-1, 1) or forward_offset != .154:
        raise ValueError('Contact uses the original two closing directions and declared own robot TCP offset')
    forward = np.asarray([0., 0., -1.])
    closing = pose_matrix(shape_pose)[:3, :3] @ np.asarray([0., 1., 0.])
    closing[2] = 0.
    closing = closing / np.linalg.norm(closing) * turn
    orientation = np.column_stack([forward, closing, np.cross(forward, closing)])
    quaternion = Rotation.from_matrix(orientation).as_quat()
    return np.r_[shape_pose[:3] - orientation[:, 0] * forward_offset, quaternion[3], quaternion[:3]]
