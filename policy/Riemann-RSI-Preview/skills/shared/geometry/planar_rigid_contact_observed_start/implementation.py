import math
import numpy as np
from scipy.spatial.transform import Rotation
from skills.shared.motion.urdf_pose_kinematics_observed_start.implementation import matrix_pose, pose_matrix


from skills.shared.geometry.planar_rigid_contact.implementation import checked_pose

def yaw(value):
    matrix = pose_matrix(checked_pose(value))
    return math.atan2(matrix[1, 0], matrix[0, 0])

def wrapped(value):
    return (value + math.pi) % (2 * math.pi) - math.pi

def errors(actual, destination):
    actual, destination = (checked_pose(actual), checked_pose(destination))
    rotation = pose_matrix(destination)[:3, :3] @ pose_matrix(actual)[:3, :3].T
    return {'xy_m': float(np.linalg.norm(actual[:2] - destination[:2])), 'orientation_deg': math.degrees(float(Rotation.from_matrix(rotation).magnitude()))}

def planar_step(current, destination, initial, translation=0.018, angle=math.radians(6)):
    current, destination, initial = map(checked_pose, (current, destination, initial))
    if type(translation) not in (int, float) or not math.isfinite(translation) or (not 0 < translation <= 2) or (type(angle) not in (int, float)) or (not math.isfinite(angle)) or (not 0 < angle <= math.pi):
        raise ValueError('A planar increment has a declared finite translation and yaw bound')
    delta = destination[:2] - current[:2]
    distance = float(np.linalg.norm(delta))
    result = initial.copy()
    result[:2] = current[:2] + delta * min(1.0, translation / distance) if distance else current[:2]
    increment = float(np.clip(wrapped(yaw(destination) - yaw(current)), -angle, angle))
    offset = wrapped(yaw(current) + increment - yaw(initial))
    rotated = Rotation.from_rotvec([0.0, 0.0, offset]).as_matrix() @ pose_matrix(initial)[:3, :3]
    quaternion = Rotation.from_matrix(rotated).as_quat()
    result[3:] = np.r_[quaternion[3], quaternion[:3]]
    return result

def capture(public_shape_pose, observed_own_wrist):
    return matrix_pose(np.linalg.inv(pose_matrix(checked_pose(observed_own_wrist))) @ pose_matrix(checked_pose(public_shape_pose)))

def own_wrist_target(destination, shape_in_wrist):
    return matrix_pose(pose_matrix(checked_pose(destination)) @ np.linalg.inv(pose_matrix(checked_pose(shape_in_wrist))))

def contact_wrist(shape_pose, turn, forward_offset=0.154):
    shape_pose = checked_pose(shape_pose)
    if turn not in (-1, 1) or forward_offset != 0.154:
        raise ValueError('Contact uses the original two closing directions and declared own robot TCP offset')
    forward = np.asarray([0.0, 0.0, -1.0])
    closing = pose_matrix(shape_pose)[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    closing[2] = 0.0
    closing = closing / np.linalg.norm(closing) * turn
    orientation = np.column_stack([forward, closing, np.cross(forward, closing)])
    quaternion = Rotation.from_matrix(orientation).as_quat()
    return np.r_[shape_pose[:3] - orientation[:, 0] * forward_offset, quaternion[3], quaternion[:3]]
