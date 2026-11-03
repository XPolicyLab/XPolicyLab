"""Shared rigid transforms and URDF joint-chain kinematics.

Pose vectors use [x, y, z, qw, qx, qy, qz]. These functions own no episode state;
callers provide the calibrated robot asset, base transform and observed joints.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def pose(transform):
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], quaternion[[3, 0, 1, 2]]].tolist()


def calculate_target_pose(real_pose, set_pose, target):
    return pose(matrix(set_pose) @ np.linalg.inv(matrix(real_pose)) @ matrix(target))


def joint_chain(urdf, joint_names):
    tree = ET.parse(urdf)
    chain = []
    for name in joint_names:
        joint = tree.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError('URDF joint is missing: ' + name)
        origin = joint.find('origin')
        transform = np.eye(4)
        transform[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
        transform[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        chain.append((transform, axis))
    return chain


def forward_kinematics(chain, base_pose, joints):
    result = matrix(base_pose)
    for angle, (origin, axis) in zip(joints, chain, strict=True):
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        result = result @ origin @ motion
    return pose(result)
