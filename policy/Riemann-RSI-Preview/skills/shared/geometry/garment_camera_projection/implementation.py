"""The v45 garment pinhole convention from public RGB, EEF and static URDF.

This module produces model rays; it supplies no surface/visibility/contact
evidence.  In particular a plane argument is a caller-declared geometric
reference, not an assertion that a current camera measured that plane.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np
from scipy.spatial.transform import Rotation

from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix


MAXIMUM_POINTS = 1024
MAXIMUM_IMAGE_RAYS = 480 * 640
PRINCIPAL_POINT = np.array([319.5, 239.5])
HEAD_FOCAL = 640 * 10 / 22.212
WRIST_FOCAL = 640 * 13 / 20.955
URDF_SHA256 = '7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423'
HEAD_FROM_OPTICAL = np.eye(4)
HEAD_FROM_OPTICAL[:3, :3] = [[1., 0., 0.], [0., -np.sqrt(3) / 2, .5], [0., -.5, -np.sqrt(3) / 2]]
HEAD_FROM_OPTICAL[:3, 3] = [0., -.41, 1.308]


@lru_cache(maxsize=1)
def wrist_optical_from_link6():
    """Read only the two fixed declared camera ancestors, never finger joints."""
    root = Path(__file__).resolve().parents[4]
    settings = json.loads((root / 'configs/mature-trajectory.json').read_text())
    source = root / settings['robot_urdf']
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != URDF_SHA256:
        raise ValueError('Garment camera geometry requires its declared fixed X5 URDF')
    joints = {joint.attrib['name']: joint for joint in ElementTree.parse(source).getroot().findall('joint')}
    transform = np.eye(4)
    for name, parent, child in (('hand_to_camera_mount', 'link6', 'camera_base'),
                                ('camera_joint', 'camera_base', 'camera')):
        joint = joints[name]
        if (joint.attrib['type'] != 'fixed' or joint.find('parent').attrib['link'] != parent
                or joint.find('child').attrib['link'] != child):
            raise ValueError('A garment wrist camera has only declared fixed URDF ancestors')
        origin = joint.find('origin')
        mounted = np.eye(4)
        mounted[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
        mounted[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        transform = transform @ mounted
    camera = np.eye(4)
    camera[:3, 3] = [0., 0., .001]
    camera[:3, :3] = Rotation.from_euler('XYZ', [0., -80., -90.], degrees=True).as_matrix()
    optical = np.eye(4)
    optical[:3, :3] = np.diag([1., -1., -1.])
    result = transform @ camera @ optical
    result.setflags(write=False)
    return result


def camera_model(api, camera):
    if api.image(camera).shape != (480, 640, 3):
        raise ValueError('Garment camera geometry requires current native 640x480 RGB')
    if camera == 'cam_head':
        return HEAD_FROM_OPTICAL.copy(), HEAD_FOCAL
    if camera in ('cam_left_wrist', 'cam_right_wrist'):
        side = camera.split('_')[1]
        return pose_matrix(api.targets()[side + '_ee_pose']) @ wrist_optical_from_link6(), WRIST_FOCAL
    raise ValueError('Garment geometry cannot introduce another camera')


def pixel_rays(api, camera, pixels):
    pixels = np.asarray(pixels, dtype=float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not 1 <= len(pixels) <= MAXIMUM_IMAGE_RAYS
            or not np.isfinite(pixels).all() or np.any(pixels < 0)
            or np.any(pixels[:, 0] >= 640) or np.any(pixels[:, 1] >= 480)):
        raise ValueError('A garment ray starts in a bounded current public image sample')
    transform, focal = camera_model(api, camera)
    local = np.column_stack([(pixels - PRINCIPAL_POINT) / focal, np.ones(len(pixels))])
    rays = local @ transform[:3, :3].T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return transform[:3, 3].copy(), rays


def project_points(api, camera, points):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= MAXIMUM_POINTS
            or not np.isfinite(points).all()):
        raise ValueError('Project a bounded finite garment point set')
    transform, focal = camera_model(api, camera)
    optical = (points - transform[:3, 3]) @ transform[:3, :3]
    valid = optical[:, 2] > 1e-8
    result = np.full((len(points), 2), np.nan)
    result[valid] = optical[valid, :2] / optical[valid, 2, None] * focal + PRINCIPAL_POINT
    return result


def points_on_plane(api, camera, pixels, height):
    pixels = np.asarray(pixels, dtype=float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not 1 <= len(pixels) <= MAXIMUM_POINTS
            or not np.isfinite(height)):
        raise ValueError('A garment reference height must be finite')
    origin, rays = pixel_rays(api, camera, pixels)
    valid = np.abs(rays[:, 2]) > 1e-8
    distance = np.full(len(rays), np.nan)
    distance[valid] = (height - origin[2]) / rays[valid, 2]
    valid &= distance > 0
    result = np.full((len(rays), 3), np.nan)
    result[valid] = origin + rays[valid] * distance[valid, None]
    return result
