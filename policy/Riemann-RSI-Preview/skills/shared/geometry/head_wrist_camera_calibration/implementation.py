"""Current RGB camera geometry from fixed calibration and measured X5 link6 poses.

Camera poses are never obtained from the simulator. URDF origins use extrinsic
xyz Euler angles; RoboDojo camera-config orientations use intrinsic XYZ angles.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from skills.shared.geometry.x5_public_frame_model.implementation import CAMERA_CENTER, CAMERA_ROTATION, FOCAL_PX, matrix, rotation


class Camera:
    def __init__(self, name, transform, focal_px):
        self.name = name
        self.center = np.asarray(transform[:3, 3]).copy()
        self.rotation = np.asarray(transform[:3, :3]).copy()
        self.focal_px = float(focal_px)

    def project(self, points):
        xyz = (np.asarray(points, float) - self.center) @ self.rotation
        distance = -xyz[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = np.stack([320 + self.focal_px * xyz[..., 0] / distance,
                           240 - self.focal_px * xyz[..., 1] / distance], axis=-1)
        return np.where((distance > 1e-5)[..., None], uv, np.nan)

    def pixels(self, uv, plane_z):
        uv = np.asarray(uv, float).reshape(-1, 2)
        rays = np.c_[(uv[:, 0] - 320) / self.focal_px,
                     -(uv[:, 1] - 240) / self.focal_px, -np.ones(len(uv))]
        rays = rays @ self.rotation.T
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = (float(plane_z) - self.center[2]) / rays[:, 2]
        points = self.center + scale[:, None] * rays
        valid = (scale > 0) & (np.abs(rays[:, 2]) > 1e-6)
        return np.where(valid[:, None], points, np.nan)


class CameraRig:
    """Only the shipped public URDF and the official fixed camera constants."""
    def __init__(self, urdf, fingers=None):
        urdf = Path(urdf)
        joints = {j.attrib['name']: j for j in ET.parse(urdf).getroot().findall('joint')}
        self.wrist_from_link6 = np.eye(4)
        for name in ('hand_to_camera_mount', 'camera_joint'):
            joint = joints[name]
            if joint.attrib['type'] != 'fixed':
                raise ValueError('Wrist camera calibration requires the fixed public X5 joints')
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler(
                'xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            self.wrist_from_link6 = self.wrist_from_link6 @ transform
        local = np.eye(4)
        local[:3, 3] = [0., 0., .001]
        local[:3, :3] = Rotation.from_euler('XYZ', [0., -80., -90.], degrees=True).as_matrix()
        self.wrist_from_link6 = self.wrist_from_link6 @ local
        meshes = np.load(fingers or urdf.parent / 'assets' / 'fingers.npz', allow_pickle=False)
        self.primary_limits = tuple(float(joints['joint7'].find('limit').attrib[key]) for key in ('lower', 'upper'))
        self.maximum_openings = np.array([max(abs(float(joints[f'joint{index}'].find('limit').attrib[key])) for key in ('lower','upper')) for index in (7,8)])
        self.fingers = []
        for index in (7, 8):
            joint = joints[f'joint{index}']
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib['xyz'], sep=' ')
            transform[:3, :3] = Rotation.from_euler(
                'xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
            self.fingers.append((transform, axis, meshes[f'link{index}'].copy()))

    def current(self, obs):
        head = np.eye(4)
        head[:3, :3], head[:3, 3] = CAMERA_ROTATION, CAMERA_CENTER
        result = {'cam_head': Camera('cam_head', head, FOCAL_PX)}
        for arm in ('left', 'right'):
            name = f'cam_{arm}_wrist'
            transform = matrix(obs.state[arm + '_ee_pose']) @ self.wrist_from_link6
            result[name] = Camera(name, transform, 640. * 13. / 20.955)
        return result


def finger_occlusion_mask(camera, state, plane_z, rig):
    """Union explicit public-asset opening hypotheses; never a measured finger pair."""
    from skills.shared.observability.public_action_observation.implementation import opening_hypotheses
    mask = np.zeros((480, 640), np.uint8)
    arms = {}
    for arm in ('left', 'right'):
        arm_mask = np.zeros_like(mask)
        wrist = np.asarray(state[arm + '_ee_pose'])
        for openings in opening_hypotheses(state, arm, rig):
            for index, (origin, axis, triangles) in enumerate(rig.fingers):
                local = triangles @ origin[:3, :3].T + origin[:3, 3] + axis * float(openings[index])
                world = local @ rotation(wrist[3:]).T + wrist[:3]
                world = world[world[:, :, 2].max(axis=1) >= plane_z]
                uv = camera.project(world)
                uv = uv[np.isfinite(uv).all(axis=(1, 2))]
                for triangle in np.rint(np.clip(uv, -10000, 10000)).astype(np.int32):
                    cv2.fillConvexPoly(arm_mask, triangle, 1)
        arm_mask = cv2.dilate(arm_mask, np.ones((3, 3), np.uint8))
        mask |= arm_mask
        arms[arm] = arm_mask
    return mask.astype(bool), arms
