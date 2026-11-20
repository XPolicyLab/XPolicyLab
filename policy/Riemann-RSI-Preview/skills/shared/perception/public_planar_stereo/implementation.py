"""Metric plane registration from public RGB, own EE poses and sensor calibration."""
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


class RGBGeometryError(Exception):
    pass


def _transform(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _normalized(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high = grey - cv2.GaussianBlur(grey, (0, 0), 3)
    scale = np.sqrt(cv2.GaussianBlur(high * high, (0, 0), 3) + 1)
    return high / scale, scale


class PublicPlanarStereo:
    def __init__(self, calibration, robot_urdf):
        self.calibration = dict(calibration)
        joints = {j.find('child').get('link'): j
                  for j in ET.parse(Path(robot_urdf)).getroot().findall('joint')}
        chain, link = [], 'camera'
        while link != 'link6':
            if link not in joints or len(chain) > 8:
                raise ValueError('Declared own-robot camera mounting chain is unavailable')
            joint = joints[link]
            if joint.get('type') != 'fixed':
                raise ValueError('The camera mounting transform must be public and fixed')
            origin = joint.find('origin')
            chain.append(_transform([float(x) for x in origin.get('xyz', '0 0 0').split()],
                                    [float(x) for x in origin.get('rpy', '0 0 0').split()]))
            link = joint.find('parent').get('link')
        self.mount = np.eye(4)
        for transform in reversed(chain):
            self.mount = self.mount @ transform
        sensor = np.eye(4)
        sensor[:3, :3] = Rotation.from_euler('XYZ', calibration['wrist_sensor_euler_deg'],
                                           degrees=True).as_matrix()
        sensor[:3, 3] = calibration['wrist_sensor_origin_m']
        self.mount = self.mount @ sensor
        cv2.setNumThreads(1)

    def cameras(self, state):
        head = np.eye(4)
        head[:3, :3] = Rotation.from_euler('xyz', self.calibration['head_euler_deg'],
                                         degrees=True).as_matrix()
        head[:3, 3] = self.calibration['head_origin_m']
        cameras = [head]
        for arm in ('left', 'right'):
            pose = np.asarray(state[arm + '_ee_pose'], dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise RGBGeometryError('Only finite public own EE poses define the camera baseline')
            hand = np.eye(4)
            hand[:3, :3] = Rotation.from_quat(np.r_[pose[4:], pose[3]]).as_matrix()
            hand[:3, 3] = pose[:3]
            cameras.append(hand @ self.mount)
        return cameras

    def estimate(self, images, state, mask, *, lower=None, upper=None,
                 minimum_support=300, minimum_score=.28):
        """Return a measured-image plane hypothesis; never fall back to robot-base Z."""
        images = [np.asarray(images[name]) for name in CAMERAS]
        if any(image.shape != (480, 640, 3) or image.dtype != np.uint8 for image in images):
            raise RGBGeometryError('Planar stereo requires current640x480 uint8 RGB')
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (480, 640) or mask.sum() < minimum_support:
            raise RGBGeometryError('Insufficient public image support for a plane')
        cameras = self.cameras(state)
        highest = min(camera[2, 3] for camera in cameras) - .03
        low = highest - 1.0 if lower is None else float(lower)
        high = highest if upper is None else min(float(upper), highest)
        if not np.isfinite([low, high]).all() or not low < high or high-low > 2.0:
            raise RGBGeometryError('Invalid bounded public-camera plane search')
        projections = []
        for i, camera in enumerate(cameras):
            focal = self.calibration['head_focal_pixels' if i == 0 else 'wrist_focal_pixels']
            # Camera projection uses continuous sensor coordinates; OpenCV
            # addresses raster samples by integer indices. The centre of a
            # rendered pixel is (index+.5), as in the existing face warper.
            intrinsic = np.array([[focal, 0, 319.5], [0, focal, 239.5], [0, 0, 1.]])
            projections.append(intrinsic @ np.diag([1., -1., -1.]) @ np.linalg.inv(camera)[:3])
        target, texture = _normalized(images[0])
        def score(height):
            first = projections[0]
            homography = np.c_[first[:, :2], first[:, 2]*height+first[:, 3]]
            inverse = np.linalg.inv(homography)
            estimates = []
            for view in (1, 2):
                second = projections[view]
                transfer = np.c_[second[:, :2], second[:, 2]*height+second[:, 3]] @ inverse
                reverse = np.linalg.inv(transfer)
                warp = cv2.warpPerspective(images[view], reverse, (640, 480))
                visible = cv2.warpPerspective(np.ones((480,640), np.uint8), reverse,
                    (640,480), flags=cv2.INTER_NEAREST).astype(bool)
                pattern, scale = _normalized(warp)
                supported = visible & mask & (texture > 1.8) & (scale > 1.8)
                count = int(supported.sum())
                if count < minimum_support:
                    continue
                numerator = cv2.GaussianBlur(target*pattern, (0,0), 2)
                a = cv2.GaussianBlur(target*target, (0,0), 2)
                b = cv2.GaussianBlur(pattern*pattern, (0,0), 2)
                correlation = numerator / np.sqrt(np.maximum(a*b, 1e-6))
                estimates.append((float(np.mean(correlation[supported])), count, view))
            return max(estimates, default=(-1.,0,0))
        coarse = [(float(height), score(height)) for height in
                  np.linspace(low, high, min(512, max(2, int(np.ceil((high-low)/.004))+1)))]
        ranked = sorted(coarse, key=lambda item:item[1][0], reverse=True)
        peaks = [ranked[0]]
        alternatives = [item for item in ranked[1:] if abs(item[0]-peaks[0][0]) > .014]
        if alternatives:
            peaks.append(alternatives[0])
        refined = []
        for height, _ in peaks:
            refined.extend((float(z), score(z)) for z in
                           np.linspace(max(low,height-.006), min(high,height+.006), 25))
        refined.sort(key=lambda item:item[1][0], reverse=True)
        height, match = refined[0]
        rivals = [item[1][0] for item in coarse+refined if abs(item[0]-height) > .014]
        gap = float(match[0]-max(rivals, default=-1.))
        if match[0] < minimum_score or gap < .045:
            raise RGBGeometryError('Public RGB has no unique qualified metric plane peak')
        return {'height_m':height, 'score':match[0], 'support_pixels':match[1],
                'view':CAMERAS[match[2]], 'competing_height_margin':gap,
                'rgb_only':True, 'robot_base_height_used':False,
                'horizontal_surface_hypothesis':True}
