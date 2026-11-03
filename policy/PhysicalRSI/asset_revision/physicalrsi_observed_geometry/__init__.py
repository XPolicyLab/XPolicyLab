"""Metric geometry estimated from RGB and the current camera calibration.

No benchmark meshes, object dimensions, support heights or simulator depth are
read. Depth is a learned estimate, not ground truth; visible bounds do not prove
complete reconstruction of hidden surfaces.
"""
from functools import lru_cache
from itertools import product
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class GeometryUnavailable(ValueError):
    pass


class Camera:
    """Camera-to-world transform with USD axes (+X right, +Y up, -Z forward)."""
    def __init__(self, image, intrinsic, camera_to_world):
        self.image = np.asarray(image)
        self.k = np.asarray(intrinsic, dtype=float)
        self.transform = np.asarray(camera_to_world, dtype=float)
        if (self.image.dtype != np.uint8 or self.image.ndim != 3
                or self.image.shape[2] != 3 or self.k.shape != (3, 3)
                or self.transform.shape != (4, 4)
                or not np.isfinite(self.k).all() or not np.isfinite(self.transform).all()
                or self.k[0, 0] <= 0 or self.k[1, 1] <= 0
                or not np.allclose(self.k[2], [0, 0, 1])
                or not np.allclose(self.transform[3], [0, 0, 0, 1])):
            raise GeometryUnavailable('Current uint8 RGB and calibrated K/T are required')
        rotation = self.transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise GeometryUnavailable('Camera transform must be a proper rigid transform')
        self.height, self.width = self.image.shape[:2]

    @classmethod
    def from_observation(cls, observation, name='cam_head'):
        try:
            entry = observation['vision'][name]
            return cls(entry['color'], entry['intrinsic_matrix'], entry['extrinsic_matrix'])
        except KeyError as exc:
            raise GeometryUnavailable('Observation is missing RGB camera calibration: ' + name) from exc

    def unproject(self, pixels, axial_depth):
        uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
        depth = np.asarray(axial_depth, dtype=float).reshape(-1)
        if (len(depth) != len(uv) or not np.isfinite(uv).all() or not np.isfinite(depth).all()
                or np.any(depth <= 0) or np.any(uv < 0)
                or np.any(uv[:, 0] >= self.width) or np.any(uv[:, 1] >= self.height)):
            raise GeometryUnavailable('Finite in-image pixels and positive metric depth required')
        rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(self.k).T
        rays[:, 1:] *= -1
        return (rays * depth[:, None]) @ self.transform[:3, :3].T + self.transform[:3, 3]

    def project(self, points):
        camera = (np.asarray(points, dtype=float).reshape(-1, 3) - self.transform[:3, 3]) @ self.transform[:3, :3]
        depth = -camera[:, 2]
        if np.any(depth <= 0):
            raise GeometryUnavailable('Point behind camera')
        optical = camera * [1, -1, -1]
        homogeneous = optical @ self.k.T
        return homogeneous[:, :2] / depth[:, None], depth


class _MetricDepth:
    def __init__(self, model_path, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device, self.lock = torch, device, Lock()
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True).to(device).eval()
        if getattr(self.model.config, 'depth_estimation_type', None) != 'metric':
            raise GeometryUnavailable('A metric depth checkpoint is required; relative depth cannot set scale')

    def __call__(self, rgb):
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
            raw = self.model(**inputs).predicted_depth
            return self.torch.nn.functional.interpolate(raw[:, None], size=rgb.shape[:2], mode='bilinear', align_corners=False)[0, 0].float().cpu().numpy()


@lru_cache(maxsize=2)
def _depth_model(path, device):
    return _MetricDepth(path, device)


def metric_depth(rgb):
    path = os.environ.get('PHYSICALRSI_METRIC_DEPTH_MODEL')
    if not path or not Path(path).is_dir():
        raise GeometryUnavailable('Set PHYSICALRSI_METRIC_DEPTH_MODEL to the installed metric depth model')
    return _depth_model(str(Path(path).resolve()), os.environ.get('PHYSICALRSI_METRIC_DEPTH_DEVICE', 'cpu'))(rgb)


class Scene:
    def __init__(self, camera, *, predictor=None):
        self.camera = camera
        self.depth = np.asarray((predictor or metric_depth)(camera.image), dtype=float)
        if (self.depth.shape != camera.image.shape[:2] or not np.isfinite(self.depth).all()
                or np.any(self.depth <= 0)):
            raise GeometryUnavailable('Metric depth model returned invalid axial depth')

    @classmethod
    def from_observation(cls, observation, name='cam_head', *, predictor=None):
        scene = cls(Camera.from_observation(observation, name), predictor=predictor)
        scene._other_cameras = [Camera.from_observation(observation, key)
                                for key in observation['vision'] if key != name]
        scene._raw_depth = scene.depth.copy()
        scene._support_cache = None
        return scene

    def mask(self, value):
        mask = np.asarray(value, dtype=bool)
        if mask.shape != self.depth.shape:
            mask = cv2.resize(mask.astype('uint8'), (self.camera.width, self.camera.height), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mask.sum() < 8:
            raise GeometryUnavailable('Insufficient visible object pixels')
        return mask

    def surface(self, mask, *, maximum=8192):
        mask = self.mask(mask)
        yy, xx = np.nonzero(mask)
        selected = np.linspace(0, len(xx)-1, min(len(xx), maximum), dtype=int)
        yy, xx = yy[selected], xx[selected]
        return self.camera.unproject(np.c_[xx, yy], self.depth[yy, xx])

    def point(self, pixel, radius=2):
        uv = np.asarray(pixel, dtype=float)
        x, y = np.rint(uv).astype(int)
        if not (0 <= x < self.camera.width and 0 <= y < self.camera.height):
            raise GeometryUnavailable('Target pixel outside current camera')
        patch = self.depth[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1]
        return self.camera.unproject([uv], [np.median(patch)])[0]

    def support_plane(self, mask):
        if len(getattr(self, '_other_cameras', [])) >= 2:
            from .rgb_support import initial_plane, refine_plane
            from .observed_depth_alignment import ObservedPlaneDepthAlignment
            import time
            mask = self.mask(mask)
            cached = self._support_cache
            if cached is not None and np.array_equal(cached[0], mask):
                return cached[1]
            cameras = [self.camera, *self._other_cameras]
            masks = [mask] + [np.ones(c.image.shape[:2], bool) for c in self._other_cameras]
            coefficients = refine_plane(cameras, masks, initial_plane(cameras, masks))
            alignment = ObservedPlaneDepthAlignment(camera=self.camera, plane=coefficients, target='axial')
            try:
                self.depth, report = alignment.align(self._raw_depth, mask, deadline=time.monotonic()+30)
            except ValueError as error:
                raise GeometryUnavailable(str(error)) from error
            a, b, c = coefficients
            normal = np.array([-a, -b, 1.]); length = np.linalg.norm(normal)
            result = {'normal': normal/length, 'offset': float(-c/length),
                      'residual_m': report['holdout']['median_residual_m'],
                      'source': 'current_calibrated_rgb_support_plane',
                      'depth_alignment': report, 'metric_accuracy_certified': False}
            self._support_cache = (mask.copy(), result)
            return result
        points = self.surface(mask, maximum=2048)
        rng = np.random.default_rng(0)
        best = None
        for _ in range(128):
            a, b, c = points[rng.choice(len(points), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            if normal[2] < 0:
                normal = -normal
            # A supported-object capability, not a fixed world height.
            if normal[2] < .9:
                continue
            offset = -normal @ a
            residual = np.abs(points @ normal + offset)
            inliers = residual < .01
            if best is None or inliers.sum() > best[0].sum():
                best = inliers, normal, offset
        if best is None or best[0].mean() < .6 or best[0].sum() < 20:
            raise GeometryUnavailable('No reliable observed support plane')
        selected = points[best[0]]
        center = selected.mean(0)
        _, _, axes = np.linalg.svd(selected-center, full_matrices=False)
        normal = axes[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -normal @ center
        return {'normal': normal, 'offset': float(offset),
                'residual_m': float(np.median(np.abs(selected @ normal + offset))),
                'source': 'current_rgb_metric_depth'}

    def geometry(self, mask, *, support_mask=None):
        # Calibration must precede unprojection of the object's pixels.
        plane = self.support_plane(support_mask) if support_mask is not None else None
        points = self.surface(mask)
        if plane is not None:
            normal, offset = plane['normal'], plane['offset']
            heights = points @ normal + offset
            foreground = points[heights > max(.002, 3*plane['residual_m'])]
            if len(foreground) < 8:
                raise GeometryUnavailable('Object cannot be separated from observed support')
            # Extend only to the measured support, never a benchmark table height.
            points = np.r_[foreground, foreground - (foreground @ normal + offset)[:, None]*normal]
        rect = cv2.minAreaRect(points[:, :2].astype('float32'))
        yaw = np.deg2rad(rect[2])
        rotation = Rotation.from_euler('z', yaw).as_matrix()
        local = points @ rotation
        low, high = np.quantile(local, [.01, .99], axis=0)
        if np.any(high-low < 1e-5):
            raise GeometryUnavailable('Observed shape is degenerate; hidden thickness is unknown')
        center = ((low+high)/2) @ rotation.T
        dims = high-low
        quat = Rotation.from_matrix(rotation).as_quat()
        return {'pose': [*center.tolist(), *quat[[3,0,1,2]].tolist()],
                'bbox_local': (np.asarray(list(product([-.5,.5], repeat=3)))*dims).tolist(),
                'dims': dims.tolist(), 'points_local': (points-center) @ rotation,
                'source': ('current_rgb_depth_aligned_to_observed_support' if getattr(self, '_support_cache', None) is not None else 'current_rgb_metric_depth'), 'hidden_surfaces_reconstructed': False, 'extended_to_observed_support': support_mask is not None}
