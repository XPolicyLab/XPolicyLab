"""RGB head calibration and explicitly selected wrist model; no simulator queries."""
from contextvars import ContextVar
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation

_CURRENT = ContextVar('physicalrsi_rgb_cameras')

def pose_matrix(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape == (4, 4):
        return p.copy()
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('Expected finite xyz+wxyz robot pose')
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    out[:3, 3] = p[:3]
    return out

def validated_projection(record, name):
    if not isinstance(record, dict):
        raise ValueError(f'Missing RGB calibration for {name}')
    quality = record.get('quality', {})
    if not (record.get('accepted') is True or quality.get('accepted') is True):
        raise ValueError(f'RGB calibration not accepted: {name}')
    p = np.asarray(record.get('projection'), dtype=float)
    if p.shape != (3, 4) or not np.isfinite(p).all() or np.linalg.matrix_rank(p) != 3:
        raise ValueError(f'Invalid 3x4 RGB projection for {name}')
    if abs(np.linalg.det(p[:, :3])) < 1e-12:
        raise ValueError(f'RGB projection has no finite pinhole center: {name}')
    return p.copy()

def project(p, points):
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    uvw = np.column_stack((xyz, np.ones(len(xyz)))) @ p.T
    if np.any(np.abs(uvw[:, 2]) < 1e-10):
        raise ValueError('RGB projection ray is at infinity')
    return uvw[:, :2] / uvw[:, 2:3]

def backproject_plane(p, pixels, world_z):
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    # General finite projective camera; rays need not use guessed focal/mount.
    center = -np.linalg.solve(p[:, :3], p[:, 3])
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(p[:, :3]).T
    if np.any(np.abs(rays[:, 2]) < 1e-10):
        raise ValueError('RGB ray cannot intersect the task support plane')
    distance = (float(world_z) - center[2]) / rays[:, 2]
    return center + distance[:, None] * rays


def upstream_static_wrist_projection():
    """Algebraic P_link6 from the archived upstream grasp_rgb.camera/project.

    This is a public static prior, NOT an estimated/accepted RGB calibration.
    No environment metadata or runtime camera matrix is read. Both arms use the
    upstream's common link6-relative camera model; its domain error is unknown.
    """
    a=Rotation.from_euler('xyz',[0,0,-3.1416])
    b=Rotation.from_euler('xyz',[0,.3491,np.pi])
    offset=np.array([.057,0,0])+a.apply([-.0275,0,.05])+(a*b).apply([0,0,.001])
    orientation=a*b*Rotation.from_euler('XYZ',[0,-80,-90],degrees=True)
    inverse=orientation.as_matrix().T
    # Original coordinates look along -Z and image y is downward.
    intrinsic=np.array([[397.04127893,0.,-320.],
                        [0.,-397.04127893,-240.],[0.,0.,-1.]])
    return intrinsic @ np.column_stack([inverse,-inverse@offset])

class CalibratedCameras:
    def __init__(self, options):
        head = options.get('rgb_calibration', {})
        if head.get('source') != 'rgb_fingertip_fk':
            raise ValueError('Head projection must come from RGB fingertip/FK calibration')
        self.head = validated_projection(head, 'head')
        model = options.get('wrist_camera_model', {})
        mode = model.get('mode', 'rgb_calibrated')
        if mode == 'upstream_static_prior':
            if options.get('rgb_wrist_calibration'):
                raise ValueError('Choose explicit static wrist prior OR RGB wrist calibration, not both')
            projection = upstream_static_wrist_projection()
            self.wrists = {arm: projection.copy() for arm in ('left', 'right')}
            self.sources = {'head': head.get('source'),
                            'wrists': 'upstream_public_static_wrist_prior',
                            'wrist_model_mode': mode,
                            'wrist_rgb_estimated': False,
                            'wrist_rgb_validated': False,
                            'wrist_source': 'rdj_swap_t/14d69d4d118fb34d/grasp_rgb.py:16-23',
                            'wrist_source_sha256': 'f728b57b882b1dfe64a23799b864ae5543840c256113944e18973fd41976a976'}
        elif mode == 'rgb_calibrated':
            wrist = options.get('rgb_wrist_calibration', {})
            if wrist.get('source') not in ('rgb_head_pose_fk', 'rgb_fingertip_fk', 'rgb_multiview_fk'):
                raise ValueError('Wrist projection needs accepted RGB/FK calibration or explicit upstream_static_prior mode')
            self.wrists = {arm: validated_projection(wrist.get(arm), arm + '_wrist')
                           for arm in ('left', 'right')}
            self.sources = {'head': head.get('source'), 'wrists': wrist.get('source'),
                            'wrist_model_mode': mode,'wrist_rgb_estimated': True,
                            'wrist_rgb_validated': True}
        else:
            raise ValueError('Unknown wrist_camera_model mode: '+str(mode))
        self.arm = 'left'
    def head_project(self, points):
        return project(self.head, points)
    def head_on_plane(self, pixels, z):
        return backproject_plane(self.head, pixels, z)
    def wrist_world_projection(self, ee):
        return self.wrists[self.arm] @ np.linalg.inv(pose_matrix(ee))
    def wrist_project(self, points, ee):
        return project(self.wrist_world_projection(ee), points)
    def wrist_on_plane(self, pixels, ee, z):
        return backproject_plane(self.wrist_world_projection(ee), pixels, z)

@contextmanager
def use_cameras(cameras):
    token = _CURRENT.set(cameras)
    try:
        yield cameras
    finally:
        _CURRENT.reset(token)

def active_cameras():
    return _CURRENT.get()
