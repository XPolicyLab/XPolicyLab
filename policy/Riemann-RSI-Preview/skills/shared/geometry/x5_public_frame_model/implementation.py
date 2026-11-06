"""Original fixed public X5 geometry, camera projection and six-joint calibration."""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ARMS = ('left', 'right')
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
FOCAL_PX = 640.0 * 10.0 / 22.212
CAMERA_CENTER = np.array([0.0, -0.41, 1.308])
CAMERA_ROTATION = Rotation.from_euler('x', 30, degrees=True).as_matrix()

def rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Expected a finite WXYZ quaternion')
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def quaternion(r):
    return Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]


def matrix(pose):
    p = np.asarray(pose, float)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('Expected a finite XYZ WXYZ pose')
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation(p[3:]), p[:3]
    return result


def pose(t):
    return np.r_[t[:3, 3], quaternion(t[:3, :3])]


def project(points):
    xyz = (np.asarray(points, float) - CAMERA_CENTER) @ CAMERA_ROTATION
    distance = -xyz[..., 2]
    if np.any(distance <= 1e-5):
        raise ValueError('Fixed head camera projection is behind camera')
    return np.stack([320 + FOCAL_PX * xyz[..., 0] / distance,
                     240 - FOCAL_PX * xyz[..., 1] / distance], axis=-1)


def backproject(uv, plane_point, plane_normal=(0, 0, 1)):
    uv = np.asarray(uv, float).reshape(-1, 2)
    ray = np.c_[(uv[:, 0] - 320) / FOCAL_PX,
                -(uv[:, 1] - 240) / FOCAL_PX, -np.ones(len(uv))]
    ray = ray @ CAMERA_ROTATION.T
    normal = np.asarray(plane_normal, float)
    denominator = ray @ normal
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError('Camera ray is parallel to recovered mounting plane')
    scale = ((np.asarray(plane_point) - CAMERA_CENTER) @ normal) / denominator
    if np.any(scale <= 0):
        raise ValueError('Recovered plane is behind current camera rays')
    return CAMERA_CENTER + scale[:, None] * ray


class X5Kinematics:
    """Parse the six public URDF joints, keeping no simulator/robot handles."""
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        by_name = {j.attrib['name']: j for j in root.findall('joint')}
        self.joints = []
        for index in range(1, 7):
            joint = by_name[f'joint{index}']; origin = joint.find('origin')
            t = np.eye(4)
            t[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
            limits = joint.find('limit')
            self.joints.append((t, axis, max(-3.14159, float(limits.attrib['lower'])),
                                min(3.14159, float(limits.attrib['upper']))))
        self.low = np.array([j[2] for j in self.joints])
        self.high = np.array([j[3] for j in self.joints])

    def fk(self, joints, *, all_links=False):
        q = np.asarray(joints, float).reshape(6)
        result = np.eye(4); links = {'base_link': result.copy()}
        for index, ((origin, axis, _, _), angle) in enumerate(zip(self.joints, q), 1):
            r = np.eye(4); r[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            result = result @ origin @ r; links[f'link{index}'] = result.copy()
        return links if all_links else result

    def recover(self, state):
        transforms = {arm: matrix(state[arm + '_ee_pose']) @ np.linalg.inv(self.fk(state[arm + '_arm_joint_state'])) for arm in ARMS}
        heights = [t[2, 3] for t in transforms.values()]
        normals = [t[:3, 2] for t in transforms.values()]
        if np.ptp(heights) > .005 or any(np.dot(n, [0, 0, 1]) < .999 for n in normals):
            raise ValueError('Independent robot-state frame recovery disagrees or mounting plane is tilted')
        return RobotFrame(self, transforms, float(np.mean(heights)))


class RobotFrame:
    def __init__(self, kinematics, bases, mounting_z):
        self.kinematics, self.bases = kinematics, bases
        self.mounting_z = mounting_z
        self.normal = np.array([0., 0., 1.])

    def point_on_plane(self, relative_z=0):
        return np.array([0., 0., self.mounting_z + float(relative_z)])

    def pixels(self, uv, relative_z=0):
        return backproject(uv, self.point_on_plane(relative_z), self.normal)

    def root_pose(self, arm):
        return pose(self.bases[arm])

    def ik(self, arm, target_pose, current, *, axis_only=False, max_axis_deg=8):
        kin = self.kinematics
        target = np.linalg.inv(self.bases[arm]) @ matrix(target_pose)
        desired_axis = target[:3, 0]
        def residual(q):
            actual = kin.fk(q)
            dp = actual[:3, 3] - target[:3, 3]
            if axis_only:
                delta = actual[:3, 0] - desired_axis
                cone = max(0., np.cos(np.deg2rad(max_axis_deg)) - actual[:3, 0] @ desired_axis)
                return np.r_[dp, .02 * delta, 3 * cone, .001 * (q - current)]
            error = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
            return np.r_[dp, .12 * error]
        # Keep the same forward elbow branch along low table carries.
        # The alternative negative-elbow root can have a small wrist angle
        # while requiring a long branch switch between adjacent waypoints.
        seeds = [np.asarray(current), np.array([0., 2., 1.5, -1., 0., 0.]),
                 np.array([0., 1., -1., 0., 1., 0.]),
                 np.array([0., -.8, .8, 0., -.8, 0.]), np.zeros(6)]
        results = []
        for seed in seeds:
            opt = least_squares(residual, np.clip(seed, kin.low + 1e-6, kin.high - 1e-6),
                                bounds=(kin.low, kin.high), max_nfev=110)
            actual = kin.fk(opt.x)
            pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            re = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
            ae = float(np.degrees(np.arccos(np.clip(actual[:3, 0] @ desired_axis, -1, 1))))
            world = self.bases[arm] @ actual
            record = dict(joints=opt.x, target_pose=pose(world), position_error_m=pe,
                          rotation_error_rad=re, axis_error_deg=ae)
            results.append(record)
            if pe < .0025 and (ae < max_axis_deg + .1 if axis_only else re < .03):
                record['valid']=True
                if opt.x[2]>=0:return record
        valid=[record for record in results if record.get('valid')]
        return valid[0] if valid else None


__all__ = ('rotation', 'quaternion', 'matrix', 'pose', 'project', 'backproject', 'X5Kinematics', 'RobotFrame', 'ARMS', 'CAMERAS', 'FOCAL_PX', 'CAMERA_CENTER', 'CAMERA_ROTATION')
