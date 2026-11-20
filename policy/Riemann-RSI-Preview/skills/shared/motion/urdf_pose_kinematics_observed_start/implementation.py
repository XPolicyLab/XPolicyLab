import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix

def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('A numerical pose contains only finite own-robot XYZ/WXYZ values')
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = (rotation_matrix(pose[3:]), pose[:3])
    return matrix

def matrix_pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quaternion[3], quaternion[:3]]

class OwnURDFKinematics:

    def __init__(self, urdf):
        source = Path(urdf)
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 2097152:
            raise ValueError('Parse only the contained declared public robot URDF')
        document = ElementTree.parse(source).getroot()
        by_name = {joint.attrib['name']: joint for joint in document.findall('joint')}
        self.joints = []
        for index in range(1, 7):
            joint = by_name['joint' + str(index)]
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
            limits = joint.find('limit')
            lower, upper = (max(-3.14159, float(limits.attrib['lower'])), min(3.14159, float(limits.attrib['upper'])))
            if not np.isfinite(transform).all() or axis.shape != (3,) or (not np.isfinite(axis).all()) or (lower >= upper):
                raise ValueError('The declared six-joint mechanism is incomplete')
            self.joints.append((transform, axis, lower, upper))
        self.lower_limits = np.asarray([joint[2] for joint in self.joints])
        self.upper_limits = np.asarray([joint[3] for joint in self.joints])

    def forward(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('Forward kinematics accepts only six finite own joints')
        result = np.eye(4)
        for (origin, axis, unused_lower, unused_upper), angle in zip(self.joints, joints):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            result = result @ origin @ rotation
        return result

    def observed_base(self, joints, observed_pose):
        return pose_matrix(observed_pose) @ np.linalg.inv(self.forward(joints))

    def solve(self, base, target_pose, current):
        current = np.asarray(current, dtype=float)
        base = np.asarray(base, dtype=float)
        if current.shape != (6,) or base.shape != (4, 4) or (not np.isfinite(current).all()) or (not np.isfinite(base).all()):
            raise ValueError('Solve only a current public own-robot frame')
        target = np.linalg.inv(base) @ pose_matrix(target_pose)

        def residual(joints):
            actual = self.forward(joints)
            displacement = actual[:3, 3] - target[:3, 3]
            error = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
            return np.r_[displacement, 0.12 * error]
        seeds = [current, np.asarray([0.0, 2.0, 1.5, -1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, -1.0, 0.0, 1.0, 0.0]), np.asarray([0.0, -0.8, 0.8, 0.0, -0.8, 0.0]), np.zeros(6)]
        accepted = []
        for seed in seeds:
            result = least_squares(residual, np.clip(seed, self.lower_limits + 1e-06, self.upper_limits - 1e-06), bounds=(self.lower_limits, self.upper_limits), max_nfev=110)
            actual = self.forward(result.x)
            distance = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            angle = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
            if distance < 0.0025 and angle < 0.03:
                record = {'joints': result.x.copy(), 'position_error_m': distance, 'rotation_error_radians': angle, 'public_target_pose': matrix_pose(base @ actual)}
                accepted.append(record)
                if result.x[2] >= 0:
                    return record
        return accepted[0] if accepted else None

    def cartesian(self, base, current, target, validate_path):
        if not callable(validate_path):
            raise ValueError('Cartesian continuity requires the native numerical collision validator')
        current = np.asarray(current, dtype=float)
        start = matrix_pose(np.asarray(base) @ self.forward(current))
        target = np.asarray(target, dtype=float)
        distance = float(np.linalg.norm(target[:3] - start[:3]))
        angle = float(Rotation.from_matrix(rotation_matrix(start[3:]).T @ rotation_matrix(target[3:])).magnitude())
        count = max(3, int(math.ceil(distance / 0.008)), int(math.ceil(angle / 0.055)))
        if count > 250:
            return None
        interpolation = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([rotation_matrix(start[3:]), rotation_matrix(target[3:])])))
        joints, path = (current.copy(), [current.copy()])
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            quaternion = interpolation(fraction).as_quat()
            waypoint = np.r_[start[:3] + fraction * (target[:3] - start[:3]), quaternion[3], quaternion[:3]]
            solution = self.solve(base, waypoint, joints)
            if solution is None or np.max(np.abs(solution['joints'] - joints)) > 0.4:
                return None
            joints = solution['joints']
            path.append(joints.copy())
        path = np.stack(path)
        return path if validate_path(path) else None
