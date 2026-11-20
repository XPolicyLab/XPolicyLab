import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix, matrix_pose


TCP_OFFSET = .145


class OwnToolModel:
    def __init__(self, source):
        source = Path(source)
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 2097152:
            raise ValueError('Own kinematics parse only the contained declared robot URDF')
        document = ElementTree.parse(source).getroot()
        by_name = {joint.attrib['name']: joint for joint in document.findall('joint')}
        self.origins, self.axes, lower, upper = [], [], [], []
        for index in range(1, 7):
            joint = by_name['joint' + str(index)]
            origin, transform = joint.find('origin'), np.eye(4)
            transform[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            self.origins.append(transform)
            self.axes.append(np.fromstring(joint.find('axis').attrib['xyz'], sep=' '))
            limits = joint.find('limit')
            lower.append(float(limits.attrib['lower']))
            upper.append(float(limits.attrib['upper']))
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.offset = np.eye(4)
        self.offset[0, 3] = TCP_OFFSET
        self.queries = 0

    def eef(self, joints):
        values = np.asarray(joints, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('Own FK takes exactly six finite public robot joints')
        transform = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, values):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ rotation
        return transform

    def root(self, joints, measured_ee):
        return pose_matrix(measured_ee) @ np.linalg.inv(self.eef(joints))

    def tcp(self, joints, root):
        return root @ self.eef(joints) @ self.offset

    def public_pose(self, tcp):
        return matrix_pose(np.asarray(tcp) @ np.linalg.inv(self.offset))

    def solve(self, target_tcp, seed, root, retries=0, tolerance=.005, rotation_tolerance=.20):
        target_tcp, seed, root = np.asarray(target_tcp), np.asarray(seed), np.asarray(root)
        if (target_tcp.shape != (4, 4) or root.shape != (4, 4) or seed.shape != (6,)
                or not np.isfinite(target_tcp).all() or not np.isfinite(root).all() or not np.isfinite(seed).all()
                or type(retries) is not int or not 0 <= retries <= 2 or not .002 <= tolerance <= .008
                or not .02 <= rotation_tolerance <= .20):
            raise ValueError('Own IK accepts only its declared finite robot targets and original local retry settings')
        if self.queries >= 512:
            return None
        self.queries += 1
        local = np.linalg.inv(root) @ target_tcp @ np.linalg.inv(self.offset)
        lower, upper = np.maximum(self.lower, seed - math.pi), np.minimum(self.upper, seed + math.pi)

        def residual(joints):
            actual = self.eef(joints)
            rotation = Rotation.from_matrix(local[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[actual[:3, 3] - local[:3, 3], .16 * rotation]

        generator = np.random.default_rng(9017)
        seeds = [seed]
        if retries:
            for bend in ((2.2, 1.8, -.9), (-2.2, -1.8, .9)):
                start = seed.copy()
                start[1:4] = bend
                seeds.append(start)
        seeds += [seed + generator.normal(0, .65, 6) for unused_index in range(retries)]
        best = None
        for start in seeds:
            fit = least_squares(residual, np.clip(start, lower + 1e-6, upper - 1e-6), bounds=(lower, upper),
                                max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            actual = self.tcp(fit.x, root)
            distance = float(np.linalg.norm(actual[:3, 3] - target_tcp[:3, 3]))
            angle = float(np.linalg.norm(Rotation.from_matrix(target_tcp[:3, :3] @ actual[:3, :3].T).as_rotvec()))
            if distance < tolerance and angle < rotation_tolerance:
                cost = float(np.linalg.norm(fit.x - seed))
                if best is None or cost < best[0]:
                    best = cost, fit.x
                if cost < 1.5:
                    break
        return None if best is None else best[1]


def run():
    root = Path(__file__).resolve().parents[4]
    settings = json.loads((root / 'configs/mature-trajectory.json').read_text())
    source = root / settings['robot_urdf']
    if hashlib.sha256(source.read_bytes()).hexdigest() != settings['robot_urdf_sha256']:
        raise ValueError('The declared own robot model changed')
    return OwnToolModel(source)
