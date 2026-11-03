"""Robot-only URDF kinematics and inter-arm sphere checks; no scene object access."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import json
from scipy.spatial.transform import Rotation


class RobotGeometry:
    def __init__(self):
        root = Path(__file__).with_name('calibration')
        tree = ET.parse(root/'X5A.urdf').getroot()
        self.joints = []
        for j in tree.findall('joint'):
            origin = j.find('origin')
            T = np.eye(4)
            if origin is not None:
                T[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = j.find('axis')
            self.joints.append((j.get('name'), j.get('type'), j.find('parent').get('link'),
                j.find('child').get('link'), T,
                np.fromstring(axis.get('xyz') if axis is not None else '0 0 1', sep=' ')))
        self.spheres = json.loads((root/'collision_spheres.json').read_text())

    def frames(self, q, finger=.044):
        values = {f'joint{i+1}': float(x) for i,x in enumerate(q)}
        values.update(joint7=finger, joint8=finger)
        frames = {'base_link': np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for item in list(pending):
                name, kind, parent, child, origin, axis = item
                if parent not in frames:
                    continue
                motion = np.eye(4)
                if kind in ('revolute', 'continuous'):
                    motion[:3,:3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
                elif kind == 'prismatic':
                    motion[:3,3] = axis*values[name]
                frames[child] = frames[parent] @ origin @ motion
                pending.remove(item); progressed = True
            if not progressed:
                raise RuntimeError('Unconnected robot URDF')
        return frames

    def world_spheres(self, q, origin, finger=.044, frame_bias=(0,0,0)):
        frames = self.frames(q, finger)
        base = Rotation.from_quat(origin[3:], scalar_first=True).as_matrix()
        centers, radii, names = [], [], []
        for link, spheres in self.spheres.items():
            T = frames[link]
            for sphere in spheres:
                local = T[:3,:3] @ sphere['center'] + T[:3,3] - frame_bias
                centers.append(origin[:3]+base@local)
                radii.append(sphere['radius']); names.append(link)
        return np.asarray(centers), np.asarray(radii), names


def pair_clearance(left, right, details=False):
    lc, lr, ln = left; rc, rr, rn = right
    distances = np.linalg.norm(lc[:,None,:]-rc[None,:,:], axis=-1)-lr[:,None]-rr[None,:]
    i,j = np.unravel_index(np.argmin(distances), distances.shape)
    result = float(distances[i,j]), (ln[i], rn[j])
    if not details:
        return result
    delta = rc[j]-lc[i]
    norm = np.linalg.norm(delta)
    direction = delta/norm if norm > 1e-12 else np.array([1.,0.,0.])
    return result + ({'left_point_m': (lc[i]+lr[i]*direction).tolist(),
                       'right_point_m': (rc[j]-rr[j]*direction).tolist()},)
