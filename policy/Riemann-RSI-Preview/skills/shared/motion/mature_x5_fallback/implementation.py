import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from harness.runtime.observation import public_observation
from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix


OPERATIONS = ('x5_pose', 'x5_place', 'x5_place_calibrated', 'x5_retreat')


def principal(values):
    return (np.asarray(values, dtype=np.float64) + math.pi) % (2 * math.pi) - math.pi


def pose_matrix(pose):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


class OriginalX5Fallback:
    def __init__(self, urdf):
        from skills.shared.motion.mature_x5_fallback.kinematics import X5MeasuredFallbackKinematics
        self.kinematics = X5MeasuredFallbackKinematics(urdf)

    def path(self, payload, interval):
        from skills.shared.motion.mature_x5_fallback.kinematics import EEF_FROM_TCP
        observation = public_observation(payload['observation'])
        side, operation = payload['side'], payload['operation']
        joints = np.asarray(observation['state'][side + '_arm_joint_state'], dtype=np.float64).reshape(6)
        observed = np.asarray(observation['state'][side + '_ee_pose'], dtype=np.float64).reshape(7)
        pose = np.asarray(payload['pose'], dtype=np.float64).copy()
        if operation == 'x5_retreat':
            pose[:2], pose[3:] = observed[:2], observed[3:]
        if operation == 'x5_place_calibrated':
            pose[:2] += np.asarray([0.012, 0.010])
        root = pose_matrix(observed) @ np.linalg.inv(self.kinematics.eef(joints))
        target_tcp = np.linalg.inv(root) @ pose_matrix(pose) @ EEF_FROM_TCP
        start_tcp = self.kinematics.tcp(joints)
        waypoints = [target_tcp]
        if operation in ('x5_place', 'x5_place_calibrated'):
            high = target_tcp.copy()
            high[2, 3] = float(start_tcp[2, 3])
            waypoints = [high, target_tcp]
        knots, seed, start = [], joints.copy(), start_tcp.copy()
        for destination in waypoints:
            distance = float(np.linalg.norm(destination[:3, 3] - start[:3, 3]))
            angle = float(np.linalg.norm(Rotation.from_matrix(destination[:3, :3] @ start[:3, :3].T).as_rotvec()))
            count = max(3, int(math.ceil(distance / 0.010)), int(math.ceil(angle / 0.070)))
            rotations = Slerp([0.0, 1.0], Rotation.from_matrix([start[:3, :3], destination[:3, :3]]))
            for index in range(1, count + 1):
                fraction = index / count
                blend = 10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5
                target = np.eye(4, dtype=np.float64)
                target[:3, 3] = (1 - blend) * start[:3, 3] + blend * destination[:3, 3]
                target[:3, :3] = rotations([blend]).as_matrix()[0]
                strict = operation in ('x5_place', 'x5_place_calibrated', 'x5_retreat')
                seed = self.kinematics.solve_tcp(target, seed,
                    position_tolerance=0.004 if strict else 0.030,
                    rotation_tolerance=math.radians(45 if strict else 15),
                    branch_jump_tolerance=0.50, rotation_weight=0.02 if strict else 0.1)
                knots.append(seed.copy())
            start = destination.copy()
        upsample = max(1, int(math.ceil(interval * 1.8)))
        previous = joints.copy()
        previous[5] = float(principal(previous[5]))
        controls = []
        for goal in knots:
            destination = np.asarray(goal, dtype=np.float64).copy()
            destination[5] = float(principal(destination[5]))
            delta = destination - previous
            delta[5] = float(principal(delta[5]))
            for index in range(1, upsample + 1):
                command = previous + index / upsample * delta
                command[5] = float(principal(command[5]))
                controls.append(command.astype(np.float32))
            previous = destination
        return np.asarray(controls, dtype=np.float32)
