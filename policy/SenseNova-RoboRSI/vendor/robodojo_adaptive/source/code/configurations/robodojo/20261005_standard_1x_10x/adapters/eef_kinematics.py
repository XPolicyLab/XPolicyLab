"""Robot-only bounded EEF conversion for the reference RoboDojo control contract.

Adapted from GPT-as-Policy 8f3d362b, robodojo_server/kinematics.py.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


class ArmFK:
    def __init__(self, urdf, joint_names, base='base_link', tip='link6'):
        self.names = list(joint_names)
        edges = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
        self.chain = []
        while tip != base:
            j = edges[tip]
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = Rotation.from_euler('xyz', rpy).as_matrix(), xyz
            axis = j.find('axis')
            axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0, 0])
            self.chain.insert(0, (j.get('name'), j.get('type'), t, axis))
            tip = j.find('parent').get('link')
        moving = [n for n, kind, _, _ in self.chain if kind != 'fixed']
        if set(moving) != set(self.names):
            raise ValueError(f'URDF chain/arm joint mismatch: {moving}, {self.names}')

    def matrix(self, q):
        q = np.asarray(q, float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError('Invalid arm joint vector')
        values, t = dict(zip(self.names, q)), np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ('revolute', 'continuous'):
                motion[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
            elif kind != 'fixed':
                raise ValueError('Only fixed/revolute arm chain supported')
            t = t @ origin @ motion
        return t

    def bounded_target(self, q, limits, root, target):
        q = np.asarray(q, float)
        now = root @ self.matrix(q)
        goal = transform(target['position'], target['quaternion_wxyz'])
        translation = goal[:3, 3] - now[:3, 3]
        rotation = Rotation.from_matrix(goal[:3, :3] @ now[:3, :3].T).as_rotvec()
        bounded = lambda x, cap: x*min(1., cap/max(np.linalg.norm(x), 1e-12))
        error = np.r_[bounded(translation, .02), bounded(rotation, .1)]
        jacobian = np.zeros((6, len(q)))
        for i in range(len(q)):
            shifted = q.copy(); shifted[i] += 1e-5
            nxt = root @ self.matrix(shifted)
            jacobian[:3, i] = (nxt[:3, 3]-now[:3, 3])/1e-5
            jacobian[3:, i] = Rotation.from_matrix(nxt[:3, :3] @ now[:3, :3].T).as_rotvec()/1e-5
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .05**2*np.eye(6), error)
        limits = np.asarray(limits, float)
        low, high = np.maximum(limits[:, 0], q-.05), np.minimum(limits[:, 1], q+.05)
        if np.any(low > high):
            raise ValueError('Joint outside bounded valid interval')
        result = np.clip(q+delta, low, high).astype(np.float32)
        return result, dict(target_position_error_m=float(np.linalg.norm(translation)),
            target_rotation_error_rad=float(np.linalg.norm(rotation)),
            proposed_joint_delta=(result-q).tolist(), bounded_task_delta=error.tolist(),
            method='robot_only_numerical_jacobian_dls', physical_tracking_verified=False)


class DualKinematics:
    def __init__(self, env):
        self.env, self.manager = env, env.robot_manager
        self.robots = {r.arm_name.split('_')[0]: r for r in self.manager.robot_list if r.type == 'target'}
        if set(self.robots) != {'left', 'right'}:
            raise ValueError('Expected left/right target X5 arms')
        self.fk = {arm: ArmFK(r.urdf_path, r.arm_joints_name, r.base_link, r.ee_link_name)
                   for arm, r in self.robots.items()}

    def root(self, arm):
        robot = self.robots[arm]
        value = self.manager.get_link_pose(robot, robot.base_link, is_relative=True)[0]
        return transform(value[:3], value[3:])

    def check(self):
        checks = {}
        for arm, r in self.robots.items():
            q = self.manager.get_joint(r)[0]
            predicted = self.root(arm) @ self.fk[arm].matrix(q)
            measured = self.manager.get_real_endpose(r)[0]
            error = np.linalg.norm(predicted[:3, 3]-measured[:3])
            angle = Rotation.from_matrix(predicted[:3, :3] @ transform(measured[:3], measured[3:])[:3, :3].T).magnitude()
            checks[arm] = dict(position_error_m=float(error), rotation_error_rad=float(angle),
                passed=bool(error < .002 and angle < .01))
        if not all(c['passed'] for c in checks.values()):
            raise ValueError(f'Robot FK does not match measured link6: {checks}')
        return dict(arms=checks, passed=True, physical_steps=0)

    def target(self, targets):
        if set(targets) != {'left', 'right'}:
            raise ValueError('Explicit left/right targets required')
        self.check()
        action, diagnostics = [], {}
        for arm in ('left', 'right'):
            robot, target = self.robots[arm], targets[arm]
            p, q = np.asarray(target['position'], float), np.asarray(target['quaternion_wxyz'], float)
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or abs(np.linalg.norm(q)-1) > 1e-4:
                raise ValueError('Finite pose and unit wxyz required')
            # Per-decision 5 cm / .35 rad bound is checked by the shared host;
            # each real ACK recomputes this bounded robot-only IK update.
            key = self.manager.robot_key[self.manager.robot_list.index(robot)]
            limits = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
            joints, diagnostics[arm] = self.fk[arm].bounded_target(
                self.manager.get_joint(robot)[0], limits, self.root(arm), target)
            if type(target['gripper_closed']) is not bool:
                raise ValueError('Explicit boolean gripper required')
            opening = target.get('gripper_opening', float(not target['gripper_closed']))
            if not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError('Invalid gripper opening')
            action.extend([*joints.tolist(), opening])
        return dict(action=action, diagnostics=diagnostics, physical_steps=0)
