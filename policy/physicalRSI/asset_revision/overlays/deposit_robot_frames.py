"""X5 kinematics calibrated from current joint and world-frame EE observations."""
from physicalrsi_skill_library.geometry import matrix, pose, calculate_target_pose, joint_chain, forward_kinematics
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

class RobotFrames:

    def __init__(self):
        root = Path(__file__).resolve().parent / 'assets/x5'
        self.chain = joint_chain(root / 'X5A.urdf', [f'joint{i}' for i in range(1, 7)])
        self.robots = {arm: SimpleNamespace(arm=arm, gripper_bias=0.145, delta_matrix=np.eye(3), inv_delta_matrix=np.eye(3), entity_origin_pose=None) for arm in ('left', 'right')}
        self.states = {}
        self.observations = {}
        self.supports = {}
        self.solvers = {}

    def bind_state(self, env_idx, state):
        # URDF supplies robot geometry; the observation supplies its world placement.
        bases = {}
        for arm in self.robots:
            joints = np.asarray(state[f'{arm}_arm_joint_state'], dtype=float).reshape(-1)
            observed = np.asarray(state[f'{arm}_ee_pose'], dtype=float).reshape(-1)
            if joints.shape != (6,) or observed.shape != (7,) or not (
                np.isfinite(joints).all() and np.isfinite(observed).all()
            ):
                raise ValueError('Finite observed joints and world EE pose required')
            local = forward_kinematics(self.chain, [0, 0, 0, 1, 0, 0, 0], joints)
            bases[arm] = pose(matrix(observed) @ np.linalg.inv(matrix(local)))
        for arm, base in bases.items():
            self.robots[arm].entity_origin_pose = base
        self.states[env_idx] = state

    def fk(self, arm, joints):
        if self.robots[arm].entity_origin_pose is None:
            raise ValueError('Bind current robot state before forward kinematics')
        return forward_kinematics(self.chain, self.robots[arm].entity_origin_pose, joints)

    def get_robot_by_arm_name(self, name):
        return self.robots[name.removesuffix('_arm')]

    def reset(self):
        self.states.clear()
        self.observations.clear()
        self.supports.clear()
        self.solvers.clear()

    def bind_observation(self, env_idx, observation):
        self.bind_state(env_idx, observation['state'])
        self.observations[env_idx] = observation

    def solve_ik(self, target_pose, env_idx, robot):
        from physicalrsi_observed_geometry import GeometryUnavailable
        from physicalrsi_observed_geometry.support_collision import patches_in_robot_frame
        if env_idx not in self.observations:
            raise GeometryUnavailable('Bind calibrated RGB before deposit IK')
        if env_idx not in self.supports:
            from physicalrsi_observed_geometry.support_observation import observe_support
            from perception_services.perception.sam3_http import segment_many_http
            self.supports[env_idx] = observe_support(
                self.observations[env_idx], segment_many_http,
                occluders=('robot arm', 'coin', 'piggy bank'))
        key = (env_idx, robot.arm)
        base = np.asarray(robot.entity_origin_pose, float)
        cached = self.solvers.get(key)
        if cached is None or not np.allclose(cached[0], base, atol=1e-5, rtol=0):
            from .policy_ik import PolicyIK
            scene = patches_in_robot_frame(self.supports[env_idx], base)
            cached = (base.copy(), PolicyIK(scene))
            self.solvers[key] = cached
        return cached[1].solve_ik_to_joint(
            self.states[env_idx][f'{robot.arm}_arm_joint_state'], target_pose,
            real_robot_pose=robot.entity_origin_pose)
