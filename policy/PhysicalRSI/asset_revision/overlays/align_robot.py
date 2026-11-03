"""CuRobo backend with robot placement measured from current proprioception."""
from pathlib import Path
import numpy as np
from physicalrsi_skill_library.geometry import matrix, pose, joint_chain, forward_kinematics
from .curobo_planner import CuroboPlanner
from physicalrsi_observed_geometry.support_collision import patches_in_robot_frame

ASSETS = Path(__file__).parent / 'assets' / 'x5'


class Robot:
    def __init__(self):
        self.names = [f'joint{i}' for i in range(1,7)]
        self.chain = joint_chain(ASSETS/'X5A.urdf', self.names)
        self.bases = {}
        self.planners = {}
        self.state = None
        self.support = None

    def bind_support(self, support):
        if not support:raise ValueError('Current observed support collision geometry required')
        self.support = support
        self.planners = {}

    def update(self, state):
        bases = {}
        for arm in ('left','right'):
            joints = np.asarray(state[f'{arm}_arm_joint_state'],float).reshape(-1)
            observed = np.asarray(state[f'{arm}_ee_pose'],float).reshape(-1)
            if joints.shape!=(6,) or observed.shape!=(7,) or not np.isfinite(np.r_[joints,observed]).all():
                raise ValueError('Current finite joint and world EE observations are required')
            local = forward_kinematics(self.chain,[0,0,0,1,0,0,0],joints)
            bases[arm] = pose(matrix(observed)@np.linalg.inv(matrix(local)))
        if self.support is None:raise ValueError('Observe and bind support before planning')
        if not self.planners:
            self.planners = {arm:CuroboPlanner(base,self.names,self.names,'cuda:0',dt=.004,
                                              yml_path=str(ASSETS/'curobo.yml'),
                                              observed_scene=patches_in_robot_frame(self.support,base))
                             for arm,base in bases.items()}
        self.bases = bases
        self.state = state

    def solve_ik(self, arm, target):
        if self.state is None:raise ValueError('Bind current robot observations before planning')
        return self.planners[arm].solve_ik_to_joint(self.state[f'{arm}_arm_joint_state'],target,
                                                  real_robot_pose=self.bases[arm])

    def path(self, arm, start, target, constraint):
        if self.state is None:raise ValueError('Bind current robot observations before planning')
        result = self.planners[arm].plan_path(start,target,self.bases[arm],constraint_pose=constraint)
        if result['status']!='Success':raise RuntimeError(f'H07 trajectory failed: {result}')
        return np.asarray(result['position'],dtype=float)
