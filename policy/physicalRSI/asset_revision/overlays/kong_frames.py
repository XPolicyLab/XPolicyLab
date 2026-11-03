"""Robot FK and IK frames derived from current observations and robot URDF."""
from physicalrsi_skill_library.geometry import matrix, pose, calculate_target_pose, joint_chain, forward_kinematics
from pathlib import Path
import numpy as np


class RobotFrames:
    def __init__(self):
        root=Path(__file__).resolve().parent/'assets/x5'
        self.chain=joint_chain(root/'X5A.urdf',[f'joint{i}' for i in range(1,7)])
        self.reset_observations()

    def reset_observations(self):
        self.observation=None;self.bases={};self.support=None;self.support_plane=None;self.solvers={}

    def fk(self,base_pose,joints):
        return forward_kinematics(self.chain,base_pose,joints)

    def bind_observation(self,observation):
        self.observation=observation
        for arm in ('left','right'):
            state=observation['state']
            local=self.fk([0,0,0,1,0,0,0],state[arm+'_arm_joint_state'])
            self.bases[arm]=pose(matrix(state[arm+'_ee_pose'])@np.linalg.inv(matrix(local)))
        return self.bases

    def get_support_plane(self):
        from physicalrsi_observed_geometry import GeometryUnavailable
        if self.observation is None:
            raise GeometryUnavailable('Current calibrated observation required for kong support')
        if self.support_plane is None:
            from physicalrsi_observed_geometry.support_observation import observe_support_geometry
            from perception_services.perception.sam3_http import segment_many_http
            geometry=observe_support_geometry(self.observation,segment_many_http,
                                              occluders=('robot arm','mahjong tile'))
            self.support=geometry['patches']
            self.support_plane=np.asarray(geometry['plane'],float).copy()
        return self.support_plane.copy()

    def solve(self,arm,seed,target):
        from physicalrsi_observed_geometry import GeometryUnavailable
        from physicalrsi_observed_geometry.support_collision import patches_in_robot_frame
        try:
            self.get_support_plane()
            base=np.asarray(self.bases[arm],float);cached=self.solvers.get(arm)
            if cached is None or not np.allclose(cached[0],base,atol=1e-5,rtol=0):
                from .policy_ik import PolicyIK
                cached=(base.copy(),PolicyIK(patches_in_robot_frame(self.support,base)))
                self.solvers[arm]=cached
            return cached[1].solve_ik_to_joint(seed,target,real_robot_pose=base.tolist())
        except GeometryUnavailable as error:
            return {'status':'ObservedGeometryUnavailable: '+str(error)}
