"""Fixed robot kinematics; robot placement is derived solely from public state."""
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def matrix(pose):
    pose=np.asarray(pose,dtype=float)
    result=np.eye(4);result[:3,3]=pose[:3]
    result[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    return result

def base_from_state(world_flange,local_flange,frame_bias):
    local=np.asarray(local_flange,dtype=float).copy()
    local[:3]-=np.asarray(frame_bias,dtype=float)
    base=matrix(world_flange)@np.linalg.inv(matrix(local))
    q=Rotation.from_matrix(base[:3,:3]).as_quat()
    return np.r_[base[:3,3],q[3],q[:3]].tolist()

class RobotOnlyPlanner:
    def __init__(self):self.observation=None;self.planner=None
    def set_observation(self,observation):self.observation=observation
    def __call__(self,*,arm,target_pose):
        if self.observation is None:raise RuntimeError('Official observation required before planning')
        if arm not in ('left','right'):raise ValueError('Unknown arm')
        if self.planner is None:
            from .robot_planner import PublicCuroboPlanner
            joints=[f'joint{i}' for i in range(1,7)]
            self.planner=PublicCuroboPlanner(robot_origin_pose=[0,0,0,1,0,0,0],active_joints_name=joints,all_joints=joints,dt=.004,yml_path=str(Path(os.environ['ROBODOJO_ROOT'])/'Assets/Robots/x5/curobo.yml'),table_height=None)
            if self.planner.scene_model is not None:raise RuntimeError('Scene geometry is forbidden')
        joints=self.observation.state[arm+'_arm_joint_state']
        state=self.planner.motion_planner.compute_kinematics(self.planner._build_joint_state(joints))
        frame=self.planner.motion_planner.tool_frames[0]
        flange=state.tool_poses.to_dict()[frame]
        local=np.r_[flange.position.detach().cpu().numpy().reshape(-1,3)[0],flange.quaternion.detach().cpu().numpy().reshape(-1,4)[0]]
        base=base_from_state(self.observation.state[arm+'_ee_pose'],local,self.planner.frame_bias)
        result=self.planner.plan_path(joints,target_pose,base)
        return {'status':result['status'],'position':np.asarray(result['position']).tolist() if result.get('position') is not None else None}

def planner_for():
    return RobotOnlyPlanner()
