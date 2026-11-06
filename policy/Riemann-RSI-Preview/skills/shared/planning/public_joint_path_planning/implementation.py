"""Original checked cuRobo planning through injected public planners; no raw environment handle."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from skills.shared.geometry.x5_public_frame_model.implementation import rotation, quaternion, pose

class Planner:
    def __init__(self,planners):self.planners=planners;self.checkers={}

    def world(self,arm,frame,digits,excluded=None):
        from curobo._src.geom.types import Cuboid, SceneCfg
        base=frame.bases[arm]
        cuboids=[Cuboid(name='table',pose=[0,0,-.05,1,0,0,0],dims=[3.,3.,.05])]
        for digit in digits:
            if digit['digit']==excluded:continue
            center=np.r_[digit['xy'],frame.mounting_z+digit.get('support_z',0.)+digit['origin_to_bottom_m']]
            local=base[:3,:3].T@(center-base[:3,3])
            cuboids.append(Cuboid(name=f"rgb_digit_{digit['digit']}",pose=[*local,1,0,0,0],
                                 dims=[.065,.065,digit['height_m']+.008]))
        # Other arm's root is public proprioception; keep free trajectories away from its pedestal.
        other='right' if arm=='left' else 'left'
        center=frame.bases[other][:3,3]+[0,0,.065]
        local=base[:3,:3].T@(center-base[:3,3])
        cuboids.append(Cuboid(name='other_robot_base',pose=[*local,1,0,0,0],dims=[.12,.12,.12]))
        scene=SceneCfg(cuboid=cuboids)
        planner=self.planners[arm]
        for solver_name in ('motion_planner','motion_planner_batch','ik_solver'):
            getattr(planner,solver_name).update_world(scene)
        return scene

    def validate(self,arm,path):
        """Check every sampled joint configuration with native cuRobo spheres."""
        import torch
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        planner=self.planners[arm]
        if arm not in self.checkers:
            config=RobotCollisionCheckerCfg.load_from_config(robot_config=planner.robot_cfg,
                device_cfg=planner.device_cfg,scene_collision_checker=planner.motion_planner.scene_collision_checker,
                collision_activation_distance=0.,self_collision_activation_distance=0.)
            self.checkers[arm]=RobotCollisionChecker(config)
        full=np.stack([planner._build_cspace_joint_values(joints) for joints in path])
        positions=torch.as_tensor(full,dtype=torch.float32,device=planner.device_cfg.device).unsqueeze(0)
        return bool(self.checkers[arm].validate(positions).all().item())

    def cartesian(self,arm,current,target,frame):
        """Sequential fixed-URDF IK preserves a level payload and vertical contact."""
        start=pose(frame.bases[arm]@frame.kinematics.fk(current))
        length=float(np.linalg.norm(np.asarray(target[:3])-start[:3]))
        angle=float(Rotation.from_matrix(rotation(start[3:]).T@rotation(target[3:])).magnitude())
        count=max(3,int(np.ceil(length/.008)),int(np.ceil(angle/.055)))
        rotations=Slerp([0.,1.],Rotation.from_matrix(np.stack([rotation(start[3:]),rotation(target[3:])])))
        joints=np.asarray(current).copy();path=[]
        for fraction in np.linspace(0.,1.,count+1)[1:]:
            waypoint=np.r_[start[:3]+fraction*(np.asarray(target[:3])-start[:3]),quaternion(rotations(fraction).as_matrix())]
            solution=frame.ik(arm,waypoint,joints)
            if solution is None or np.max(np.abs(solution['joints']-joints))>.4:return None
            joints=solution['joints'];path.append(joints)
        path=np.vstack([current,path])
        return path if self.validate(arm,path) else None

    def joint(self,arm,current,target):
        result=self.planners[arm].plan_joint(np.asarray(current,dtype=np.float32),np.asarray(target,dtype=np.float32))
        if result.get('status')!='Success':raise RuntimeError(f'cuRobo rejected {arm} joint route')
        return np.asarray(result['position'],float)

    def pose(self,arm,current,target,frame):
        result=self.planners[arm].plan_path(np.asarray(current,dtype=np.float32),np.asarray(target,float).tolist(),
                                          frame.root_pose(arm).tolist())
        return None if result.get('status')!='Success' else np.asarray(result['position'],float)
