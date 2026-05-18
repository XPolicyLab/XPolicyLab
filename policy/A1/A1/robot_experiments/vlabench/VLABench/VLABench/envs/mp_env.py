import numpy as np
import math
from dm_control import composer
from VLABench.utils.utils import euler_to_quaternion, expand_mask
from VLABench.utils.depth2cloud import rotMatList2NPRotMat, quat2Mat, posRotMat2Mat, PointCloudGenerator
from VLABench.envs.dm_env import LM4ManipDMEnv
from VLABench.utils.skill_lib import SkillLib
from VLABench.tasks.components.container import CommonContainer
from VLABench.utils.utils import *

def reach(env,point,quat,gripper_state,max_n_substep=100,tolerance=0.01):
    action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)[:7]
    action = np.concatenate([action, gripper_state])
    waypoint = np.concatenate([point, quaternion_to_euler(quat), gripper_state])
    for _ in range(max_n_substep):
        timestep = env.step(action,check_next_stage=False)
        current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
        if np.max(current_qpos - np.array(action[:7])) < tolerance \
            and np.min(current_qpos - np.array(action[:7])) > -tolerance:
            break

class MPEnv(LM4ManipDMEnv):
    def __init__(self, **kwargs):
        self.eval = kwargs.get("eval", True)
        if 'eval' in kwargs:
            kwargs.pop('eval')
        super().__init__(**kwargs)
        self.current_stage = 0
        
        
    def reset(self):
        timestep = super().reset()
        self.current_stage = 0
        self.current_stage_name = self.task.current_stage_name
        stage_condition_config = self.task.config_manager.config["task"]["stage_conditions"][self.current_stage]
        self.stage = stage_condition_config[0]
        self.obj = stage_condition_config[3]['obj']
        self.target = stage_condition_config[3]['target']
        # print('stage_condition_config',stage_condition_config)
        # if self.eval:
        #     if stage_condition_config[2]['reach']:
        #         target = stage_condition_config[2]['target']
        #         gripper_closed = stage_condition_config[2]['grasp']
        #         lift = stage_condition_config[2]['lift']
        #         pos = np.array(self.task.entities[target].get_xpos(self.physics))
        #         pos[-1]+=0.3

        #         if not self.eval:
        #             ranges = [(-0.05, 0.05), (-0.05, 0.05), (0, 0.10)]
        #             random_array = np.array([np.random.uniform(low, high) for low, high in ranges])
        #             pos += random_array

        #         if gripper_closed:
        #             pos[-1]+=0.1
        #             gripper_state = np.zeros(2)
        #         else:
        #             gripper_state = np.ones(2) * 0.04
        #         if lift:
        #             SkillLib.lift(self, lift_height=0.3, gripper_state=gripper_state)
        #         SkillLib.moveto(self, pos,gripper_state=gripper_state)
        return timestep
        
    def get_observation(self):
        observation = super().get_observation()
        if self.obj is not None:
            geom_ids = [self.physics.bind(geom).element_id for geom in self.task.entities[self.obj].geoms]
            obj_geom_ids = np.array(geom_ids,np.float32)
        else:
            obj_geom_ids = np.array([],np.float32)
        if self.target is not None:
            geom_ids = [self.physics.bind(geom).element_id for geom in self.task.entities[self.target].geoms]
            target_geom_ids = np.array(geom_ids,np.float32)
        else:
            target_geom_ids = np.array([],np.float32)
        observation["obj_geom_ids"] = obj_geom_ids
        observation["target_geom_ids"] = target_geom_ids
        observation['current_stage_name']=self.current_stage_name
        observation['stage'] = self.stage
        
        return observation

    def step(self, action=None,check_next_stage=True):
        if action is None:
            timestep = self.physics.step()
        else:
            timestep = super().step(action)
        # if check_next_stage and self.task.current_stage > self.current_stage:
        #     self.current_stage = self.task.current_stage
        #     self.current_stage_name = self.task.current_stage_name
        #     stage_condition_config = self.task.config_manager.config["task"]["stage_conditions"][self.current_stage]
        #     self.stage = stage_condition_config[0]
        #     self.obj = stage_condition_config[3]['obj']
        #     self.target = stage_condition_config[3]['target']
        #     # print('stage_condition_config',stage_condition_config)
        #     print('get into next stage, current stage is',self.current_stage,stage_condition_config[0])
        #     if stage_condition_config[2]['reach']:
        #         target = stage_condition_config[2]['target']
        #         gripper_closed = stage_condition_config[2]['grasp']
        #         lift = stage_condition_config[2]['lift']
        #         pos = np.array(self.task.entities[target ].get_xpos(self.physics))
        #         pos[-1]+=0.3
        #         if gripper_closed:
        #             pos[-1]+=0.1
        #             gripper_state = np.zeros(2)
        #         else:
        #             gripper_state = np.ones(2) * 0.04
        #         print('gripper_closed',gripper_closed,gripper_state)
        #         if lift:
        #             SkillLib.lift(self, lift_height=0.3, gripper_state=gripper_state)
        #         SkillLib.moveto(self, pos,gripper_state=gripper_state)
        return timestep
    