import cv2
import numpy as np
import torch
from .detr.act_policy import ACT
from argparse import Namespace

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self.model = self.get_model(model_cfg=model_cfg)
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg'])
        self.action_type = model_cfg['action_type']

    def get_model(self, model_cfg):
        return ACT(model_cfg, Namespace(**model_cfg))

    def update_obs(self, obs):
        encoded_obs = encode_obs(obs, self.action_type, self.robot_action_dim_info)
        self.model.update_obs(encoded_obs)
    
    # def update_obs_batch(self, obs_list): # TODO
    #     pass
    
    def get_action(self):
        actions = self.model.get_action()
        action_list = unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type='obs')
        return action_list

    # def get_action_batch(self, env_idx_list): # TODO
    #     pass

    def reset(self):
        # Reset temporal aggregation state if enabled
        if self.model.temporal_agg:
            self.model.all_time_actions = torch.zeros([
                self.model.max_timesteps,
                self.model.max_timesteps + self.model.num_queries,
                self.model.state_dim,
            ]).to(self.model.device)
            self.model.t = 0
            print("Reset temporal aggregation state")
        else:
            self.model.t = 0

def encode_obs(observation, action_type, robot_action_dim_info):
    head_cam = cv2.resize(observation["vision"]["cam_head"]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = cv2.resize(observation["vision"]["cam_left_wrist"]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = cv2.resize(observation["vision"]["cam_right_wrist"]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
    
    qpos = pack_robot_state(observation, action_type, action_type, "obs")

    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }