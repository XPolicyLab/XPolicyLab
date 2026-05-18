import torch
import os
import numpy as np
import yaml
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import hydra
from omegaconf import OmegaConf
from VLABench.evaluation.model.policy.base import Policy 
from VLABench.utils.utils import quaternion_to_euler
from gea.cfgs.actions_ids import actions_name2id
from collections import deque

class gea_policy(Policy):
    def __init__(self, 
                 model_path,
                 **kwargs):
        config = model_path+'/.hydra/config.yaml'
        
        config = OmegaConf.load(config)
        agent_cfg = config.agent
        agent_cfg.depth = config.depth
        agent_cfg.dino = config.dino
        agent = hydra.utils.instantiate(agent_cfg)
        pretrain_cfg = config.pretrain
        pretrain_cfg.path = model_path+'/latest.pt'
        agent.load_pretrained_weights(
            pretrain_cfg.path, pretrain_cfg.just_encoder_decoders
        )
        agent.train(False)
        self.num_frames = 2
        self._frames = deque([], maxlen=self.num_frames)
        self.action_length = 3
        self.actions = deque(maxlen=self.action_length)
        self.last_gripper = 1
        super().__init__(agent)
    
    def reset(self):
        self._frames.clear()
        self.actions.clear()
        self.last_gripper = 0

    def process_observation(self, obs, unnorm_key):
        instruction = obs['instruction']
        stage = obs['stage']
        observation = obs
        rgb = observation['rgb']
        depth = observation['depth'][-1]
        segmentation = observation['segmentation'][...,0]
        obj_geom_id = observation['obj_geom_ids']
        target_geom_id = observation['target_geom_ids']
        robot_mask = observation['robot_mask']
        robot_mask = robot_mask.astype(np.bool_)
        mask = np.zeros_like(rgb).astype(np.float32)
        mask[~robot_mask,:] = (1,0,0)
        if len(obj_geom_id)>0:
            obj_mask = np.where((segmentation <= max(obj_geom_id))&(segmentation >= min(obj_geom_id)), 0, 1).astype(np.bool_)
            mask[~obj_mask,:] = (0,1,0)
        else:
            obj_mask = np.ones_like(segmentation)
        if len(target_geom_id)>0:
            target_mask = np.where((segmentation <= max(target_geom_id))&(segmentation >= min(target_geom_id)), 0, 1).astype(np.bool_)
            mask[~target_mask,:] = (0,0,1)
        else:
            target_mask = np.ones_like(segmentation)
        depth = np.where((obj_mask[-1]==0) | (target_mask[-1]==0) | (robot_mask[-1]==0), depth, 1)
        mask = mask[:-1]
        mask = mask.transpose(0,3,1,2)
        mask = mask.reshape(-1,mask.shape[2],mask.shape[3])
        depth = depth[np.newaxis, :, :]
        mask = np.concatenate((mask, depth), axis=0)
        stage = actions_name2id[stage]
        self._frames.append(mask)
        if len(self._frames)<self.num_frames:
            self._frames.append(mask)
        state = observation['ee_state'].astype(np.float32)
        state[:3] -= np.array([0, -0.4, 0.78])

        if len(state.shape)>1:
            state = state[:,0]
        # state[-3:] = [-3.13,0,0]
        # state = np.concatenate((state, np.array([self.last_gripper]))).astype(np.float32)
        task_id = obs['task_id']
        obs = {
            'pixels':np.concatenate(list(self._frames), axis=0),
            'task':instruction,
            'action': np.array([task_id],np.int32),
            'state': state
        }
        return obs
    
    def predict(self, obs, unnorm_key=None, process_obs=True, **kwargs):
        current_ee_state = obs["ee_state"]
        # robot_position = obs['robot_position']
        # print('obs',obs['ee_state'][:3])
        if process_obs:
            obs = self.process_observation(obs, unnorm_key)
        if len(self.actions) == 0:
            with torch.no_grad():
                pred_actions = self.model.act(
                    obs
                )
                pred_actions[:,:3] += np.array([0, -0.4, 0.78])
                self.actions.extend(pred_actions[:self.action_length])
        
        # 需要取第一个action
        pred_action = self.actions.popleft()
        
        
        # if len(current_ee_state) == 8:
        #     pos, quat = current_ee_state[:3], current_ee_state[3:7]
        #     euler = quaternion_to_euler(quat)
        # elif len(current_ee_state) == 7:
        #     pos, euler = current_ee_state[:3], current_ee_state[3:6]
        # print('delta_action',delta_action[:3],delta_action[3:6],delta_action[-1])
        # print('pred_action',pred_action[:3])
        target_pos = pred_action[:3] # np.array(pos) # + delta_action[:3]
        target_euler = pred_action[3:6] # delta_action[3:6] # euler # (euler+delta_action[3:6])/2 #  # absolute angle
        print('target_pos',pred_action)
        print('current_ee_state',current_ee_state)
        # target_euler[-1] = 0
        # target_euler[-2] = 0
        # target_euler[-3] = -3.13
        # print('target_euler',target_euler)
        gripper_open = pred_action[-1]
        # print('current_ee_state',current_ee_state)
        # print('action',action)
        # target_pos = action[:3]+robot_position
        # target_euler = action[3:6]
        # gripper_open = action[-1]
        # print('gripper_open',gripper_open)

        gripper_threshold = 0.03
        gripper_state = np.ones(2)*0.04 if gripper_open >= gripper_threshold else np.zeros(2)
        if gripper_open >= gripper_threshold:
            self.last_gripper = 1
        else:
            self.last_gripper = 0
        return target_pos, target_euler, gripper_state     
    
    @property
    def name(self):
        return "gea_policy"

if __name__ == "__main__":
    model = gea_policy("/mnt/data/zhangkaidong/VLABench/gea/exp_local/real_world_finetune_2025.04.30_11:17:48_2871954")