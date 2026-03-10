import torch
import yaml
import cv2
import numpy as np
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner
from XPolicyLab.model_template import ModelTemplate

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        action_dim = model_cfg['action_dim']
        load_config_path = os.path.join(parent_dir, f'diffusion_policy/config/robot_dp_{action_dim}.yaml')
        with open(load_config_path, "r", encoding="utf-8") as f:
            model_training_config = yaml.safe_load(f)
        
        n_obs_steps = model_training_config['n_obs_steps']
        n_action_steps = model_training_config['n_action_steps']
        self.action_type = model_cfg['action_type']

        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        self.model = self.get_model(model_cfg=model_cfg)

    def get_model(self, model_cfg):
        ckpt_file = os.path.join(parent_dir, f"checkpoints/{model_cfg['task_name']}-{model_cfg['env_cfg']}-{model_cfg['expert_data_num']}-{model_cfg['action_type']}-{model_cfg['seed']}/{model_cfg['checkpoint_num']}.ckpt")

        # load checkpoint and workspace
        payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device("cuda:0")
        policy.to(device)
        policy.eval()
        
        return policy

    def update_obs(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]
        obs_list = [encode_obs(obs, self.action_type) for obs in obs_list]
        self.runner.update_obs(obs_list, env_idx_list)
    
    def reset(self):
        self.runner.reset_obs()

    def get_action(self, env_idx_list):
        actions = self.runner.get_action(self.model, env_idx_list)
        action_dict_list = []

        if self.action_type == "joint": # TODO
            left_key = "left_arm_joint_state"
            right_key = "right_arm_joint_state"
        elif self.action_type == "ee":
            left_key = "left_ee_pose"
            right_key = "right_ee_pose"
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        for i in range(len(env_idx_list)):
            env_action, current_env_list = actions[i], []
            for action in env_action:
                action_dict = { # TODO
                    left_key: action[0:7],
                    "left_ee_joint_state": action[7:13],
                    right_key: action[13:20],
                    "right_ee_joint_state": action[20:26],
                }
                current_env_list.append(action_dict)
            action_dict_list.append(current_env_list)
            
        return action_dict_list

def encode_obs(observation, action_type):
    head_img = (np.moveaxis(observation["vision"]["cam_head"]["color"], -1, 0) / 255)
    head_img = np.transpose(cv2.resize(np.transpose(head_img, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    # resize
    # left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    # right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict( # TODO
        head_cam=head_img,
        # left_cam=left_cam,
        # right_cam=right_cam,
    )
    if action_type == 'joint':
        if "joint_states" in observation['state'].keys(): # single arm
            obs["agent_pos"] = np.concatenate([observation["state"]["joint_state"], observation["state"]["ee_joint_state"]], axis=-1)
        else:
            assert "left_arm_joint_state" in observation['state'].keys() and "right_arm_joint_state" in observation['state'].keys(), "Expected joint states for both arms in the observationset."
            left_arm_joint_state = observation['state']["left_arm_joint_state"]
            right_arm_joint_state = observation['state']["right_arm_joint_state"]
            left_ee_joint_state = observation['state']["left_ee_joint_state"]
            right_ee_joint_state = observation['state']["right_ee_joint_state"]
            obs["agent_pos"] = np.concatenate([left_arm_joint_state, left_ee_joint_state, right_arm_joint_state, right_ee_joint_state], axis=-1)
    elif action_type == 'ee':
        if "ee_pose" in observation['state'].keys(): # dual arm
            ee_pose = observation['state']["ee_pose"]
            ee_joint_state = observation['state']["ee_joint_state"]
            obs["agent_pos"] = np.concatenate([ee_pose, ee_joint_state], axis=-1)
        else:
            assert "left_ee_pose" in observation['state'].keys() and "right_ee_pose" in observation['state'].keys(), "Expected ee poses for both arms in the observationset."
            left_ee_pose = observation['state']["left_ee_pose"]
            right_ee_pose = observation['state']["right_ee_pose"]
            left_ee_joint_state = observation['state']["left_ee_joint_state"]
            right_ee_joint_state = observation['state']["right_ee_joint_state"]
            obs["agent_pos"] = np.concatenate([left_ee_pose, left_ee_joint_state, right_ee_pose, right_ee_joint_state], axis=-1)
    return obs