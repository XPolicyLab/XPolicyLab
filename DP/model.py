import torch
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

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, action_type):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        self.action_type = action_type

    def update_obs(self, observation):
        self.runner.update_obs(encode_obs(observation, self.action_type))
    
    def reset(self):
        self.runner.reset_obs()

    def get_action(self):
        action = self.runner.get_action(self.policy)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint and workspace
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy

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
            assert "left_arm_joint_states" in observation['state'].keys() and "right_arm_joint_states" in observation['state'].keys(), "Expected joint states for both arms in the observationset."
            left_arm_joint_states = observation['state']["left_arm_joint_states"]
            right_arm_joint_states = observation['state']["right_arm_joint_states"]
            left_ee_joint_states = observation['state']["left_ee_joint_states"]
            right_ee_joint_states = observation['state']["right_ee_joint_states"]
            obs["agent_pos"] = np.concatenate([left_arm_joint_states, left_ee_joint_states, right_arm_joint_states, right_ee_joint_states], axis=-1)
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