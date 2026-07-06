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
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info, get_action_dim

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        load_config_path = os.path.join(parent_dir, f'diffusion_policy/config/robot_dp.yaml')
        with open(load_config_path, "r", encoding="utf-8") as f:
            model_training_config = yaml.safe_load(f)
        
        model_training_config['action_dim'] = get_action_dim(model_cfg['env_cfg_type'])
        model_training_config['bench_name'] = model_cfg['bench_name']
        model_training_config['task'] = model_cfg['task_name']
        n_obs_steps = model_training_config['n_obs_steps']
        n_action_steps = model_training_config['n_action_steps']
        self.action_type = model_cfg['action_type']

        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        self.model = self.get_model(model_cfg=model_cfg)

        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        self._latest_env_idx_list = None

    def get_model(self, model_cfg):
        ckpt_setting = str(model_cfg['ckpt_name'])
        ckpt_file = self._resolve_checkpoint_file(ckpt_setting, model_cfg.get('checkpoint_num', 'latest'))

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

    def _resolve_checkpoint_file(self, ckpt_setting, checkpoint_num):
        ckpt_dir = os.path.join(parent_dir, "checkpoints", ckpt_setting)
        checkpoint_num = "latest" if checkpoint_num is None else str(checkpoint_num)

        if checkpoint_num.lower() not in {"", "latest", "none"}:
            ckpt_file = os.path.join(ckpt_dir, f"{checkpoint_num}.ckpt")
            if not os.path.isfile(ckpt_file):
                raise FileNotFoundError(f"DP checkpoint not found: {ckpt_file}")
            return ckpt_file

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"DP checkpoint directory not found: {ckpt_dir}")

        candidates = []
        for name in os.listdir(ckpt_dir):
            if not name.endswith(".ckpt"):
                continue
            stem = name[:-5]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(ckpt_dir, name)))

        if not candidates:
            raise FileNotFoundError(f"No numeric DP checkpoints found under: {ckpt_dir}")

        return max(candidates, key=lambda item: item[0])[1]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]
        obs_list = [encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list]
        self.runner.update_obs(obs_list, env_idx_list)
        self._latest_env_idx_list = env_idx_list

    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")

        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])
            
        return action_list[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        actions = self.runner.get_action(self.model, env_idx_list)
        action_dict_list = []

        for i in range(len(env_idx_list)):
            current_env_action_list = unpack_robot_state(actions[i], self.action_type, self.robot_action_dim_info, source_type='obs')
            action_dict_list.append(current_env_action_list)
            
        return action_dict_list

    def reset(self):
        self.runner.reset_obs()
        self._latest_env_idx_list = None

def encode_obs(observation, action_type, robot_action_dim_info):
    head_img = (np.moveaxis(observation["vision"]["cam_head"]["color"], -1, 0) / 255)
    head_img = np.transpose(cv2.resize(np.transpose(head_img, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    left_cam = (np.moveaxis(observation["vision"]["cam_left_wrist"]["color"], -1, 0) / 255)
    left_cam = np.transpose(cv2.resize(np.transpose(left_cam, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    right_cam = (np.moveaxis(observation["vision"]["cam_right_wrist"]["color"], -1, 0) / 255)
    right_cam = np.transpose(cv2.resize(np.transpose(right_cam, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1))
    obs = dict(
        head_cam=head_img,
        left_cam=left_cam,
        right_cam=right_cam,
        agent_pos=pack_robot_state(observation, action_type, robot_action_dim_info, source_type='obs'),
    )
    return obs