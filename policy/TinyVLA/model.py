import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


TINYVLA_DIR = Path(__file__).resolve().parent / "tinyvla"
if str(TINYVLA_DIR) not in sys.path:
    sys.path.append(str(TINYVLA_DIR))

from eval_real_franka import llava_pythia_act_policy



class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.action_type = model_cfg["action_type"]
        self.camera_keys = list(model_cfg["camera_keys"])
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])

        self.policy = llava_pythia_act_policy({
            "model_path": model_cfg["model_path"],
            "model_base": model_cfg["model_base"],
            "enable_lora": model_cfg["enable_lora"],
            "conv_mode": model_cfg["conv_mode"],
        })
        self.policy.policy.eval()

        with open(model_cfg["stats_path"], "rb") as f:
            self.stats = pickle.load(f)

        self.latest_obs = {}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.latest_obs[int(obs["env_idx"])] = self._encode_obs(obs)

    def get_action(self):
        return self.get_action_batch([0])[0]

    def get_action_batch(self, env_idx_list):
        action_dict_list = []
        for env_idx in env_idx_list:
            env_idx = int(env_idx)
            curr_image, robot_state, raw_lang = self.latest_obs[env_idx]

            batch = self.policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
            with torch.inference_mode():
                all_actions = self.policy.policy(**batch, eval=True)

            action_chunk = all_actions[0].detach().cpu().to(torch.float32).numpy()
            action_chunk = (
                action_chunk * self.stats["action_std"] + self.stats["action_mean"]
            )

            action_dict_list.append(
                unpack_robot_state(
                    action_chunk,
                    action_type=self.action_type,
                    robot_action_dim_info=self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return action_dict_list

    def reset(self):
        self.latest_obs.clear()

    def _encode_obs(self, obs):
        cam_chws = []
        for cam_key in self.camera_keys:
            rgb = np.asarray(obs["vision"][cam_key]["color"])
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
            cam_chws.append(np.transpose(rgb, (2, 0, 1)))

        stacked = np.stack(cam_chws, axis=0).astype(np.float32) / 255.0
        curr_image = torch.from_numpy(stacked).cuda().unsqueeze(0)

        state_vec = pack_robot_state(
            obs=obs,
            action_type=self.action_type,
            robot_action_dim_info=self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        state_vec = (state_vec - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        robot_state = torch.from_numpy(state_vec).cuda().unsqueeze(0)

        raw = obs.get("instruction") or obs["instructions"]
        if isinstance(raw, (list, tuple, np.ndarray)):
            raw = raw[0]
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            raw = raw.decode("utf-8")
        raw_lang = str(raw)

        return curr_image, robot_state, raw_lang
