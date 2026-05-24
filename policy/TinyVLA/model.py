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


TARGET_SIZE = (320, 240)


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
            "action_head": "act",
        })
        self.policy.policy.eval()

        with open(model_cfg["stats_path"], "rb") as f:
            self.stats = pickle.load(f)

        self.chunk_size = int(self.policy.policy.config.chunk_size)
        self.action_dim = int(self.policy.policy.config.action_dim)

        # Buffers for temporal-aggregation, keyed by env_idx.
        self.latest_obs = {}
        self.all_time_actions = {}
        self.step_counter = {}
        self.max_timesteps = 10000

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

            agg_action = self._temporal_aggregate(env_idx, all_actions)
            action_np = agg_action.detach().cpu().to(torch.float32).numpy()
            action_np = action_np * self.stats["action_std"] + self.stats["action_mean"]

            # Keep 2-D shape [1, action_dim] so unpack_robot_state returns a
            # list[dict], matching the chunk-of-steps contract that XPolicyLab
            # deploy.py expects from get_action_batch (see policy/DP/model.py).
            action_dict_list.append(
                unpack_robot_state(
                    action_np[None, :],
                    action_type=self.action_type,
                    robot_action_dim_info=self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return action_dict_list

    def reset(self):
        self.latest_obs.clear()
        self.all_time_actions.clear()
        self.step_counter.clear()

    def _encode_obs(self, obs):
        cam_chws = []
        for cam_key in self.camera_keys:
            bgr = np.asarray(obs["vision"][cam_key]["color"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
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

        instructions = obs.get("instruction") or obs["instructions"]
        raw_lang = instructions if isinstance(instructions, str) else str(instructions[0])

        return curr_image, robot_state, raw_lang

    def _temporal_aggregate(self, env_idx, all_actions):
        if env_idx not in self.all_time_actions:
            self.all_time_actions[env_idx] = torch.zeros(
                self.max_timesteps,
                self.max_timesteps + self.chunk_size,
                self.action_dim,
                dtype=torch.float16,
                device=all_actions.device,
            )
            self.step_counter[env_idx] = 0

        buf = self.all_time_actions[env_idx]
        t = self.step_counter[env_idx]
        buf[[t], t : t + self.chunk_size] = all_actions

        col = buf[:, t]
        populated = torch.all(col != 0, dim=1)
        col = col[populated]
        weights = torch.from_numpy(np.exp(-0.01 * np.arange(len(col)))).to(col.device)
        weights = (weights / weights.sum()).unsqueeze(1)

        self.step_counter[env_idx] = t + 1
        return (col * weights).sum(dim=0)
