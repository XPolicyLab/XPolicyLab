import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[3]
_WAN_VA_ROOT = _CUR_DIR / "lingbot_va" / "wan_va"

for _path in (str(_REPO_ROOT), str(_WAN_VA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from configs import VA_CONFIGS
from distributed.util import init_distributed
from wan_va_server import VA_Server


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    images = {
        "cam_high": extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"]),
        "cam_left_wrist": extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"]),
        "cam_right_wrist": extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"]),
    }

    if robot_action_dim_info is None:
        state = np.zeros((1,), dtype=np.float32)
    else:
        state = pack_robot_state(
            observation,
            action_type,
            robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)

    prompt = observation.get("prompt", default_prompt)

    return {
        "observation.images.cam_high": images["cam_high"],
        "observation.images.cam_left_wrist": images["cam_left_wrist"],
        "observation.images.cam_right_wrist": images["cam_right_wrist"],
        "observation.state": state,
        "task": prompt,
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg) -> None:
        self.model_cfg = dict(model_cfg)

        self.task_name = model_cfg.get("task_name", "default_task")
        self.action_type = model_cfg["action_type"]
        self.default_prompt = model_cfg.get("prompt", self.task_name)

        env_cfg = model_cfg.get("env_cfg")
        if env_cfg is None:
            self.robot_action_dim_info = None
        else:
            try:
                self.robot_action_dim_info = get_robot_action_dim_info(env_cfg)
            except FileNotFoundError:
                print(f"[WARN] env_cfg '{env_cfg}' not found, fallback to raw action mode.")
                self.robot_action_dim_info = None

        self.observation_window: list[dict[str, Any]] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self.vla = self.get_model(self.model_cfg)

    def get_model(self, model_cfg):
        config_name = model_cfg["config_name"]
        if config_name not in VA_CONFIGS:
            raise KeyError(f"Unknown config_name: {config_name}")

        job_config = copy.deepcopy(VA_CONFIGS[config_name])

        if model_cfg.get("save_root") is not None:
            job_config.save_root = model_cfg["save_root"]

        checkpoint_path = (
            model_cfg.get("checkpoint_path")
            or model_cfg.get("wan22_pretrained_model_name_or_path")
        )
        if checkpoint_path is not None:
            job_config.wan22_pretrained_model_name_or_path = checkpoint_path

        rank = int(os.getenv("RANK", 0))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))

        init_distributed(world_size, local_rank, rank)

        job_config.rank = rank
        job_config.local_rank = local_rank
        job_config.world_size = world_size

        model = VA_Server(job_config)
        model._reset(prompt=self.default_prompt)
        return model

    def _to_engine_obs(self, observation):
        image_dict = {
            key: observation[key]
            for key in self.vla.job_config.obs_cam_keys
        }
        return {
            "obs": [image_dict],
            "state": observation["observation.state"],
            "prompt": observation["task"],
        }

    def _format_action_chunk(self, action):
        action = np.asarray(action)

        if action.ndim == 3:
            action = np.transpose(action, (1, 2, 0)).reshape(-1, action.shape[0])
        elif action.ndim == 2 and action.shape[0] == self.vla.job_config.action_dim:
            action = action.T

        return action

    def infer(self, observation):
        if "reset" in observation and observation["reset"]:
            self.reset(
                checkpoint_path=observation["checkpoint_path"] if "checkpoint_path" in observation else None
            )
            return dict(action=None)

        model_obs = self._to_engine_obs(observation)
        result = self.vla.infer(model_obs)
        action = self._format_action_chunk(result["action"])
        return dict(action=action)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt)
            for obs in obs_list
        ]
        self.observation_window = encoded_obs_list

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            action_chunk = self.infer(self.observation_window[batch_index])["action"]

            if self.robot_action_dim_info is None:
                action_list.append(action_chunk)
            else:
                action_list.append(
                    unpack_robot_state(
                        action_chunk,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self, checkpoint_path=None) -> None:
        if checkpoint_path is not None:
            self.model_cfg["checkpoint_path"] = checkpoint_path
            self.vla = self.get_model(self.model_cfg)
        else:
            self.vla._reset(prompt=self.default_prompt)

        self.observation_window = None
        self._latest_env_idx_list = [0]


def _make_fake_obs(prompt: str):
    h, w = 224, 224
    return {
        "vision": {
            "cam_high": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        },
        "joint_action": {
            "left_gripper": 0.0,
            "left_arm": [0.0] * 6,
            "right_gripper": 0.0,
            "right_arm": [0.0] * 6,
        },
        "endpose": {
            "left_endpose": [0.0] * 6,
            "right_endpose": [0.0] * 6,
        },
        "prompt": prompt,
        "env_idx": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--action-type", type=str, required=True)
    parser.add_argument("--env-cfg", type=str, default=None)
    parser.add_argument("--task-name", type=str, default="robotwin_task")
    parser.add_argument("--prompt", type=str, default="pick up the object")
    parser.add_argument("--save_root", type=str, default="visualization/")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    args = parser.parse_args()

    model_cfg = {
        "config_name": args.config_name,
        "task_name": args.task_name,
        "action_type": args.action_type,
        "env_cfg": args.env_cfg,
        "prompt": args.prompt,
        "save_root": args.save_root,
    }
    if args.checkpoint_path is not None:
        model_cfg["checkpoint_path"] = args.checkpoint_path

    model = Model(model_cfg)

    fake_obs = _make_fake_obs(args.prompt)

    print("Testing update_obs ...")
    model.update_obs(fake_obs)

    print("Testing get_action ...")
    action = model.get_action()
    print("get_action output type:", type(action))
    print("get_action output:", action)

    print("Testing update_obs_batch ...")
    model.update_obs_batch([fake_obs, fake_obs])

    print("Testing get_action_batch ...")
    actions = model.get_action_batch(env_idx_list=[0, 1])
    print("get_action_batch output type:", type(actions))
    print("get_action_batch output:", actions)

    print("Testing reset ...")
    model.reset()

    print("Model interface test passed.")


if __name__ == "__main__":
    main()
