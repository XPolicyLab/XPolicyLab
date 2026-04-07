import os
import numpy as np
from dataclasses import dataclass
from typing import Any

from .openvla_oft.prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from .openvla_oft.experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

@dataclass
class InferenceConfig:
    pretrained_checkpoint: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = True
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 3
    center_crop: bool = True
    unnorm_key: str = ""
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32

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
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": observation["images"]["cam_high"],
            "cam_left_wrist": observation["images"]["cam_left_wrist"],
            "cam_right_wrist": observation["images"]["cam_right_wrist"],
        }
        prompt = observation.get("prompt", default_prompt)
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg is required when encoding raw environment observations.")

    images = {
        "cam_high": extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"]),
        "cam_left_wrist": extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"]),
        "cam_right_wrist": extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"]),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("prompt", default_prompt)
    return {
        "full_image": images["cam_high"],
        "left_wrist_image": images["cam_left_wrist"],
        "right_wrist_image": images["cam_right_wrist"],
        "state": state,
        "instruction": prompt,
    }

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = self.get_model(model_cfg)

        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        self.action_head = None
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if self.cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg, self.vla.llm_dim, PROPRIO_DIM
            )
        
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.default_prompt = model_cfg.get("prompt", self.task_name)
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg"]) if model_cfg.get("env_cfg") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]
        self.observation_window = encoded_obs_list

    def infer(self, observation: dict):
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=observation,
            task_label=observation["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions
    
    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        print("action_list", action_list[0][0]["left_arm_joint_state"])  # Debug print to check the structure of action_list

        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list

        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            action_chunk = self.infer(self.observation_window[batch_index])
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
    
    def reset(self):
        return
    # TODO
    def get_model(self, model_cfg: dict[str, Any]):
        config_args = {
            "pretrained_checkpoint": model_cfg["checkpoint_path"],
            "use_l1_regression": model_cfg.get("use_l1_regression", True),
            "use_diffusion": model_cfg.get("use_diffusion", False),
            "use_film": model_cfg.get("use_film", True),
            "use_proprio": model_cfg.get("use_proprio", True),
            "load_in_8bit": model_cfg.get("load_in_8bit", False),
            "load_in_4bit": model_cfg.get("load_in_4bit", False),
            "num_images_in_input": model_cfg.get("num_images_in_input", 3),
            "center_crop": model_cfg.get("center_crop", True),
            "unnorm_key": model_cfg["unnorm_key"],
            "num_open_loop_steps": model_cfg.get("num_open_loop_steps", NUM_ACTIONS_CHUNK),
            "lora_rank": model_cfg.get("lora_rank", 32),
        }

        cfg = InferenceConfig(**config_args)
        return cfg