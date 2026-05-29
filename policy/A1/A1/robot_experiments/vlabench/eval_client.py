import collections
import dataclasses
import logging
import math
import pathlib
import json
import imageio
import numpy as np
from typing import Optional, Union
from pathlib import Path
import os
import draccus
from dataclasses import dataclass,replace
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import requests
import base64
from PIL import Image
from io import BytesIO
# import tyro
# install vlabench in venv first
from a1.data import build_mm_preprocessor
from a1.data.collator import MMCollatorForAction
from robot_experiments.utils import *
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler
from VLABench.envs import load_env
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import Policy 
from VLABench.robots import *
from VLABench.tasks import *

VLABENCH_DUMMY_ACTION = [0.0] * 6 + [0.04, 0.04]
VLABENCH_ENV_RESOLUTION = 480  # resolution used to render training data

from robot_experiments.robot_utils import (
    DATE_TIME,
    # get_action,
    get_image_resize_size,
    # get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


from robot_experiments.vla_utils import resize_image_for_policy,prepare_images_for_vla

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class GenerateConfig:
    #################################################################################################################
    # VLABench environment-specific parameters
    #################################################################################################################
    tasks: str="select_toy select_fruit select_painting select_mahjong"
    eval_track: str = None
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    n_episode: int = 50  # Number of rollouts per task
    episode_config_path:str = "robot_experiments/vlabench/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json"
    intention_score_threshold: float=0.2
    #################################################################################################################
    # Utils
    #################################################################################################################
    metrics: str = "success_rate intention_score progress_score"
    save_dir: str = "outputs/vlabench/A1/track_1"  # Path to save videos
    visulization: bool = True
    seed: int = 7  # Random Seed (for reproducibility

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "a1"                    # Model family
    # pretrained_checkpoint: Union[str, Path] = "/mnt/data3/zhangjian/a1/libero_spatial_qwen3-4b_l1_regression_wrist_proprio_ft_ah_lora_r8_llm_bs108/step13000-unsharded"     # Pretrained checkpoint path
    url: str = "http://localhost:8000"
    fsdp: bool = False                               # Whether to use FSDP for model loading
    # llm: str = "qwen2_7b"                            # LLM model name (e.g., "qwen2_7b", "olmoe", "qwen2_72b")
    # vision_backbone: str = "openai"                  # Vision backbone name (e.g., "openai", "siglip", "dinov2_large_336")
    sequence_length: int = 768                       # Sequence length for the model

    use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = True                       # If True, uses continuous action head with diffusion modeling objective (DDIM)

    action_head_diffusion_inference_steps: int = 30
    llm_causal_attention: bool = False                # default: False, as openvla-oft's parallel decoding, If True, uses causal attention in the transformer model

    # num_diffusion_steps: int = 10                    # (When `diffusion==True`) Number of diffusion steps for inference
    # use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_wrist_image: bool = True                     # Whether to use wrist image in input
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

class A1(Policy):
    def __init__(self, url, cfg,replan_steps=5):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.cfg = cfg
        self.replan_steps=replan_steps
        self.action_plan = collections.deque(maxlen=replan_steps)
    
    def reset(self):
        self.action_plan.clear()
    
    def encode_image_to_base64(self, image_data, idx=0):
        image = Image.fromarray(image_data)
            
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def prepare_observation(self, obs):
        """Prepare observation for policy input."""
        # Get preprocessed images
        _, _, image, image_wrist = obs["rgb"]

        # Resize images to size expected by model
        state = obs["ee_state"]
        pos, quat, gripper_state = state[:3], state[3:7], state[-1]
        ee_euler = quaternion_to_euler(quat)
        pos -= np.array([0, -0.4, 0.78])
        state = np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)])
            

        # Prepare observations dict
        observation = {
            "full_image": image,
            "wrist_image": image_wrist,
            "state": state
        }

        return observation  # Return both processed observation and original image for replay

    def predict(self, obs, **kwargs):
        if len(self.action_plan) == 0:
            observation = self.prepare_observation(obs)
            payload = {
                "instruction": obs["instruction"],
                "images": [
                    self.encode_image_to_base64(observation["full_image"],0),
                    self.encode_image_to_base64(observation["wrist_image"],1),
                ],
                "proprio_data": [observation["state"].tolist()],
            }
            response = self.session.post(f"{self.url}/inference", json=payload)
            response.raise_for_status()
            actions = response.json()["predicted_actions"]
            
            actions = np.array(actions)
            # actions[:, 3] = observation["state"][3]
            actions[:, 4] = observation["state"][4]
            actions = actions.tolist()
            self.action_plan.extend(actions[:self.replan_steps])
        action = self.action_plan.popleft()
        target_pos, target_euler, gripper = action[:3], action[3:6], action[6]
        if obs['ee_state'][6] >= 0.5 and gripper < 0.5:
            target_pos[2] -=0.01

        if gripper >= 0.5:
            gripper_state = np.ones(2)*0.04
        else:
            gripper_state = np.zeros(2)
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state
    
    @property
    def name(self):
        return "A1"

@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    # VLABench environment-specific parameters
    if cfg.eval_track is not None:
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", cfg.eval_track), "r") as f:
            episode_configs = json.load(f)
            tasks = list(episode_configs.keys())
    else:
        tasks = cfg.tasks.split(" ")
    metrics = cfg.metrics.split(" ")
    assert isinstance(tasks, list)

    with open(cfg.episode_config_path, "r")  as f:
        episode_configs=json.load(f)
    
    # Set random seed
    set_seed_everywhere(cfg.seed)

    model = A1(cfg.url,cfg,cfg.num_open_loop_steps)
    

    
    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=cfg.n_episode,
        episode_config=episode_configs,
        max_substeps=5,   
        save_dir=cfg.save_dir,
        visulization=cfg.visulization,
        metrics=metrics,
        intention_score_threshold=cfg.intention_score_threshold
    )
    

    evaluator.evaluate(model)
    
if __name__ == "__main__":
    main()