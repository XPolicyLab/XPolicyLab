"""
LEGACY (RMBench joint_action.vector API). Use policy/Mem_0/model.py + deploy.py for XPolicyLab.

Mem_0 deployment entry.

Responsibilities:
- Preprocess observation: camera to PIL, reorder qpos layout, normalize arms.
- Load low-level policy checkpoint and state/action stats.
- Postprocess model actions: denormalize arms, reorder for task_env, clip grippers.
- Run eval loop and manage MemoryBank reset.
"""

import os
import json
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint
from PIL import Image

from .scripts.tools_for_deploy.image_utils import to_pil
from .scripts.tools_for_deploy.layout_utils import env_to_model_layout, model_to_env_layout
from .scripts.tools_for_deploy.normlization import denormalize_arms, load_stats, normalize_arms
from .source.agent import MemoryMattersAgent

# Normalization method
NORM_WAY = "minmax"  # options: "quantile", "minmax", "meanstd"

# Runtime knobs filled by get_model, read inside encode_obs
_RUNTIME_SETTINGS: Dict[str, object] = {
    "camera_key": "head_camera",
    "image_size": (224, 224),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Stats for normalization/denormalization
_STATS: Dict[str, Optional[np.ndarray]] = {
    "state_mean": None,
    "state_std": None,
    "action_mean": None,
    "action_std": None,
    "state_min": None,
    "state_max": None,
    "action_min": None,
    "action_max": None,
    "state_q01": None,
    "state_q99": None,
    "action_q01": None,
    "action_q99": None,
}

# 0: original 16 dims; 1: padded to 16 dims
PADDING_HAPPEND = 0

def _extract_state(observation: dict) -> np.ndarray:
    """Extract 16-d joint vector; pad or truncate to 16 dims."""
    joint_vec = observation.get("joint_action", {}).get("vector")
    if joint_vec is None:
        cprint("[deploy] joint_action.vector missing, using zeros", "yellow")
        return np.zeros((16,), dtype=np.float32)
    joint_arr = np.asarray(joint_vec, dtype=np.float32).reshape(-1)
    
    if joint_arr.size < 16:
        padded = np.zeros((16,), dtype=np.float32)
        padded[ : joint_arr.size // 2 - 1] = joint_arr[ : joint_arr.size // 2 - 1]
        padded[7] = joint_arr[joint_arr.size // 2 - 1] 
        padded[8 : 8 + joint_arr.size // 2 - 1] = joint_arr[joint_arr.size // 2 : -1]
        padded[15] = joint_arr[joint_arr.size - 1] 
        
        joint_arr = padded
        
        global PADDING_HAPPEND
        if PADDING_HAPPEND == 0:
            cprint("[deploy] joint vector padded to 16 dims", "yellow")
        PADDING_HAPPEND = 1 # indicate padding happened
        
    elif joint_arr.size > 16:
        joint_arr = joint_arr[:16]
        cprint("[deploy] joint vector truncated to 16 dims", "yellow")
        
    return joint_arr


def _normalize_state(state_vec: np.ndarray) -> np.ndarray:
    """Normalize state (model layout)"""
    if NORM_WAY == "quantile":
        q01 = _STATS.get("state_q01")
        q99 = _STATS.get("state_q99")
        if q01 is not None and q99 is not None:
            return normalize_arms(
                state_vec,
                None,
                None,
                arm_dims=state_vec.shape[-1],
                quantile=True,
                q01=q01,
                q99=q99,
            )
    
    if NORM_WAY == "minmax":
        min = _STATS.get("state_min")
        max = _STATS.get("state_max")
        if min is not None and max is not None:
            return normalize_arms(state_vec, None, None, min, max, arm_dims=14)

    if NORM_WAY == "meanstd":
        mean = _STATS.get("state_mean")
        std = _STATS.get("state_std")
        if mean is not None and std is not None:
            return normalize_arms(state_vec, mean, std, arm_dims=14)
    
    return state_vec


def _denormalize_action(action_vec: np.ndarray) -> np.ndarray:
    """Denormalize actions"""
    if NORM_WAY == "quantile":
        q01 = _STATS.get("action_q01")
        q99 = _STATS.get("action_q99")
        if q01 is not None and q99 is not None:
            return denormalize_arms(
                action_vec,
                None,
                None,
                arm_dims=action_vec.shape[-1],
                quantile=True,
                q01=q01,
                q99=q99,
            )
    
    if NORM_WAY == "minmax":
        min = _STATS.get("action_min")
        max = _STATS.get("action_max")
        if min is not None and max is not None:
            return denormalize_arms(action_vec, None, None, min, max, arm_dims=14)

    if NORM_WAY == "meanstd":
        mean = _STATS.get("action_mean")
        std = _STATS.get("action_std")
        if mean is not None and std is not None:
            return denormalize_arms(action_vec, mean, std, arm_dims=14)
        
    return action_vec


def _load_stats(stats_path: str) -> None:
    """
    Load state/action stats from JSON.
    Supports mean/std or quantile stats (q01/q99) or min/max
    """
    if not stats_path:
        cprint("[deploy] stats path not provided; skipping stats load", "yellow")
        return
    try:
        stats = load_stats(stats_path)
        _STATS.update(stats)
        has_quantile = (
            _STATS["state_q01"] is not None
            and _STATS["state_q99"] is not None
            and _STATS["action_q01"] is not None
            and _STATS["action_q99"] is not None
        )
        has_mean_std = _STATS["state_mean"] is not None and _STATS["action_mean"] is not None
        has_min_max = _STATS["state_min"] is not None and _STATS["action_min"] is not None

        cprint(f"[deploy] loaded stats from {stats_path}", "cyan")
        if has_quantile:
            cprint(
                f"[deploy] quantile state q01 head={_STATS['state_q01'][:3]}; q99 head={_STATS['state_q99'][:3]}",
                "cyan",
            )
            
        if has_mean_std:
            cprint(
                f"[deploy] state_std head={_STATS['state_std'][:3] if _STATS['state_std'] is not None else 'None'}",
                "cyan",
            )
            
        if has_min_max:
            cprint(
                f"[deploy] min/max state min head={_STATS['state_min'][:3]}; max head={_STATS['state_max'][:3]}",
                "cyan",
            )
            
    except Exception as exc:
        cprint(f"[deploy] failed to load stats ({stats_path}): {exc}", "red")


def _postprocess_action_chunk(actions_model: np.ndarray) -> np.ndarray:
    """
    Denormalize arms, reorder layout, clip grippers, and return env-ready chunk.
    actions_model: (T, 16) in model layout (normalized), optionally with leading batch/horizon dims.
    """
    chunk = np.array(actions_model, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    flat_actions = chunk.reshape(-1, chunk.shape[-1])

    processed = []
    for step in flat_actions:
        denorm = _denormalize_action(step)
        env_order = model_to_env_layout(denorm)
        if PADDING_HAPPEND == 1:
            env_order = np.concatenate((env_order[0 : 6], env_order[7 : 8], env_order[8 : 14], env_order[15 : 16]), axis = 0)
        processed.append(env_order)
        
    processed = np.stack(processed, axis=0)
    return processed

def encode_obs(observation: dict) -> Dict[str, object]:
    """
    Post-process raw observation into model-ready payload:
    - pick camera
    - reorder state to model layout
    - normalize arm dims with stats
    """
    cam_key: str = _RUNTIME_SETTINGS.get("camera_key", "head_camera")
    target_size: Tuple[int, int] = _RUNTIME_SETTINGS.get("image_size", (224, 224))

    image_array = None
    obs_block = observation.get("observation", {}) 
    image_array = obs_block[cam_key]["rgb"]
    if image_array is None:
        raise KeyError("No RGB camera found in observation payload")

    pil_image = to_pil(np.array(image_array), target_size)
    raw_state_env = _extract_state(observation)
    model_state = env_to_model_layout(raw_state_env)
    norm_stats = _normalize_state(model_state)
    return {"image": pil_image, "state": norm_stats.reshape(1, -1), "instruction": observation.get("instruction", "")}


def get_model(usr_args: dict) -> MemoryMattersAgent:
    """
    Build MemoryMatters agent from deploy_policy.yml + overrides.
    """
    device_str = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    _RUNTIME_SETTINGS["camera_key"] = usr_args.get("camera_key", "head_camera")
    _RUNTIME_SETTINGS["image_size"] = tuple(usr_args.get("image_size", (224, 224)))
    _RUNTIME_SETTINGS["device"] = device

    ckpt_path = usr_args.get("execution_ckpt", "")
    stats_path = usr_args.get("state_stats_path", "")
    _load_stats(stats_path)
    cfg = OmegaConf.create(usr_args)

    cprint(f"[deploy] device: {device}; camera: {_RUNTIME_SETTINGS['camera_key']}", "cyan")
    cprint(f"[deploy] ckpt: {ckpt_path}", "cyan")
    agent = MemoryMattersAgent(cfg, ckpt_path=ckpt_path, device=device)
    return agent


def eval(TASK_ENV, model: MemoryMattersAgent, observation: dict):
    """
    Execute one chunk of actions in the environment.
    """
    if model.is_init == 0:
        model.is_init = 1
        image = TASK_ENV.now_obs["observation"]["head_camera"]["rgb"]
        Image.fromarray (image).save ("./_tmp_visual/init.png")
        
        # --- For Mn Tasks: use planner to get the first subtask instruction
        if model.task_type == "Mn":
            model.init_high_with_image ()
        # --- The End
        
        # --- For M1 Tasks: comment out the 2 lines above and instead set instruction directly:
        if model.task_type == "M1":
            model.instruction = model.config.get ("global_task", "")
            model._set_video_ffmpeg()
        # --- The End
        
    instruction = model.instruction
    observation["instruction"] = instruction
    encoded_obs = encode_obs(observation)
    
    if model.action_count == 0:
        model.update_obs (encoded_obs)
    result = model.get_action() 
    
    if result is None or len(result) == 0:
        cprint("[deploy] no actions produced", "red")
        raise SystemExit("Empty actions from model; aborting eval.")
    else:
        model.accumulate_actions_chunk(result["normalized_actions"])  # updates internal time-indexed history
        smoothed_model_actions = model.get_smoothed_actions(model.iter, model.action_strip)
        actions = _postprocess_action_chunk(smoothed_model_actions)
    
    model.ffmpeg.stdin.write(TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes())
    
    # Execute only `action_strip` smoothed steps for current eval
    steps_to_run = min(model.action_strip, actions.shape[0])
    for idx in range(steps_to_run):
        action = actions[idx]
        TASK_ENV.take_action(action, action_type="qpos")
        
        observation = TASK_ENV.get_obs()
        observation["instruction"] = instruction
        encoded_obs = encode_obs(observation)
        
        sub_end_flag = model.update_obs(encoded_obs)
        
        # --- For Mn Tasks
        if model.task_type == "Mn":
            if sub_end_flag == 1:
                cprint (f"[deploy] subtask end signal += 1 on [{model.iter}]", "yellow") 
            if model.end_signal_count >= model.threshold:
                image = TASK_ENV.now_obs["observation"]["head_camera"]["rgb"]
                Image.fromarray (image).save (f"./_tmp_visual/image_{model.stage}.png")
                break
        # --- The End
        
    # Advance iteration by number of executed steps
    model.iter += steps_to_run
    
    # --- For Mn Tasks
    if model.task_type == "Mn":
        if model.end_signal_count >= model.threshold:
            model.ffmpeg.stdin.write (TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes ())
            model._del_video_ffmpeg ()
            cprint (f"[deploy] subtask end detected; moving to stage {model.stage + 1}, action_count {model.action_count}", "green")
            model.update_high_observation ()
            
            model.end_signal_count = 0
            model.action_count = 0
            model.executor.memory_bank.reset ()
            
            model.ffmpeg.stdin.write (TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes ())
            instruction = model.instruction
    # --- The End

def reset_model(model: Optional[MemoryMattersAgent] = None):
    """Clear memory at the beginning of every episode."""
    if model is not None:
        model.reset()
