import os
import sys

import cv2
import numpy as np

# Add AgiBot-World to sys.path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGIBOT_DIR = os.path.join(_SCRIPT_DIR, "AgiBot-World")
if _AGIBOT_DIR not in sys.path:
    sys.path.insert(0, _AGIBOT_DIR)

from evaluate.deploy import GO1Infer

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def _find_latest_checkpoint(run_dir):
    """Find the latest checkpoint-N subdirectory in a run directory."""
    if not os.path.isdir(run_dir):
        return None
    ckpts = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, d))]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(run_dir, ckpts[-1])


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.task_name = model_cfg.get("task_name", "")
        self.default_prompt = model_cfg.get("prompt", self.task_name)

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), \
            "Arm and EE action dimensions must match"

        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 30))
        self.ctrl_freq = int(model_cfg.get("ctrl_freq", 30))

        self.model = self._load_model(model_cfg)

        self._obs_buffer = None
        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]

        print(f"[GO1 Model] Initialized | action_type={self.action_type} | "
              f"action_chunk_size={self.action_chunk_size} | ctrl_freq={self.ctrl_freq}")

    def _load_model(self, model_cfg):
        model_path = model_cfg.get("model_path", None) or None
        run_dir = None

        if model_path is None:
            # Try trained checkpoint first, then fall back to pretrained model
            task_name = model_cfg.get("task_name", "")
            seed = model_cfg.get("seed", "0")
            runname = f"{task_name}-go1-seed{seed}"
            run_dir = os.path.join(_SCRIPT_DIR, "checkpoints", runname)
            if os.path.isdir(run_dir):
                model_path = _find_latest_checkpoint(run_dir)
            if model_path is None:
                model_path = os.path.join(_SCRIPT_DIR, "AgiBot-World", "go1", "models", "GO-1")
                if not os.path.isdir(model_path):
                    model_path = "/mnt/pfs/pg4hw0/qiwei/models/GO-1"
        else:
            # If model_path points to a run dir (not a checkpoint subdir), find latest
            latest = _find_latest_checkpoint(model_path)
            if latest is not None:
                run_dir = model_path
                model_path = latest

        data_stats_path = model_cfg.get("data_stats_path", None) or None
        if data_stats_path is None:
            # Look for dataset_stats.json in run dir, then in model_path
            for candidate_dir in [run_dir, model_path]:
                if candidate_dir is None:
                    continue
                candidate = os.path.join(candidate_dir, "dataset_stats.json")
                if os.path.exists(candidate):
                    data_stats_path = candidate
                    break

        print(f"[GO1 Model] Loading model from: {model_path}")
        if data_stats_path:
            print(f"[GO1 Model] Loading data stats from: {data_stats_path}")

        return GO1Infer(model_path=model_path, data_stats_path=data_stats_path)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            encoded = _encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt, self.ctrl_freq)
            self._obs_buffer_batch[env_idx] = encoded

    def get_action(self):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])
        return action_list[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list

        action_batch = []
        for env_idx in env_idx_list:
            if env_idx not in self._obs_buffer_batch:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}. Call update_obs first.")

            payload = self._obs_buffer_batch[env_idx]
            actions = self.model.inference(payload)

            action_steps = []
            for step_idx in range(actions.shape[0]):
                step_action = actions[step_idx]
                action_dict = unpack_robot_state(
                    step_action,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
                action_steps.append(action_dict)

            action_batch.append(action_steps)

        return action_batch

    def reset(self):
        self._obs_buffer = None
        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]
        print("[GO1 Model] Reset")


def _encode_obs(observation, action_type, robot_action_dim_info, default_prompt, ctrl_freq):
    """Convert XPolicyLab observation dict to GO1 inference payload."""
    vision = observation.get("vision", {})

    top_img = _extract_and_prepare_image(vision, ["cam_head"])
    left_img = _extract_and_prepare_image(vision, ["cam_left_wrist"])
    right_img = _extract_and_prepare_image(vision, ["cam_right_wrist"])

    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")
    state = state.astype(np.float32).reshape(1, -1)  # (state_dim,) → (1, state_dim)

    instruction = observation.get("instruction", observation.get("instructions", default_prompt))
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else default_prompt

    payload = {
        "top": top_img,
        "instruction": instruction,
        "state": state,
        "ctrl_freqs": np.array([ctrl_freq], dtype=np.float32),
    }

    if right_img is not None:
        payload["right"] = right_img
    if left_img is not None:
        payload["left"] = left_img

    return payload


def _extract_and_prepare_image(vision, candidate_names):
    """Extract image from vision dict, decode if needed, convert BGR->RGB, return HWC uint8 numpy."""
    for name in candidate_names:
        if name not in vision:
            continue
        cam_data = vision[name]
        if isinstance(cam_data, dict):
            img = cam_data.get("color", cam_data.get("rgb", None))
        else:
            img = cam_data

        if img is None:
            continue

        img = np.asarray(img)

        if img.ndim == 1 and img.dtype == np.uint8:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if img is None:
                continue

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.uint8)

    return None
