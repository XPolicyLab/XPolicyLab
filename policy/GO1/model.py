import os
import sys

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGIBOT_DIR = os.path.join(_SCRIPT_DIR, "AgiBot-World")
_DEFAULT_GO1_MODEL_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../../models/GO-1"))
if _AGIBOT_DIR not in sys.path:
    sys.path.insert(0, _AGIBOT_DIR)

from evaluate.deploy import GO1Infer

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def _normalize_optional_path(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return os.path.abspath(value)


def _find_latest_checkpoint(run_dir):
    """Find the latest checkpoint-N subdirectory in a run directory."""
    if not os.path.isdir(run_dir):
        return None
    ckpts = [
        d for d in os.listdir(run_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(run_dir, d))
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(run_dir, ckpts[-1])


def _list_candidate_run_dirs(checkpoints_dir, run_basename):
    candidates = []
    latest_file = os.path.join(checkpoints_dir, f"{run_basename}.latest")
    if os.path.isfile(latest_file):
        with open(latest_file, "r", encoding="utf-8") as f:
            latest_dir = f.read().strip()
        if os.path.isdir(latest_dir):
            candidates.append(latest_dir)

    preferred_dir = os.path.join(checkpoints_dir, run_basename)
    if os.path.isdir(preferred_dir) and preferred_dir not in candidates:
        candidates.append(preferred_dir)

    prefix = f"{run_basename}-"
    if os.path.isdir(checkpoints_dir):
        run_dirs = []
        for entry in os.listdir(checkpoints_dir):
            path = os.path.join(checkpoints_dir, entry)
            if entry.startswith(prefix) and os.path.isdir(path):
                run_dirs.append(path)
        run_dirs.sort(key=os.path.getmtime, reverse=True)
        for path in run_dirs:
            if path not in candidates:
                candidates.append(path)

    return candidates


def _resolve_model_assets(model_cfg):
    checkpoints_dir = os.path.join(_SCRIPT_DIR, "checkpoints")
    model_path = _normalize_optional_path(model_cfg.get("model_path"))
    data_stats_path = _normalize_optional_path(model_cfg.get("data_stats_path"))
    run_dir = None

    if model_path:
        latest_ckpt = _find_latest_checkpoint(model_path)
        if latest_ckpt is not None:
            run_dir = model_path
            model_path = latest_ckpt
        elif os.path.basename(model_path).startswith("checkpoint-"):
            run_dir = os.path.dirname(model_path)
    else:
        bench_name = model_cfg.get("bench_name", "")
        ckpt_name = model_cfg.get("ckpt_name", "")
        action_type = model_cfg.get("action_type", "")
        env_cfg_type = model_cfg.get("env_cfg_type", "")
        expert_data_num = model_cfg.get("expert_data_num", "")
        seed = model_cfg.get("seed", "0")
        run_basename = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{expert_data_num}-{action_type}-{seed}"
        for candidate_run_dir in _list_candidate_run_dirs(checkpoints_dir, run_basename):
            candidate_model_path = _find_latest_checkpoint(candidate_run_dir)
            if candidate_model_path is not None:
                run_dir = candidate_run_dir
                model_path = candidate_model_path
                break

    if data_stats_path is None:
        for candidate_dir in (run_dir, model_path):
            if not candidate_dir:
                continue
            candidate = os.path.join(candidate_dir, "dataset_stats.json")
            if os.path.isfile(candidate):
                data_stats_path = candidate
                break

    if model_path is None:
        bundled_model_path = os.path.join(_SCRIPT_DIR, "AgiBot-World", "go1", "models", "GO-1")
        if os.path.isdir(bundled_model_path):
            model_path = bundled_model_path
        elif os.path.isdir(_DEFAULT_GO1_MODEL_PATH):
            model_path = _DEFAULT_GO1_MODEL_PATH
        else:
            raise FileNotFoundError(
                "No finetuned checkpoint found and default GO-1 model path is missing: "
                f"{_DEFAULT_GO1_MODEL_PATH}. Set MODEL_PATH explicitly or train first."
            )

    return model_path, data_stats_path


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.default_prompt = model_cfg.get("prompt") or "Do your job."

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        assert len(self.robot_action_dim_info["arm_dim"]) == len(self.robot_action_dim_info["ee_dim"]), \
            "Arm and EE action dimensions must match"

        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 25))
        self.ctrl_freq = int(model_cfg.get("ctrl_freq", 25))

        self.model = self._load_model(model_cfg)

        self._obs_buffer = None
        self._obs_buffer_batch = {}
        self._latest_env_idx_list = [0]

        print(
            f"[GO1 Model] Initialized | action_type={self.action_type} | "
            f"action_chunk_size={self.action_chunk_size} | ctrl_freq={self.ctrl_freq}"
        )

    def _load_model(self, model_cfg):
        model_path, data_stats_path = _resolve_model_assets(model_cfg)

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
            encoded = _encode_obs(
                obs, self.action_type, self.robot_action_dim_info, self.default_prompt, self.ctrl_freq
            )
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
    state = state.astype(np.float32).reshape(1, -1)

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
    """Extract an HWC uint8 RGB image from the observation."""
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

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        return img.astype(np.uint8)

    return None
