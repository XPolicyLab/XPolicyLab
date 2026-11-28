#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_OPENPI_SRC = _POLICY_DIR / "openpi" / "src"
if str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_RL_TRAIN_MODES = ("ppo", "grpo")
# Fields produced by the XPolicyLab/OpenPI transforms before native replay packing.
_MODEL_INPUT_KEYS = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask")


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            extra_digits = max(0, len(str(candidate_step)) - len(str(desired_step)))
            scaled_step = desired_step * 10**extra_digits
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    RL_MODES = ("eval", "ppo", "grpo")
    ACTION_TYPE = "joint"

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.task_name = model_cfg.get("task_name")
        self.action_type = model_cfg.get("action_type", self.ACTION_TYPE)
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        if model_cfg.get("rl_backend") not in (None, "rlinf_v0_4"):
            raise ValueError("rl_backend must be rlinf_v0_4 or unset for ordinary OpenPI evaluation.")
        if model_cfg.get("rl_backend") == "rlinf_v0_4" or any(mode in self.get_rl_modes() for mode in _RL_TRAIN_MODES):
            self.policy = self.get_rl_model(model_cfg=model_cfg)
        else:
            self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def get_rl_model(self, model_cfg: dict[str, Any]):
        """Load the PyTorch RL policy and the same transforms ``create_trained_policy`` uses."""
        import openpi

        from .rl import Pi05RLConfig, load_pi05_rl_policy

        if _OPENPI_SRC not in Path(openpi.__file__).resolve().parents:
            raise ImportError(
                f"Pi_05 RL needs the vendored openpi at {_OPENPI_SRC}, but openpi was imported from "
                f"{openpi.__file__}. Import XPolicyLab.policy.Pi_05 before any other openpi package."
            )
        model_root = _resolve_pi05_model_root(model_cfg)
        assets_root = model_root if model_root.is_dir() else model_root.parent

        train_config = _config.get_config(model_cfg.get("train_config_name", "pi05_aloha"))
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats_path = model_cfg.get("norm_stats_path") or assets_root / "assets" / str(model_cfg.get("repo_id", "1118"))
        norm_stats = _normalize.load(Path(norm_stats_path))
        self._input_transform = _transforms.compose(
            [
                _transforms.InjectDefaultPrompt(model_cfg.get("prompt")),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

        rl_head = dict(model_cfg.get("rl_head") or {})
        rl_head.setdefault("action_chunk", int(model_cfg.get("num_action_chunks", train_config.model.action_horizon)))
        rl_head.setdefault("add_value_head", bool(model_cfg.get("add_value_head", False)))
        if rl_head["add_value_head"]:
            rl_head.setdefault("value_after_vlm", True)
        if self.robot_action_dim_info is not None:
            rl_head.setdefault(
                "action_env_dim",
                sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"]),
            )
        rl_head.setdefault("config_name", model_cfg.get("train_config_name", "pi05_aloha"))
        self.rl_config = Pi05RLConfig.from_dict(rl_head)
        print(f"[Pi_05] RL policy | modes={self.get_rl_modes()} | {self.rl_config}")
        return load_pi05_rl_policy(
            train_config.model, str(model_root), self.rl_config,
            output_transform=self._output_transform, precision=model_cfg.get("precision"),
            embed_dtype=model_cfg.get("embed_dtype"),
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        if hasattr(self, "rl_config"):
            return self._get_action_batch_rl(env_idx_list)
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()

    def predict_action_batch_rl(self, obs_list, mode="train", compute_values=True, **kwargs):
        if not hasattr(self, "rl_config"):
            raise NotImplementedError(self._error_msg("Pi_05 RL policy is not loaded; set rl.modes to include ppo or grpo."))
        encoded_obs_list = [encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list]
        inputs = self._transform_inputs(encoded_obs_list)
        actions, extras = self.policy.sample_preprocessed(
            inputs, mode=mode, compute_values=compute_values, **kwargs
        )
        return actions.detach().float().cpu().numpy(), extras

    def compute_logprob_value(self, forward_inputs, compute_values=False, **kwargs):
        if not hasattr(self, "rl_config"):
            raise NotImplementedError("Pi_05 RL backend is not loaded.")
        return self.policy.default_forward(forward_inputs, compute_values=compute_values, **kwargs)

    def _get_action_batch_rl(self, env_idx_list):
        obs_list = [slice_stacked_obs(self.observation_window, batch_index) for batch_index, _ in enumerate(env_idx_list)]
        inputs = self._transform_inputs(obs_list)
        actions, _ = self.policy.sample_preprocessed(inputs, mode="eval", compute_values=False)
        actions = actions.detach().float().cpu().numpy()
        if self.robot_action_dim_info is None:
            return list(actions)
        return [
            unpack_robot_state(chunk, self.action_type, self.robot_action_dim_info, source_type="obs")
            for chunk in actions
        ]

    def _transform_inputs(self, encoded_obs_list):
        """Run the openpi input transforms per sample; return flat batched tensors on the policy device."""
        device = next(self.policy.parameters()).device
        samples = [_flatten_inputs(self._input_transform(dict(obs))) for obs in encoded_obs_list]
        inputs = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        return {
            key: torch.from_numpy(value.astype(np.float32) if value.dtype == np.float64 else value).to(device)
            for key, value in inputs.items()
        }


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def _flatten_inputs(inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten transformed model inputs to ``{"image/<cam>": array, ...}`` for RL rollout buffers."""
    flat = {}
    for key in _MODEL_INPUT_KEYS:
        if key not in inputs:
            continue
        value = inputs[key]
        if isinstance(value, dict):
            flat.update({f"{key}/{name}": np.asarray(item) for name, item in value.items()})
        else:
            flat[key] = np.asarray(value)
    return flat


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


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
