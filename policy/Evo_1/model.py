"""Evo-1's released RoboTwin policy behind the XPolicyLab observation contract."""

from pathlib import Path
import json
import os
import sys

import cv2
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info


POLICY_DIR = Path(__file__).resolve().parent
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = (
    "left_arm_joint_state", "left_ee_joint_state",
    "right_arm_joint_state", "right_ee_joint_state",
)


def smooth_actions(actions, kernel_size=9):
    """Upstream RoboTwin Gaussian smoothing, including boundary renormalization."""
    if kernel_size == 1 or len(actions) <= kernel_size:
        return actions
    half = kernel_size // 2
    sigma = half / 2.0
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - half) ** 2 / sigma ** 2)
    kernel /= kernel.sum()
    smoothed = np.copy(actions)
    for t in range(len(actions)):
        start, end = max(0, t - half), min(len(actions), t + half + 1)
        weights = kernel[half - (t - start):half + (end - t)]
        weights = weights / weights.sum()
        smoothed[t] = (actions[start:end] * weights[:, None]).sum(axis=0)
    return smoothed


def load_backend(checkpoint_dir, model_cfg):
    """Reuse the pinned upstream network and normalizer without its JSON transport."""
    source = Path(model_cfg.get("upstream_dir") or os.environ.get("EVO1_SOURCE_DIR")
                  or POLICY_DIR / "upstream").expanduser()
    if not source.is_absolute():
        source = POLICY_DIR / source
    source = source.resolve() / "Evo_1"
    if not (source / "scripts" / "Evo1_server.py").is_file():
        raise FileNotFoundError(f"Evo-1 source missing at {source}; run install.sh first.")
    sys.path.insert(0, str(source))
    from config import EvoConfig
    from scripts.Evo1 import EVO1
    from scripts.Evo1_server import Normalizer

    if not torch.cuda.is_available():
        raise RuntimeError("Evo-1 inference requires a CUDA GPU and its policy environment.")
    config = json.loads((checkpoint_dir / "config.json").read_text())
    config.update(device="cuda", finetune_vlm=False, finetune_action_head=False,
                  num_inference_timesteps=int(model_cfg.get("num_inference_timesteps", 50)))
    if model_cfg.get("vlm_path"):
        vlm_path = Path(model_cfg["vlm_path"]).expanduser()
        config["vlm_name"] = str((vlm_path if vlm_path.is_absolute() else POLICY_DIR / vlm_path).resolve())
    if config["num_inference_timesteps"] < 1:
        raise ValueError("num_inference_timesteps must be positive.")
    for key, expected in (("per_action_dim", 24), ("state_dim", 24), ("image_size", 448)):
        if config.get(key) != expected:
            raise ValueError(f"RoboTwin checkpoint requires {key}={expected}, got {config.get(key)}")
    model = EVO1(EvoConfig.from_dict(config)).eval()
    ds_weights = checkpoint_dir / "mp_rank_00_model_states.pt"
    if ds_weights.is_file():
        payload = torch.load(ds_weights, map_location="cpu", weights_only=False)
        state_dict = payload["module"]
    else:
        payload = torch.load(checkpoint_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        state_dict = payload["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")
    normalizer = Normalizer(str(checkpoint_dir / "norm_stats.json"),
                            normalization_type=config.get("normalization_type", "bounds"))
    return model, normalizer


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if model_cfg.get("bench_name") != "RoboTwin" or model_cfg.get("action_type") != "joint":
            raise ValueError("Evo_1 currently supports bench_name=RoboTwin, action_type=joint.")
        self.env_cfg_type = model_cfg.get("env_cfg_type")
        if self.env_cfg_type not in ("arx_x5", "aloha_agilex"):
            raise ValueError("Use RoboTwin arx_x5 (or its aloha_agilex configuration).")
        dims = get_robot_action_dim_info(self.env_cfg_type)
        if len(dims["arm_dim"]) != 2 or len(dims["ee_dim"]) != 2:
            raise ValueError("This checkpoint requires a bimanual robot.")
        self.state_dims = tuple(v for pair in zip(dims["arm_dim"], dims["ee_dim"]) for v in pair)
        self.action_dim = sum(self.state_dims)
        self.action_horizon = int(model_cfg.get("action_horizon", 37))
        self.smoothing_kernel = int(model_cfg.get("smoothing_kernel", 9))
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive.")
        if self.smoothing_kernel < 1 or self.smoothing_kernel % 2 != 1:
            raise ValueError("smoothing_kernel must be a positive odd integer.")
        self.arm_key = str(model_cfg.get("arm_key") or "aloha_joint")
        checkpoint_dir = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints")
        self.stats = json.loads((checkpoint_dir / "norm_stats.json").read_text())
        if self.arm_key not in self.stats:
            raise ValueError(f"Missing arm_key={self.arm_key!r} in norm_stats.json")
        self.default_dataset_key = self._dataset_key(model_cfg.get("task_name"))
        self._validate_stats(self.default_dataset_key)
        self.current_dataset_key = self.default_dataset_key
        if model_cfg.get("seed") is not None:
            torch.manual_seed(int(model_cfg["seed"]))
        self.model, self.normalizer = load_backend(checkpoint_dir, model_cfg)
        self.device = next(self.model.parameters()).device
        self.reset()

    def _dataset_key(self, task_name):
        explicit = self.model_cfg.get("dataset_key")
        if explicit:
            return str(explicit)
        if not task_name:
            raise ValueError("task_name is required to select per-task normalization.")
        suffix = str(self.model_cfg.get("dataset_key_suffix") or "")
        if suffix not in ("", "_clean", "_rand"):
            raise ValueError("dataset_key_suffix must be empty, _clean, or _rand.")
        return f"robotwin_{task_name}{suffix}"

    def _validate_stats(self, dataset_key):
        group = self.stats[self.arm_key]
        if dataset_key not in group:
            raise ValueError(f"Missing {self.arm_key}/{dataset_key} normalization; "
                             "select the checkpoint's exact task key (no aggregate fallback).")
        for feature in ("observation.state", "action"):
            for metric, values in group[dataset_key][feature].items():
                array = np.asarray(values)
                if array.ndim != 1 or len(array) not in (self.action_dim, 24):
                    raise ValueError(f"Invalid {feature}.{metric} dimension for {dataset_key}")
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite {feature}.{metric} for {dataset_key}")

    def update_obs(self, obs):
        state_parts = []
        for key, dim in zip(STATE_KEYS, self.state_dims):
            value = np.asarray(obs["state"][key], dtype=np.float32).reshape(-1)
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain {dim} finite values.")
            state_parts.append(value)
        images = []
        for camera in CAMERAS:
            rgb = np.asarray(obs["vision"][camera]["color"])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{camera}.color must be a decoded HWC uint8 RGB image.")
            images.append(rgb)
        prompt = obs.get("instruction", obs.get("instructions"))
        if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise ValueError("instruction must be a string (or a one-element instructions list).")
        task = obs.get("task_name") or (obs.get("additional_info") or {}).get("task_name")
        dataset_key = self._dataset_key(task) if task else self.current_dataset_key
        self._validate_stats(dataset_key)
        self._observation = (images, np.concatenate(state_parts), prompt, dataset_key)

    def get_action(self):
        if self._observation is None:
            raise RuntimeError("Call update_obs after reset and before get_action.")
        images, state, prompt, dataset_key = self._observation
        # Resize only at inference time, not on each intermediate control-step update.
        tensors = []
        for rgb in images:
            resized = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div_(255))
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self.normalizer.normalize_state(state, self.arm_key, dataset_key).float()
        mask = torch.zeros((1, 24), dtype=torch.int32, device=self.device)
        mask[:, :self.action_dim] = 1
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                                                   dtype=torch.bfloat16,
                                                   enabled=self.device.type == "cuda"):
            action = self.model.run_inference(
                images=[image.to(self.device) for image in tensors],
                image_mask=torch.ones(3, dtype=torch.int32, device=self.device),
                prompt=prompt, state_input=state, action_mask=mask,
            )
            action = action.reshape(-1, 24)
            action = self.normalizer.denormalize_action(action, self.arm_key, dataset_key)
        # The old JSON client reconstructed float64 before smoothing; preserve that order.
        actions = action.float().cpu().numpy().astype(np.float64)
        if len(actions) < self.action_horizon or not np.isfinite(actions).all():
            raise ValueError("Model returned an undersized or nonfinite action chunk.")
        actions = smooth_actions(actions, self.smoothing_kernel)[:self.action_horizon]
        offsets = np.cumsum((0,) + self.state_dims)
        return [{key: row[offsets[i]:offsets[i + 1]].copy()
                 for i, key in enumerate(STATE_KEYS)} for row in actions]

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Evo_1 uses single-environment rollouts; eval_batch=false.")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Evo_1 does not enable batched evaluation.")

    def prepare_case(self, case_meta=None):
        if case_meta and case_meta.get("task_name"):
            # Some environment clients swallow prepare_case errors. Invalidate the old
            # selection first so a later observation cannot use the previous task's norm.
            self.current_dataset_key = None
            self._observation = None
            dataset_key = self._dataset_key(case_meta["task_name"])
            self.current_dataset_key = dataset_key
            self._validate_stats(dataset_key)

    def reset(self):
        # RoboTwin calls prepare_case before reset. Keep its task selection and RNG.
        self._observation = None
