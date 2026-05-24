from __future__ import annotations

import sys
import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_GIGAWORLD_ROOT = _CUR_DIR / "giga_world_policy"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (
    str(_REPO_ROOT),
    str(_CUR_DIR),
    str(_GIGAWORLD_ROOT),
    str(_GIGAWORLD_ROOT / "third_party" / "giga-train"),
    str(_GIGAWORLD_ROOT / "third_party" / "giga-models"),
    str(_GIGAWORLD_ROOT / "third_party" / "giga-datasets"),
    str(_GIGAWORLD_ROOT / "third_party" / "wan"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_path(value: str | None, base_dir: Path = _CUR_DIR) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.resolve()


def _ensure_stats_compatible(stats_path: Path) -> Path:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    changed = False
    for key in ("observation.state", "action"):
        entry = stats.get("norm_stats", {}).get(key, {})
        if "min" not in entry and "q01" in entry:
            entry["min"] = entry["q01"]
            changed = True
        if "max" not in entry and "q99" in entry:
            entry["max"] = entry["q99"]
            changed = True

    if not changed:
        return stats_path

    output_path = Path(tempfile.gettempdir()) / f"{stats_path.stem}_gigaworld_compatible.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    return output_path.resolve()


def _pad_or_trim_np(value: np.ndarray, dim: int) -> np.ndarray:
    if value.shape[-1] == dim:
        return value
    if value.shape[-1] > dim:
        return value[..., :dim]
    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, dim - value.shape[-1])
    return np.pad(value, pad_width, mode="constant").astype(np.float32)


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def _patch_transformer_loader(inference_server_module: Any, action_dim: int) -> None:
    transformer_cls = inference_server_module.CasualWorldActionTransformer
    if getattr(transformer_cls, "_xpolicylab_action_dim", None) == action_dim:
        return

    original_from_pretrained = getattr(
        transformer_cls,
        "_xpolicylab_original_from_pretrained",
        transformer_cls.from_pretrained,
    )
    transformer_cls._xpolicylab_original_from_pretrained = original_from_pretrained

    def _from_pretrained_with_action_dim(cls, pretrained_model_name_or_path, *args, **kwargs):
        import torch
        import torch.nn as nn
        from world_action_model.models.transformer_wa_casual import WanRotaryPosEmbed1D

        checkpoint_dir = Path(pretrained_model_name_or_path)
        weight_path = checkpoint_dir / "diffusion_pytorch_model.bin"
        if not weight_path.is_file():
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        torch_dtype = kwargs.pop("torch_dtype", None)
        transformer = cls.from_config(pretrained_model_name_or_path)
        inner_dim = _config_value(transformer.config, "num_attention_heads") * _config_value(
            transformer.config, "attention_head_dim"
        )
        transformer.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, inner_dim),
        )
        transformer.action_decoder = nn.Sequential(
            nn.Linear(inner_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )
        transformer.action_rope = WanRotaryPosEmbed1D(
            _config_value(transformer.config, "attention_head_dim"),
            _config_value(transformer.config, "rope_max_seq_len"),
        )

        state_dict = torch.load(weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        transformer.load_state_dict(state_dict, strict=True)
        if torch_dtype is not None:
            transformer = transformer.to(torch_dtype)
        return transformer.eval()

    transformer_cls.from_pretrained = classmethod(_from_pretrained_with_action_dim)
    transformer_cls._xpolicylab_action_dim = action_dim


def _patch_pipeline_action_dim(inference_server_module: Any, action_dim: int) -> None:
    pipeline_cls = inference_server_module.WAPipeline
    if getattr(pipeline_cls, "_xpolicylab_action_dim", None) == action_dim:
        return

    original_prepare_latents = getattr(
        pipeline_cls,
        "_xpolicylab_original_prepare_latents",
        pipeline_cls.prepare_latents,
    )
    original_call = getattr(
        pipeline_cls,
        "_xpolicylab_original_call",
        pipeline_cls.__call__,
    )
    pipeline_cls._xpolicylab_original_prepare_latents = original_prepare_latents
    pipeline_cls._xpolicylab_original_call = original_call

    def _prepare_latents_with_action_dim(self, *args, **kwargs):
        kwargs.setdefault("action_dim", action_dim)
        return original_prepare_latents(self, *args, **kwargs)

    def _call_with_expand_timesteps(self, *args, **kwargs):
        self.register_to_config(expand_timesteps=True)
        return original_call(self, *args, **kwargs)

    pipeline_cls.prepare_latents = _prepare_latents_with_action_dim
    pipeline_cls.__call__ = _call_with_expand_timesteps
    pipeline_cls._xpolicylab_action_dim = action_dim


def _resolve_checkpoint_root(model_cfg: dict[str, Any]) -> Path:
    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name:
        local_candidate = (_CHECKPOINTS_DIR / str(ckpt_name)).resolve()
        if local_candidate.exists():
            return local_candidate

    return (_CHECKPOINTS_DIR / "RoboDojo_sim_arx_seed_0").resolve()


def _extract_image(observation: dict[str, Any], candidate_names: list[str]) -> np.ndarray:
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


def _ensure_chw01(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")
    if image.shape[0] in (1, 3):
        chw = image
    elif image.shape[-1] in (1, 3):
        chw = image.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if np.issubdtype(chw.dtype, np.floating):
        return np.clip(chw.astype(np.float32), 0.0, 1.0)
    return (chw.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _manual_unpack_ee(packed_state: np.ndarray, robot_action_dim_info: dict[str, list[int]]) -> dict[str, np.ndarray]:
    packed = np.asarray(packed_state, dtype=np.float32)
    ee_dims = robot_action_dim_info["ee_dim"]
    if len(ee_dims) == 1:
        return {
            "ee_pose": packed[:7],
            "ee_joint_state": packed[7 : 7 + ee_dims[0]],
        }
    if len(ee_dims) == 2:
        left_end = 7 + ee_dims[0]
        right_pose_end = left_end + 7
        return {
            "left_ee_pose": packed[:7],
            "left_ee_joint_state": packed[7:left_end],
            "right_ee_pose": packed[left_end:right_pose_end],
            "right_ee_joint_state": packed[right_pose_end : right_pose_end + ee_dims[1]],
        }
    raise ValueError(f"Unsupported arm count: {len(ee_dims)}")


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "debug_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_chunk = int(self.model_cfg.get("action_chunk") or 1)
        self.xpolicylab_action_dim = self._packed_dim()
        self.model_state_dim = int(self.model_cfg.get("model_state_dim") or self.xpolicylab_action_dim)
        self.model_action_dim = int(self.model_cfg.get("model_action_dim") or self.xpolicylab_action_dim)
        self.load_model = _parse_bool(self.model_cfg.get("load_model", False))
        self.device = self.model_cfg.get("device", "cuda")
        self.dtype = self.model_cfg.get("dtype", "bf16")

        self.checkpoint_root = _resolve_checkpoint_root(self.model_cfg)
        self.checkpoint_dir = self._resolve_checkpoint_dir(self.checkpoint_root)
        self.transformer_path = self._resolve_transformer_path(self.checkpoint_dir)
        self.base_model_path = _resolve_path(self.model_cfg.get("base_model_path")) or Path(
            "/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B-Diffusers/"
        )
        self.stats_path = _ensure_stats_compatible(
            _resolve_path(self.model_cfg.get("stats_path")) or (_GIGAWORLD_ROOT / "norm_stats_delta.json")
        )
        self.t5_embedding_pkl = _resolve_path(self.model_cfg.get("t5_embedding_pkl"))

        self._latest_env_idx_list = [0]
        self._latest_observation: dict[str, Any] | None = None
        self._latest_observations: dict[int, dict[str, Any]] = {}
        self.policy = self._load_policy() if self.load_model else None

        print(
            "[GigaWorldPolicy] initialized",
            f"load_model={self.load_model}",
            f"xpolicylab_action_dim={self.xpolicylab_action_dim}",
            f"model_state_dim={self.model_state_dim}",
            f"model_action_dim={self.model_action_dim}",
            f"checkpoint_root={self.checkpoint_root}",
            f"checkpoint_dir={self.checkpoint_dir}",
            f"transformer_path={self.transformer_path}",
        )

    def _resolve_checkpoint_dir(self, checkpoint_root: Path) -> Path:
        models_dir = checkpoint_root / "models"
        checkpoint_num = self.model_cfg.get("checkpoint_num")

        if models_dir.is_dir():
            if checkpoint_num is not None:
                checkpoint_candidate = models_dir / str(checkpoint_num)
                if checkpoint_candidate.is_dir():
                    return checkpoint_candidate.resolve()

                step_digits = "".join(ch for ch in str(checkpoint_num) if ch.isdigit())
                if step_digits:
                    for candidate in sorted(models_dir.iterdir()):
                        if candidate.is_dir() and candidate.name.endswith(f"step_{step_digits}"):
                            return candidate.resolve()
                raise FileNotFoundError(
                    f"checkpoint_num={checkpoint_num!r} not found under {models_dir}"
                )

            checkpoint_dirs = [path for path in models_dir.iterdir() if path.is_dir()]
            if checkpoint_dirs:
                return max(checkpoint_dirs, key=lambda path: path.stat().st_mtime).resolve()

        return checkpoint_root.resolve()

    def _resolve_transformer_path(self, checkpoint_root: Path) -> Path:
        transformer_subdir = self.model_cfg.get("transformer_subdir", "transformer_ema")
        candidates = [
            checkpoint_root / str(transformer_subdir),
            checkpoint_root / "transformer_ema",
            checkpoint_root / "transformer",
            checkpoint_root,
        ]
        for candidate in candidates:
            if (candidate / "config.json").is_file():
                return candidate
        raise FileNotFoundError(f"Could not resolve GigaWorld transformer checkpoint under {checkpoint_root}")

    def _load_policy(self):
        if self.t5_embedding_pkl is None:
            raise ValueError("t5_embedding_pkl is required when load_model=true.")

        inference_server = importlib.import_module("giga_world_policy.scripts.inference_server")
        _patch_transformer_loader(inference_server, self.model_action_dim)
        _patch_pipeline_action_dim(inference_server, self.model_action_dim)
        get_policy = inference_server.get_policy

        args = SimpleNamespace(
            model_id=str(self.base_model_path),
            transformer_path=str(self.transformer_path),
            stats_path=str(self.stats_path),
            t5_embedding_pkl=str(self.t5_embedding_pkl),
            t5_len=int(self.model_cfg.get("t5_len", 64)),
            device=self.device,
            dtype=self.dtype,
            dst_width=int(self.model_cfg.get("dst_width", 768)),
            dst_height=int(self.model_cfg.get("dst_height", 192)),
            action_chunk=int(self.model_cfg.get("action_chunk", 48)),
            num_frames=int(self.model_cfg.get("num_frames", 5)),
            num_inference_steps=int(self.model_cfg.get("num_inference_steps", 10)),
            guidance_scale=float(self.model_cfg.get("guidance_scale", 0.0)),
            norm_mode=self.model_cfg.get("norm_mode", "zscore"),
            crop_mode=self.model_cfg.get("crop_mode", "center"),
            return_images=False,
            vis_dir=self.model_cfg.get("vis_dir", "./vis"),
            vis_fps=int(self.model_cfg.get("vis_fps", 5)),
            state_dim=self.model_state_dim,
            action_dim=self.model_action_dim,
            delta_mask=self.model_cfg.get("model_delta_mask") or self.model_cfg.get("delta_mask") or self._default_delta_mask(),
        )
        return get_policy(args)

    def _packed_dim(self) -> int:
        if self.action_type == "ee":
            return 7 * len(self.robot_action_dim_info["ee_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        return sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])

    def _default_delta_mask(self) -> str:
        dim = self._packed_dim()
        if self.action_type == "joint":
            if dim == 14:
                mask = ["1", "1", "1", "1", "1", "1", "0", "1", "1", "1", "1", "1", "1", "0"]
            else:
                mask = ["1"] * dim
            if self.model_action_dim > len(mask):
                mask.extend(["0"] * (self.model_action_dim - len(mask)))
            return ",".join(mask[: self.model_action_dim])
        mask: list[str] = []
        for ee_dim in self.robot_action_dim_info["ee_dim"]:
            mask.extend(["1", "1", "1", "1", "1", "1", "0"])
            mask.extend(["1"] * ee_dim)
        if self.model_action_dim > len(mask):
            mask.extend(["0"] * (self.model_action_dim - len(mask)))
        return ",".join(mask[: self.model_action_dim])

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._latest_observations = {
            env_idx: self._encode_observation(obs) for env_idx, obs in zip(self._latest_env_idx_list, obs_list)
        }
        self._latest_observation = self._latest_observations[self._latest_env_idx_list[0]]

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        actions = []
        for env_idx in env_idx_list:
            obs = self._latest_observations.get(env_idx)
            if obs is None:
                obs = self._latest_observation
            if obs is None:
                raise AssertionError("update_obs or update_obs_batch must be called before get_action.")
            actions.append(self._predict_action_sequence(obs))
        return actions

    def reset(self):
        self._latest_env_idx_list = [0]
        self._latest_observation = None
        self._latest_observations = {}

    def _encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = pack_robot_state(observation, self.action_type, self.robot_action_dim_info, source_type="obs").astype(np.float32)
        state = _pad_or_trim_np(state, self.model_state_dim)
        return {
            "observation.state": state,
            "observation.images.cam_high": _ensure_chw01(
                _extract_image(observation, ["cam_head", "cam_high", "head_camera", "top_camera"])
            ),
            "observation.images.cam_left_wrist": _ensure_chw01(
                _extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
            ),
            "observation.images.cam_right_wrist": _ensure_chw01(
                _extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
            ),
        }

    def _predict_action_sequence(self, encoded_obs: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        if self.policy is None:
            packed_actions = np.zeros((1, self.model_action_dim), dtype=np.float32)
        else:
            pred = self.policy.inference(encoded_obs)
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu().numpy()
            packed_actions = np.asarray(pred, dtype=np.float32)
            if packed_actions.ndim == 1:
                packed_actions = packed_actions[None, :]

        action_list = []
        for packed_action in packed_actions:
            packed_action = packed_action[: self.xpolicylab_action_dim]
            if self.action_type == "ee":
                action_list.append(_manual_unpack_ee(packed_action, self.robot_action_dim_info))
            else:
                action_list.append(
                    unpack_robot_state(
                        packed_action,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
        return action_list
