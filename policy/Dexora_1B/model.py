from __future__ import annotations

import os
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_CUR_DIR = Path(__file__).resolve().parent
_DEFAULT_DEXORA_ROOT = Path("/root/crx/Dexora")


def _optional_path(value: str | None, *base_dirs: Path) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base_dir in base_dirs:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (base_dirs[0] / path).resolve()


def _decode_image(image: Any) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed image bytes.")
        image = decoded

    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _extract_camera(observation: dict[str, Any], camera_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for camera_name in camera_names:
        if camera_name not in vision:
            continue
        camera_obs = vision[camera_name]
        if isinstance(camera_obs, dict):
            for image_key in ("color", "rgb", "colors"):
                if image_key in camera_obs:
                    return _decode_image(camera_obs[image_key])
        else:
            return _decode_image(camera_obs)
    raise KeyError(f"Missing camera from candidates: {camera_names}")


class Model(ModelTemplate):
    """XPolicyLab adapter for the 14-D Dexora/RDT-1B policy."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("Dexora_1B currently supports action_type='joint' only.")

        self.dexora_root = _optional_path(
            self.model_cfg.get("dexora_root"),
            _CUR_DIR,
            _DEFAULT_DEXORA_ROOT,
        ) or _DEFAULT_DEXORA_ROOT
        if not self.dexora_root.exists():
            raise FileNotFoundError(f"Dexora root does not exist: {self.dexora_root}")
        if str(self.dexora_root) not in sys.path:
            sys.path.insert(0, str(self.dexora_root))

        self._configure_hf_cache()

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = self._resolve_robot_action_dim_info()
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.dtype = (
            torch.bfloat16
            if self.model_cfg.get("dtype", "bfloat16") == "bfloat16"
            else torch.float32
        )

        self.config_path = self._resolve_config_path()
        self.config = self._load_yaml(self.config_path)
        self._validate_config_dims()
        self.stats = self._load_statistics()

        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        from models.multimodal_encoder.t5_encoder import T5Embedder
        from models.rdt_runner import RDTRunner

        self.vision_encoder = SiglipVisionTower(
            vision_tower=self.model_cfg.get(
                "vision_encoder_path", "google/siglip-so400m-patch14-384"
            ),
            args=None,
        )
        self.image_processor = self.vision_encoder.image_processor

        self.policy = self._build_policy(RDTRunner)
        self._load_checkpoint(self.policy, self._resolve_checkpoint_path())
        self.policy.to(self.device, dtype=self.dtype).eval()
        self.vision_encoder.vision_tower.to(self.device, dtype=self.dtype).eval()

        text_embedder = T5Embedder(
            from_pretrained=self.model_cfg.get("text_encoder_path", "google/t5-v1_1-xxl"),
            model_max_length=self.config["dataset"]["tokenizer_max_length"],
            device=self.device,
            local_files_only=bool(self.model_cfg.get("local_files_only", False)),
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
        self.text_encoder.eval()

        self.camera_candidates = [
            ["cam_head", "cam_high", "head_camera"],
            ["cam_right_wrist", "right_camera"],
            ["cam_left_wrist", "left_camera"],
        ]
        self.img_history_size = int(self.config["common"].get("img_history_size", 1))
        self.ctrl_freq = int(self.model_cfg.get("ctrl_freq", 25))
        self.default_prompt = self.model_cfg.get("prompt") or self.model_cfg.get("task_name", "")
        self.input_color_order = self.model_cfg.get("input_color_order", "rgb").lower()

        self._obs_windows: dict[int, deque[dict[str, Any]]] = {}
        self._latest_env_idx_list: list[int] = [0]
        print(
            f"[Dexora_1B] loaded policy from {self.model_cfg.get('checkpoint_path') or self.model_cfg.get('ckpt_name')}; "
            f"state/action dim={self.action_dim}, img_history={self.img_history_size}"
        )

    def _configure_hf_cache(self) -> None:
        hf_home = self.model_cfg.get("hf_home")
        hf_hub_cache = self.model_cfg.get("hf_hub_cache")
        if hf_home:
            os.environ.setdefault("HF_HOME", str(_optional_path(hf_home, _CUR_DIR) or hf_home))
        if hf_hub_cache:
            os.environ.setdefault(
                "HF_HUB_CACHE", str(_optional_path(hf_hub_cache, _CUR_DIR) or hf_hub_cache)
            )
        if self.model_cfg.get("hf_offline", False):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _get_device(self, device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _resolve_robot_action_dim_info(self) -> dict[str, list[int]]:
        explicit = self.model_cfg.get("robot_action_dim_info")
        if isinstance(explicit, dict):
            return {
                "arm_dim": [int(x) for x in explicit["arm_dim"]],
                "ee_dim": [int(x) for x in explicit["ee_dim"]],
            }
        if self.env_cfg_type:
            try:
                return get_robot_action_dim_info(self.env_cfg_type)
            except Exception as exc:
                if not bool(self.model_cfg.get("allow_default_robot_dims", True)):
                    raise
                print(
                    f"[Dexora_1B][WARN] failed to load env_cfg_type={self.env_cfg_type!r}: {exc}; "
                    "falling back to dual-arm 6+1 / 6+1 dims."
                )
        return {"arm_dim": [6, 6], "ee_dim": [1, 1]}

    def _resolve_config_path(self) -> Path:
        return _optional_path(
            self.model_cfg.get("config_path"),
            self.dexora_root,
            _CUR_DIR,
        ) or (self.dexora_root / "configs" / "base.yaml")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    def _load_statistics(self) -> dict[str, dict[str, np.ndarray]] | None:
        stats_path = _optional_path(
            self.model_cfg.get("stats_file") or self.model_cfg.get("stats_path"),
            self.dexora_root,
            _CUR_DIR,
        )
        if stats_path is None:
            return None
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file does not exist: {stats_path}")

        with open(stats_path, "r", encoding="utf-8") as fp:
            raw_stats = json.load(fp)

        stats: dict[str, dict[str, np.ndarray]] = {}
        for key in ("state", "action"):
            if key not in raw_stats:
                raise KeyError(f"Statistics file missing '{key}' section: {stats_path}")
            p1 = np.asarray(raw_stats[key]["percentile_1"], dtype=np.float32)
            p99 = np.asarray(raw_stats[key]["percentile_99"], dtype=np.float32)
            if p1.shape[-1] != self.action_dim or p99.shape[-1] != self.action_dim:
                raise ValueError(
                    f"{key} stats dim mismatch: expected {self.action_dim}, "
                    f"got percentile_1={p1.shape}, percentile_99={p99.shape}"
                )
            scale = p99 - p1
            scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
            stats[key] = {"min": p1, "scale": scale}
        print(f"[Dexora_1B] loaded normalization statistics from {stats_path}")
        return stats

    def _validate_config_dims(self) -> None:
        cfg_state_dim = int(self.config["common"]["state_dim"])
        cfg_token_dim = int(self.config["model"]["state_token_dim"])
        if cfg_state_dim != self.action_dim or cfg_token_dim != self.action_dim:
            raise ValueError(
                "Dexora config/action dim mismatch: "
                f"env action_dim={self.action_dim}, common.state_dim={cfg_state_dim}, "
                f"model.state_token_dim={cfg_token_dim}. "
                "Pass robot_action_dim_info or a matching config_path."
            )

    def _build_policy(self, runner_cls):
        img_cond_len = (
            int(self.config["common"]["img_history_size"])
            * int(self.config["common"]["num_cameras"])
            * self.vision_encoder.num_patches
        )
        return runner_cls(
            action_dim=self.config["common"]["state_dim"],
            pred_horizon=self.config["common"]["action_chunk_size"],
            config=self.config["model"],
            lang_token_dim=self.config["model"]["lang_token_dim"],
            img_token_dim=self.config["model"]["img_token_dim"],
            state_token_dim=self.config["model"]["state_token_dim"],
            max_lang_cond_len=self.config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                (
                    "image",
                    (
                        self.config["common"]["img_history_size"],
                        self.config["common"]["num_cameras"],
                        -self.vision_encoder.num_patches,
                    ),
                )
            ],
            lang_pos_embed_config=[
                ("lang", -self.config["dataset"]["tokenizer_max_length"])
            ],
            dtype=self.dtype,
        )

    def _resolve_checkpoint_path(self) -> Path:
        value = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("model_path")
        if not value:
            ckpt_name = self.model_cfg.get("ckpt_name")
            if ckpt_name:
                value = str(self.dexora_root / "checkpoints" / str(ckpt_name))
        path = _optional_path(value, self.dexora_root, _CUR_DIR) if value else None
        if path is None:
            raise ValueError("checkpoint_path, model_path, or ckpt_name is required.")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        if path.is_file():
            return path

        candidates = [
            path / "ema" / "model.safetensors",
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "pytorch_model" / "mp_rank_00_model_states.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No supported weights found under {path}. Expected one of: "
            "ema/model.safetensors, model.safetensors, pytorch_model.bin, "
            "pytorch_model/mp_rank_00_model_states.pt."
        )

    def _load_checkpoint(self, policy, path: Path) -> None:
        strict = bool(self.model_cfg.get("strict_load", True))
        print(f"[Dexora_1B] loading weights from {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_model

            load_model(policy, str(path), strict=strict)
            return

        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("module")
                or checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            state_dict = checkpoint
        missing, unexpected = policy.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(
                f"[Dexora_1B][WARN] load_state_dict missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = []
        for candidates in self.camera_candidates:
            image = _extract_camera(observation, candidates)
            if self.input_color_order == "bgr":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if instruction in (None, ""):
            instruction = self.default_prompt

        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        if self.stats is not None:
            state = (state - self.stats["state"]["min"]) / self.stats["state"]["scale"]
        return {"images": images, "state": state, "prompt": str(instruction)}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for fallback_idx, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", fallback_idx))
            self._latest_env_idx_list.append(env_idx)
            window = self._obs_windows.setdefault(env_idx, deque(maxlen=self.img_history_size))
            window.append(self._convert_obs(obs))

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(image, mode="RGB")
        return self.image_processor.preprocess(
            pil_image, return_tensors="pt"
        )["pixel_values"][0]

    def _encode_images(self, window: deque[dict[str, Any]]) -> torch.Tensor:
        frames = list(window)
        if not frames:
            raise AssertionError("update_obs must be called before get_action.")
        while len(frames) < self.img_history_size:
            frames.insert(0, frames[0])

        tensors = []
        for frame in frames[-self.img_history_size :]:
            tensors.extend(self._preprocess_image(image) for image in frame["images"])
        image_batch = torch.stack(tensors, dim=0).to(self.device, dtype=self.dtype)
        image_embeds = self.vision_encoder(image_batch).detach()
        return image_embeds.reshape(1, -1, self.vision_encoder.hidden_size)

    def _encode_text(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        text_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )["last_hidden_state"].detach()
        return text_embeds.to(dtype=self.dtype), attn_mask

    @torch.inference_mode()
    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        if env_idx not in self._obs_windows:
            raise AssertionError("update_obs must be called before get_action.")
        window = self._obs_windows[env_idx]
        current = window[-1]

        img_tokens = self._encode_images(window)
        lang_tokens, lang_attn_mask = self._encode_text(current["prompt"])
        state_tokens = torch.from_numpy(current["state"]).to(self.device, dtype=self.dtype)
        state_tokens = state_tokens.view(1, 1, -1)
        action_mask = torch.ones(
            (1, 1, self.action_dim), device=self.device, dtype=self.dtype
        )
        ctrl_freqs = torch.tensor([self.ctrl_freq], device=self.device)

        actions = self.policy.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
        )
        actions_np = actions.squeeze(0).float().cpu().numpy()
        if self.stats is not None:
            if bool(self.model_cfg.get("clip_normalized_actions", True)):
                actions_np = np.clip(actions_np, 0.0, 1.0)
            actions_np = (
                actions_np * self.stats["action"]["scale"] + self.stats["action"]["min"]
            ).astype(np.float32)
        return actions_np

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [
            unpack_robot_state(
                self._infer_chunk(int(env_idx)),
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for env_idx in env_idx_list
        ]

    def reset(self):
        self._obs_windows = {}
        self._latest_env_idx_list = [0]
        torch.cuda.empty_cache()
