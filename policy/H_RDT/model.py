from __future__ import annotations

import os
import sys
import json
from pathlib import Path

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
_HRDT_ROOT = _CUR_DIR / "H_RDT"

for _path in (str(_HRDT_ROOT), str(_HRDT_ROOT / "models")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone
from models.hrdt_runner import HRDTRunner


def _optional_str(value):
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _resolve_path(value, *base_dirs):
    value = _optional_str(value)
    if value is None:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    for base_dir in base_dirs:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return str(candidate)

    return str(Path(base_dirs[0]) / path)


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _decode_image(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed image buffer.")
        image = decoded

    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")

    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}.")

    return image


def _extract_image(observation, camera_names):
    vision = observation.get("vision", {})
    for camera_name in camera_names:
        if camera_name not in vision:
            continue
        camera_obs = vision[camera_name]
        if isinstance(camera_obs, dict):
            for image_key in ("color", "rgb"):
                if image_key in camera_obs:
                    return camera_obs[image_key]
        else:
            return camera_obs
    raise KeyError(f"Could not find camera image from candidates: {camera_names}")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.policy_name = self.model_cfg.get("policy_name", "H_RDT")
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("H_RDT currently supports only action_type='joint'.")

        self.env_cfg_type = self.model_cfg.get("env_cfg_type") or self.model_cfg.get("env_cfg")
        if self.env_cfg_type is None:
            raise ValueError("H_RDT requires env_cfg_type so action dimensions can be resolved.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.dtype = torch.bfloat16 if self.model_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32
        self.input_color_order = self.model_cfg.get("input_color_order", "bgr").lower()

        self.config_path = _resolve_path(
            self.model_cfg.get("config_path"),
            _CUR_DIR,
            _HRDT_ROOT,
        ) or str(_HRDT_ROOT / "configs" / "hrdt_finetune.yaml")
        self.config = _load_yaml(self.config_path)
        self._override_action_dims()
        self.action_q01, self.action_q99 = self._load_action_stats()
        self.action_scale = self.action_q99 - self.action_q01
        self.action_scale = np.where(self.action_scale < 1e-6, 1.0, self.action_scale)

        self.checkpoint_path = self._resolve_checkpoint_path()
        self.lang_tokens, self.lang_attn_mask = self._load_language_embedding()
        self.vision_encoder = self._build_vision_encoder()
        self.image_transform = self.vision_encoder.get_image_transform()
        self.policy = self._build_policy()
        self.policy.to(self.device, dtype=self.dtype).eval()
        self.model = self.policy

        self.max_img_cache_size = int(self.config["common"].get("img_history_size", 1))
        self.obs_cache_by_env = {}
        self._latest_env_idx_list = [0]

        print(
            f"[H_RDT] initialized task={self.task_name}, env_cfg={self.env_cfg_type}, "
            f"action_dim={self.action_dim}, checkpoint={self.checkpoint_path}"
        )

    def _get_device(self, device_arg):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _override_action_dims(self):
        common_cfg = self.config.setdefault("common", {})
        common_cfg["state_dim"] = self.action_dim
        common_cfg["action_dim"] = self.action_dim
        self.config.setdefault("model", {}).setdefault("hrdt", {})["output_size"] = self.action_dim

    def _load_action_stats(self):
        stats_path = _resolve_path(
            self.model_cfg.get("stats_path") or (_HRDT_ROOT / "datasets" / "xpolicylab" / "stats.json"),
            _CUR_DIR,
            _HRDT_ROOT,
        )
        with open(stats_path, "r", encoding="utf-8") as fp:
            stats = json.load(fp)["xpolicylab"]

        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        if q01.shape[0] != self.action_dim or q99.shape[0] != self.action_dim:
            raise ValueError(
                f"stats dim mismatch: expected {self.action_dim}, got q01={q01.shape[0]}, q99={q99.shape[0]}"
            )
        print(f"[H_RDT] loaded q01/q99 stats from: {stats_path}")
        return q01, q99

    def _normalize_action(self, action):
        clipped = np.clip(action, self.action_q01, self.action_q99)
        return ((clipped - self.action_q01) / self.action_scale * 2.0 - 1.0).astype(np.float32)

    def _denormalize_action(self, action):
        clipped = np.clip(action, -1.0, 1.0)
        return ((clipped + 1.0) * 0.5 * self.action_scale + self.action_q01).astype(np.float32)

    def _resolve_checkpoint_path(self):
        checkpoint_value = (
            self.model_cfg.get("checkpoint_path")
            or self.model_cfg.get("ckpt_path")
            or self.model_cfg.get("model_path")
            or self.model_cfg.get("ckpt_name")
        )
        checkpoint_path = _resolve_path(
            checkpoint_value,
            _CUR_DIR,
            _CUR_DIR / "checkpoints",
            _HRDT_ROOT,
        )
        if checkpoint_path is None:
            raise ValueError("H_RDT requires checkpoint_path, ckpt_path, model_path, or ckpt_name.")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"H_RDT checkpoint path does not exist: {checkpoint_path}")
        return checkpoint_path

    def _load_language_embedding(self):
        embedding_path = _resolve_path(
            self.model_cfg.get("lang_embedding_path"),
            _CUR_DIR,
            _HRDT_ROOT,
        )

        if embedding_path is None:
            embedding_dir = _resolve_path(
                self.model_cfg.get("lang_embedding_dir")
                or (_HRDT_ROOT / "datasets" / "robotwin2" / "lang_embeddings"),
                _CUR_DIR,
                _HRDT_ROOT,
            )
            if embedding_dir is not None:
                embedding_path = str(Path(embedding_dir) / f"{self.task_name}.pt")

        if embedding_path is None or not Path(embedding_path).exists():
            if self.model_cfg.get("allow_dummy_lang_embedding", False):
                token_len = int(self.config["dataset"]["tokenizer_max_length"])
                feature_dim = int(self.config["model"]["text"]["feature_dim"])
                tokens = torch.zeros(token_len, feature_dim, dtype=self.dtype, device=self.device)
                mask = torch.zeros(token_len, dtype=torch.bool, device=self.device)
                print("[H_RDT] using dummy zero language embedding")
                return tokens, mask
            raise FileNotFoundError(
                "H_RDT language embedding is required. Set lang_embedding_path or "
                f"lang_embedding_dir. Missing path: {embedding_path}"
            )

        embedding_data = torch.load(embedding_path, map_location=self.device)
        embeddings = embedding_data.get("embeddings") if isinstance(embedding_data, dict) else embedding_data
        if embeddings is None:
            raise KeyError(f"No embeddings tensor found in {embedding_path}")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)

        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
        attn_mask = torch.ones(embeddings.shape[0], dtype=torch.bool, device=self.device)
        print(f"[H_RDT] loaded language embedding from: {embedding_path}")
        return embeddings, attn_mask

    def _build_vision_encoder(self):
        image_aspect_ratio = self.config["dataset"].get("image_aspect_ratio", "pad")
        image_resize_strategy = "letterbox" if image_aspect_ratio == "pad" else "resize-naive"
        encoder = DinoSigLIPViTBackbone(
            vision_backbone_id=self.model_cfg.get("vision_backbone_id", "dino-siglip"),
            image_resize_strategy=image_resize_strategy,
            default_image_size=int(self.model_cfg.get("vision_image_size", 384)),
        )
        encoder.to(self.device, dtype=self.dtype)
        encoder.eval()
        return encoder

    def _build_policy(self):
        common_cfg = self.config["common"]
        pred_horizon = int(common_cfg["action_chunk_size"])
        return HRDTRunner.from_pretrained(
            pretrained_model_name_or_path=self.checkpoint_path,
            state_dim=int(common_cfg["state_dim"]),
            action_dim=int(common_cfg["action_dim"]),
            pred_horizon=pred_horizon,
            config=self.config["model"],
            act_pos_emb_config=[
                ("state", 1),
                ("action", pred_horizon),
            ],
            img_pos_emb_config=[
                (
                    "image",
                    (
                        int(common_cfg["img_history_size"]),
                        int(common_cfg["num_cameras"]),
                        -self.vision_encoder.num_patches,
                    ),
                ),
            ],
            lang_pos_emb_config=[
                ("lang", -int(self.config["dataset"]["tokenizer_max_length"])),
            ],
            max_img_len=int(common_cfg["img_history_size"])
            * int(common_cfg["num_cameras"])
            * self.vision_encoder.num_patches,
            max_lang_len=int(self.config["dataset"]["tokenizer_max_length"]),
            dtype=self.dtype,
        )

    def _encode_obs(self, observation):
        head_cam = _decode_image(_extract_image(observation, ["cam_head", "head_camera"]))
        left_cam = _decode_image(_extract_image(observation, ["cam_left_wrist", "left_camera"]))
        right_cam = _decode_image(_extract_image(observation, ["cam_right_wrist", "right_camera"]))

        if self.input_color_order == "bgr":
            head_cam = cv2.cvtColor(head_cam, cv2.COLOR_BGR2RGB)
            left_cam = cv2.cvtColor(left_cam, cv2.COLOR_BGR2RGB)
            right_cam = cv2.cvtColor(right_cam, cv2.COLOR_BGR2RGB)

        return {
            "head_cam": head_cam,
            "left_cam": left_cam,
            "right_cam": right_cam,
            "agent_pos": self._normalize_action(
                pack_robot_state(
                    observation,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                ).astype(np.float32)
            ),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = obs.get("env_idx", 0)
            self._latest_env_idx_list.append(env_idx)
            encoded_obs = self._encode_obs(obs)
            obs_cache = self.obs_cache_by_env.setdefault(env_idx, [])
            obs_cache.append(encoded_obs)
            if len(obs_cache) > self.max_img_cache_size:
                obs_cache.pop(0)

    def _build_image_tokens(self, current_obs):
        camera_images = [
            current_obs["head_cam"],
            current_obs["right_cam"],
            current_obs["left_cam"],
        ]

        transformed_images = [self.image_transform(Image.fromarray(image)) for image in camera_images]
        image_inputs = {}
        for key in transformed_images[0]:
            stacked = torch.stack([image[key] for image in transformed_images], dim=0)
            image_inputs[key] = stacked.unsqueeze(0).to(self.device, dtype=self.dtype)

        first_key = next(iter(image_inputs))
        batch_size, seq_len, channels, height, width = image_inputs[first_key].shape
        for key in image_inputs:
            image_inputs[key] = image_inputs[key].view(-1, channels, height, width)

        image_features = self.vision_encoder(image_inputs)
        return image_features.view(batch_size, -1, self.vision_encoder.embed_dim)

    @torch.inference_mode()
    def _infer(self, env_idx=0):
        obs_cache = self.obs_cache_by_env.get(env_idx)
        if not obs_cache:
            raise AssertionError("update_obs must be called before get_action.")

        current_obs = obs_cache[-1]
        state_tokens = torch.as_tensor(
            current_obs["agent_pos"],
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)
        image_tokens = self._build_image_tokens(current_obs)
        action_pred = self.policy.predict_action(
            state_tokens=state_tokens,
            image_tokens=image_tokens,
            lang_tokens=self.lang_tokens.unsqueeze(0),
            lang_attn_mask=self.lang_attn_mask.unsqueeze(0),
        )
        return self._denormalize_action(action_pred.float().cpu().numpy()[0])

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [
            unpack_robot_state(
                self._infer(env_idx),
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.obs_cache_by_env = {}
        self._latest_env_idx_list = [0]
