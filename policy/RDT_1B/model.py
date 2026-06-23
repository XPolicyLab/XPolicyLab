from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PImage

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_RDT_ROOT = _CUR_DIR / "rdt"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_REPO_ROOT), str(_CUR_DIR), str(_RDT_ROOT), str(_RDT_ROOT / "models")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, unpack_robot_state

from .rdt.scripts.robodojo_model import create_model
from .rdt.models.multimodal_encoder.t5_encoder import T5Embedder


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_CUR_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


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


def ensure_hwc_uint8(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_compressed_image(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_compressed_image(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def decode_compressed_image(image_buffer):
    return decode_image_bit(image_buffer)


def encode_obs(observation, default_prompt):
    images = {
        "cam_high": ensure_hwc_uint8(extract_image(observation, ["cam_head"])),
        "cam_right_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_right_wrist"])
        ),
        "cam_left_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_left_wrist"])
        ),
    }

    state_dict = observation["state"]
    state = np.concatenate(
        [
            np.asarray(state_dict["left_arm_joint_state"], dtype=np.float32),
            np.asarray(state_dict["left_ee_joint_state"], dtype=np.float32),
            np.asarray(state_dict["right_arm_joint_state"], dtype=np.float32),
            np.asarray(state_dict["right_ee_joint_state"], dtype=np.float32),
        ],
        axis=-1,
    )
    prompt = observation.get("instruction", default_prompt)
    return {"images": images, "state": state, "prompt": prompt}


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("RDT-1b in XPolicyLab currently supports only action_type='joint'.")

        self.env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg) if self.env_cfg is not None else None
        self.default_prompt = self.model_cfg.get("prompt", self.task_name)

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.dtype = torch.bfloat16 if self.model_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32

        self.config = self._build_runtime_config()
        self.args = self._build_model_args()
        self.policy = create_model(
            args=self.args["config"],
            dtype=self.dtype,
            pretrained=self.args["pretrained_model_name_or_path"],
            pretrained_vision_encoder_name_or_path=self.args["pretrained_vision_encoder_name_or_path"],
            control_frequency=self.args["ctrl_freq"],
        )

        self.tokenizer, self.text_encoder = self._load_text_embedder()
        self.observation_window = None
        self._observation_windows: dict[int, deque] = {}
        self._latest_encoded_obs_list = []
        self.lang_embeddings = None
        self._latest_env_idx_list: list[int] = [0]
        self.model = self.policy

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _build_runtime_config(self):
        if self.robot_action_dim_info is None:
            raise ValueError("RDT-1b requires env_cfg or env_cfg_type so action dimensions can be resolved.")

        arm_dim = self.robot_action_dim_info.get("arm_dim")
        if arm_dim is None or len(arm_dim) != 2:
            raise ValueError(
                f"RDT-1b expects a dual-arm env_cfg with joint-state dimensions, got env_cfg={self.env_cfg!r}, arm_dim={arm_dim!r}."
            )

        left_arm_dim, right_arm_dim = arm_dim
        return {
            "episode_len": int(self.model_cfg.get("episode_len", 10000)),
            "state_dim": int(left_arm_dim + 1 + right_arm_dim + 1),
            "chunk_size": int(self.model_cfg.get("chunk_size", 64)),
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        }

    def _resolve_checkpoint_root(self) -> Path | None:
        ckpt_name = self.model_cfg.get("ckpt_name")
        if ckpt_name:
            checkpoint_root = (_CHECKPOINTS_DIR / str(ckpt_name)).expanduser().resolve()
            if not checkpoint_root.is_dir():
                return checkpoint_root

            candidate_dirs = []
            if any((checkpoint_root / marker).exists() for marker in ("config.json", "pytorch_model.bin", "pytorch_model")):
                candidate_dirs.append(checkpoint_root)
            candidate_dirs.extend(
                child
                for child in sorted(checkpoint_root.iterdir())
                if child.is_dir() and any((child / marker).exists() for marker in ("config.json", "pytorch_model.bin", "pytorch_model"))
            )
            if not candidate_dirs:
                return checkpoint_root

            checkpoint_num = self.model_cfg.get("checkpoint_num")
            desired_step = _extract_step_number(checkpoint_num)
            if desired_step is not None:
                for candidate in candidate_dirs:
                    candidate_step = _extract_step_number(candidate.name)
                    if candidate_step is None:
                        continue
                    scaled_step = desired_step
                    while len(str(scaled_step)) < len(str(candidate_step)):
                        scaled_step *= 10
                    if candidate_step in {desired_step, scaled_step}:
                        return candidate

            numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
            if numeric_dirs:
                return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
            return candidate_dirs[0]

        for key in ("checkpoint_path", "model_path", "model_root"):
            resolved = _resolve_path(self.model_cfg.get(key))
            if resolved is not None:
                return resolved
        return None

    def _resolve_indexed_path(self, base_dir: Path | None, explicit_value: str | None, candidate_relpaths: list[str]) -> str | None:
        explicit_path = _resolve_path(explicit_value)
        if explicit_path is not None:
            return str(explicit_path)
        search_roots = []
        for root in (base_dir, _CHECKPOINTS_DIR):
            if root is not None and root not in search_roots:
                search_roots.append(root)
        if not search_roots:
            return None

        for root in search_roots:
            fallback_path = root
            for relative_path in candidate_relpaths:
                candidate = root / relative_path if relative_path else root
                fallback_path = candidate
                if candidate.exists():
                    return str(candidate)
            if root is base_dir:
                return str(fallback_path)
        return str(fallback_path)

    def _with_weights_fallback(self, explicit_value: str | None, resolved: str | None, weight_dirname: str) -> str | None:
        if explicit_value:
            return resolved
        if resolved is not None and Path(resolved).exists():
            return resolved
        fallback = _CUR_DIR / "weights" / "RDT" / weight_dirname
        if fallback.exists():
            return str(fallback)
        return resolved

    def _default_model_paths(self):
        checkpoint_root = self._resolve_checkpoint_root()
        model_root = _resolve_path(self.model_cfg.get("model_root")) or checkpoint_root or _RDT_ROOT
        default_config_path = model_root / "configs" / "base.yaml"
        if not default_config_path.exists():
            default_config_path = _RDT_ROOT / "configs" / "base.yaml"

        return {
            "config_path": self.model_cfg.get("config_path") or str(default_config_path),
            "text_encoder_path": self._with_weights_fallback(
                self.model_cfg.get("text_encoder_path"),
                self._resolve_indexed_path(
                    model_root,
                    self.model_cfg.get("text_encoder_path"),
                    [
                        "shared/t5-v1_1-xxl",
                        "text_encoder",
                        "weights/RDT/t5-v1_1-xxl",
                        "google/t5-v1_1-xxl",
                        "t5-v1_1-xxl",
                    ],
                ),
                "t5-v1_1-xxl",
            ),
            "vision_encoder_path": self._with_weights_fallback(
                self.model_cfg.get("vision_encoder_path"),
                self._resolve_indexed_path(
                    model_root,
                    self.model_cfg.get("vision_encoder_path"),
                    [
                        "shared/siglip-so400m-patch14-384",
                        "vision_encoder",
                        "weights/RDT/siglip-so400m-patch14-384",
                        "google/siglip-so400m-patch14-384",
                        "siglip-so400m-patch14-384",
                    ],
                ),
                "siglip-so400m-patch14-384",
            ),
            "checkpoint_path": self._resolve_indexed_path(
                checkpoint_root,
                self.model_cfg.get("checkpoint_path") or self.model_cfg.get("model_path"),
                ["", "checkpoint", "model", "pretrained_model"],
            ),
        }

    def _build_model_args(self):
        paths = self._default_model_paths()
        if paths["checkpoint_path"] is None:
            raise ValueError("ckpt_name, checkpoint_path, or model_path is required for RDT-1b.")

        return {
            "max_publish_step": int(self.model_cfg.get("max_publish_step", 10000)),
            "seed": self.model_cfg.get("seed"),
            "ctrl_freq": int(self.model_cfg.get("ctrl_freq", 25)),
            "chunk_size": int(self.model_cfg.get("chunk_size", 64)),
            "config_path": paths["config_path"],
            "pretrained_model_name_or_path": paths["checkpoint_path"],
            "pretrained_vision_encoder_name_or_path": paths["vision_encoder_path"],
            "text_encoder_path": paths["text_encoder_path"],
            "config": self._load_yaml(paths["config_path"]),
        }

    def _load_yaml(self, config_path):
        with open(config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
        config["arm_dim"] = {
            "left_arm_dim": self.robot_action_dim_info["arm_dim"][0],
            "right_arm_dim": self.robot_action_dim_info["arm_dim"][1],
        }
        return config

    def _load_text_embedder(self):
        text_embedder = T5Embedder(
            from_pretrained=self.args["text_encoder_path"],
            model_max_length=self.args["config"]["dataset"]["tokenizer_max_length"],
            device=self.device,
            use_offload_folder=None,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        text_encoder.eval()
        return tokenizer, text_encoder

    def _set_language_instruction(self, instruction: str):
        device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            tokens = self.tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )["input_ids"].to(device)
            tokens = tokens.view(1, -1)
            output = self.text_encoder(tokens)
            self.lang_embeddings = output.last_hidden_state.detach().cpu()
        torch.cuda.empty_cache()

    def _jpeg_mapping(self, img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    def _resize_img(self, img):
        img_size = tuple(self.model_cfg.get("image_size", (640, 480)))
        return cv2.resize(img, img_size)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._latest_encoded_obs_list = [encode_obs(obs, self.default_prompt) for obs in obs_list]
        if self.lang_embeddings is None:
            self._set_language_instruction(self._latest_encoded_obs_list[0]["prompt"])

        for env_idx, encoded_obs in zip(self._latest_env_idx_list, self._latest_encoded_obs_list):
            window = self._observation_windows.get(env_idx)
            if window is None:
                window = deque(maxlen=2)
                window.append(
                    {
                        "qpos": None,
                        "images": {
                            self.config["camera_names"][0]: None,
                            self.config["camera_names"][1]: None,
                            self.config["camera_names"][2]: None,
                        },
                    }
                )
                self._observation_windows[env_idx] = window

            img_front = self._jpeg_mapping(self._resize_img(encoded_obs["images"]["cam_high"]))
            img_right = self._jpeg_mapping(self._resize_img(encoded_obs["images"]["cam_right_wrist"]))
            img_left = self._jpeg_mapping(self._resize_img(encoded_obs["images"]["cam_left_wrist"]))
            qpos = torch.from_numpy(np.asarray(encoded_obs["state"], dtype=np.float32)).float().to(self.device)

            window.append(
                {
                    "qpos": qpos,
                    "images": {
                        self.config["camera_names"][0]: img_front,
                        self.config["camera_names"][1]: img_right,
                        self.config["camera_names"][2]: img_left,
                    },
                }
            )
        self.observation_window = self._observation_windows[self._latest_env_idx_list[0]]

    @torch.inference_mode()
    def infer(self, observation_window=None):
        observation_window = observation_window or self.observation_window
        if observation_window is None or self.lang_embeddings is None:
            raise AssertionError("update_obs must be called before get_action.")

        image_arrs = [
            observation_window[-2]["images"][self.config["camera_names"][0]],
            observation_window[-2]["images"][self.config["camera_names"][1]],
            observation_window[-2]["images"][self.config["camera_names"][2]],
            observation_window[-1]["images"][self.config["camera_names"][0]],
            observation_window[-1]["images"][self.config["camera_names"][1]],
            observation_window[-1]["images"][self.config["camera_names"][2]],
        ]
        images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
        proprio = observation_window[-1]["qpos"].unsqueeze(0)
        actions = self.policy.step(proprio=proprio, images=images, text_embeds=self.lang_embeddings)
        return actions.squeeze(0).float().cpu().numpy()

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        for env_idx in env_idx_list:
            raw_actions = self.infer(self._observation_windows[env_idx])
            action_list.append(unpack_robot_state(raw_actions, self.action_type, self.robot_action_dim_info, source_type="obs"))
        return action_list

    def reset(self):
        self.lang_embeddings = None
        self.observation_window = None
        self._observation_windows = {}
        self._latest_encoded_obs_list = []
        self._latest_env_idx_list = [0]
