from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import FastWAMNativeSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _default_video_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": True,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def _default_action_dit_config(action_dim: int) -> dict[str, Any]:
    return {
        "action_dim": action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": True,
    }


@DatasetConfig.register_subclass("fastwam")
@dataclass
class FastWAMDatasetConfig(DatasetConfig):
    repo_id: str = "fastwam_local"
    dataset_dirs: list[str] = field(default_factory=list)

    image_keys: list[str] = field(default_factory=lambda: ["image", "wrist_image"])
    image_raw_shapes: list[list[int]] = field(default_factory=lambda: [[3, 512, 512], [3, 512, 512]])
    image_shapes: list[list[int]] = field(default_factory=lambda: [[3, 224, 224], [3, 224, 224]])

    action_keys: list[str] = field(default_factory=lambda: ["default"])
    action_raw_shapes: list[int] = field(default_factory=lambda: [7])
    action_shapes: list[int] = field(default_factory=lambda: [7])

    state_keys: list[str] = field(default_factory=lambda: ["default"])
    state_raw_shapes: list[int] = field(default_factory=lambda: [8])
    state_shapes: list[int] = field(default_factory=lambda: [8])

    num_frames: int = 33
    global_sample_stride: int = 1
    action_video_freq_ratio: int = 4
    video_size: tuple[int, int] = (224, 448)
    camera_key: str | None = None
    val_set_proportion: float = 0.0
    skip_padding_as_possible: bool = False
    max_padding_retry: int = 3
    concat_multi_camera: str = "horizontal"
    override_instruction: str | None = None

    processor_num_output_cameras: int = 2
    processor_action_output_dim: int = 7
    processor_proprio_output_dim: int = 8
    processor_resize_shape: tuple[int, int] | None = None
    processor_use_stepwise_action_norm: bool = False
    processor_norm_default_mode: str = "min/max"
    processor_use_zh_instruction: bool = False
    processor_delta_action_dim_mask: dict[str, list[bool]] | None = field(
        default_factory=lambda: {"default": [True, True, True, True, True, True, False]}
    )

    text_embedding_cache_dir: str | None = None
    context_len: int = 128
    normalization_stats_path: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.dataset_dirs) == 0 and self.repo_id_file is None:
            raise ValueError(
                "FastWAM dataset needs either `dataset.dataset_dirs` or `dataset.repo_id_file` "
                "with one local LeRobot v3.0 dataset path per line."
            )
        if not (
            len(self.image_keys) == len(self.image_raw_shapes) == len(self.image_shapes)
        ):
            raise ValueError("Image metadata lengths must match.")
        if not (
            len(self.action_keys) == len(self.action_raw_shapes) == len(self.action_shapes)
        ):
            raise ValueError("Action metadata lengths must match.")
        if not (
            len(self.state_keys) == len(self.state_raw_shapes) == len(self.state_shapes)
        ):
            raise ValueError("State metadata lengths must match.")

    @property
    def shape_meta(self) -> dict[str, Any]:
        return {
            "images": [
                {
                    "key": key,
                    "raw_shape": raw_shape,
                    "shape": shape,
                }
                for key, raw_shape, shape in zip(
                    self.image_keys, self.image_raw_shapes, self.image_shapes, strict=True
                )
            ],
            "action": [
                {
                    "key": key,
                    "raw_shape": raw_shape,
                    "shape": shape,
                }
                for key, raw_shape, shape in zip(
                    self.action_keys, self.action_raw_shapes, self.action_shapes, strict=True
                )
            ],
            "state": [
                {
                    "key": key,
                    "raw_shape": raw_shape,
                    "shape": shape,
                }
                for key, raw_shape, shape in zip(
                    self.state_keys, self.state_raw_shapes, self.state_shapes, strict=True
                )
            ],
        }


@PreTrainedConfig.register_subclass("fastwam")
@dataclass
class FastWAMConfig(PreTrainedConfig):
    variant: str = "fastwam"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    tokenizer_max_len: int = 128
    load_text_encoder: bool = False
    redirect_common_files: bool = True
    mot_checkpoint_mixed_attn: bool = True
    skip_dit_load_from_pretrain: bool = False
    action_dit_pretrained_path: str = "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    native_checkpoint_path: str | None = None
    dtype: str = "bfloat16"
    device: str | None = None

    action_dim: int = 7
    proprio_dim: int = 8
    action_horizon: int = 32
    n_action_steps: int = 1
    num_inference_steps: int = 10

    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    video_scheduler: dict[str, Any] = field(
        default_factory=lambda: {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        }
    )
    action_scheduler: dict[str, Any] = field(
        default_factory=lambda: {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        }
    )
    lambda_video: float = 1.0
    lambda_action: float = 1.0
    action_norm_use_stepwise: bool = False
    action_norm_default_mode: str = "min/max"
    action_norm_exception_mode: dict[str, dict[str, str]] | None = None
    action_stats_path: str | None = None

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-2
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "cosine"
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_ratio: float = 0.05
    scheduler_decay_lr: float | None = None
    train_num_epochs: int | None = None
    train_max_steps: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.variant not in {"fastwam", "fastwam_joint", "fastwam_idm"}:
            raise ValueError(
                f"Unsupported FastWAM variant '{self.variant}'. "
                "Expected one of: fastwam, fastwam_joint, fastwam_idm."
            )
        if self.video_dit_config is None:
            self.video_dit_config = _default_video_dit_config(self.action_dim)
        else:
            self.video_dit_config = dict(self.video_dit_config)
            self.video_dit_config.setdefault("action_dim", self.action_dim)
        if self.action_dit_config is None:
            self.action_dit_config = _default_action_dit_config(self.action_dim)
        else:
            self.action_dit_config = dict(self.action_dit_config)
            self.action_dit_config.setdefault("action_dim", self.action_dim)
        if self.scheduler_decay_lr is None:
            self.scheduler_decay_lr = self.optimizer_lr * 0.01
        self.video_dit_config["use_gradient_checkpointing"] = bool(self.mot_checkpoint_mixed_attn)
        self.action_dit_config["use_gradient_checkpointing"] = bool(self.mot_checkpoint_mixed_attn)

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.proprio_dim,),
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return FastWAMNativeSchedulerConfig(
            peak_lr=self.optimizer_lr,
            min_lr_ratio=self.scheduler_decay_lr / self.optimizer_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            scheduler_type=self.scheduler_type,
            warmup_ratio=self.scheduler_warmup_ratio,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype '{self.dtype}'.")
        return mapping[self.dtype]
