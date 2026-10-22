from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import WSALargeNativeSchedulerConfig
from lerobot.policies.WSA_Base.da3_teacher import resolve_da3_backbone_defaults
from lerobot.policies.names import WSA_LARGE
from lerobot.utils.constants import ACTION, OBS_STATE


FUTURE_3D_QUERY_MODES = ("query_token", "noised_query_token", "slot_noise")
FUTURE_3D_QUERY_SIGMA_SOURCES = ("constant", "video")


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


def _default_future_3d_config(
    da3_query_dim: int,
    da3_num_views: int,
    da3_tokens_per_view: int,
    future_3d_tokens_per_view: int,
    query_mode: str,
    query_noise_scale: float,
    query_noise_min_sigma: float,
    query_noise_max_sigma: float,
    query_sigma_source: str,
    slot_pos_scale: float,
) -> dict[str, Any]:
    return {
        "hidden_dim": 768,
        "ffn_dim": 3072,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "num_query_tokens": da3_num_views * future_3d_tokens_per_view,
        "da3_num_views": da3_num_views,
        "da3_tokens_per_view": da3_tokens_per_view,
        "da3_query_dim": da3_query_dim,
        "query_layer_indices": (14, 19, 24, 29),
        "future_query_init_std": 0.02,
        "query_mode": query_mode,
        "query_noise_scale": query_noise_scale,
        "query_noise_min_sigma": query_noise_min_sigma,
        "query_noise_max_sigma": query_noise_max_sigma,
        "query_sigma_source": query_sigma_source,
        "slot_pos_scale": slot_pos_scale,
        "use_gradient_checkpointing": True,
    }


def _percent_aligned_future_3d_query_layers(
    da3_teacher_layers: tuple[int, ...],
    future_3d_num_layers: int,
) -> tuple[int, ...]:
    if future_3d_num_layers <= 0:
        raise ValueError("future_3d_config.num_layers must be positive")
    if len(da3_teacher_layers) == 0:
        raise ValueError("da3_teacher_layers must not be empty")
    max_teacher_layer = max(int(idx) for idx in da3_teacher_layers)
    if max_teacher_layer <= 0:
        raise ValueError(f"Cannot percent-align DA3 teacher layers {da3_teacher_layers}.")
    max_future_layer = future_3d_num_layers - 1
    return tuple(
        min(max_future_layer, max(0, round(int(layer_idx) / max_teacher_layer * max_future_layer)))
        for layer_idx in da3_teacher_layers
    )


@DatasetConfig.register_subclass(WSA_LARGE)
@DatasetConfig.register_subclass("wsa_large")
# Compatibility aliases for old development checkpoints/configs.
@DatasetConfig.register_subclass("TBot_SA1_Wan")
@DatasetConfig.register_subclass("tbot_sa1_wan")
@DatasetConfig.register_subclass("magicbot-r0")
@DatasetConfig.register_subclass("magicbot_r0")
@DatasetConfig.register_subclass("MagicBot_R0")
@dataclass
class WSALargeDatasetConfig(DatasetConfig):
    _canonical_type: ClassVar[str] = WSA_LARGE

    repo_id: str = "wsa_large_local"
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
    standardize_video_size_by_cameras: bool = True
    camera_key: str | None = None
    val_set_proportion: float = 0.0
    skip_padding_as_possible: bool = False
    max_padding_retry: int = 3
    concat_multi_camera: str = "auto"
    override_instruction: str | None = None
    return_future_3d_images: bool = True
    future_3d_target_index: int = -1

    processor_num_output_cameras: int = 2
    processor_action_output_dim: int = 24
    processor_proprio_output_dim: int = 24
    processor_resize_shape: tuple[int, int] | None = None
    processor_use_stepwise_action_norm: bool = False
    processor_norm_default_mode: str = "min/max"
    processor_use_zh_instruction: bool = False
    processor_delta_action_dim_mask: dict[str, list[bool]] | None = field(
        default_factory=lambda: {"default": [True, True, True, True, True, True, False]}
    )
    pretrain_multi_embodiment: bool = False

    text_embedding_cache_dir: str | None = None
    text_embedding_cache_max_entries: int = 0
    cache_in_memory: bool = False
    context_len: int = 128
    normalization_stats_path: str | None = None
    dataset_sampling_weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.dataset_dirs) == 0 and self.repo_id_file is None:
            raise ValueError(
                "WSA_Large dataset needs either `dataset.dataset_dirs` or `dataset.repo_id_file` "
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
        if len(self.video_size) != 2:
            raise ValueError("video_size must be a pair of (height, width).")
        self.video_size = tuple(int(value) for value in self.video_size)
        if self.video_size[0] <= 0 or self.video_size[1] <= 0:
            raise ValueError("video_size values must be positive.")
        if self.concat_multi_camera not in {"auto", "horizontal", "vertical", "robotwin"}:
            raise ValueError("concat_multi_camera must be one of: auto, horizontal, vertical, robotwin.")
        self.text_embedding_cache_max_entries = int(self.text_embedding_cache_max_entries)
        if self.text_embedding_cache_max_entries < 0:
            raise ValueError("text_embedding_cache_max_entries must be non-negative.")

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


@PreTrainedConfig.register_subclass(WSA_LARGE)
@PreTrainedConfig.register_subclass("wsa_large")
# Compatibility aliases for old development checkpoints/configs.
@PreTrainedConfig.register_subclass("TBot_SA1_Wan")
@PreTrainedConfig.register_subclass("tbot_sa1_wan")
@PreTrainedConfig.register_subclass("magicbot-r0")
@PreTrainedConfig.register_subclass("magicbot_r0")
@PreTrainedConfig.register_subclass("MagicBot_R0")
@dataclass
class WSALargeConfig(PreTrainedConfig):
    _canonical_type: ClassVar[str] = WSA_LARGE

    variant: str = "wsa_large"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    tokenizer_max_len: int = 128
    load_text_encoder: bool = False
    redirect_common_files: bool = True
    mot_checkpoint_mixed_attn: bool = True
    skip_dit_load_from_pretrain: bool = False
    action_dit_pretrained_path: str = "checkpoints/wsa_large/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    future_3d_pretrained_path: str | None = (
        "checkpoints/wsa_large/Future3DExpert_linear_interp_Wan22_alphascale_768hdim.pt"
    )
    native_checkpoint_path: str | None = None
    dtype: str = "bfloat16"
    device: str | None = None

    action_dim: int = 24
    proprio_dim: int = 24
    action_horizon: int = 32
    n_action_steps: int = 1
    num_inference_steps: int = 10

    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    future_3d_config: dict[str, Any] | None = None
    lambda_3d: float = 0.05
    da3_variant: str = "large"
    da3_teacher_layers: tuple[int, ...] | None = None
    da3_query_dim: int | None = None
    da3_tokens_per_view: int = 1296
    da3_num_views: int = 2
    future_3d_tokens_per_view: int = 144
    future_3d_view_attention_layout: str = "auto"
    da3_model_path_or_name: str = "depth-anything/DA3-LARGE-1.1"
    da3_model_name: str | None = None
    da3_code_root: str | None = None
    da3_teacher_process_res: int = 504
    da3_layer_weights: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6)
    future_query_init_std: float = 0.02
    future_3d_query_mode: str = "slot_noise"
    future_3d_query_noise_scale: float = 0.5
    future_3d_query_noise_min_sigma: float = 0.0
    future_3d_query_noise_max_sigma: float = 0.5
    future_3d_query_sigma_source: str = "constant"
    future_3d_slot_pos_scale: float = 0.5
    future_3d_target_index: int = -1
    log_da3_teacher_timing: bool = False
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
    mask_action_dim_padding_loss: bool = False
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
        variant_aliases = {
            "wsa_large": "wsa_large",
            "wsa_large_joint": "wsa_large_joint",
            "WSA_Large": "wsa_large",
            "WSA_Large_joint": "wsa_large_joint",
            "tbot_sa1_wan": "wsa_large",
            "tbot_sa1_wan_joint": "wsa_large_joint",
            "TBot_SA1_Wan": "wsa_large",
            "TBot_SA1_Wan_joint": "wsa_large_joint",
            "magicbot_r0": "wsa_large",
            "magicbot_r0_joint": "wsa_large_joint",
        }
        self.variant = variant_aliases.get(self.variant, self.variant)
        if self.variant not in {"wsa_large", "wsa_large_joint"}:
            raise ValueError(
                f"Unsupported WSA_Large variant '{self.variant}'. "
                "Expected one of: wsa_large, wsa_large_joint."
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
        if self.da3_model_name is not None:
            self.da3_model_path_or_name = self.da3_model_name
        da3_defaults = resolve_da3_backbone_defaults(self.da3_model_path_or_name, self.da3_variant)
        if self.da3_teacher_layers is None:
            self.da3_teacher_layers = tuple(int(layer_idx) for layer_idx in da3_defaults["teacher_layers"])
        if self.da3_query_dim is None:
            self.da3_query_dim = int(da3_defaults["query_dim"])
        if self.da3_num_views <= 0:
            raise ValueError("da3_num_views must be positive")
        if self.da3_tokens_per_view <= 0:
            raise ValueError("da3_tokens_per_view must be positive")
        if self.future_3d_tokens_per_view <= 0:
            raise ValueError("future_3d_tokens_per_view must be positive")
        if self.future_3d_view_attention_layout not in {"auto", "horizontal", "vertical", "robotwin", "full"}:
            raise ValueError(
                "future_3d_view_attention_layout must be one of: "
                "auto, horizontal, vertical, robotwin, full"
            )
        if self.future_3d_query_mode not in FUTURE_3D_QUERY_MODES:
            raise ValueError(
                f"future_3d_query_mode must be one of {FUTURE_3D_QUERY_MODES}, "
                f"got {self.future_3d_query_mode!r}"
            )
        if self.future_3d_query_sigma_source not in FUTURE_3D_QUERY_SIGMA_SOURCES:
            raise ValueError(
                f"future_3d_query_sigma_source must be one of {FUTURE_3D_QUERY_SIGMA_SOURCES}, "
                f"got {self.future_3d_query_sigma_source!r}"
            )
        if self.future_3d_query_noise_scale < 0:
            raise ValueError("future_3d_query_noise_scale must be non-negative")
        if not (0.0 <= self.future_3d_query_noise_min_sigma <= self.future_3d_query_noise_max_sigma <= 1.0):
            raise ValueError(
                "future_3d_query_noise_min_sigma and future_3d_query_noise_max_sigma must satisfy "
                "0 <= min <= max <= 1"
            )
        if len(self.da3_teacher_layers) != len(self.da3_layer_weights):
            raise ValueError("da3_layer_weights must align with da3_teacher_layers")
        has_explicit_future_3d_query_layers = (
            self.future_3d_config is not None and "query_layer_indices" in self.future_3d_config
        )
        if self.future_3d_config is None:
            self.future_3d_config = _default_future_3d_config(
                da3_query_dim=self.da3_query_dim,
                da3_num_views=self.da3_num_views,
                da3_tokens_per_view=self.da3_tokens_per_view,
                future_3d_tokens_per_view=self.future_3d_tokens_per_view,
                query_mode=self.future_3d_query_mode,
                query_noise_scale=self.future_3d_query_noise_scale,
                query_noise_min_sigma=self.future_3d_query_noise_min_sigma,
                query_noise_max_sigma=self.future_3d_query_noise_max_sigma,
                query_sigma_source=self.future_3d_query_sigma_source,
                slot_pos_scale=self.future_3d_slot_pos_scale,
            )
        else:
            default_future_3d_config = _default_future_3d_config(
                da3_query_dim=self.da3_query_dim,
                da3_num_views=self.da3_num_views,
                da3_tokens_per_view=self.da3_tokens_per_view,
                future_3d_tokens_per_view=self.future_3d_tokens_per_view,
                query_mode=self.future_3d_query_mode,
                query_noise_scale=self.future_3d_query_noise_scale,
                query_noise_min_sigma=self.future_3d_query_noise_min_sigma,
                query_noise_max_sigma=self.future_3d_query_noise_max_sigma,
                query_sigma_source=self.future_3d_query_sigma_source,
                slot_pos_scale=self.future_3d_slot_pos_scale,
            )
            default_future_3d_config.update(dict(self.future_3d_config))
            self.future_3d_config = default_future_3d_config
            self.future_3d_config.setdefault("da3_query_dim", self.da3_query_dim)
            self.future_3d_config.setdefault("da3_num_views", self.da3_num_views)
            self.future_3d_config.setdefault("da3_tokens_per_view", self.da3_tokens_per_view)
            self.future_3d_config.setdefault("future_query_init_std", self.future_query_init_std)
        self.future_3d_config["future_query_init_std"] = float(self.future_query_init_std)
        self.future_3d_config["query_mode"] = self.future_3d_query_mode
        self.future_3d_config["query_noise_scale"] = float(self.future_3d_query_noise_scale)
        self.future_3d_config["query_noise_min_sigma"] = float(self.future_3d_query_noise_min_sigma)
        self.future_3d_config["query_noise_max_sigma"] = float(self.future_3d_query_noise_max_sigma)
        self.future_3d_config["query_sigma_source"] = self.future_3d_query_sigma_source
        self.future_3d_config["slot_pos_scale"] = float(self.future_3d_slot_pos_scale)
        future_3d_num_layers = int(self.future_3d_config["num_layers"])
        if has_explicit_future_3d_query_layers:
            query_layer_indices = self.future_3d_config["query_layer_indices"]
        else:
            query_layer_indices = _percent_aligned_future_3d_query_layers(
                self.da3_teacher_layers,
                future_3d_num_layers,
            )
        self.future_3d_config["query_layer_indices"] = tuple(int(idx) for idx in query_layer_indices)
        if len(self.future_3d_config["query_layer_indices"]) != len(self.da3_teacher_layers):
            raise ValueError("future_3d_config.query_layer_indices and da3_teacher_layers must have the same length")
        invalid_query_layers = [
            idx for idx in self.future_3d_config["query_layer_indices"] if idx < 0 or idx >= future_3d_num_layers
        ]
        if invalid_query_layers:
            raise ValueError(
                "future_3d_config.query_layer_indices must be valid 0-based layer indices for "
                f"num_layers={future_3d_num_layers}; got invalid indices {invalid_query_layers}."
            )
        if self.future_3d_config["num_query_tokens"] % self.da3_num_views != 0:
            raise ValueError("future_3d_config.num_query_tokens must be divisible by da3_num_views")
        if self.scheduler_decay_lr is None:
            self.scheduler_decay_lr = self.optimizer_lr * 0.01
        self.video_dit_config["use_gradient_checkpointing"] = bool(self.mot_checkpoint_mixed_attn)
        self.action_dit_config["use_gradient_checkpointing"] = bool(self.mot_checkpoint_mixed_attn)
        self.future_3d_config["use_gradient_checkpointing"] = bool(self.mot_checkpoint_mixed_attn)

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
        return WSALargeNativeSchedulerConfig(
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
