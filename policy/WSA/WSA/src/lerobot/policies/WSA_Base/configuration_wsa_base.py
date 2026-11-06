#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field, replace
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.names import WSA_BASE
from lerobot.policies.WSA_Base.da3_teacher import resolve_da3_backbone_defaults
from lerobot.policies.WSA_Base.transform_wsa_base import (
    Qwen3_VLProcessorTransformFn,
    UnifyWSABaseInputsTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import OBS_IMAGES


ATTENTION_MASK_MODES = ("default", "causal")


@DatasetConfig.register_subclass(WSA_BASE)
@DatasetConfig.register_subclass("wsa_base")
# Compatibility aliases for old development checkpoints/configs.
@DatasetConfig.register_subclass("TBot_SA1")
@DatasetConfig.register_subclass("tbot_sa1")
@DatasetConfig.register_subclass("magicbot")
@DatasetConfig.register_subclass("cubev2")
@dataclass
class WSABaseDatasetConfig(DatasetConfig):
    _canonical_type: ClassVar[str] = WSA_BASE

    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_vl_processor_path: str = "Qwen/Qwen3-VL-2B-Instruct"

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=WSABaseDatasetConfig.height,
                    width=WSABaseDatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=WSABaseDatasetConfig.max_state_dim,
                    max_action_dim=WSABaseDatasetConfig.max_action_dim,
                ),
                Qwen3_VLProcessorTransformFn(),
                UnifyWSABaseInputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        uses_pi05_image_aug = (
            self.image_transforms.enable
            and self.image_transforms.preset in {"pi05", "pi0.5", "pi05_style"}
        )
        inputs = [t for t in inputs if not isinstance(t, Pi05ImageAugmentFn)]
        if uses_pi05_image_aug:
            insert_idx = next(
                (idx + 1 for idx, transform in enumerate(inputs) if isinstance(transform, ResizeImagesWithPadFn)),
                0,
            )
            inputs.insert(insert_idx, Pi05ImageAugmentFn())

        for idx, transform in enumerate(inputs):
            if isinstance(transform, Qwen3_VLProcessorTransformFn):
                inputs[idx] = replace(
                    transform,
                    pretrained_model_name_or_path=self.qwen3_vl_processor_path,
                )
        # Persist transform overrides even when the delta transform set is unchanged.
        self.data_transforms = replace(self.data_transforms, inputs=inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta":
            if not has_delta:
                inputs = [DeltaActionTransformFn(), *inputs]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)
        else:
            if has_delta:
                inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)


@DatasetConfig.register_subclass("robochallenge_raw_w1")
@dataclass
class RoboChallengeRawW1DatasetConfig(WSABaseDatasetConfig):
    """WSABase dataset config for direct RoboChallenge DOS-W1 raw data loading."""

    repo_id: str = "robochallenge_raw_w1"
    raw_root: str = ""
    embodiment: str = "DOS-W1"
    action_representation: str = "joint"
    frame_interval: int = 1
    task_regex: str | None = None
    task_preset: str | None = None
    weighted_task_sampling: bool = False
    task_sampling_mode: str = "per_task"
    task_sampling_gamma: float = 1.0
    regular_task_weight: float = 1.0
    extra_task_weight: float = 0.8
    regular_task_total_weight: float | None = None
    extra_task_total_weight: float | None = None
    state_cache_dir: str | None = None
    state_cache_size: int = 32
    validate_videos: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        if self.action_representation != "joint":
            raise ValueError("robochallenge_raw_* currently supports action_representation='joint' only")
        self.task_sampling_mode = str(self.task_sampling_mode or "none").strip().lower()
        if self.task_sampling_mode not in {"none", "per_task", "group_frames_pow"}:
            raise ValueError("task_sampling_mode must be one of: none, per_task, group_frames_pow")
        if self.task_sampling_gamma < 0:
            raise ValueError("task_sampling_gamma must be non-negative")
        if self.regular_task_weight <= 0:
            raise ValueError("regular_task_weight must be positive")
        if self.extra_task_weight <= 0:
            raise ValueError("extra_task_weight must be positive")
        if self.regular_task_total_weight is not None and self.regular_task_total_weight <= 0:
            raise ValueError("regular_task_total_weight must be positive when set")
        if self.extra_task_total_weight is not None and self.extra_task_total_weight <= 0:
            raise ValueError("extra_task_total_weight must be positive when set")


@DatasetConfig.register_subclass("robochallenge_raw_aloha")
@dataclass
class RoboChallengeRawAlohaDatasetConfig(RoboChallengeRawW1DatasetConfig):
    """WSABase dataset config for direct RoboChallenge ALOHA raw data loading."""

    repo_id: str = "robochallenge_raw_aloha"
    embodiment: str = "ALOHA"
    task_preset: str | None = "table30v2_aloha"
    weighted_task_sampling: bool = True
    task_sampling_mode: str = "group_frames_pow"
    task_sampling_gamma: float = 0.8
    regular_task_total_weight: float | None = 4.0
    extra_task_total_weight: float | None = 4.0


@PreTrainedConfig.register_subclass(WSA_BASE)
@PreTrainedConfig.register_subclass("wsa_base")
# Compatibility aliases for old development checkpoints/configs.
@PreTrainedConfig.register_subclass("TBot_SA1")
@PreTrainedConfig.register_subclass("tbot_sa1")
@PreTrainedConfig.register_subclass("magicbot")
@PreTrainedConfig.register_subclass("cubev2")
@dataclass
class WSABaseConfig(PreTrainedConfig):
    _canonical_type: ClassVar[str] = WSA_BASE

    qwen3_vl_variant: str = "qwen3_vl_2b"
    action_expert_variant: str = "qwen3_600m"
    qwen3_vl_pretrained_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32
    mask_action_dim_padding_loss: bool = False
    action_loss_valid_dim: int | None = None

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    attention_mask_mode: str = "default"

    image_resolution: tuple[int, int] = (224, 224)
    image_delta_indices: list[int] = field(default_factory=lambda: [0, 0, 15])
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_vlm_only: bool = False
    lora_modules: tuple[str, ...] = ()
    lora_unselected_mode: str = "full"
    lora_targets: tuple[str, ...] = ("attn", "ffn")
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_rank_und: int | None = None
    lora_alpha_und: float | None = None
    lora_rank_gen: int | None = None
    lora_alpha_gen: float | None = None
    lora_rank_act: int | None = None
    lora_alpha_act: float | None = None
    lora_dropout: float = 0.0

    scale_factor: int = 8
    lambda_gen: float = 0.01
    cosmos_tokenizer_path_or_name: str = "nvidia/Cosmos-Tokenizer-CI8x8"

    enable_3d_queries: bool = False
    num_3d_query_tokens: int = 1296  # compressed future-3D bottleneck queries
    da3_alignment_mode: str = "query_decoder"
    da3_query_resampler_layers: int = 1  # kept for config compatibility; fixed to 1
    da3_query_resampler_ff_mult: int = 1  # kept for config compatibility; fixed to 1
    query_layer_indices: tuple[int, ...] = (13, 19, 23, 27)
    da3_variant: str = "auto"
    da3_teacher_layers: tuple[int, ...] | None = None
    da3_query_dim: int | None = None
    da3_tokens_per_view: int = 1296
    da3_num_views: int = 3
    lambda_3d: float = 0.05
    da3_model_path_or_name: str = "depth-anything/DA3-LARGE-1.1"
    da3_model_name: str | None = None
    da3_code_root: str | None = None
    da3_teacher_process_res: int = 504
    da3_layer_weights: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6)
    future_query_init_std: float = 0.02
    log_da3_teacher_timing: bool = False

    def __post_init__(self):
        super().__post_init__()

        self.lora_modules = tuple(dict.fromkeys(module_name.lower() for module_name in self.lora_modules))
        self.lora_targets = tuple(dict.fromkeys(target_name.lower() for target_name in self.lora_targets))

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.action_loss_valid_dim is not None:
            self.action_loss_valid_dim = int(self.action_loss_valid_dim)
            if self.action_loss_valid_dim <= 0:
                raise ValueError("action_loss_valid_dim must be positive when set")
            if self.action_loss_valid_dim > self.max_action_dim:
                raise ValueError(
                    f"action_loss_valid_dim ({self.action_loss_valid_dim}) cannot exceed "
                    f"max_action_dim ({self.max_action_dim})"
                )
        if self.mask_action_dim_padding_loss and self.action_loss_valid_dim is None:
            raise ValueError("action_loss_valid_dim must be set when mask_action_dim_padding_loss=true")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.attention_mask_mode not in ATTENTION_MASK_MODES:
            raise ValueError(
                f"attention_mask_mode must be one of {ATTENTION_MASK_MODES}, got {self.attention_mask_mode!r}"
            )
        self.image_delta_indices = [int(idx) for idx in self.image_delta_indices]
        if len(self.image_delta_indices) != 3:
            raise ValueError("image_delta_indices must contain exactly 3 frame offsets.")

        supported_lora_modules = {"und", "gen", "act"}
        unsupported_lora_modules = set(self.lora_modules) - supported_lora_modules
        if unsupported_lora_modules:
            raise ValueError(
                f"Unsupported LoRA modules: {sorted(unsupported_lora_modules)}. "
                f"Expected a subset of {sorted(supported_lora_modules)}."
            )

        supported_lora_targets = {"attn", "ffn"}
        unsupported_lora_targets = set(self.lora_targets) - supported_lora_targets
        if unsupported_lora_targets:
            raise ValueError(
                f"Unsupported LoRA target groups: {sorted(unsupported_lora_targets)}. "
                f"Expected a subset of {sorted(supported_lora_targets)}."
            )

        if self.lora_unselected_mode not in {"full", "freeze"}:
            raise ValueError("lora_unselected_mode must be one of: 'full', 'freeze'")

        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        per_module_lora_values = {
            "lora_rank_und": self.lora_rank_und,
            "lora_alpha_und": self.lora_alpha_und,
            "lora_rank_gen": self.lora_rank_gen,
            "lora_alpha_gen": self.lora_alpha_gen,
            "lora_rank_act": self.lora_rank_act,
            "lora_alpha_act": self.lora_alpha_act,
        }
        for field_name, field_value in per_module_lora_values.items():
            if field_value is None:
                continue
            if field_value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")

        if self.lora_enabled and (self.train_expert_only or self.train_vlm_only):
            raise ValueError(
                "train_expert_only/train_vlm_only cannot be combined with LoRA expert selection. "
                "Use lora_modules to choose which experts are adapted."
            )

        if self.enable_3d_queries and self.num_3d_query_tokens <= 0:
            raise ValueError("num_3d_query_tokens must be positive when 3D queries are enabled")

        if self.enable_3d_queries and self.num_3d_query_tokens % self.da3_num_views != 0:
            raise ValueError(
                "num_3d_query_tokens must be divisible by da3_num_views for view-aware 3D alignment"
            )
        if self.attention_mask_mode == "causal" and not self.enable_3d_queries:
            raise ValueError("attention_mask_mode='causal' requires enable_3d_queries=true")

        if self.da3_alignment_mode not in {"query_decoder", "upsample"}:
            raise ValueError(
                "da3_alignment_mode must be one of: 'query_decoder', 'upsample'"
            )
        if self.da3_query_resampler_layers != 1:
            raise ValueError("da3_query_resampler_layers is fixed to 1 in the current WSABase query decoder")
        if self.da3_query_resampler_ff_mult != 1:
            raise ValueError("da3_query_resampler_ff_mult is fixed to 1 in the current WSABase query decoder")

        if self.da3_model_name is not None:
            self.da3_model_path_or_name = self.da3_model_name

        da3_defaults = resolve_da3_backbone_defaults(
            self.da3_model_path_or_name,
            self.da3_variant,
        )
        if self.da3_teacher_layers is None:
            self.da3_teacher_layers = tuple(int(layer_idx) for layer_idx in da3_defaults["teacher_layers"])
        if self.da3_query_dim is None:
            self.da3_query_dim = int(da3_defaults["query_dim"])

        if len(self.query_layer_indices) != len(self.da3_teacher_layers):
            raise ValueError("query_layer_indices and da3_teacher_layers must have the same length")

        if len(self.query_layer_indices) != len(self.da3_layer_weights):
            raise ValueError("da3_layer_weights must align with query_layer_indices")

    @property
    def lora_enabled(self) -> bool:
        return len(self.lora_modules) > 0

    def _get_lora_rank_overrides(self) -> dict[str, int | None]:
        return {
            "und": self.lora_rank_und,
            "gen": self.lora_rank_gen,
            "act": self.lora_rank_act,
        }

    def _get_lora_alpha_overrides(self) -> dict[str, float | None]:
        return {
            "und": self.lora_alpha_und,
            "gen": self.lora_alpha_gen,
            "act": self.lora_alpha_act,
        }

    def get_lora_rank_for(self, module_name: str) -> int:
        module_name = module_name.lower()
        override = self._get_lora_rank_overrides().get(module_name)
        return self.lora_rank if override is None else override

    def get_lora_alpha_for(self, module_name: str) -> float:
        module_name = module_name.lower()
        override = self._get_lora_alpha_overrides().get(module_name)
        return self.lora_alpha if override is None else override

    def get_lora_hparams_for(self, module_name: str) -> tuple[int, float]:
        return self.get_lora_rank_for(module_name), self.get_lora_alpha_for(module_name)

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
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
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
