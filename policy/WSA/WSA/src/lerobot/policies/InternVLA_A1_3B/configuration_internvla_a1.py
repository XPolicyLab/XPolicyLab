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

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import (
    Qwen3_VLProcessorTransformFn,
    UnifyQwenA1InputsTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import OBS_IMAGES


@DatasetConfig.register_subclass("qwena1")
@dataclass
class QwenA1DatasetConfig(DatasetConfig):
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
                    height=QwenA1DatasetConfig.height,
                    width=QwenA1DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                Qwen3_VLProcessorTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=QwenA1DatasetConfig.max_state_dim,
                    max_action_dim=QwenA1DatasetConfig.max_action_dim,
                ),
                UnifyQwenA1InputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        for idx, transform in enumerate(inputs):
            if isinstance(transform, Qwen3_VLProcessorTransformFn):
                inputs[idx] = replace(
                    transform,
                    pretrained_model_name_or_path=self.qwen3_vl_processor_path,
                )
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


@PreTrainedConfig.register_subclass("qwena1")
@dataclass
class QwenA1Config(PreTrainedConfig):
    qwen3_vl_variant: str = "qwen3_vl_2b"
    action_expert_variant: str = "qwen3_600m"
    qwen3_vl_pretrained_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
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

    scale_factor: int = 8
    lambda_gen: float = 0.01
    cosmos_tokenizer_path_or_name: str = "nvidia/Cosmos-Tokenizer-CI8x8"
    cosmos_device: str | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

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

    @property
    def image_delta_indices(self) -> list | None:
        return [-15, 0, 15]
