#!/usr/bin/env python

from dataclasses import dataclass, field, replace

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.qwenaction.transform_qwenaction import (
    QwenActionProcessorTransformFn,
    UnifyQwenActionInputsTransformFn,
)
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DeltaActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    TransformGroup,
)
from lerobot.utils.constants import OBS_IMAGES


@DatasetConfig.register_subclass("qwenaction")
@dataclass
class QwenActionDatasetConfig(DatasetConfig):
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
                    height=QwenActionDatasetConfig.height,
                    width=QwenActionDatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                QwenActionProcessorTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=QwenActionDatasetConfig.max_state_dim,
                    max_action_dim=QwenActionDatasetConfig.max_action_dim,
                ),
                UnifyQwenActionInputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        for idx, transform in enumerate(inputs):
            if isinstance(transform, QwenActionProcessorTransformFn):
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
        elif has_delta:
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
            self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("qwenaction")
@dataclass
class QwenActionConfig(PreTrainedConfig):
    qwen3_vl_variant: str = "qwen3_vl_28l"
    action_expert_variant: str = "qwen3_28l"
    qwen3_vl_pretrained_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    qwen3_vl_processor_path: str | None = None
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

    @property
    def image_delta_indices(self) -> list | None:
        return [0]
