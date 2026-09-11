"""Native, model-agnostic orchestration for WorldScape Policy training."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

from worldscape_policy.conditioning.prompt_format import render_instruction_template
from worldscape_policy.checkpoint.weights_io import (
    DEFAULT_MAX_SHARD_SIZE,
    MODEL_FILENAME,
    MODEL_INDEX_FILENAME,
    checkpoint_weight_files,
    load_checkpoint_state_dict,
    save_checkpoint_state_dict,
)
from worldscape_policy.checkpoint.loader import validate_native_checkpoint_artifacts
from worldscape_policy.data import ConditionMode, TrainingBatch
from worldscape_policy.embodiment import expand_embodiment_ids
from worldscape_policy.training.callbacks import CallbackList, TrainingCallback
from worldscape_policy.training.objective import (
    CompositeObjective,
    ObjectiveInputs,
    ObjectiveResult,
)
from worldscape_policy.training.prompt_schedule import PromptSchedule
from worldscape_policy.training.runtime import (
    NativeDistributedConfig,
    NativeTrainingRuntime,
)
from worldscape_policy.types import (
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    WorldActionOutput,
)
from worldscape_policy.wam.wan22 import Wan22TrainingInputs


@dataclass(frozen=True)
class ModelReadyTrainingBatch:
    """Explicit bridge between native data and the policy's tensor contract."""

    mode: InteractionMode
    observation: ObservationBatch
    prompts: PromptBatch
    clean_video: Tensor
    clean_video_latents: Tensor
    clean_action: Tensor
    clean_video_normalized: bool = False
    mode_mask: Tensor | None = None
    condition_modes: tuple[ConditionMode, ...] | None = None
    condition_ids: Tensor | None = None
    action_mask: Tensor | None = None
    video_weight: Tensor | None = None
    action_weight: Tensor | None = None
    action_dim_mask: Tensor | None = None
    has_real_action: Tensor | None = None
    semantic_target: Tensor | None = None
    semantic_mask: Tensor | None = None
    planning_labels: Tensor | None = None
    planning_labels_text: list[str | None] | None = None

    def validate(self) -> None:
        self.observation.validate()
        self.prompts.validate(self.clean_action.shape[0])
        batch_size = self.clean_action.shape[0]
        if self.mode_mask is not None and (
            self.mode_mask.dtype is not torch.bool
            or self.mode_mask.shape != (batch_size,)
        ):
            raise ValueError(f"mode_mask must be bool with shape [{batch_size}]")
        if (self.condition_modes is None) != (self.condition_ids is None):
            raise ValueError(
                "condition_modes and condition_ids must be provided together"
            )
        if self.condition_modes is not None:
            if len(self.condition_modes) != batch_size:
                raise ValueError("condition_modes must be batch-sized")
            assert self.condition_ids is not None
            if self.condition_ids.shape != (batch_size,) or (
                self.condition_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("condition_ids must be integer with shape [B]")
            expected = torch.tensor(
                [ConditionMode.parse(mode).id for mode in self.condition_modes],
                dtype=self.condition_ids.dtype,
                device=self.condition_ids.device,
            )
            if not torch.equal(self.condition_ids, expected):
                raise ValueError("condition_ids do not match condition_modes")
        if self.clean_video.ndim != 5:
            raise ValueError("clean_video must have shape [B,T,C,H,W]")
        if self.clean_video_latents.ndim != 5:
            raise ValueError("clean_video_latents must have shape [B,C,T,H,W]")
        for name, tensor in (
            ("clean_video", self.clean_video),
            ("clean_video_latents", self.clean_video_latents),
            ("clean_action", self.clean_action),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(f"{name} must have batch size {batch_size}")
        if self.action_dim_mask is not None and self.action_dim_mask.shape not in {
            (self.clean_action.shape[-1],),
            (batch_size, self.clean_action.shape[-1]),
        }:
            raise ValueError("action_dim_mask must have shape [D] or [B,D]")
        if self.has_real_action is not None and (
            self.has_real_action.dtype is not torch.bool
            or self.has_real_action.shape != (batch_size,)
        ):
            raise ValueError(
                f"has_real_action must be bool with shape [{batch_size}]"
            )


LOGGER = logging.getLogger(__name__)


@runtime_checkable
class TrainingBatchAdapter(Protocol):
    def __call__(self, batch: TrainingBatch) -> ModelReadyTrainingBatch: ...


@runtime_checkable
class TrainingNoiseKernel(Protocol):
    def prepare_training_inputs(
        self,
        *,
        clean_video_latents: Tensor,
        clean_action: Tensor,
        generator: torch.Generator,
    ) -> Wan22TrainingInputs: ...


class NativeWan22BatchAdapter:
    """Convert a homogeneous native batch without guessing model semantics.

    Images may be ``[B,T,H,W,C]``, ``[B,T,C,H,W]``, or their multi-view
    variants. Diffusion targets may use the legacy three-view mosaic while
    head-only consumers keep the selected raw view.
    """

    def __init__(
        self,
        *,
        video_latent_encoder: Callable[[Tensor], Tensor],
        diffusion_video_preprocessor: Callable[[Tensor], Tensor] | None = None,
        embodiment_ids: Mapping[str, int],
        image_key: str = "video",
        diffusion_image_key: str | None = None,
        head_view_index: int = 0,
        channels_last: bool = True,
        action_dim: int | None = None,
        state_dim: int | None = None,
        diffusion_view_layout: str = "mosaic_2x2",
    ) -> None:
        self.video_latent_encoder = video_latent_encoder
        self.diffusion_video_preprocessor = diffusion_video_preprocessor
        self.embodiment_ids = expand_embodiment_ids(embodiment_ids)
        self.image_key = image_key
        self.diffusion_image_key = diffusion_image_key
        self.head_view_index = int(head_view_index)
        self.channels_last = bool(channels_last)
        self.action_dim = action_dim
        self.state_dim = state_dim
        if diffusion_view_layout not in {"mosaic_2x2", "robotwin_concat"}:
            raise ValueError(
                "diffusion_view_layout must be 'mosaic_2x2' or "
                "'robotwin_concat'"
            )
        self.diffusion_view_layout = diffusion_view_layout
        if self.diffusion_video_preprocessor is None:
            raise ValueError("diffusion_video_preprocessor is required")
        preprocessor_owner = getattr(
            self.diffusion_video_preprocessor,
            "__self__",
            None,
        )
        codec_layout = getattr(
            preprocessor_owner,
            "diffusion_view_layout",
            diffusion_view_layout,
        )
        if codec_layout != diffusion_view_layout:
            raise ValueError(
                "batch adapter diffusion_view_layout does not match visual codec: "
                f"{diffusion_view_layout!r} != {codec_layout!r}"
            )
        for name, value in (("action_dim", action_dim), ("state_dim", state_dim)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def __call__(self, batch: TrainingBatch) -> ModelReadyTrainingBatch:
        batch.validate()
        self._validate_unmasked_inputs(batch)
        if self.image_key not in batch.observations:
            raise KeyError(f"TrainingBatch has no image modality {self.image_key!r}")
        images = self._images_to_policy_layout(batch.observations[self.image_key])
        diffusion_images = (
            images
            if self.diffusion_image_key is None
            else self._images_to_policy_layout(
                batch.observations[self.diffusion_image_key]
            )
        )
        if not 0 <= self.head_view_index < images.shape[2]:
            raise ValueError("head_view_index is outside the available views")
        head_video = images[:, :, self.head_view_index]
        clean_video = self.diffusion_video_preprocessor(diffusion_images)
        assert clean_video is not None
        history_images = (
            self._history_to_vlm_layout(batch.history_head_frames)
            if batch.history_head_frames is not None
            else None
        )
        try:
            embodiment_id = torch.tensor(
                [self.embodiment_ids[name] for name in batch.embodiments],
                device=batch.robot_state.device,
                dtype=torch.long,
            )
        except KeyError as exc:
            raise KeyError(f"No embodiment id configured for {exc.args[0]!r}") from exc
        high_level_instructions = [
            high or event or ""
            for event, high in zip(
                batch.event_instructions, batch.high_level_instructions, strict=True
            )
        ]
        event_instructions = [
            event or high or ""
            for event, high in zip(
                batch.event_instructions, batch.high_level_instructions, strict=True
            )
        ]
        goal_images = (
            self._visual_prompt_to_policy_layout(
                batch.goal_images, is_goal=True
            )
            if batch.goal_images is not None
            else None
        )
        demo_videos = (
            self._visual_prompt_to_policy_layout(
                batch.demo_videos, is_goal=False
            )
            if batch.demo_videos is not None
            else None
        )
        raw_action_dim = batch.actions.shape[-1]
        clean_action = self._pad_feature_dim(
            batch.actions, self.action_dim, name="actions"
        )
        proprioception = self._pad_feature_dim(
            batch.robot_state, self.state_dim, name="robot_state"
        )
        action_dim_mask = batch.action_dim_mask
        if self.action_dim is not None:
            if action_dim_mask is None:
                action_dim_mask = torch.ones(
                    raw_action_dim,
                    dtype=clean_action.dtype,
                    device=clean_action.device,
                )
            action_dim_mask = self._pad_feature_dim(
                action_dim_mask, self.action_dim, name="action_dim_mask"
            )
        prompts = PromptBatch(
            vlm_planning_text=high_level_instructions,
            language_instruction=event_instructions,
            planning_labels_text=(
                list(batch.planning_labels_text)
                if batch.planning_labels_text is not None
                else event_instructions
            ),
            goal_images=goal_images,
            goal_image_mask=batch.goal_image_mask,
            demo_videos=demo_videos,
            demo_video_mask=batch.demo_video_mask,
            visual_prompt=(
                "none"
                if batch.goal_images is None and batch.demo_videos is None
                else (
                    "mixed"
                    if batch.goal_images is not None
                    and batch.demo_videos is not None
                    else "goal_or_demo"
                )
            ),
        )
        condition_modes, condition_ids = _batch_condition_modes(batch)
        ready = ModelReadyTrainingBatch(
            mode=batch.modes[0],
            mode_mask=batch.mode_mask,
            condition_modes=condition_modes,
            condition_ids=condition_ids,
            observation=ObservationBatch(
                images=images,
                head_view=head_video[:, :1],
                proprioception=proprioception,
                embodiment_id=embodiment_id,
                vlm_history_images=history_images,
                vlm_history_mask=batch.history_mask,
            ),
            prompts=prompts,
            clean_video=clean_video,
            clean_video_latents=self.video_latent_encoder(clean_video),
            clean_action=clean_action,
            clean_video_normalized=True,
            action_mask=batch.action_mask,
            action_dim_mask=action_dim_mask,
            has_real_action=batch.has_real_action,
            semantic_target=batch.semantic_target,
            semantic_mask=batch.semantic_mask,
            planning_labels=batch.planning_labels,
            planning_labels_text=(
                list(batch.planning_labels_text)
                if batch.planning_labels_text is not None
                else event_instructions
            ),
        )
        ready.validate()
        return ready

    @staticmethod
    def _pad_feature_dim(
        value: Tensor,
        target_dim: int | None,
        *,
        name: str,
    ) -> Tensor:
        if target_dim is None or value.shape[-1] == target_dim:
            return value
        if value.shape[-1] > target_dim:
            raise ValueError(
                f"{name} dimension {value.shape[-1]} exceeds target {target_dim}"
            )
        return torch.nn.functional.pad(value, (0, target_dim - value.shape[-1]))

    def _validate_unmasked_inputs(self, batch: TrainingBatch) -> None:
        """Fail closed where the native policy has no padding-mask input."""

        if self.image_key not in batch.observation_masks:
            raise KeyError(f"TrainingBatch has no image modality {self.image_key!r}")
        diffusion_image_key = self.diffusion_image_key or self.image_key
        for name, mask in (
            (
                f"observation_masks[{self.image_key!r}]",
                batch.observation_masks[self.image_key],
            ),
            (
                f"observation_masks[{diffusion_image_key!r}]",
                batch.observation_masks[diffusion_image_key],
            ),
            ("robot_state_mask", batch.robot_state_mask),
        ):
            if not bool(mask.all()):
                raise ValueError(
                    f"{name} contains padding, but NativeWan22BatchAdapter cannot "
                    "propagate that mask; batch these fields at equal temporal length"
                )

    def _images_to_policy_layout(self, value: Tensor) -> Tensor:
        if value.ndim not in (5, 6):
            raise ValueError(
                "image modality must be single- or multi-view batched video"
            )
        if self.channels_last:
            if value.shape[-1] not in (1, 3, 4):
                raise ValueError("channels-last image tensor has invalid channel count")
            value = value.movedim(-1, -3)
        elif value.shape[-3] not in (1, 3, 4):
            raise ValueError("channels-first image tensor has invalid channel count")
        if value.ndim == 5:
            value = value.unsqueeze(2)
        return value

    def _history_to_vlm_layout(self, value: Tensor) -> Tensor:
        if value.ndim != 5:
            raise ValueError(
                "history_head_frames must have shape [B,T,H,W,C] or [B,T,C,H,W]"
            )
        if self.channels_last:
            if value.shape[-1] not in (1, 3, 4):
                raise ValueError("channels-last VLM history has invalid channel count")
            return value.movedim(-1, 2)
        if value.shape[2] not in (1, 3, 4):
            raise ValueError("channels-first VLM history has invalid channel count")
        return value

    def _visual_prompt_to_policy_layout(
        self, value: Tensor, *, is_goal: bool
    ) -> Tensor:
        expected_ndim = (4, 5) if is_goal else (5, 6)
        if value.ndim not in expected_ndim:
            kind = "goal image" if is_goal else "demo video"
            raise ValueError(
                f"{kind} must be single- or multi-view channels-last visual data"
            )
        if self.channels_last:
            if value.shape[-1] not in (1, 3, 4):
                raise ValueError("channels-last visual prompt has invalid channels")
            value = value.movedim(-1, -3)
        else:
            channel_axis = -3
            if value.shape[channel_axis] not in (1, 3, 4):
                raise ValueError("channels-first visual prompt has invalid channels")
        return value.unsqueeze(1) if is_goal else value


class NativeTrainer:
    """Single-process native trainer with deterministic, strict resume."""

    DIRECTORY_CHECKPOINT_VERSION = 2
    LEGACY_DEEPSPEED_CHECKPOINT_VERSION = 1
    DEEPSPEED_CHECKPOINT_VERSION = 2
    DEEPSPEED_METADATA = "trainer_state.pt"
    PORTABLE_MODEL = MODEL_FILENAME
    PORTABLE_MODEL_INDEX = MODEL_INDEX_FILENAME
    LEGACY_DEEPSPEED_POLICY = "policy.pt"
    DEEPSPEED_COMPLETE = ".complete"
    DEEPSPEED_ENGINE_DIR = "deepspeed"
    DEEPSPEED_ENGINE_TAG = "checkpoint"
    CHECKPOINT_KEYS = frozenset(
        {
            "format_version",
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "step",
            "data_batches_consumed",
            "prompt_schedule",
            "config",
            "callbacks",
            "distributed",
            "scaler",
        }
    )
    RNG_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda", "generator"})

    def __init__(
        self,
        *,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        objective: CompositeObjective,
        noise_kernel: TrainingNoiseKernel,
        scheduler: Any | None = None,
        batch_adapter: TrainingBatchAdapter | None = None,
        generator: torch.Generator | None = None,
        max_grad_norm: float | None = None,
        skip_grad_norm_threshold: float | None = None,
        skip_grad_norm_threshold_start_ratio: float = 0.0,
        config_metadata: Mapping[str, Any] | None = None,
        prompt_schedule: PromptSchedule | None = None,
        max_steps: int | None = None,
        prompt_schedule_per_sample: bool = True,
        t5_high_level_prompt_prob: float = 1.0 / 3.0,
        t5_prompt_template: str | None = None,
        vlm_prompt_template: str | None = None,
        log_prompt_text: bool = False,
        projector_only_end: float = 0.0,
        projector_module_paths: tuple[str, ...] = (
            "condition_router.auto.projector",
            "condition_router.interactive.projector",
        ),
        callbacks: TrainingCallback | Sequence[TrainingCallback] | None = None,
        runtime: NativeTrainingRuntime | None = None,
        precision: str = "fp32",
        gradient_accumulation_steps: int = 1,
        checkpoint_max_shard_size: int | str = DEFAULT_MAX_SHARD_SIZE,
    ) -> None:
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if (
            skip_grad_norm_threshold is not None
            and skip_grad_norm_threshold <= 0
        ):
            raise ValueError("skip_grad_norm_threshold must be positive")
        if not 0 <= skip_grad_norm_threshold_start_ratio <= 1:
            raise ValueError(
                "skip_grad_norm_threshold_start_ratio must be in [0, 1]"
            )
        if (
            skip_grad_norm_threshold is not None
            and skip_grad_norm_threshold_start_ratio > 0
            and (max_steps is None or max_steps <= 0)
        ):
            raise ValueError(
                "delayed gradient-norm threshold requires a positive max_steps"
            )
        if prompt_schedule is not None and (max_steps is None or max_steps <= 0):
            raise ValueError("prompt scheduling requires a positive max_steps")
        if not 0 <= t5_high_level_prompt_prob <= 1:
            raise ValueError("t5_high_level_prompt_prob must be in [0, 1]")
        if t5_prompt_template is not None:
            try:
                t5_prompt_template.format(instruction="test")
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "t5_prompt_template must be format-compatible with {instruction}"
                ) from exc
        if not 0 <= projector_only_end <= 1:
            raise ValueError("projector_only_end must be in [0, 1]")
        if projector_only_end > 0 and prompt_schedule is None:
            raise ValueError("projector-only staging requires a prompt schedule")
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.policy = policy
        self.optimizer = optimizer
        if runtime is None:
            first_parameter = next(policy.parameters(), None)
            policy_device = (
                "cuda"
                if first_parameter is not None and first_parameter.device.type == "cuda"
                else "cpu"
            )
            runtime = NativeTrainingRuntime(
                NativeDistributedConfig(backend="single", device=policy_device)
            )
        self.runtime = runtime
        self.forward_policy, self.optimizer = self.runtime.setup(policy, optimizer)
        self.objective = objective
        self.noise_kernel = noise_kernel
        self.scheduler = scheduler
        self.batch_adapter = batch_adapter
        self.generator = generator or torch.Generator(device=self.runtime.device).manual_seed(
            torch.initial_seed()
        )
        self.max_grad_norm = max_grad_norm
        self.skip_grad_norm_threshold = skip_grad_norm_threshold
        self.skip_grad_norm_threshold_start_ratio = float(
            skip_grad_norm_threshold_start_ratio
        )
        self._accumulated_batch_diagnostics: list[dict[str, object]] = []
        self.config_metadata = dict(config_metadata or {})
        self.prompt_schedule = prompt_schedule
        self.max_steps = max_steps
        self.prompt_schedule_per_sample = bool(prompt_schedule_per_sample)
        self.t5_high_level_prompt_prob = float(t5_high_level_prompt_prob)
        self.t5_prompt_template = t5_prompt_template
        self.vlm_prompt_template = vlm_prompt_template
        self.log_prompt_text = bool(log_prompt_text)
        self.projector_only_end = float(projector_only_end)
        self.projector_module_paths = tuple(projector_module_paths)
        self.callbacks = CallbackList(callbacks)
        self.precision = precision
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.checkpoint_max_shard_size = checkpoint_max_shard_size
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=precision == "fp16" and self.runtime.device.type == "cuda",
        )
        self.data_source: Any | None = None
        self._accumulation_index = 0
        self.last_training_step_time = 0.0
        self.last_model_forward_time = 0.0
        self.last_step_timing: dict[str, float] = {}
        self._base_trainability = {
            name: parameter.requires_grad
            for name, parameter in self.policy.named_parameters()
        }
        if self.projector_only_end > 0 and not any(
            trainable and self._is_projector_parameter(name)
            for name, trainable in self._base_trainability.items()
        ):
            raise ValueError(
                "projector-only stage has no trainable parameters under configured "
                "projector module paths"
            )
        self.step = 0
        self.data_batches_consumed = 0

    @property
    def rank(self) -> int:
        return self.runtime.rank

    @property
    def is_rank_zero(self) -> bool:
        return self.runtime.is_rank_zero

    def attach_data_source(self, data_source: Any) -> None:
        """Attach the rank-local iterator owner for exact checkpoint resume."""

        self.data_source = data_source

    def train_step(
        self, batch: TrainingBatch | ModelReadyTrainingBatch
    ) -> dict[str, float]:
        training_step_start = time.perf_counter()
        self.callbacks.on_before_step(self, batch)
        self._accumulated_batch_diagnostics.extend(
            _batch_dataset_diagnostics(batch)
        )
        ready = self._prepare_batch(batch)
        schedule_metrics = self._apply_prompt_schedule(ready)
        ready = schedule_metrics.pop("batch")
        self._apply_projector_stage()
        self.policy.train()
        if self._accumulation_index == 0:
            self.optimizer.zero_grad(set_to_none=True)
        noise = self.noise_kernel.prepare_training_inputs(
            clean_video_latents=ready.clean_video_latents,
            clean_action=ready.clean_action,
            generator=self.generator,
        )
        prepare_time = time.perf_counter() - training_step_start
        device_type = self.runtime.device.type
        autocast_enabled = self.precision != "fp32"
        autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            model_forward_start = time.perf_counter()
            output = self._training_forward(ready, noise)
            forward_time = time.perf_counter() - model_forward_start
        self.last_model_forward_time = forward_time
        auxiliary_index = (
            ready.mode_mask
            if ready.mode_mask is not None
            else torch.full(
                (ready.clean_action.shape[0],),
                ready.mode is InteractionMode.AUTO,
                dtype=torch.bool,
                device=ready.clean_action.device,
            )
        )
        semantic_target = ready.semantic_target
        semantic_mask = ready.semantic_mask
        planning_labels = ready.planning_labels
        if bool(auxiliary_index.any()) and not bool(auxiliary_index.all()):
            semantic_target = _select_optional(semantic_target, auxiliary_index)
            semantic_mask = _select_optional(semantic_mask, auxiliary_index)
            planning_labels = _select_optional(planning_labels, auxiliary_index)
        result = self.objective(
            ObjectiveInputs(
                video_prediction=output.video_velocity,
                video_target=noise.video_velocity_target,
                action_prediction=output.action_velocity,
                action_target=noise.action_velocity_target,
                video_weight=(
                    noise.video_weight
                    if noise.video_weight is not None
                    else ready.video_weight
                ),
                action_weight=(
                    noise.action_weight
                    if noise.action_weight is not None
                    else ready.action_weight
                ),
                action_mask=ready.action_mask,
                action_dim_mask=ready.action_dim_mask,
                has_real_action=ready.has_real_action,
                semantic_prediction=output.metrics.get("semantic_prediction"),
                semantic_target=(
                    semantic_target
                    if semantic_target is not None
                    else output.metrics.get("semantic_target")
                ),
                semantic_mask=(
                    semantic_mask
                    if semantic_mask is not None
                    else output.metrics.get("semantic_mask")
                ),
                semantic_active=bool(auxiliary_index.any()),
                planning_logits=output.metrics.get("planning_logits"),
                planning_labels=(
                    planning_labels
                    if planning_labels is not None
                    else output.metrics.get("planning_labels")
                ),
            )
        )
        loss = result.loss / self.gradient_accumulation_steps
        backward_start = time.perf_counter()
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            self.runtime.backward(loss)
        backward_time = time.perf_counter() - backward_start
        self._accumulation_index += 1
        optimizer_step = self._accumulation_index == self.gradient_accumulation_steps
        grad_norm = 0.0
        optimizer_time = 0.0
        gradient_finite = True
        gradient_threshold_exceeded = False
        gradient_threshold_active = False
        optimizer_update_applied = False
        if optimizer_step:
            optimizer_start = time.perf_counter()
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            gradient_threshold_active = self._gradient_threshold_active()
            effective_threshold = (
                self.skip_grad_norm_threshold
                if gradient_threshold_active
                else None
            )
            gradient_health = self.runtime.inspect_gradients(
                max_norm_threshold=effective_threshold
            )
            gradient_finite = gradient_health.finite
            gradient_threshold_exceeded = gradient_health.threshold_exceeded
            if not gradient_finite or gradient_threshold_exceeded:
                grad_norm = gradient_health.global_norm
                rank_diagnostics = self.runtime.gather_rank_state(
                    {
                        "rank": self.runtime.rank,
                        "samples": tuple(self._accumulated_batch_diagnostics),
                    }
                )
                self.runtime.skip_gradient_step(self.optimizer)
                if self.is_rank_zero:
                    if not gradient_finite:
                        LOGGER.error(
                            "Skipping optimizer step %d because gradients are "
                            "non-finite; %s",
                            self.step + 1,
                            gradient_health.first_nonfinite,
                        )
                    else:
                        LOGGER.error(
                            "Skipping optimizer step %d because finite global "
                            "gradient norm %.6g exceeds threshold %.6g",
                            self.step + 1,
                            gradient_health.global_norm,
                            effective_threshold,
                        )
                    LOGGER.error(
                        "Gradient anomaly dataset samples by rank: %s",
                        rank_diagnostics,
                    )
            elif self.runtime.manages_gradient_clipping:
                # ZeRO partitions gradients away from parameter.grad and applies
                # the configured global clipping inside engine.step().
                self.runtime.step(self.optimizer)
                grad_norm = self.runtime.last_gradient_norm()
                optimizer_update_applied = True
            else:
                self._apply_semantic_gate_clip()
                grad_norm = self._apply_gradient_clip()
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.runtime.step(self.optimizer)
                optimizer_update_applied = True
            if self.scheduler is not None and optimizer_update_applied:
                self.scheduler.step()
            self.step += 1
            self._accumulation_index = 0
            self._accumulated_batch_diagnostics.clear()
            optimizer_time = time.perf_counter() - optimizer_start
        postprocess_start = time.perf_counter()
        self.data_batches_consumed += 1
        metrics = self._metrics(result, grad_norm)
        metrics["optimizer_step"] = float(optimizer_step)
        metrics["gradient_finite"] = float(gradient_finite)
        metrics["optimizer_update_applied"] = float(optimizer_update_applied)
        metrics["optimizer_step_skipped_nonfinite"] = float(
            optimizer_step and not gradient_finite
        )
        metrics["gradient_norm_threshold_active"] = float(
            optimizer_step and gradient_threshold_active
        )
        metrics["gradient_norm_threshold_exceeded"] = float(
            optimizer_step and gradient_threshold_exceeded
        )
        metrics["optimizer_step_skipped_large_gradient"] = float(
            optimizer_step and gradient_threshold_exceeded
        )
        metrics.update(schedule_metrics)
        metrics = self.runtime.all_reduce_metrics(metrics)
        if self.is_rank_zero:
            self.callbacks.on_metrics(self, metrics)
        self.callbacks.on_after_step(self, batch, metrics)
        postprocess_time = time.perf_counter() - postprocess_start
        total_time = time.perf_counter() - training_step_start
        self.last_training_step_time = total_time
        self.last_step_timing = {
            "step_time/prepare": prepare_time,
            "step_time/forward": forward_time,
            "step_time/backward": backward_time,
            "step_time/optimizer": optimizer_time,
            "step_time/postprocess": postprocess_time,
            "step_time/total": total_time,
        }
        return metrics

    def train_start(self) -> None:
        """Notify callbacks that the entrypoint training loop is starting."""

        self.callbacks.on_train_start(self)

    def train_end(self) -> None:
        """Notify callbacks that the entrypoint training loop has ended."""

        self.callbacks.on_train_end(self)

    def _training_forward(
        self,
        ready: ModelReadyTrainingBatch,
        noise: Wan22TrainingInputs,
    ) -> WorldActionOutput:
        mode_mask = ready.mode_mask
        if mode_mask is None:
            mode_mask = torch.full(
                (ready.clean_action.shape[0],),
                ready.mode is InteractionMode.AUTO,
                dtype=torch.bool,
                device=ready.clean_action.device,
            )
        self._validate_training_modes(mode_mask)
        condition_ids = ready.condition_ids
        if condition_ids is None:
            condition_ids = torch.full(
                (ready.clean_action.shape[0],),
                ConditionMode.T2VA.id,
                dtype=torch.long,
                device=ready.clean_action.device,
            )
        else:
            condition_ids = condition_ids.to(device=ready.clean_action.device)
        outputs: list[tuple[Tensor, WorldActionOutput]] = []
        for mode, selected_mode in (
            (InteractionMode.AUTO, mode_mask),
            (InteractionMode.INTERACTIVE, ~mode_mask),
        ):
            for condition_mode, demo_length in _condition_groups(
                condition_ids,
                selected_mode,
                ready.prompts.demo_video_mask,
            ):
                selected = selected_mode & (condition_ids == condition_mode.id)
                if demo_length is not None:
                    assert ready.prompts.demo_video_mask is not None
                    lengths = ready.prompts.demo_video_mask.sum(dim=1).to(
                        device=selected.device
                    )
                    selected &= lengths == demo_length
                if not bool(selected.any()):
                    continue
                subset = _select_ready_batch(
                    ready,
                    selected,
                    mode,
                    condition_mode,
                    demo_length=demo_length,
                )
                output = self.runtime.forward(
                    mode=mode,
                    observation=subset.observation,
                    prompts=subset.prompts,
                    clean_video=subset.clean_video,
                    clean_action=subset.clean_action,
                    noisy_video=noise.noisy_video[selected],
                    noisy_action=noise.noisy_action[selected],
                    video_timestep=noise.video_timestep[selected],
                    action_timestep=noise.action_timestep[selected],
                    planning_supervision=(
                        mode is InteractionMode.AUTO
                        and self.objective.planning_ce_weight > 0
                    ),
                    clean_video_latents=subset.clean_video_latents,
                    clean_video_normalized=subset.clean_video_normalized,
                )
                if not isinstance(output, WorldActionOutput):
                    raise TypeError(
                        "policy.training_forward must return WorldActionOutput"
                    )
                if output.video_velocity is None or output.action_velocity is None:
                    raise RuntimeError(
                        "training output must contain video and action velocity"
                    )
                outputs.append((selected, output))

        if not outputs:
            raise RuntimeError("training batch produced no routed WAM outputs")
        video_velocity = torch.zeros_like(
            noise.video_velocity_target,
            dtype=outputs[0][1].video_velocity.dtype,
        )
        action_velocity = torch.zeros_like(
            noise.action_velocity_target,
            dtype=outputs[0][1].action_velocity.dtype,
        )
        for selected, output in outputs:
            indices = selected.nonzero(as_tuple=False).flatten()
            video_velocity = video_velocity.index_copy(
                0, indices, output.video_velocity
            )
            action_velocity = action_velocity.index_copy(
                0, indices, output.action_velocity
            )
        metrics = _merge_auto_metrics(outputs, mode_mask)
        return WorldActionOutput(
            video_velocity=video_velocity,
            action_velocity=action_velocity,
            metrics=metrics,
        )

    def _validate_training_modes(self, mode_mask: Tensor) -> None:
        configured = getattr(self.policy, "configured_mode", None)
        supported = getattr(self.policy, "training_supported_modes", None)
        if supported is None:
            supported_modes = (
                {InteractionMode.parse(configured)}
                if configured is not None
                else {InteractionMode.AUTO, InteractionMode.INTERACTIVE}
            )
        else:
            supported_modes = {InteractionMode.parse(mode) for mode in supported}
        requested = set()
        if bool(mode_mask.any()):
            requested.add(InteractionMode.AUTO)
        if bool((~mode_mask).any()):
            requested.add(InteractionMode.INTERACTIVE)
        unsupported = requested - supported_modes
        if unsupported:
            names = ", ".join(sorted(mode.value for mode in unsupported))
            raise ValueError(
                f"Policy checkpoint mode does not support training mode(s): {names}"
            )

    def _apply_prompt_schedule(
        self, ready: ModelReadyTrainingBatch
    ) -> dict[str, Any]:
        if self.prompt_schedule is None:
            return {"batch": ready}
        assert self.max_steps is not None
        progress = min(self.step / self.max_steps, 1.0)
        batch_size = ready.clean_action.shape[0]
        sampled = self.prompt_schedule.sample(
            batch_size,
            progress,
            generator=self.generator,
            device=self.generator.device,
            per_sample=self.prompt_schedule_per_sample,
        )
        projector_only = progress < self.projector_only_end
        mode_mask = (
            torch.ones_like(sampled.mode_mask)
            if projector_only
            else sampled.mode_mask
        ).to(device=ready.clean_action.device)
        # DeepSpeed ZeRO gathers parameters in module-execution order. If ranks
        # independently sample Auto/T5 routing, they can enter different
        # branches and deadlock in mismatched NCCL collectives. Rank zero owns
        # the routing decision so every process executes identical subsets.
        mode_mask = self.runtime.broadcast_from_rank_zero(mode_mask)
        high_level = ready.prompts.vlm_planning_text
        event_level = ready.prompts.language_instruction
        if high_level is None and event_level is None:
            raise ValueError(
                "prompt scheduling requires Auto or Interactive instruction text"
            )
        high_level = list(high_level or event_level or ())
        event_level = list(event_level or high_level)
        use_high_level = (
            torch.rand(
                batch_size,
                generator=self.generator,
                device=self.generator.device,
            )
            < self.t5_high_level_prompt_prob
        ).tolist()
        t5_text = [
            high if use_high else event
            for high, event, use_high in zip(
                high_level, event_level, use_high_level, strict=True
            )
        ]
        if self.t5_prompt_template is not None:
            t5_text = [
                self.t5_prompt_template.format(instruction=text.lower())
                for text in t5_text
            ]
        if self.log_prompt_text and self.runtime.is_rank_zero:
            is_auto = bool(mode_mask[0])
            vlm_goal_text = high_level[0] if is_auto else None
            vlm_planning_text = vlm_goal_text
            if (
                vlm_planning_text is not None
                and self.vlm_prompt_template is not None
            ):
                vlm_planning_text = render_instruction_template(
                    self.vlm_prompt_template,
                    vlm_planning_text,
                )
            print(
                json.dumps(
                    {
                        "text_inputs": {
                            "step": self.step,
                            "mode": "auto" if is_auto else "interactive",
                            "vlm_planning": vlm_planning_text,
                            "vlm_goal": vlm_goal_text,
                            "t5": t5_text[0],
                        }
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        prompts = replace(
            ready.prompts,
            vlm_planning_text=high_level,
            language_instruction=t5_text,
        )
        return {
            "batch": replace(
                ready,
                mode=(
                    InteractionMode.AUTO
                    if bool(mode_mask[0])
                    else InteractionMode.INTERACTIVE
                ),
                mode_mask=mode_mask,
                prompts=prompts,
            ),
            "prompt_schedule/progress": progress,
            "prompt_schedule/stage": float(sampled.stage_index),
            "prompt_schedule/auto": float(mode_mask.float().mean().item()),
            "prompt_schedule/projector_only": float(projector_only),
        }

    def _apply_projector_stage(self) -> None:
        if self.prompt_schedule is None or self.projector_only_end <= 0:
            return
        assert self.max_steps is not None
        projector_only = self.step / self.max_steps < self.projector_only_end
        for name, parameter in self.policy.named_parameters():
            base_trainable = self._base_trainability[name]
            parameter.requires_grad_(
                base_trainable
                and (not projector_only or self._is_projector_parameter(name))
            )

    def _is_projector_parameter(self, name: str) -> bool:
        return any(
            name == path or name.startswith(path + ".")
            for path in self.projector_module_paths
        )

    def _prepare_batch(
        self, batch: TrainingBatch | ModelReadyTrainingBatch
    ) -> ModelReadyTrainingBatch:
        if isinstance(batch, TrainingBatch):
            batch.validate()
            if self.batch_adapter is None:
                raise TypeError(
                    "TrainingBatch requires an explicit TrainingBatchAdapter"
                )
            ready = self.batch_adapter(_move_tensors(batch, self.runtime.device))
        elif isinstance(batch, ModelReadyTrainingBatch):
            ready = batch
        else:
            raise TypeError(
                "batch must be TrainingBatch or ModelReadyTrainingBatch"
            )
        ready = _move_tensors(ready, self.runtime.device)
        ready.validate()
        return ready

    def _apply_gradient_clip(self) -> float:
        parameters = [
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not parameters:
            raise RuntimeError("training loss produced no policy gradients")
        if self.max_grad_norm is not None:
            value = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            return float(value.detach().cpu())
        total = torch.stack(
            [parameter.grad.detach().float().norm(2) for parameter in parameters]
        ).norm(2)
        return float(total.cpu())

    def _gradient_threshold_active(self) -> bool:
        if self.skip_grad_norm_threshold is None:
            return False
        if self.skip_grad_norm_threshold_start_ratio <= 0:
            return True
        assert self.max_steps is not None
        return self.step / self.max_steps >= self.skip_grad_norm_threshold_start_ratio

    def _apply_semantic_gate_clip(self) -> None:
        router = getattr(self.policy, "condition_router", None)
        auto = getattr(router, "auto", None)
        if auto is None or not bool(
            getattr(auto, "semantic_gate_only", False)
        ):
            return
        max_norm = float(getattr(auto, "semantic_grad_clip_norm", 0.5))
        if max_norm <= 0:
            return
        modules = [auto.projector]
        for name in ("gate", "goal_fuser"):
            module = getattr(auto.event_memory, name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is None:
                    continue
                grad_norm = torch.linalg.vector_norm(parameter.grad.detach())
                if not bool(torch.isfinite(grad_norm)):
                    parameter.grad.zero_()
                elif bool(grad_norm > max_norm):
                    parameter.grad.mul_(
                        max_norm / (grad_norm + 1e-6)
                    )

    def _metrics(
        self, result: ObjectiveResult, grad_norm: float
    ) -> dict[str, float]:
        metrics = {
            name: float(value.detach().float().cpu())
            for name, value in result.metrics.items()
        }
        metrics["grad_norm"] = grad_norm
        metrics["step"] = float(self.step)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.runtime.engine is not None:
            return self._save_deepspeed_checkpoint(Path(path))
        destination = Path(path)
        if destination.suffix == ".pt":
            return self._save_legacy_file_checkpoint(destination)
        return self._save_directory_checkpoint(destination)

    def _save_legacy_file_checkpoint(self, destination: Path) -> Path:
        """Write the format-5 single-file checkpoint for backward compatibility."""
        if self._accumulation_index:
            raise RuntimeError("checkpointing during gradient accumulation is unsupported")
        if self.is_rank_zero:
            destination.parent.mkdir(parents=True, exist_ok=True)
        # Save hooks are pre-snapshot hooks: callback mutations made here must
        # be represented by the checkpoint that is about to be serialized.
        if self.is_rank_zero:
            self.callbacks.on_checkpoint_save(self, destination)
        rank_state = {
            "rank": self.rank,
            "rng": self._rng_state(),
            "data": self._data_state(),
            "backend": self.runtime.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        rank_states = self.runtime.gather_rank_state(rank_state)
        payload = {
            "format_version": 5,
            "model": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "rng": rank_states[0]["rng"],
            "step": self.step,
            "data_batches_consumed": self.data_batches_consumed,
            "prompt_schedule": (
                {
                    "position": self.step,
                    "max_steps": self.max_steps,
                }
                if self.prompt_schedule is not None
                else None
            ),
            "config": self.config_metadata,
            "callbacks": dict(self.callbacks.state_dict()),
            "scaler": self.scaler.state_dict(),
            "distributed": {
                "format_version": 1,
                "backend": self.runtime.backend,
                "world_size": self.runtime.world_size,
                "rank_states": rank_states,
            },
        }
        if self.is_rank_zero:
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        self.runtime.barrier()
        return destination

    def _save_directory_checkpoint(self, destination: Path) -> Path:
        """Write one HF-shaped directory with portable weights and resume state."""
        if self._accumulation_index:
            raise RuntimeError("checkpointing during gradient accumulation is unsupported")
        if self.is_rank_zero:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(
                    f"training checkpoint destination already exists: {destination}"
                )
            destination.mkdir()
            self.callbacks.on_checkpoint_save(self, destination)
        self.runtime.barrier()

        rank_file = self._deepspeed_rank_filename(self.rank)
        _atomic_torch_save(
            {
                "format_version": self.DIRECTORY_CHECKPOINT_VERSION,
                "rank": self.rank,
                "rng": self._rng_state(),
                "data": self._data_state(),
                "backend": self.runtime.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            destination / rank_file,
        )
        self.runtime.barrier()

        if self.is_rank_zero:
            model_files = save_checkpoint_state_dict(
                self.policy.state_dict(),
                destination,
                max_shard_size=self.checkpoint_max_shard_size,
            )
            model_file = model_files[0].name
            metadata = {
                "format_version": self.DIRECTORY_CHECKPOINT_VERSION,
                "backend": self.runtime.backend,
                "world_size": self.runtime.world_size,
                "step": self.step,
                "data_batches_consumed": self.data_batches_consumed,
                "prompt_schedule": (
                    {"position": self.step, "max_steps": self.max_steps}
                    if self.prompt_schedule is not None
                    else None
                ),
                "config": self.config_metadata,
                "callbacks": dict(self.callbacks.state_dict()),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "rank_files": [
                    self._deepspeed_rank_filename(rank)
                    for rank in range(self.runtime.world_size)
                ],
                "model_file": model_file,
            }
            _atomic_torch_save(metadata, destination / self.DEEPSPEED_METADATA)
            _write_hf_config(destination, self.config_metadata)
            _write_complete_marker(
                destination,
                f"native-training-checkpoint-v{self.DIRECTORY_CHECKPOINT_VERSION}",
            )
        self.runtime.barrier()
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        notify_callbacks: bool = True,
        restore_data_state: bool = True,
    ) -> Mapping[str, Any]:
        source = Path(path)
        if source.is_dir():
            if (source / "checkpoint_manifest.json").is_file():
                validate_native_checkpoint_artifacts(source)
            metadata_path = source / self.DEEPSPEED_METADATA
            if not metadata_path.is_file():
                raise ValueError("directory checkpoint trainer metadata is missing")
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
            if isinstance(metadata, Mapping) and metadata.get("backend") == "deepspeed":
                if self.runtime.engine is None:
                    raise ValueError(
                        "DeepSpeed directory checkpoint requires a DeepSpeed runtime"
                    )
                return self._load_deepspeed_checkpoint(
                    source,
                    notify_callbacks=notify_callbacks,
                    restore_data_state=restore_data_state,
                )
            return self._load_directory_checkpoint(
                source,
                metadata=metadata,
                notify_callbacks=notify_callbacks,
                restore_data_state=restore_data_state,
            )
        self.runtime.barrier()
        payload = torch.load(source, map_location="cpu", weights_only=True)
        self._validate_checkpoint(payload)
        self.policy.load_state_dict(payload["model"], strict=True)
        if self.runtime.engine is None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        distributed = payload["distributed"]
        rank_state = distributed["rank_states"][self.rank]
        self._set_rng_state(rank_state["rng"])
        if restore_data_state:
            self._load_data_state(rank_state["data"])
        self.runtime.load_state_dict(rank_state["backend"])
        self.scaler.load_state_dict(rank_state["scaler"])
        self.step = payload["step"]
        self.data_batches_consumed = payload["data_batches_consumed"]
        self.callbacks.load_state_dict(payload["callbacks"])
        if notify_callbacks and self.is_rank_zero:
            self.callbacks.on_checkpoint_load(self, source)
        return payload["config"]

    def _load_directory_checkpoint(
        self,
        source: Path,
        *,
        metadata: object,
        notify_callbacks: bool,
        restore_data_state: bool,
    ) -> Mapping[str, Any]:
        self.runtime.barrier()
        _validate_complete_marker(
            source,
            {
                f"native-training-checkpoint-v{self.DIRECTORY_CHECKPOINT_VERSION}",
            },
        )
        if not isinstance(metadata, dict):
            raise TypeError("directory trainer metadata must contain a mapping")
        _require_exact_keys(
            metadata,
            frozenset(
                {
                    "format_version",
                    "backend",
                    "world_size",
                    "step",
                    "data_batches_consumed",
                    "prompt_schedule",
                    "config",
                    "callbacks",
                    "optimizer",
                    "scheduler",
                    "rank_files",
                    "model_file",
                }
            ),
            "directory trainer metadata",
        )
        if metadata["format_version"] != self.DIRECTORY_CHECKPOINT_VERSION:
            raise ValueError("unsupported directory checkpoint format_version")
        if (
            metadata["backend"] != self.runtime.backend
            or metadata["world_size"] != self.runtime.world_size
        ):
            raise ValueError("checkpoint distributed runtime does not match current run")
        if metadata["config"] != self.config_metadata:
            raise ValueError("checkpoint config metadata does not match trainer config")
        expected_rank_files = [
            self._deepspeed_rank_filename(rank)
            for rank in range(self.runtime.world_size)
        ]
        if metadata["rank_files"] != expected_rank_files or any(
            not (source / name).is_file() for name in expected_rank_files
        ):
            raise ValueError("directory checkpoint rank sidecars are incomplete")
        if metadata["model_file"] not in {
            self.PORTABLE_MODEL,
            self.PORTABLE_MODEL_INDEX,
        }:
            raise ValueError("directory checkpoint model_file is invalid")
        try:
            checkpoint_weight_files(source)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError("directory checkpoint portable model is invalid") from error
        rank_state = torch.load(
            source / self._deepspeed_rank_filename(self.rank),
            map_location="cpu",
            weights_only=True,
        )
        self._validate_directory_rank_state(rank_state)
        self.policy.load_state_dict(load_checkpoint_state_dict(source), strict=True)
        self.optimizer.load_state_dict(metadata["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(metadata["scheduler"])
        self._set_rng_state(rank_state["rng"])
        if restore_data_state:
            self._load_data_state(rank_state["data"])
        self.runtime.load_state_dict(rank_state["backend"])
        self.scaler.load_state_dict(rank_state["scaler"])
        self.step = metadata["step"]
        self.data_batches_consumed = metadata["data_batches_consumed"]
        self.callbacks.load_state_dict(metadata["callbacks"])
        if notify_callbacks and self.is_rank_zero:
            self.callbacks.on_checkpoint_load(self, source)
        self.runtime.barrier()
        return metadata["config"]

    def _save_deepspeed_checkpoint(self, destination: Path) -> Path:
        if self._accumulation_index:
            raise RuntimeError("checkpointing during gradient accumulation is unsupported")
        engine = self.runtime.engine
        if engine is None:
            raise RuntimeError("DeepSpeed checkpoint requested without an engine")
        if self.is_rank_zero:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(
                    f"DeepSpeed checkpoint destination already exists: {destination}"
                )
            destination.mkdir()
            self.callbacks.on_checkpoint_save(self, destination)
        self.runtime.barrier()

        self.runtime.save_deepspeed_checkpoint(
            str(destination / self.DEEPSPEED_ENGINE_DIR),
            tag=self.DEEPSPEED_ENGINE_TAG,
        )

        rank_file = self._deepspeed_rank_filename(self.rank)
        _atomic_torch_save(
            {
                "format_version": self.DEEPSPEED_CHECKPOINT_VERSION,
                "rank": self.rank,
                "rng": self._rng_state(),
                "data": self._data_state(),
                "scaler": self.scaler.state_dict(),
            },
            destination / rank_file,
        )
        self.runtime.barrier()

        export_dir = destination / ".policy-export"
        self.runtime.save_deepspeed_16bit_model(
            str(export_dir), save_filename=self.LEGACY_DEEPSPEED_POLICY
        )
        self.runtime.barrier()

        if self.is_rank_zero:
            export_path = export_dir / self.LEGACY_DEEPSPEED_POLICY
            if not export_path.is_file():
                raise RuntimeError("DeepSpeed 16-bit policy export was not created")
            exported_state = torch.load(
                export_path, map_location="cpu", weights_only=True
            )
            policy_state = _normalize_deepspeed_policy_state(
                exported_state, self.policy.state_dict()
            )
            model_files = save_checkpoint_state_dict(
                policy_state,
                destination,
                max_shard_size=self.checkpoint_max_shard_size,
            )
            shutil.rmtree(export_dir)
            metadata = {
                "format_version": self.DEEPSPEED_CHECKPOINT_VERSION,
                "backend": "deepspeed",
                "world_size": self.runtime.world_size,
                "step": self.step,
                "data_batches_consumed": self.data_batches_consumed,
                "prompt_schedule": (
                    {"position": self.step, "max_steps": self.max_steps}
                    if self.prompt_schedule is not None
                    else None
                ),
                "config": self.config_metadata,
                "callbacks": dict(self.callbacks.state_dict()),
                "scheduler": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "rank_files": [
                    self._deepspeed_rank_filename(rank)
                    for rank in range(self.runtime.world_size)
                ],
                "policy_file": model_files[0].name,
            }
            _atomic_torch_save(metadata, destination / self.DEEPSPEED_METADATA)
            _write_hf_config(destination, self.config_metadata)
        self.runtime.barrier()
        if self.is_rank_zero:
            _write_complete_marker(
                destination,
                f"deepspeed-checkpoint-v{self.DEEPSPEED_CHECKPOINT_VERSION}",
            )
        self.runtime.barrier()
        return destination

    def _load_deepspeed_checkpoint(
        self,
        source: Path,
        *,
        notify_callbacks: bool,
        restore_data_state: bool,
    ) -> Mapping[str, Any]:
        self.runtime.barrier()
        _validate_complete_marker(
            source,
            {
                (
                    "deepspeed-checkpoint-v"
                    f"{self.LEGACY_DEEPSPEED_CHECKPOINT_VERSION}"
                ),
                f"deepspeed-checkpoint-v{self.DEEPSPEED_CHECKPOINT_VERSION}",
            },
        )
        metadata_path = source / self.DEEPSPEED_METADATA
        if not metadata_path.is_file():
            raise ValueError("DeepSpeed checkpoint trainer metadata is missing")
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        self._validate_deepspeed_metadata(metadata, source)
        rank_path = source / self._deepspeed_rank_filename(self.rank)
        rank_state = torch.load(rank_path, map_location="cpu", weights_only=True)
        self._validate_deepspeed_rank_state(rank_state)

        if self.runtime.engine is None:
            raise RuntimeError("DeepSpeed checkpoint requested without an engine")
        self.runtime.load_deepspeed_checkpoint(
            str(source / self.DEEPSPEED_ENGINE_DIR),
            tag=self.DEEPSPEED_ENGINE_TAG,
        )
        self._set_rng_state(rank_state["rng"])
        if restore_data_state:
            self._load_data_state(rank_state["data"])
        self.scaler.load_state_dict(rank_state["scaler"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(metadata["scheduler"])
        self.step = metadata["step"]
        self.data_batches_consumed = metadata["data_batches_consumed"]
        self.callbacks.load_state_dict(metadata["callbacks"])
        if notify_callbacks and self.is_rank_zero:
            self.callbacks.on_checkpoint_load(self, source)
        self.runtime.barrier()
        return metadata["config"]

    @staticmethod
    def _deepspeed_rank_filename(rank: int) -> str:
        return f"rank-{rank:05d}.pt"

    def _validate_deepspeed_metadata(
        self, metadata: object, source: Path
    ) -> None:
        if not isinstance(metadata, dict):
            raise TypeError("DeepSpeed trainer metadata must contain a mapping")
        _require_exact_keys(
            metadata,
            frozenset(
                {
                    "format_version",
                    "backend",
                    "world_size",
                    "step",
                    "data_batches_consumed",
                    "prompt_schedule",
                    "config",
                    "callbacks",
                    "scheduler",
                    "rank_files",
                    "policy_file",
                }
            ),
            "DeepSpeed trainer metadata",
        )
        if metadata["format_version"] not in {
            self.LEGACY_DEEPSPEED_CHECKPOINT_VERSION,
            self.DEEPSPEED_CHECKPOINT_VERSION,
        }:
            raise ValueError("unsupported DeepSpeed checkpoint format_version")
        if metadata["backend"] != "deepspeed" or self.runtime.backend != "deepspeed":
            raise ValueError("checkpoint training backend does not match runtime")
        if metadata["world_size"] != self.runtime.world_size:
            raise ValueError("checkpoint training world_size does not match runtime")
        expected_rank_files = [
            self._deepspeed_rank_filename(rank)
            for rank in range(self.runtime.world_size)
        ]
        if metadata["rank_files"] != expected_rank_files or any(
            not (source / name).is_file() for name in expected_rank_files
        ):
            raise ValueError("DeepSpeed checkpoint rank sidecars are incomplete")
        if metadata["format_version"] == self.LEGACY_DEEPSPEED_CHECKPOINT_VERSION:
            if (
                metadata["policy_file"] != self.LEGACY_DEEPSPEED_POLICY
                or not (source / self.LEGACY_DEEPSPEED_POLICY).is_file()
            ):
                raise ValueError("DeepSpeed checkpoint portable policy artifact is missing")
        else:
            if metadata["policy_file"] not in {
                self.PORTABLE_MODEL,
                self.PORTABLE_MODEL_INDEX,
            }:
                raise ValueError("DeepSpeed checkpoint portable policy artifact is invalid")
            try:
                checkpoint_weight_files(source)
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(
                    "DeepSpeed checkpoint portable policy artifact is missing"
                ) from error
        if not isinstance(metadata["step"], int) or metadata["step"] < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        if (
            not isinstance(metadata["data_batches_consumed"], int)
            or metadata["data_batches_consumed"] < 0
        ):
            raise ValueError(
                "checkpoint data_batches_consumed must be a non-negative integer"
            )
        if metadata["config"] != self.config_metadata:
            raise ValueError("checkpoint config metadata does not match trainer config")
        schedule_state = metadata["prompt_schedule"]
        if (self.prompt_schedule is None) != (schedule_state is None):
            raise ValueError("checkpoint prompt schedule presence does not match trainer")
        if schedule_state is not None and schedule_state != {
            "position": metadata["step"],
            "max_steps": self.max_steps,
        }:
            raise ValueError("checkpoint prompt schedule state does not match trainer")
        if (self.scheduler is None) != (metadata["scheduler"] is None):
            raise ValueError("checkpoint scheduler presence does not match trainer")
        if not isinstance(metadata["callbacks"], dict):
            raise TypeError("checkpoint callback state must be a mapping")

    def _validate_directory_rank_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise TypeError("directory rank sidecar must contain a mapping")
        _require_exact_keys(
            state,
            frozenset(
                {"format_version", "rank", "rng", "data", "backend", "scaler"}
            ),
            "directory rank sidecar",
        )
        if state["format_version"] != self.DIRECTORY_CHECKPOINT_VERSION:
            raise ValueError("unsupported directory rank sidecar format_version")
        if state["rank"] != self.rank:
            raise ValueError("directory rank sidecar does not match runtime rank")
        if not isinstance(state["rng"], dict):
            raise TypeError("rank RNG state must be a mapping")
        _require_exact_keys(state["rng"], self.RNG_KEYS, "rank RNG state")
        self._validate_rng_state(state["rng"])
        if not isinstance(state["backend"], Mapping):
            raise TypeError("rank backend state must be a mapping")
        if not isinstance(state["scaler"], dict):
            raise TypeError("rank scaler state must be a mapping")

    def _validate_deepspeed_rank_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise TypeError("DeepSpeed rank sidecar must contain a mapping")
        _require_exact_keys(
            state,
            frozenset({"format_version", "rank", "rng", "data", "scaler"}),
            "DeepSpeed rank sidecar",
        )
        if state["format_version"] not in {
            self.LEGACY_DEEPSPEED_CHECKPOINT_VERSION,
            self.DEEPSPEED_CHECKPOINT_VERSION,
        }:
            raise ValueError("unsupported DeepSpeed rank sidecar format_version")
        if state["rank"] != self.rank:
            raise ValueError("DeepSpeed rank sidecar does not match runtime rank")
        if not isinstance(state["rng"], dict):
            raise TypeError("rank RNG state must be a mapping")
        _require_exact_keys(state["rng"], self.RNG_KEYS, "rank RNG state")
        self._validate_rng_state(state["rng"])
        if not isinstance(state["scaler"], dict):
            raise TypeError("rank scaler state must be a mapping")

    def _validate_checkpoint(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise TypeError("training checkpoint must contain a mapping")
        _require_exact_keys(payload, self.CHECKPOINT_KEYS, "checkpoint")
        if payload["format_version"] != 5:
            raise ValueError("unsupported training checkpoint format_version")
        distributed = payload["distributed"]
        if not isinstance(distributed, dict):
            raise TypeError("checkpoint distributed state must be a mapping")
        _require_exact_keys(
            distributed,
            frozenset({"format_version", "backend", "world_size", "rank_states"}),
            "distributed state",
        )
        if distributed["format_version"] != 1:
            raise ValueError("unsupported distributed checkpoint format_version")
        if distributed["world_size"] != self.runtime.world_size:
            raise ValueError("checkpoint training world_size does not match runtime")
        if distributed["backend"] != self.runtime.backend:
            raise ValueError("checkpoint training backend does not match runtime")
        rank_states = distributed["rank_states"]
        if not isinstance(rank_states, list) or len(rank_states) != self.runtime.world_size:
            raise ValueError("checkpoint must contain exactly one state per rank")
        if [state.get("rank") for state in rank_states] != list(range(self.runtime.world_size)):
            raise ValueError("checkpoint rank states are not ordered by rank")
        for state in rank_states:
            if not isinstance(state, dict):
                raise TypeError("individual rank state must be a mapping")
            _require_exact_keys(
                state,
                frozenset({"rank", "rng", "data", "backend", "scaler"}),
                "rank state",
            )
            if not isinstance(state["rng"], dict):
                raise TypeError("rank RNG state must be a mapping")
            _require_exact_keys(state["rng"], self.RNG_KEYS, "rank RNG state")
            self._validate_rng_state(state["rng"])
            if not isinstance(state["backend"], dict) or not isinstance(
                state["scaler"], dict
            ):
                raise TypeError("rank backend/scaler state must be mappings")
        if not isinstance(payload["scaler"], dict):
            raise TypeError("checkpoint scaler state must be a mapping")
        if not isinstance(payload["model"], dict):
            raise TypeError("checkpoint model state must be a mapping")
        target_model = self.policy.state_dict()
        _require_exact_keys(payload["model"], frozenset(target_model), "model state")
        for key, target in target_model.items():
            source = payload["model"][key]
            if not isinstance(source, Tensor):
                raise TypeError(f"model state {key!r} must be a tensor")
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"model state {key!r} metadata mismatch: "
                    f"{tuple(source.shape)}/{source.dtype} != "
                    f"{tuple(target.shape)}/{target.dtype}"
                )
        if not isinstance(payload["optimizer"], dict):
            raise TypeError("checkpoint optimizer state must be a mapping")
        if self.runtime.engine is None:
            _require_exact_keys(
                payload["optimizer"],
                frozenset({"state", "param_groups"}),
                "optimizer state",
            )
            self._validate_optimizer_state(payload["optimizer"])
        scheduler_state = payload["scheduler"]
        if (self.scheduler is None) != (scheduler_state is None):
            raise ValueError("checkpoint scheduler presence does not match trainer")
        if scheduler_state is not None and not isinstance(scheduler_state, dict):
            raise TypeError("checkpoint scheduler state must be a mapping")
        if scheduler_state is not None:
            _require_exact_keys(
                scheduler_state,
                frozenset(self.scheduler.state_dict()),
                "scheduler state",
            )
        if not isinstance(payload["rng"], dict):
            raise TypeError("checkpoint RNG state must be a mapping")
        _require_exact_keys(payload["rng"], self.RNG_KEYS, "RNG state")
        self._validate_rng_state(payload["rng"])
        if not isinstance(payload["step"], int) or payload["step"] < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        if (
            not isinstance(payload["data_batches_consumed"], int)
            or payload["data_batches_consumed"] < 0
        ):
            raise ValueError(
                "checkpoint data_batches_consumed must be a non-negative integer"
            )
        schedule_state = payload["prompt_schedule"]
        if (self.prompt_schedule is None) != (schedule_state is None):
            raise ValueError("checkpoint prompt schedule presence does not match trainer")
        if schedule_state is not None:
            if not isinstance(schedule_state, dict):
                raise TypeError("checkpoint prompt schedule state must be a mapping")
            _require_exact_keys(
                schedule_state,
                frozenset({"position", "max_steps"}),
                "prompt schedule state",
            )
            if schedule_state["position"] != payload["step"]:
                raise ValueError("checkpoint prompt schedule position differs from step")
            if schedule_state["max_steps"] != self.max_steps:
                raise ValueError("checkpoint prompt schedule max_steps differs")
        if not isinstance(payload["config"], dict):
            raise TypeError("checkpoint config metadata must be a mapping")
        if payload["config"] != self.config_metadata:
            raise ValueError("checkpoint config metadata does not match trainer config")
        callback_state = payload["callbacks"]
        if not isinstance(callback_state, dict):
            raise TypeError("checkpoint callback state must be a mapping")
        expected_callback_keys = {
            str(index) for index in range(len(self.callbacks.callbacks))
        }
        _require_exact_keys(
            callback_state,
            frozenset(expected_callback_keys),
            "callback state",
        )
        if any(not isinstance(state, dict) for state in callback_state.values()):
            raise TypeError("individual checkpoint callback state must be a mapping")

    def _validate_optimizer_state(self, state: Mapping[str, Any]) -> None:
        groups = state["param_groups"]
        current_groups = self.optimizer.state_dict()["param_groups"]
        if not isinstance(state["state"], dict) or not isinstance(groups, list):
            raise TypeError("checkpoint optimizer state has invalid containers")
        if len(groups) != len(current_groups):
            raise ValueError("checkpoint optimizer parameter-group count differs")
        for index, (group, current) in enumerate(
            zip(groups, current_groups, strict=True)
        ):
            if not isinstance(group, dict):
                raise TypeError("checkpoint optimizer parameter group must be a mapping")
            _require_exact_keys(
                group,
                frozenset(current),
                f"optimizer parameter group {index}",
            )
            if len(group["params"]) != len(current["params"]):
                raise ValueError(
                    f"optimizer parameter group {index} parameter count differs"
                )

    def _validate_rng_state(self, state: Mapping[str, Any]) -> None:
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, dict):
            raise TypeError("checkpoint NumPy RNG state must be a mapping")
        _require_exact_keys(
            numpy_state,
            frozenset(
                {
                    "bit_generator",
                    "state",
                    "position",
                    "has_gauss",
                    "cached_gaussian",
                }
            ),
            "NumPy RNG state",
        )
        try:
            random.Random().setstate(state["python"])
            numpy_probe = np.random.RandomState()
            numpy_probe.set_state(
                (
                    numpy_state["bit_generator"],
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    numpy_state["position"],
                    numpy_state["has_gauss"],
                    numpy_state["cached_gaussian"],
                )
            )
            torch.Generator().set_state(state["torch_cpu"])
            torch.Generator(device=self.generator.device).set_state(
                state["generator"]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint RNG state is invalid") from exc
        cuda_state = state["torch_cuda"]
        if not isinstance(cuda_state, list):
            raise TypeError("checkpoint CUDA RNG state must be a list")
        if cuda_state and not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if any(not isinstance(value, Tensor) for value in cuda_state):
            raise TypeError("checkpoint CUDA RNG states must be tensors")

    def _rng_state(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "state": numpy_state[1].tolist(),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generator": self.generator.get_state(),
        }

    def _set_rng_state(self, state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        _require_exact_keys(
            numpy_state,
            frozenset(
                {
                    "bit_generator",
                    "state",
                    "position",
                    "has_gauss",
                    "cached_gaussian",
                }
            ),
            "NumPy RNG state",
        )
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"]:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        self.generator.set_state(state["generator"])

    def _data_state(self) -> Mapping[str, Any] | None:
        source = _stateful_data_source(self.data_source)
        if source is None:
            return None
        state = source.state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("data source state_dict must return a mapping")
        return dict(state)

    def _load_data_state(self, state: Mapping[str, Any] | None) -> None:
        source = _stateful_data_source(self.data_source)
        if state is None:
            return
        if source is None:
            raise ValueError("checkpoint contains data state but trainer has no stateful data source")
        source.load_state_dict(dict(state))


def _stateful_data_source(source: Any) -> Any | None:
    while source is not None:
        if hasattr(source, "state_dict") and hasattr(source, "load_state_dict"):
            return source
        for name in ("batch_sampler", "sampler", "dataset"):
            child = getattr(source, name, None)
            if child is not None and child is not source:
                found = _stateful_data_source(child)
                if found is not None:
                    return found
        return None
    return None


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _move_tensors(getattr(value, field.name), device)
                for field in fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def validate_homogeneous_mode(
    modes: tuple[InteractionMode, ...] | list[InteractionMode],
) -> InteractionMode:
    if not modes:
        raise ValueError("training batch must contain at least one mode")
    parsed = tuple(InteractionMode.parse(mode) for mode in modes)
    if any(mode is not parsed[0] for mode in parsed[1:]):
        raise ValueError(
            "native training requires homogeneous checkpoint mode per batch"
        )
    return parsed[0]


def _select_optional(value: Tensor | None, mask: Tensor) -> Tensor | None:
    if value is None:
        return None
    return value[mask.to(device=value.device)]


def _batch_condition_modes(
    batch: TrainingBatch,
) -> tuple[tuple[ConditionMode, ...], Tensor]:
    if batch.condition_modes is not None:
        assert batch.condition_ids is not None
        return (
            tuple(ConditionMode.parse(mode) for mode in batch.condition_modes),
            batch.condition_ids,
        )
    modes: list[ConditionMode] = []
    for index in range(batch.batch_size):
        has_goal = (
            batch.goal_image_mask is not None
            and bool(batch.goal_image_mask[index].any().item())
        )
        has_demo = (
            batch.demo_video_mask is not None
            and bool(batch.demo_video_mask[index].any().item())
        )
        if has_goal and has_demo:
            raise ValueError(
                f"sample {index} cannot carry both goal and demo prompts"
            )
        modes.append(
            ConditionMode.GOAL_IMAGE_TO_VA
            if has_goal
            else (
                ConditionMode.VIDEO_TO_VA
                if has_demo
                else ConditionMode.T2VA
            )
        )
    return (
        tuple(modes),
        torch.tensor(
            [mode.id for mode in modes],
            dtype=torch.long,
            device=batch.actions.device,
        ),
    )


def _condition_groups(
    condition_ids: Tensor,
    selected_mode: Tensor,
    demo_video_mask: Tensor | None,
) -> list[tuple[ConditionMode, int | None]]:
    groups: list[tuple[ConditionMode, int | None]] = []
    for condition_mode in ConditionMode:
        selected = selected_mode & (condition_ids == condition_mode.id)
        if not bool(selected.any()):
            continue
        if condition_mode is not ConditionMode.VIDEO_TO_VA:
            groups.append((condition_mode, None))
            continue
        if demo_video_mask is None:
            raise ValueError("video-to-VA samples require demo_video_mask")
        lengths = demo_video_mask.sum(dim=1).to(device=selected.device)
        unique_lengths = torch.unique(lengths[selected], sorted=True)
        for length in unique_lengths.detach().cpu().tolist():
            if int(length) <= 0:
                raise ValueError("video-to-VA prompts cannot be empty")
            groups.append((condition_mode, int(length)))
    return groups


def _merge_auto_metrics(
    outputs: list[tuple[Tensor, WorldActionOutput]],
    auto_mask: Tensor,
) -> dict[str, Tensor]:
    metrics: dict[str, Tensor] = {}
    for name in (
        "semantic_prediction",
        "semantic_target",
        "semantic_mask",
        "planning_logits",
        "planning_labels",
    ):
        metric_mask = auto_mask
        metric_indices = metric_mask.nonzero(as_tuple=False).flatten()
        pieces = [
            (selected, output.metrics[name])
            for selected, output in outputs
            if name in output.metrics and bool((selected & metric_mask).any())
        ]
        if not pieces:
            continue
        first = pieces[0][1]
        if any(value.ndim != first.ndim for _, value in pieces):
            raise ValueError(f"metric {name!r} has inconsistent ranks across groups")
        trailing_shape = tuple(
            max(value.shape[dimension] for _, value in pieces)
            for dimension in range(1, first.ndim)
        )
        fill_value = -100 if name == "planning_labels" else 0
        merged = torch.full(
            (len(metric_indices), *trailing_shape),
            fill_value,
            dtype=first.dtype,
            device=first.device,
        )
        for selected, value in pieces:
            selected = selected & metric_mask
            indices = selected.nonzero(as_tuple=False).flatten().to(
                device=metric_indices.device
            )
            positions = torch.searchsorted(metric_indices, indices)
            if value.shape[1:] != trailing_shape:
                padded = torch.full(
                    (value.shape[0], *trailing_shape),
                    fill_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                slices = (slice(None),) + tuple(
                    slice(0, size) for size in value.shape[1:]
                )
                padded[slices] = value
                value = padded
            merged = merged.index_copy(
                0,
                positions.to(device=merged.device),
                value,
            )
        metrics[name] = merged
    return metrics


def _select_text(
    values: list[str] | None, indices: list[int]
) -> list[str] | None:
    return None if values is None else [values[index] for index in indices]


def _batch_dataset_diagnostics(
    batch: TrainingBatch | ModelReadyTrainingBatch,
) -> list[dict[str, object]]:
    if not isinstance(batch, TrainingBatch):
        return [{"dataset_index": "unavailable:model-ready-batch"}]
    raw_provenance = batch.metadata.get("provenance", ())
    provenance = (
        raw_provenance
        if isinstance(raw_provenance, (tuple, list))
        else ()
    )
    diagnostics: list[dict[str, object]] = []
    for index, (episode_id, event_id) in enumerate(
        zip(batch.episode_ids, batch.event_ids, strict=True)
    ):
        sample_provenance = (
            provenance[index]
            if index < len(provenance)
            and isinstance(provenance[index], Mapping)
            else {}
        )
        diagnostics.append(
            {
                "dataset_index": sample_provenance.get(
                    "dataset_index", "unavailable"
                ),
                "dataset_episode_index": sample_provenance.get(
                    "dataset_episode_index", "unavailable"
                ),
                "temporal_anchor_index": sample_provenance.get(
                    "temporal_anchor_index", "unavailable"
                ),
                "episode_id": episode_id,
                "event_id": event_id,
            }
        )
    return diagnostics


def _select_ready_batch(
    batch: ModelReadyTrainingBatch,
    mask: Tensor,
    mode: InteractionMode,
    condition_mode: ConditionMode,
    *,
    demo_length: int | None = None,
) -> ModelReadyTrainingBatch:
    indices_tensor = mask.nonzero(as_tuple=False).flatten()
    indices = indices_tensor.detach().cpu().tolist()

    def select(value: Tensor) -> Tensor:
        return value.index_select(
            0, indices_tensor.to(device=value.device)
        )

    prompts = batch.prompts
    goal_images = (
        select(prompts.goal_images)
        if condition_mode is ConditionMode.GOAL_IMAGE_TO_VA
        and prompts.goal_images is not None
        else None
    )
    demo_videos = (
        select(prompts.demo_videos)[:, :demo_length]
        if condition_mode is ConditionMode.VIDEO_TO_VA
        and prompts.demo_videos is not None
        and demo_length is not None
        else None
    )
    selected = ModelReadyTrainingBatch(
        mode=mode,
        mode_mask=torch.full(
            (len(indices),),
            mode is InteractionMode.AUTO,
            dtype=torch.bool,
            device=batch.clean_action.device,
        ),
        condition_modes=(condition_mode,) * len(indices),
        condition_ids=torch.full(
            (len(indices),),
            condition_mode.id,
            dtype=torch.long,
            device=batch.clean_action.device,
        ),
        observation=ObservationBatch(
            images=select(batch.observation.images),
            head_view=select(batch.observation.head_view),
            proprioception=select(batch.observation.proprioception),
            embodiment_id=select(batch.observation.embodiment_id),
            vlm_history_images=(
                None
                if batch.observation.vlm_history_images is None
                else select(batch.observation.vlm_history_images)
            ),
            vlm_history_mask=(
                None
                if batch.observation.vlm_history_mask is None
                else select(batch.observation.vlm_history_mask)
            ),
        ),
        prompts=PromptBatch(
            vlm_planning_text=_select_text(prompts.vlm_planning_text, indices),
            language_instruction=_select_text(
                prompts.language_instruction, indices
            ),
            negative_vlm_text=_select_text(prompts.negative_vlm_text, indices),
            negative_language_instruction=_select_text(
                prompts.negative_language_instruction, indices
            ),
            planning_labels_text=_select_text(
                prompts.planning_labels_text, indices
            ),
            goal_images=goal_images,
            goal_image_mask=(
                torch.ones(len(indices), dtype=torch.bool, device=goal_images.device)
                if goal_images is not None
                else None
            ),
            demo_videos=demo_videos,
            demo_video_mask=(
                torch.ones(
                    (len(indices), demo_length),
                    dtype=torch.bool,
                    device=demo_videos.device,
                )
                if demo_videos is not None and demo_length is not None
                else None
            ),
            visual_prompt=(
                "none"
                if condition_mode is ConditionMode.T2VA
                else "goal_or_demo"
            ),
        ),
        clean_video=select(batch.clean_video),
        clean_video_latents=select(batch.clean_video_latents),
        clean_action=select(batch.clean_action),
        clean_video_normalized=batch.clean_video_normalized,
        action_mask=_select_optional(batch.action_mask, mask),
        video_weight=_select_optional(batch.video_weight, mask),
        action_weight=_select_optional(batch.action_weight, mask),
        action_dim_mask=(
            _select_optional(batch.action_dim_mask, mask)
            if batch.action_dim_mask is not None
            and batch.action_dim_mask.ndim > 1
            else batch.action_dim_mask
        ),
        has_real_action=_select_optional(batch.has_real_action, mask),
        semantic_target=_select_optional(batch.semantic_target, mask),
        semantic_mask=_select_optional(batch.semantic_mask, mask),
        planning_labels=_select_optional(batch.planning_labels, mask),
        planning_labels_text=_select_text(batch.planning_labels_text, indices),
    )
    selected.validate()
    return selected


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} keys failed validation: missing={missing}, unexpected={unexpected}"
        )


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _write_hf_config(
    destination: Path, config_metadata: Mapping[str, Any]
) -> None:
    model = config_metadata.get("model")
    model_config = model.get("model_config") if isinstance(model, Mapping) else None
    generation_config = (
        model.get("generation_config") if isinstance(model, Mapping) else None
    )
    payload = {
        "architectures": ["WorldScapePolicy"],
        "model_type": "worldscape_policy",
        "worldscape_policy_model_config": model_config,
        "generation_config": generation_config,
    }
    temporary = destination / f".config.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination / "config.json")


def _write_complete_marker(destination: Path, value: str) -> None:
    marker = destination / NativeTrainer.DEEPSPEED_COMPLETE
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(value + "\n")
    os.replace(temporary, marker)


def _validate_complete_marker(source: Path, allowed: set[str]) -> None:
    marker = source / NativeTrainer.DEEPSPEED_COMPLETE
    actual = marker.read_text().strip() if marker.is_file() else None
    if actual not in allowed:
        raise ValueError("training checkpoint is incomplete or has an invalid marker")


def _normalize_deepspeed_policy_state(
    exported: object,
    target: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    if (
        isinstance(exported, dict)
        and frozenset(exported) != frozenset(target)
        and set(exported) == {"module"}
    ):
        exported = exported["module"]
    if not isinstance(exported, dict):
        raise TypeError("DeepSpeed 16-bit model export must contain a state mapping")
    normalized: dict[str, Tensor] = {}
    for key, value in exported.items():
        normalized_key = str(key)
        for prefix in ("module.policy.", "policy."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
                break
        if normalized_key in normalized:
            raise ValueError(
                f"DeepSpeed policy export has duplicate key {normalized_key!r}"
            )
        normalized[normalized_key] = value
    _require_exact_keys(
        normalized, frozenset(target), "DeepSpeed portable policy state"
    )
    for key, target_tensor in target.items():
        source = normalized[key]
        if not isinstance(source, Tensor):
            raise TypeError(f"portable policy state {key!r} must be a tensor")
        if source.shape != target_tensor.shape or source.dtype != target_tensor.dtype:
            raise ValueError(
                f"portable policy state {key!r} metadata mismatch: "
                f"{tuple(source.shape)}/{source.dtype} != "
                f"{tuple(target_tensor.shape)}/{target_tensor.dtype}"
            )
    return normalized


Trainer = NativeTrainer

__all__ = [
    "ModelReadyTrainingBatch",
    "NativeTrainer",
    "NativeWan22BatchAdapter",
    "Trainer",
    "TrainingBatchAdapter",
    "TrainingNoiseKernel",
    "validate_homogeneous_mode",
]
