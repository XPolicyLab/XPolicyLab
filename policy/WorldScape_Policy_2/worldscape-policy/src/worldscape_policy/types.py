from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal

import torch
from torch import Tensor


class InteractionMode(str, Enum):
    """The mutually exclusive language-conditioning modes."""

    AUTO = "auto"
    INTERACTIVE = "interactive"

    @classmethod
    def parse(cls, value: InteractionMode | str) -> InteractionMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Unknown interaction mode {value!r}; expected one of: {choices}") from exc


@dataclass
class ObservationBatch:
    """Policy observations with an explicit batch-first contract."""

    images: Tensor
    head_view: Tensor
    proprioception: Tensor
    embodiment_id: Tensor
    vlm_history_images: Tensor | None = None
    vlm_history_mask: Tensor | None = None

    def validate(self) -> None:
        if self.images.ndim != 6:
            raise ValueError(
                "images must have shape [B, T, V, C, H, W], "
                f"got {tuple(self.images.shape)}"
            )
        if self.head_view.ndim != 5:
            raise ValueError(
                "head_view must have shape [B, 1, C, H, W], "
                f"got {tuple(self.head_view.shape)}"
            )
        batch_size = self.images.shape[0]
        for name, tensor in (
            ("head_view", self.head_view),
            ("proprioception", self.proprioception),
            ("embodiment_id", self.embodiment_id),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} does not match images batch size "
                    f"{batch_size}"
                )
        if (self.vlm_history_images is None) != (self.vlm_history_mask is None):
            raise ValueError(
                "vlm_history_images and vlm_history_mask must be provided together"
            )
        if self.vlm_history_images is not None:
            if self.vlm_history_images.ndim != 5:
                raise ValueError(
                    "vlm_history_images must have shape [B,T,C,H,W]"
                )
            if self.vlm_history_images.shape[0] != batch_size:
                raise ValueError("VLM history batch size must match images")
            assert self.vlm_history_mask is not None
            if (
                self.vlm_history_mask.dtype is not torch.bool
                or self.vlm_history_mask.shape
                != self.vlm_history_images.shape[:2]
            ):
                raise ValueError(
                    "vlm_history_mask must be bool with shape [B,T]"
                )


@dataclass
class PromptBatch:
    """Mode-specific language and optional persistent visual prompts.

    ``vlm_planning_text`` is consumed only by Auto mode. It is the prompt used
    by the VLM to produce planning tokens, not a generated event label.
    ``language_instruction`` is consumed only by Interactive mode and is
    encoded directly by the text conditioner.
    """

    vlm_planning_text: list[str] | None = None
    language_instruction: list[str] | None = None
    negative_vlm_text: list[str] | None = None
    negative_language_instruction: list[str] | None = None
    planning_labels_text: list[str | None] | None = None
    goal_images: Tensor | None = None
    goal_image_mask: Tensor | None = None
    demo_videos: Tensor | None = None
    demo_video_mask: Tensor | None = None
    visual_prompt: Literal["none", "goal_or_demo", "mixed"] = "goal_or_demo"

    def validate(self, batch_size: int) -> None:
        if self.visual_prompt not in {"none", "goal_or_demo", "mixed"}:
            raise ValueError(
                "visual_prompt must be 'none', 'goal_or_demo', or 'mixed'"
            )
        if self.visual_prompt == "none" and (
            self.goal_images is not None or self.demo_videos is not None
        ):
            raise ValueError("visual_prompt='none' cannot carry goal/demo tensors")
        for name, values in (
            ("vlm_planning_text", self.vlm_planning_text),
            ("language_instruction", self.language_instruction),
            ("negative_vlm_text", self.negative_vlm_text),
            ("negative_language_instruction", self.negative_language_instruction),
            ("planning_labels_text", self.planning_labels_text),
        ):
            if values is not None and len(values) != batch_size:
                raise ValueError(f"{name} must contain {batch_size} strings, got {len(values)}")
        for name, tensor in (
            ("goal_images", self.goal_images),
            ("demo_videos", self.demo_videos),
        ):
            if tensor is not None and tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} does not match observations "
                    f"batch size {batch_size}"
                )
        for value_name, mask_name, temporal in (
            ("goal_images", "goal_image_mask", False),
            ("demo_videos", "demo_video_mask", True),
        ):
            value = getattr(self, value_name)
            mask = getattr(self, mask_name)
            if value is None and mask is not None:
                raise ValueError(f"{mask_name} requires {value_name}")
            if mask is None:
                continue
            expected_shape = (
                (batch_size, value.shape[1])
                if temporal
                else (batch_size,)
            )
            if mask.dtype is not torch.bool or mask.shape != expected_shape:
                raise ValueError(
                    f"{mask_name} must be bool with shape {list(expected_shape)}"
                )
        if self.goal_images is not None and self.demo_videos is not None:
            if self.goal_image_mask is None or self.demo_video_mask is None:
                raise ValueError(
                    "mixed goal/demo prompts require explicit per-sample masks"
                )
            demo_present = self.demo_video_mask.any(dim=1)
            if bool((self.goal_image_mask & demo_present).any()):
                raise ValueError(
                    "goal and demo prompts must be mutually exclusive per sample"
                )

    def condition_signature(
        self, mode: InteractionMode | str
    ) -> tuple[str, ...]:
        parsed = InteractionMode.parse(mode)
        if parsed is InteractionMode.AUTO:
            return tuple(self.vlm_planning_text or ()) + tuple(
                self.negative_vlm_text or ()
            )
        return tuple(self.language_instruction or ()) + tuple(
            self.negative_language_instruction or ()
        )


@dataclass
class EventMemoryState:
    perception_tokens: Tensor | None = None
    planning_tokens: Tensor | None = None
    valid_mask: Tensor | None = None
    pending_perception_tokens: Tensor | None = None
    pending_planning_tokens: Tensor | None = None
    pending_valid_mask: Tensor | None = None
    cached_cross_attention_tokens: Tensor | None = None
    cached_negative_cross_attention_tokens: Tensor | None = None
    prompt_signature: tuple[str, ...] | None = None


@dataclass
class WanI2VCondition:
    """Window-stable Wan image-to-video conditioning.

    These tensors come from the raw reference frame through the image encoder
    and the masked VAE path. They are not interchangeable with chunk latents.
    """

    clip_features: Tensor
    masked_latent_y: Tensor


@dataclass
class WAMInferenceState:
    """Plugin-owned causal state that is committed with an executed action."""

    cache_owner_rank: int | None = None
    cache_world_size: int | None = None
    i2v_condition: WanI2VCondition | None = None
    current_start_frame: int = 0
    rebase_observation_window: bool = False
    positive_kv_cache: Any | None = None
    negative_kv_cache: Any | None = None
    positive_cross_attention_cache: Any | None = None
    negative_cross_attention_cache: Any | None = None
    condition_tokens: Tensor | None = None
    negative_condition_tokens: Tensor | None = None
    prompt_signature: tuple[str, ...] | None = None

    def fork(self) -> WAMInferenceState:
        """Create an isolated candidate state for transactional prediction."""

        i2v = self.i2v_condition
        if i2v is not None:
            i2v = WanI2VCondition(
                clip_features=i2v.clip_features.clone(),
                masked_latent_y=i2v.masked_latent_y.clone(),
            )
        return replace(
            self,
            i2v_condition=i2v,
            positive_kv_cache=_clone_runtime_value(self.positive_kv_cache),
            negative_kv_cache=_clone_runtime_value(self.negative_kv_cache),
            positive_cross_attention_cache=_clone_runtime_value(
                self.positive_cross_attention_cache
            ),
            negative_cross_attention_cache=_clone_runtime_value(
                self.negative_cross_attention_cache
            ),
            condition_tokens=_clone_runtime_value(self.condition_tokens),
            negative_condition_tokens=_clone_runtime_value(
                self.negative_condition_tokens
            ),
        )

    def without_caches(self) -> WAMInferenceState:
        return replace(
            self,
            positive_kv_cache=None,
            negative_kv_cache=None,
            positive_cross_attention_cache=None,
            negative_cross_attention_cache=None,
        )


@dataclass
class VisualMemoryState:
    persistent_prompt_latents: Tensor | None = None
    persistent_prompt_version: int = 0
    recent_observation_latents: Tensor | None = None
    wam_state: WAMInferenceState = field(default_factory=WAMInferenceState)

    def without_runtime_cache(self) -> VisualMemoryState:
        return replace(self, wam_state=self.wam_state.without_caches())


@dataclass
class Conditioning:
    cross_attention_tokens: Tensor
    negative_cross_attention_tokens: Tensor | None = None
    event_memory: EventMemoryState | None = None
    visual_memory: VisualMemoryState = field(default_factory=VisualMemoryState)
    semantic_prediction: Tensor | None = None
    semantic_target: Tensor | None = None
    semantic_mask: Tensor | None = None
    planning_logits: Tensor | None = None
    planning_labels: Tensor | None = None


@dataclass
class WorldActionOutput:
    """Predictions plus candidate state to commit after successful execution."""

    action_velocity: Tensor | None = None
    video_velocity: Tensor | None = None
    action: Tensor | None = None
    video: Tensor | None = None
    next_memory: EventMemoryState | None = None
    next_visual_memory: VisualMemoryState | None = None
    metrics: dict[str, Tensor] = field(default_factory=dict)

    def require_action(self) -> Tensor:
        if self.action is None:
            raise RuntimeError("WAM output does not contain a sampled action")
        return self.action

    def result_snapshot(self) -> WorldActionOutput:
        """Copy predictions to CPU without retaining transactional GPU state."""

        def snapshot(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.detach().cpu()

        return WorldActionOutput(
            action_velocity=snapshot(self.action_velocity),
            video_velocity=snapshot(self.video_velocity),
            action=snapshot(self.action),
            video=snapshot(self.video),
            metrics={key: value.detach().cpu() for key, value in self.metrics.items()},
        )


def empty_cross_attention(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a shape-stable empty condition for tests and unconditional plugins."""

    return torch.empty(batch_size, 0, 0, device=device, dtype=dtype)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    return value
