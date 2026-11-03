from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from worldscape_policy.types import VisualMemoryState, WorldActionOutput


@runtime_checkable
class VisualCodec(Protocol):
    """Non-WAM owner for converting public visual tensors to WAM latents."""

    def encode_visual(self, video: Tensor) -> Tensor: ...


class VisualCodecProvider(Protocol):
    """Return the uniquely registered policy visual codec without owning it."""

    def __call__(self) -> VisualCodec: ...


@runtime_checkable
class WAMPlugin(Protocol):
    """Condition-agnostic generator; visual encoding belongs to VisualCodec."""

    def training_forward(
        self,
        *,
        clean_video: Tensor,
        clean_action: Tensor,
        noisy_video: Tensor,
        noisy_action: Tensor,
        video_timestep: Tensor,
        action_timestep: Tensor,
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        persistent_prefill: Tensor | None,
        recent_visual_prefill: Tensor | None,
        clean_video_latents: Tensor | None = None,
        clean_video_normalized: bool = False,
    ) -> WorldActionOutput: ...

    def sample(
        self,
        *,
        reference_frame: Tensor,
        reference_frame_normalized: bool = False,
        chunk_latents: Tensor,
        observation_num_frames: int,
        prompt_signature: tuple[str, ...],
        state: Tensor,
        embodiment_id: Tensor,
        cross_attention_tokens: Tensor,
        negative_cross_attention_tokens: Tensor | None,
        visual_memory: VisualMemoryState,
        generator: torch.Generator,
    ) -> WorldActionOutput: ...
