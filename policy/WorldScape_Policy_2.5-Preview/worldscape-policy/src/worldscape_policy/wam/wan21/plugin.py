from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from worldscape_policy.types import VisualMemoryState, WorldActionOutput


@dataclass(frozen=True)
class Wan21WAMConfig:
    """Configuration marker for the future native Wan2.1 adapter."""


class Wan21WAMPlugin(nn.Module):
    """Protocol-complete Wan2.1 shell with no hidden legacy implementation."""

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
    ) -> WorldActionOutput:
        del (
            clean_video,
            clean_action,
            noisy_video,
            noisy_action,
            video_timestep,
            action_timestep,
            state,
            embodiment_id,
            cross_attention_tokens,
            negative_cross_attention_tokens,
            persistent_prefill,
            recent_visual_prefill,
            clean_video_latents,
            clean_video_normalized,
        )
        raise NotImplementedError(
            "Wan2.1 native training is not implemented; use a supported WAM plugin"
        )

    @torch.no_grad()
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
    ) -> WorldActionOutput:
        del (
            reference_frame,
            reference_frame_normalized,
            chunk_latents,
            observation_num_frames,
            prompt_signature,
            state,
            embodiment_id,
            cross_attention_tokens,
            negative_cross_attention_tokens,
            visual_memory,
            generator,
        )
        raise NotImplementedError(
            "Wan2.1 sampling is not implemented by the native WAM adapter"
        )


__all__ = ["Wan21WAMConfig", "Wan21WAMPlugin"]
