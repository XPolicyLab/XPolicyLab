from __future__ import annotations

import torch
from torch import Tensor, nn

from worldscape_policy.types import PromptBatch, VisualMemoryState, WAMInferenceState
from worldscape_policy.wam.protocol import VisualCodec


class VisualPrefillManager(nn.Module):
    """Own the visual codec and visual-memory lifecycle.

    Making this an ``nn.Module`` is intentional: the codec is now part of the
    native policy module tree, so converted VAE weights are visible to
    ``state_dict`` and can be validated before loading.
    """

    def __init__(
        self,
        codec: nn.Module,
        *,
        persistent_prompt: str = "goal_or_demo",
    ) -> None:
        super().__init__()
        if not isinstance(codec, nn.Module):
            raise TypeError("codec must be an nn.Module")
        if not isinstance(codec, VisualCodec):
            raise TypeError("codec must implement encode_visual(video)")
        if persistent_prompt not in {"none", "goal_or_demo"}:
            raise ValueError(
                "persistent_prompt must be 'none' or 'goal_or_demo'"
            )
        self.codec = codec
        self.persistent_prompt = persistent_prompt

    def encode_persistent_prompt(
        self,
        goal_images: Tensor | None,
        demo_videos: Tensor | None,
    ) -> Tensor | None:
        if self.persistent_prompt == "none":
            if goal_images is not None or demo_videos is not None:
                raise ValueError(
                    "persistent visual prefill is disabled by model config"
                )
            return None
        if goal_images is not None and demo_videos is not None:
            raise ValueError("Only one of goal_images or demo_videos may be provided")
        visual_prompt = goal_images if goal_images is not None else demo_videos
        if visual_prompt is None:
            return None
        return self.codec.encode_visual(visual_prompt)

    def encode_recent_observations(self, images: Tensor) -> Tensor:
        prepare_diffusion_video = getattr(
            self.codec, "prepare_diffusion_video", None
        )
        encode_normalized = getattr(self.codec, "encode_normalized", None)
        if (
            images.ndim == 6
            and images.shape[2] == 3
            and callable(prepare_diffusion_video)
            and callable(encode_normalized)
        ):
            return encode_normalized(prepare_diffusion_video(images))
        return self.codec.encode_visual(images)

    def prepare(
        self,
        *,
        images: Tensor,
        prompts: PromptBatch,
        previous_state: VisualMemoryState | None = None,
        recent_observation_latents: Tensor | None = None,
    ) -> VisualMemoryState:
        previous_state = previous_state or VisualMemoryState()
        prompts.validate(images.shape[0])
        disabled = (
            self.persistent_prompt == "none"
            or prompts.visual_prompt == "none"
        )
        if disabled:
            recent = (
                recent_observation_latents
                if recent_observation_latents is not None
                else self.encode_recent_observations(images)
            )
            prompt_changed = previous_state.persistent_prompt_latents is not None
            return VisualMemoryState(
                persistent_prompt_latents=None,
                persistent_prompt_version=(
                    previous_state.persistent_prompt_version + int(prompt_changed)
                ),
                recent_observation_latents=recent,
                wam_state=(
                    WAMInferenceState()
                    if prompt_changed
                    else previous_state.wam_state
                ),
            )
        has_new_prompt = prompts.goal_images is not None or prompts.demo_videos is not None
        persistent = (
            self.encode_persistent_prompt(prompts.goal_images, prompts.demo_videos)
            if has_new_prompt
            else previous_state.persistent_prompt_latents
        )
        prompt_changed = has_new_prompt and not self._same_prompt(
            previous_state.persistent_prompt_latents,
            persistent,
        )
        recent = (
            recent_observation_latents
            if recent_observation_latents is not None
            else self.encode_recent_observations(images)
        )
        return VisualMemoryState(
            persistent_prompt_latents=persistent,
            persistent_prompt_version=(
                previous_state.persistent_prompt_version + int(prompt_changed)
            ),
            recent_observation_latents=recent,
            wam_state=(
                WAMInferenceState()
                if prompt_changed
                else previous_state.wam_state
            ),
        )

    @staticmethod
    def reset_persistent_prompt(state: VisualMemoryState) -> VisualMemoryState:
        return VisualMemoryState(
            recent_observation_latents=state.recent_observation_latents,
            persistent_prompt_version=state.persistent_prompt_version + 1,
            wam_state=WAMInferenceState(),
        )

    @staticmethod
    def reset_episode() -> VisualMemoryState:
        return VisualMemoryState()

    @staticmethod
    def _same_prompt(previous: Tensor | None, current: Tensor | None) -> bool:
        if previous is None or current is None:
            return previous is current
        return previous.shape == current.shape and torch.equal(
            previous.to(device=current.device, dtype=current.dtype),
            current,
        )
