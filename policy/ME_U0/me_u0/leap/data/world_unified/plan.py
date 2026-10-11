"""Supervision task plans for legacy and variable temporal geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from leap.data.world_unified.types import (
    ACTION_STEPS,
    RGB_FRAMES,
    VAE_TEMPORAL_DOWNSAMPLE,
    WorldUnifiedTask,
)


@dataclass(frozen=True)
class WorldUnifiedPlan:
    """Clean-conditioning and flow-loss ownership for one canonical sample."""

    task: WorldUnifiedTask
    clean_video_frames: Tuple[int, ...]
    video_loss_frames: Tuple[int, ...]
    clean_action_steps: Tuple[int, ...]
    action_loss_steps: Tuple[int, ...]
    rgb_frames: int = RGB_FRAMES
    action_steps: int = ACTION_STEPS

    @property
    def task_id(self) -> int:
        return 2

    @property
    def latent_frames(self) -> int:
        return 1 + (self.rgb_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE

    def action_temporal_position(self, action_step: int) -> Tuple[int, int]:
        """Return ``(latent_transition, substep)`` for an action step.

        Actions are distributed uniformly over future latent transitions. For
        example, the legacy 33-RGB/H32 route and the 13-RGB/H12 route both use
        four action substeps per transition, while 13-RGB/H36 uses twelve.
        """

        if not 0 <= action_step < self.action_steps:
            raise IndexError(
                f"action_step must be in [0,{self.action_steps}), got {action_step}"
            )
        future_latents = self.latent_frames - 1
        if future_latents <= 0 or self.action_steps % future_latents:
            raise AssertionError("action/latent geometry is internally inconsistent")
        actions_per_transition = self.action_steps // future_latents
        transition, substep = divmod(action_step, actions_per_transition)
        latent_transition = transition + 1
        if latent_transition >= self.latent_frames:
            raise AssertionError("action/latent geometry is internally inconsistent")
        return latent_transition, substep


def build_task_plan(
    *,
    rgb_frames: int = RGB_FRAMES,
    action_steps: int = ACTION_STEPS,
) -> WorldUnifiedPlan:
    """Build current-observation conditioning and future RGB/action targets."""

    if rgb_frames < 2:
        raise ValueError("task-plan RGB frame count must be at least two")
    if (rgb_frames - 1) % VAE_TEMPORAL_DOWNSAMPLE:
        raise ValueError(
            "RGB/latent geometry is incompatible with Wan's temporal stride"
        )
    latent_frames = 1 + (rgb_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE
    if action_steps < 1:
        raise ValueError("task-plan action step count must be positive")
    if action_steps % (latent_frames - 1):
        raise ValueError("action steps must divide evenly across future latents")
    return WorldUnifiedPlan(
        task="video_action_pred",
        clean_video_frames=(0,),
        video_loss_frames=tuple(range(1, rgb_frames)),
        clean_action_steps=(),
        action_loss_steps=tuple(range(action_steps)),
        rgb_frames=rgb_frames,
        action_steps=action_steps,
    )
