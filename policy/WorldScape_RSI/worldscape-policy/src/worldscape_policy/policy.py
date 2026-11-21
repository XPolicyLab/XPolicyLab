from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from worldscape_policy.conditioning.router import ConditionRouter
from worldscape_policy.memory.visual.prefill import VisualPrefillManager
from worldscape_policy.types import (
    Conditioning,
    EventMemoryState,
    InteractionMode,
    ObservationBatch,
    PromptBatch,
    VisualMemoryState,
    WorldActionOutput,
)
from worldscape_policy.wam.protocol import WAMPlugin


class WorldScapePolicy(nn.Module):
    """Compose condition routing, explicit memory, and a pluggable WAM."""

    def __init__(
        self,
        *,
        condition_router: ConditionRouter,
        wam: nn.Module,
        visual_memory: VisualPrefillManager,
        configured_mode: InteractionMode | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(wam, WAMPlugin):
            raise TypeError("wam must implement the WAMPlugin protocol")
        self.condition_router = condition_router
        self.wam = wam
        if not isinstance(visual_memory, VisualPrefillManager):
            raise TypeError("visual_memory must be a VisualPrefillManager")
        self.visual_memory = visual_memory
        self.configured_mode = (
            InteractionMode.parse(configured_mode)
            if configured_mode is not None
            else None
        )
        self.training_supported_modes: tuple[InteractionMode, ...] = (
            (InteractionMode.AUTO, InteractionMode.INTERACTIVE)
            if self.configured_mode is InteractionMode.AUTO
            else (
                (InteractionMode.INTERACTIVE,)
                if self.configured_mode is InteractionMode.INTERACTIVE
                else (InteractionMode.AUTO, InteractionMode.INTERACTIVE)
            )
        )

    def reset_episode(self) -> None:
        """Reset plugin-owned caches; explicit memory is reset by the runtime."""

        reset = getattr(self.wam, "reset_episode", None)
        if callable(reset):
            reset()

    def condition(
        self,
        *,
        mode: InteractionMode | str,
        observation: ObservationBatch,
        prompts: PromptBatch,
        event_memory: EventMemoryState | None = None,
        visual_memory: VisualMemoryState | None = None,
        training: bool | None = None,
        planning_supervision: bool = False,
        enforce_configured_mode: bool = True,
        recent_observation_latents: Tensor | None = None,
    ) -> Conditioning:
        parsed_mode = InteractionMode.parse(mode)
        if (
            enforce_configured_mode
            and self.configured_mode is not None
            and parsed_mode is not self.configured_mode
        ):
            raise ValueError(
                f"Policy checkpoint is configured for {self.configured_mode.value!r}, "
                f"not {parsed_mode.value!r}"
            )
        visual_state = self.visual_memory.prepare(
            images=observation.images,
            prompts=prompts,
            previous_state=visual_memory,
            recent_observation_latents=recent_observation_latents,
        )
        if (
            visual_memory is not None
            and visual_state.persistent_prompt_version
            != visual_memory.persistent_prompt_version
        ):
            event_memory = None
        condition = self.condition_router(
            mode=parsed_mode,
            observation=observation,
            prompts=prompts,
            event_memory=event_memory,
            training=training,
            planning_supervision=planning_supervision,
        )
        if (
            training
            and parsed_mode is InteractionMode.AUTO
            and condition.semantic_prediction is not None
            and condition.semantic_target is None
            and prompts.language_instruction is not None
        ):
            teacher = self.condition_router.interactive.t5
            with torch.no_grad():
                semantic_target = teacher.encode_text(
                    prompts.language_instruction
                )
            condition = replace(
                condition,
                semantic_target=semantic_target.detach(),
            )
        return replace(condition, visual_memory=visual_state)

    def training_forward(
        self,
        *,
        mode: InteractionMode | str,
        observation: ObservationBatch,
        prompts: PromptBatch,
        clean_video: Tensor,
        clean_action: Tensor,
        noisy_video: Tensor,
        noisy_action: Tensor,
        video_timestep: Tensor,
        action_timestep: Tensor,
        event_memory: EventMemoryState | None = None,
        visual_memory: VisualMemoryState | None = None,
        planning_supervision: bool = False,
        clean_video_latents: Tensor | None = None,
        clean_video_normalized: bool = False,
    ) -> WorldActionOutput:
        parsed_mode = InteractionMode.parse(mode)
        condition = self.condition(
            mode=parsed_mode,
            observation=observation,
            prompts=prompts,
            event_memory=event_memory,
            visual_memory=visual_memory,
            training=True,
            planning_supervision=planning_supervision,
            enforce_configured_mode=False,
            recent_observation_latents=clean_video_latents,
        )
        output = self.wam.training_forward(
            clean_video=clean_video,
            clean_action=clean_action,
            noisy_video=noisy_video,
            noisy_action=noisy_action,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            state=observation.proprioception,
            embodiment_id=observation.embodiment_id,
            cross_attention_tokens=condition.cross_attention_tokens,
            negative_cross_attention_tokens=condition.negative_cross_attention_tokens,
            persistent_prefill=condition.visual_memory.persistent_prompt_latents,
            recent_visual_prefill=condition.visual_memory.recent_observation_latents,
            clean_video_latents=clean_video_latents,
            clean_video_normalized=clean_video_normalized,
        )
        if not isinstance(output, WorldActionOutput):
            raise TypeError(
                f"WAM must return WorldActionOutput, got {type(output).__name__}"
            )
        semantic_condition = (
            condition if parsed_mode is InteractionMode.AUTO else None
        )
        auxiliary = {
            "semantic_prediction": (
                semantic_condition.semantic_prediction
                if semantic_condition is not None
                else None
            ),
            "semantic_target": (
                semantic_condition.semantic_target
                if semantic_condition is not None
                else None
            ),
            "semantic_mask": (
                semantic_condition.semantic_mask
                if semantic_condition is not None
                else None
            ),
            "planning_logits": condition.planning_logits,
            "planning_labels": condition.planning_labels,
        }
        for name, value in auxiliary.items():
            if value is not None:
                if name in output.metrics:
                    raise ValueError(f"WAM output already defines reserved metric {name!r}")
                output.metrics[name] = value
        return self._attach_memory(output, condition)

    @torch.no_grad()
    def sample(
        self,
        *,
        mode: InteractionMode | str,
        observation: ObservationBatch,
        prompts: PromptBatch,
        generator: torch.Generator,
        event_memory: EventMemoryState | None = None,
        visual_memory: VisualMemoryState | None = None,
    ) -> WorldActionOutput:
        parsed_mode = InteractionMode.parse(mode)
        starts_new_window = getattr(self.wam, "starts_new_window", None)
        if callable(starts_new_window) and starts_new_window(
            visual_memory,
            observation_num_frames=observation.images.shape[1],
        ):
            if observation.images.shape[1] == 1:
                event_memory = None
            else:
                event_memory = self.condition_router.promote_pending_event_memory(
                    parsed_mode,
                    event_memory,
                )
        condition = self.condition(
            mode=parsed_mode,
            observation=observation,
            prompts=prompts,
            event_memory=event_memory,
            visual_memory=visual_memory,
            training=False,
        )
        observation_latents = condition.visual_memory.recent_observation_latents
        if observation_latents is None:
            raise RuntimeError("VisualPrefillManager did not produce recent observation latents")
        reference_frame, reference_frame_normalized = self._sample_reference_frame(
            observation
        )
        output = self.wam.sample(
            reference_frame=reference_frame,
            reference_frame_normalized=reference_frame_normalized,
            chunk_latents=observation_latents,
            observation_num_frames=observation.images.shape[1],
            prompt_signature=prompts.condition_signature(parsed_mode),
            state=observation.proprioception,
            embodiment_id=observation.embodiment_id,
            cross_attention_tokens=condition.cross_attention_tokens,
            negative_cross_attention_tokens=condition.negative_cross_attention_tokens,
            visual_memory=condition.visual_memory,
            generator=generator,
        )
        return self._attach_memory(output, condition)

    def _sample_reference_frame(self, observation: ObservationBatch) -> tuple[Tensor, bool]:
        prepare_diffusion_video = getattr(
            self.visual_memory.codec,
            "prepare_diffusion_video",
            None,
        )
        if (
            observation.images.ndim == 6
            and observation.images.shape[2] == 3
            and callable(prepare_diffusion_video)
        ):
            reference_video = prepare_diffusion_video(observation.images)
            if reference_video.ndim != 5 or reference_video.shape[2] not in (1, 3, 4):
                raise ValueError(
                    "prepared diffusion reference must have shape [B,T,C,H,W]"
                )
            return reference_video[:, :1], True
        return observation.head_view, False

    @staticmethod
    def _attach_memory(
        output: WorldActionOutput,
        condition: Conditioning,
    ) -> WorldActionOutput:
        if not isinstance(output, WorldActionOutput):
            raise TypeError(f"WAM must return WorldActionOutput, got {type(output).__name__}")
        if output.next_memory is None:
            output.next_memory = condition.event_memory
        if output.next_visual_memory is None:
            output.next_visual_memory = condition.visual_memory
        return output
