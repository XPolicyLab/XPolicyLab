from __future__ import annotations

from typing import Protocol

from torch import nn

from worldscape_policy.types import (
    Conditioning,
    EventMemoryState,
    InteractionMode,
    ObservationBatch,
    PromptBatch,
)


class Conditioner(Protocol):
    def __call__(
        self,
        *,
        observation: ObservationBatch,
        prompts: PromptBatch,
        event_memory: EventMemoryState | None,
        training: bool,
        planning_supervision: bool = False,
    ) -> Conditioning: ...


class ConditionRouter(nn.Module):
    """Select exactly one condition path before invoking the WAM."""

    def __init__(
        self,
        auto_conditioner: nn.Module,
        interactive_conditioner: nn.Module,
    ) -> None:
        super().__init__()
        self.auto = auto_conditioner
        self.interactive = interactive_conditioner

    def forward(
        self,
        mode: InteractionMode | str,
        observation: ObservationBatch,
        prompts: PromptBatch,
        event_memory: EventMemoryState | None = None,
        training: bool | None = None,
        planning_supervision: bool = False,
    ) -> Conditioning:
        mode = InteractionMode.parse(mode)
        observation.validate()
        prompts.validate(observation.images.shape[0])
        is_training = self.training if training is None else training

        if mode is InteractionMode.AUTO:
            if prompts.vlm_planning_text is None:
                raise ValueError("Auto mode requires prompts.vlm_planning_text")
            auto_kwargs = dict(
                observation=observation,
                prompts=prompts,
                event_memory=event_memory,
                training=is_training,
            )
            if planning_supervision:
                auto_kwargs["planning_supervision"] = True
            output = self.auto(**auto_kwargs)
        else:
            if prompts.language_instruction is None:
                raise ValueError(
                    "Interactive mode requires prompts.language_instruction"
                )
            output = self.interactive(
                observation=observation,
                prompts=prompts,
                event_memory=None,
                training=is_training,
            )
            if output.event_memory is not None:
                raise ValueError("Interactive conditioner must not create or update event memory")

        if not isinstance(output, Conditioning):
            raise TypeError(
                f"{mode.value} conditioner must return Conditioning, got {type(output).__name__}"
            )
        if output.cross_attention_tokens.shape[0] != observation.images.shape[0]:
            raise ValueError(
                "Condition batch size does not match observations: "
                f"{output.cross_attention_tokens.shape[0]} != {observation.images.shape[0]}"
            )
        if (
            output.negative_cross_attention_tokens is not None
            and output.negative_cross_attention_tokens.shape[0]
            != observation.images.shape[0]
        ):
            raise ValueError("Negative condition batch size does not match observations")
        if not is_training and output.semantic_target is not None:
            raise ValueError("semantic_target is training-only")
        return output

    def promote_pending_event_memory(
        self,
        mode: InteractionMode | str,
        event_memory: EventMemoryState | None,
    ) -> EventMemoryState | None:
        if InteractionMode.parse(mode) is InteractionMode.INTERACTIVE:
            return None
        promote = getattr(self.auto, "promote_pending", None)
        return promote(event_memory) if callable(promote) else event_memory
