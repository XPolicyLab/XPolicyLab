from __future__ import annotations

from torch import Tensor, nn

from worldscape_policy.conditioning.semantic_forcing import (
    LEGACY_SEMANTIC_TARGET_BUILDER,
    SemanticTargetBuilder,
)
from worldscape_policy.types import (
    Conditioning,
    EventMemoryState,
    ObservationBatch,
    PromptBatch,
)


class InteractiveConditioner(nn.Module):
    """Encode a user instruction directly, without VLM event memory."""

    def __init__(
        self,
        *,
        t5: nn.Module,
        projector: nn.Module | None = None,
        output_norm: nn.Module | None = None,
        semantic_target_builder: SemanticTargetBuilder = (
            LEGACY_SEMANTIC_TARGET_BUILDER
        ),
    ) -> None:
        super().__init__()
        if not hasattr(t5, "encode_text"):
            raise TypeError("t5 must implement encode_text(instructions)")
        self.t5 = t5
        self.projector = projector or nn.Identity()
        self.output_norm = output_norm or nn.Identity()
        self.semantic_target_builder = semantic_target_builder

    def forward(
        self,
        *,
        observation: ObservationBatch,
        prompts: PromptBatch,
        event_memory: EventMemoryState | None,
        training: bool,
    ) -> Conditioning:
        del observation, event_memory
        instructions = prompts.language_instruction
        if instructions is None:
            raise ValueError("InteractiveConditioner requires language_instruction")
        tokens = self.t5.encode_text(instructions)
        if not isinstance(tokens, Tensor):
            raise TypeError(
                f"encode_text must return a Tensor, got {type(tokens).__name__}"
            )
        projected = self.output_norm(self.projector(tokens))
        negative = projected
        if prompts.negative_language_instruction is not None:
            negative_tokens = self.t5.encode_text(
                prompts.negative_language_instruction
            )
            negative = self.output_norm(self.projector(negative_tokens))
        return Conditioning(
            cross_attention_tokens=projected,
            negative_cross_attention_tokens=negative,
            event_memory=None,
            semantic_target=self.semantic_target_builder(
                training=training,
                explicit_target=None,
            ),
        )
