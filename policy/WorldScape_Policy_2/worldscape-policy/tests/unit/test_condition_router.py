from __future__ import annotations

import torch
from torch import nn

from worldscape_policy.conditioning import (
    ConditionRouter,
    InteractiveConditioner,
    LegacySemanticTargetBuilder,
    SemanticTargetBuilder,
    build_semantic_target,
)
from worldscape_policy.types import Conditioning, ObservationBatch, PromptBatch


class _Conditioner(nn.Module):
    def __init__(self, value: float, *, keep_memory: bool) -> None:
        super().__init__()
        self.value = value
        self.keep_memory = keep_memory

    def forward(self, *, observation, prompts, event_memory, training):
        del prompts, training
        tokens = torch.full((observation.images.shape[0], 1, 2), self.value)
        return Conditioning(
            cross_attention_tokens=tokens,
            event_memory=event_memory if self.keep_memory else None,
        )


class _TextEncoder(nn.Module):
    def encode_text(self, instructions):
        return torch.ones(len(instructions), 1, 2)


def _observation() -> ObservationBatch:
    return ObservationBatch(
        images=torch.ones(1, 1, 1, 3, 2, 2),
        head_view=torch.ones(1, 1, 3, 2, 2),
        proprioception=torch.zeros(1, 1, 2),
        embodiment_id=torch.zeros(1, dtype=torch.long),
    )


def test_condition_router_keeps_auto_and_interactive_paths_exclusive() -> None:
    router = ConditionRouter(
        auto_conditioner=_Conditioner(1.0, keep_memory=True),
        interactive_conditioner=_Conditioner(2.0, keep_memory=False),
    )
    auto = router(
        "auto",
        _observation(),
        PromptBatch(vlm_planning_text=["plan"]),
        training=False,
    )
    interactive = router(
        "interactive",
        _observation(),
        PromptBatch(language_instruction=["move"]),
        training=False,
    )
    assert torch.all(auto.cross_attention_tokens == 1.0)
    assert torch.all(interactive.cross_attention_tokens == 2.0)


def test_semantic_target_builder_preserves_training_only_fallback() -> None:
    builder = LegacySemanticTargetBuilder()
    assert isinstance(builder, SemanticTargetBuilder)
    explicit = torch.tensor([[1.0]])
    fallback = torch.tensor([[2.0]])
    assert build_semantic_target(
        training=True,
        explicit_target=explicit,
        fallback_target=fallback,
    ) is explicit
    assert builder(
        training=True,
        explicit_target=None,
        fallback_target=fallback,
    ) is fallback
    assert builder(
        training=False,
        explicit_target=explicit,
        fallback_target=fallback,
    ) is None


def test_interactive_uses_shared_target_policy_without_creating_target() -> None:
    conditioner = InteractiveConditioner(t5=_TextEncoder())
    result = conditioner(
        observation=_observation(),
        prompts=PromptBatch(language_instruction=["move"]),
        event_memory=None,
        training=True,
    )
    assert result.semantic_target is None
