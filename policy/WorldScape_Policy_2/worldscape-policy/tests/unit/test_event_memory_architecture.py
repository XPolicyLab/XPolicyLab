from __future__ import annotations

import torch

from worldscape_policy.memory.event import (
    EventMemoryFusion,
    EventMemoryManager,
    GlobalHistoryBuilder,
    HistoryCompressor,
    LocalActiveSelector,
)
from worldscape_policy.types import EventMemoryState


def test_composed_event_memory_is_seed_stable():
    torch.manual_seed(17)
    reference = EventMemoryFusion(
        context_dim=8,
        goal_slots=2,
        active_slots=3,
        done_slots=2,
        done_min_gap=1,
        perception_gist_tokens=2,
        residual_scale=0.2,
        dropout=0.0,
    )
    composed = EventMemoryFusion(
        context_dim=8,
        goal_slots=2,
        active_slots=3,
        done_slots=2,
        done_min_gap=1,
        perception_gist_tokens=2,
        residual_scale=0.2,
        dropout=0.0,
    )
    composed.load_state_dict(reference.state_dict(), strict=True)

    current = torch.randn(2, 4, 8)
    perception = torch.randn(2, 5, 6, 8)
    planning = torch.randn(2, 5, 2, 8)
    valid = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    task = torch.randn(2, 3, 8)

    expected, expected_details = reference(
        current,
        history_tokens=perception,
        history_planning_tokens=planning,
        history_mask=valid,
        task_embeddings=task,
    )
    actual, actual_details = composed(
        current,
        history_tokens=perception,
        history_planning_tokens=planning,
        history_mask=valid,
        task_embeddings=task,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_details.keys() == expected_details.keys()
    for name in actual_details:
        torch.testing.assert_close(
            actual_details[name],
            expected_details[name],
            rtol=0,
            atol=0,
        )
    assert composed.state_dict().keys() == reference.state_dict().keys()
    assert isinstance(composed.history_compressor, HistoryCompressor)
    assert isinstance(composed.global_history_builder, GlobalHistoryBuilder)
    assert isinstance(composed.local_active_selector, LocalActiveSelector)


def test_event_memory_manager_preserves_candidate_commit_state():
    manager = EventMemoryManager(
        max_history_steps=2,
        detach_inference_history=True,
    )
    committed = EventMemoryState(
        perception_tokens=torch.full((1, 1, 2, 3), 1.0),
        planning_tokens=torch.full((1, 1, 1, 3), 1.0),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    current_perception = torch.full((1, 2, 3), 2.0, requires_grad=True)
    current_planning = torch.full((1, 1, 3), 2.0, requires_grad=True)

    candidate = manager.stage(
        committed,
        current_perception,
        current_planning,
        training=False,
    )
    assert candidate.perception_tokens is committed.perception_tokens
    assert candidate.pending_perception_tokens.grad_fn is None
    assert candidate.pending_planning_tokens.grad_fn is None

    promoted = manager.promote_pending(candidate)
    assert promoted is not None
    torch.testing.assert_close(
        promoted.perception_tokens[:, 0],
        committed.perception_tokens[:, 0],
    )
    torch.testing.assert_close(
        promoted.perception_tokens[:, 1],
        current_perception.detach(),
    )
    assert promoted.pending_perception_tokens is None
    assert promoted.cached_cross_attention_tokens is None


def test_prompt_change_resets_event_candidate_before_staging():
    manager = EventMemoryManager(max_history_steps=2)
    previous = EventMemoryState(
        perception_tokens=torch.ones(1, 1, 2, 3),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        cached_cross_attention_tokens=torch.ones(1, 2, 3),
        prompt_signature=("old",),
    )

    reset = manager.begin(
        previous,
        prompt_signature=("new",),
        has_planning=False,
        training=False,
    )

    assert reset.perception_tokens is None
    assert reset.cached_cross_attention_tokens is None
    assert reset.prompt_signature is None
