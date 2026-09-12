from __future__ import annotations

import torch

from worldscape_policy.memory.event import (
    EventBoundarySelector,
    EventMemoryFusion,
    LocalActiveSelector,
    MemoryGate,
    MemoryRetriever,
)


def test_event_memory_exposes_explicit_legacy_exact_composition() -> None:
    torch.manual_seed(31)
    memory = EventMemoryFusion(
        context_dim=4,
        goal_slots=1,
        active_slots=2,
        done_slots=2,
        perception_gist_tokens=2,
        residual_scale=0.1,
        dropout=0.0,
    )
    expected_keys = tuple(memory.state_dict())
    current = torch.randn(2, 3, 4)
    history = torch.randn(2, 4, 3, 4)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    expected, expected_details = super(EventMemoryFusion, memory).forward(
        current,
        history_tokens=history,
        history_mask=mask,
    )
    actual, actual_details = memory(
        current,
        history_tokens=history,
        history_mask=mask,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_details.keys() == expected_details.keys()
    for name, expected_value in expected_details.items():
        torch.testing.assert_close(
            actual_details[name],
            expected_value,
            rtol=0,
            atol=0,
        )
    assert tuple(memory.state_dict()) == expected_keys
    assert isinstance(memory.event_boundary_selector, EventBoundarySelector)
    assert isinstance(memory.memory_retriever, MemoryRetriever)
    assert isinstance(memory.memory_gate, MemoryGate)
    assert isinstance(memory.local_active_selector, LocalActiveSelector)


def test_event_memory_empty_history_preserves_output_contract() -> None:
    memory = EventMemoryFusion(context_dim=4)
    current = torch.randn(1, 2, 4)
    output, details = memory(current, history_tokens=None)
    assert output is current
    assert details == {}
