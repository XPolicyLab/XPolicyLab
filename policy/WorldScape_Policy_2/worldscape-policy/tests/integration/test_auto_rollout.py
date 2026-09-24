from __future__ import annotations

import torch

from worldscape_policy.types import InteractionMode


def test_auto_rollout_uses_real_policy_composition(composed_rollout) -> None:
    result, executor = composed_rollout(InteractionMode.AUTO)

    result.raise_for_error()
    assert result.record.status == "completed"
    assert result.record.completed_steps == 2
    assert len(executor.actions) == 2
    torch.testing.assert_close(executor.actions[0], torch.full((1, 2, 3), 2.0))
