from types import SimpleNamespace

import pytest
import torch

from worldscape_policy.wam.wan22.model import CausalWanModel


def _resolve(max_num_embodiments: int, ids: torch.Tensor | None, batch_size: int):
    model = SimpleNamespace(max_num_embodiments=max_num_embodiments)
    return CausalWanModel._resolve_embodiment_ids(
        model,
        ids,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )


def test_pretrain_multi_adapter_uses_batch_category_ids():
    ids = torch.tensor([2, 4])
    resolved = _resolve(8, ids, 2)
    torch.testing.assert_close(resolved, ids)


def test_posttrain_single_adapter_always_uses_local_zero():
    resolved = _resolve(1, torch.tensor([4, 2]), 2)
    torch.testing.assert_close(resolved, torch.zeros(2, dtype=torch.long))


def test_multi_adapter_rejects_missing_or_out_of_range_ids():
    with pytest.raises(ValueError, match="requires embodiment_id"):
        _resolve(8, None, 1)
    with pytest.raises(ValueError, match="outside"):
        _resolve(8, torch.tensor([8]), 1)
