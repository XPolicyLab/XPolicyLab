from __future__ import annotations

import pytest
import torch

from worldscape_policy.checkpoint.adapter_rows import (
    export_single_adapter_row,
    select_pretrained_adapter_row,
)


def _state(categories: int) -> dict[str, torch.Tensor]:
    return {
        "wam.core.action_encoder.W1.W": torch.arange(
            categories * 6, dtype=torch.float32
        ).reshape(categories, 2, 3),
        "wam.core.action_encoder.W1.b": torch.arange(
            categories * 3, dtype=torch.float32
        ).reshape(categories, 3),
        "wam.core.state_encoder.layer1.W": torch.arange(
            categories * 6, dtype=torch.float32
        ).reshape(categories, 2, 3),
        "wam.core.action_decoder.layer2.b": torch.arange(
            categories * 2, dtype=torch.float32
        ).reshape(categories, 2),
        "wam.core.blocks.0.weight": torch.ones(3, 3),
    }


@pytest.mark.parametrize("source_row", [2, 4])
def test_selects_one_pretrained_row_and_preserves_other_tensors(source_row: int):
    source = _state(8)
    target = _state(1)

    selected = select_pretrained_adapter_row(
        source,
        target,
        source_row=source_row,
    )

    for key, value in source.items():
        if "encoder." in key or "decoder." in key:
            assert selected[key].shape[0] == 1
            torch.testing.assert_close(selected[key][0], value[source_row])
        else:
            assert selected[key] is value


def test_exact_single_row_state_is_not_reinterpreted_as_source_index():
    source = _state(1)
    selected = select_pretrained_adapter_row(source, _state(1), source_row=4)
    assert all(selected[key] is value for key, value in source.items())


def test_export_single_adapter_row_slices_without_target_model():
    source = _state(8)

    selected = export_single_adapter_row(source, source_row=2)

    for key, value in source.items():
        if "encoder." in key or "decoder." in key:
            assert selected[key].shape[0] == 1
            torch.testing.assert_close(selected[key][0], value[2])
        else:
            assert selected[key] is value


def test_export_single_adapter_row_keeps_existing_single_row_checkpoint():
    source = _state(1)

    selected = export_single_adapter_row(source, source_row=2)

    assert all(selected[key] is value for key, value in source.items())


def test_rejects_out_of_range_source_row():
    with pytest.raises(ValueError, match="outside"):
        select_pretrained_adapter_row(_state(8), _state(1), source_row=8)


def test_rejects_adapter_trailing_shape_mismatch():
    source = _state(8)
    target = _state(1)
    target["wam.core.action_encoder.W1.W"] = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="trailing shape mismatch"):
        select_pretrained_adapter_row(source, target, source_row=2)
