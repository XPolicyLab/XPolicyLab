from __future__ import annotations

from typing import Literal

import torch


VisualInputRange = Literal["zero_one", "minus_one_one", "uint8"]


def normalize_visual(
    value: torch.Tensor,
    *,
    input_range: VisualInputRange,
) -> torch.Tensor:
    """Convert an explicitly declared public visual range to ``[-1, 1]``."""

    if input_range == "uint8":
        if value.dtype != torch.uint8:
            raise ValueError("visual_input_range='uint8' requires torch.uint8 input")
        return value.float().div(255.0).mul(2).sub(1)

    result = value.float()
    if result.numel() == 0:
        return result
    minimum = result.detach().amin()
    maximum = result.detach().amax()
    if input_range == "zero_one":
        if minimum < 0 or maximum > 1:
            raise ValueError("zero_one visual input must stay within [0, 1]")
        return result.mul(2).sub(1)
    if input_range == "minus_one_one":
        if minimum < -1 or maximum > 1:
            raise ValueError("minus_one_one visual input must stay within [-1, 1]")
        return result
    raise ValueError(f"Unknown visual input range: {input_range!r}")
