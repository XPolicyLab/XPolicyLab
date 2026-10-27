# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical80 gripper-weighted MSE with an unchanged valid-dimension denominator."""

import torch


def weighted_action_per_sample(
    element_loss: torch.Tensor,
    dims: torch.Tensor,
    steps: torch.Tensor,
    gripper_multiplier: float,
) -> torch.Tensor:
    """Normalize over14 valid action dimensions/time steps; apply gripper weights to the numerator."""
    assert element_loss.ndim == 3 and element_loss.shape[-1] == 80
    assert dims.shape == (element_loss.shape[0], 80) and torch.all(dims.sum(1) == 14)
    assert steps.shape == element_loss.shape[:2]
    assert torch.all(dims[:, [16, 45]] == 1) and gripper_multiplier in (1.0, 2.0)
    weights = torch.ones(80, device=element_loss.device, dtype=element_loss.dtype)
    weights[[16, 45]] = gripper_multiplier
    per_step = (element_loss * dims[:, None] * weights).sum(2) / dims.sum(1).clamp_min(1)[:, None]
    return (per_step * steps).sum(1) / steps.sum(1).clamp_min(1)
