from __future__ import annotations

import math

import torch
from torch import Tensor

from lerobot.configs.types import RTCAttentionSchedule

from .configuration_rtc import RTCConfig


class RTCProcessor:
    """Runtime guidance processor for flow-matching chunk generation."""

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int | None,
        time: float | Tensor,
        original_denoise_step_partial,
        execution_horizon: int | None = None,
    ) -> Tensor:
        tau = 1 - time

        if prev_chunk_left_over is None or inference_delay is None:
            return original_denoise_step_partial(x_t)

        x_t = x_t.clone().detach()
        squeezed = False

        if x_t.ndim < 3:
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        execution_horizon = min(execution_horizon, prev_chunk_left_over.shape[1])

        batch_size, action_chunk_size, action_dim = x_t.shape
        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim, device=x_t.device, dtype=x_t.dtype)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over.to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
            prev_chunk_left_over = padded
        else:
            prev_chunk_left_over = prev_chunk_left_over.to(device=x_t.device, dtype=x_t.dtype)

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(device=x_t.device, dtype=x_t.dtype)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            x_t.requires_grad_(True)
            v_t = original_denoise_step_partial(x_t)
            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            correction = torch.autograd.grad(x1_t, x_t, err.detach(), retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight, device=x_t.device, dtype=x_t.dtype)
        tau_tensor = torch.as_tensor(tau, device=x_t.device, dtype=x_t.dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / squared_one_minus_tau
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction
        if squeezed:
            result = result.squeeze(0)
        return result

    def get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
            return weights

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
            return weights

        lin_weights = self._linweights(start, end, total)
        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)

        weights = self._add_trailing_zeros(lin_weights, total, end)
        weights = self._add_leading_ones(weights, start, total)
        return weights

    def _linweights(self, start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start
        if end <= start or linspace_steps <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zeros_len)])

    def _add_leading_ones(self, weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return torch.cat([torch.ones(ones_len), weights])
