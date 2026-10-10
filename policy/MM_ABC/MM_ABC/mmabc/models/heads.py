"""Masked action regression and future-feature alignment losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    action: torch.Tensor
    future: torch.Tensor | None
    per_head: dict[str, torch.Tensor]
    per_head_count: dict[str, torch.Tensor]


def masked_action_loss(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    head_slices: dict[str, tuple[int, int]],
    *,
    weight: torch.Tensor | None = None,
    head_active: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Args:
        predictions: head -> (B, T, width) clean-sample predictions.
        target: (B, T, 80) model-space targets.
        mask: (B, T, 80) 1.0 where supervised.
        head_slices: head -> (lo, hi) canonical slice.
        weight: (B,) per-sample loss weight, e.g. the flow velocity weight.
        head_active: head -> (B,) 1.0 where the head has any label."""
    B = target.shape[0]
    device = target.device
    per_head: dict[str, torch.Tensor] = {}
    per_head_sum: dict[str, torch.Tensor] = {}
    per_head_count: dict[str, torch.Tensor] = {}

    numerator = torch.zeros(B, device=device, dtype=torch.float32)
    denominator = torch.zeros(B, device=device, dtype=torch.float32)

    for name, (lo, hi) in head_slices.items():
        pred = predictions[name].float()
        tgt = target[..., lo:hi].float()
        m = mask[..., lo:hi].float()

        # Mask operands before arithmetic: masking an inf-valued loss still yields NaN gradients.
        keep = m > 0
        zero = torch.zeros((), device=device, dtype=pred.dtype)
        err = (torch.where(keep, pred, zero) - torch.where(keep, tgt, zero)) ** 2
        count = m.sum(dim=(1, 2))
        # Per-sample mean over supervised entries only.
        sample_loss = err.sum(dim=(1, 2)) / count.clamp(min=1.0)

        active = (count > 0).float()
        if head_active is not None and name in head_active:
            active = active * (head_active[name] > 0).float()
        sample_loss = torch.where(active > 0, sample_loss, torch.zeros_like(sample_loss))

        numerator = numerator + sample_loss
        denominator = denominator + active

        per_head_sum[name] = sample_loss.sum()
        per_head_count[name] = active.sum()
        per_head[name] = per_head_sum[name] / per_head_count[name].clamp(min=1.0)

    # Average across active heads per sample, then across the batch.
    sample_total = numerator / denominator.clamp(min=1.0)
    if weight is not None:
        sample_total = sample_total * weight.float()
    has_any = (denominator > 0).float()
    # Same reasoning as above: a sample with no active head must be dropped, not
    # multiplied by zero, or its unconstrained value survives as a NaN.
    loss = torch.where(
        has_any > 0, sample_total, torch.zeros_like(sample_total)
    ).sum() / has_any.sum().clamp(min=1.0)
    return loss, per_head, per_head_count


def future_alignment_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """One minus cosine similarity, averaged over tokens."""
    pred = prediction.float()
    tgt = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)

    keep = torch.isfinite(target.float()).all(dim=-1) & (tgt.norm(dim=-1) > 1e-6)
    if valid is not None:
        v = valid.float()
        while v.dim() < keep.dim():
            v = v.unsqueeze(-1)
        keep = keep & (v.expand_as(keep) > 0)
    weight = keep.float()

    eps = 1e-12
    pred_n = pred * pred.pow(2).sum(dim=-1, keepdim=True).add(eps).rsqrt()
    tgt_n = tgt * tgt.pow(2).sum(dim=-1, keepdim=True).add(eps).rsqrt()
    sim = (pred_n * tgt_n).sum(dim=-1)  # (B, N)

    return (1.0 - sim).mul(weight).sum() / weight.sum().clamp(min=1.0)
