"""Geometry-conditioned residual refiner.

This module implements the paper's residual refinement equations while keeping
unspecified internals small and configurable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from georefiner.config import GeoRefinerConfig
from georefiner.modules.common import FeedForward, sanitize_padding_mask


@dataclass(slots=True)
class GeometryConditionedResidualRefinerOutput:
    """Refined action outputs and debug tensors from the residual refiner."""

    refined_action: Tensor
    gate: Tensor
    residual: Tensor
    applied_correction: Tensor
    fused_action_tokens: Tensor
    geometry_context: Tensor


class RefinerBlock(nn.Module):
    """Pre-norm block for geometry-conditioned action-token fusion."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = FeedForward(hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trajectory_tokens: Tensor,
        geometry_context: Tensor,
        action_padding_mask: Tensor | None = None,
        context_padding_mask: Tensor | None = None,
    ) -> Tensor:
        action_padding_mask = sanitize_padding_mask(action_padding_mask)
        context_padding_mask = sanitize_padding_mask(context_padding_mask)

        normalized_tokens = self.self_norm(trajectory_tokens)
        self_out, _ = self.self_attn(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            key_padding_mask=action_padding_mask,
            need_weights=False,
        )
        tokens = trajectory_tokens + self.dropout(self_out)

        normalized_context = self.context_norm(geometry_context)
        cross_out, _ = self.cross_attn(
            self.cross_query_norm(tokens),
            normalized_context,
            normalized_context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(cross_out)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class TemporalResidualBlock(nn.Module):
    """Simple temporal Conv1d residual block.

    This Conv1d temporal residual block is an implementation assumption because
    the paper does not specify its exact internal structure.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer; got {kernel_size}."
            )
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must have shape [B, T, D]; got {tuple(inputs.shape)}.")
        normalized = self.norm(inputs)
        conv_input = normalized.transpose(1, 2)
        conv_output = self.conv(conv_input).transpose(1, 2)
        return inputs + conv_output


class GateHead(nn.Module):
    """Token-wise refinement gate head."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        head_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = head_hidden_dim or hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, action_dim),
        )

    def forward(self, fused_action_tokens: Tensor) -> Tensor:
        gate = torch.sigmoid(self.mlp(self.norm(fused_action_tokens)))
        return gate.clamp(0.0, 1.0)


class ResidualHead(nn.Module):
    """Bounded action residual head."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        residual_scale: float | Sequence[float],
        head_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = head_hidden_dim or hidden_dim
        self.action_dim = action_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, action_dim),
        )
        scale = self._make_scale_tensor(residual_scale, action_dim)
        self.register_buffer("residual_scale", scale, persistent=True)

    def forward(self, temporal_tokens: Tensor) -> Tensor:
        raw_residual = torch.tanh(self.mlp(self.norm(temporal_tokens)))
        return raw_residual * self.residual_scale.to(
            device=raw_residual.device,
            dtype=raw_residual.dtype,
        )

    @staticmethod
    def _make_scale_tensor(
        residual_scale: float | Sequence[float],
        action_dim: int,
    ) -> Tensor:
        if isinstance(residual_scale, (int, float)):
            if residual_scale <= 0:
                raise ValueError(
                    f"residual_scale must be positive; got {residual_scale}."
                )
            return torch.tensor(float(residual_scale), dtype=torch.float32).view(1, 1, 1)

        if len(residual_scale) != action_dim:
            raise ValueError(
                "residual_scale sequence length must match action_dim; "
                f"got {len(residual_scale)} and action_dim={action_dim}."
            )
        scale = torch.tensor(list(residual_scale), dtype=torch.float32)
        if torch.any(scale <= 0):
            raise ValueError(f"residual_scale values must be positive; got {residual_scale}.")
        return scale.view(1, 1, action_dim)


class GeometryConditionedResidualRefiner(nn.Module):
    """Predict gated bounded residuals conditioned on geometry context."""

    def __init__(self, config: GeoRefinerConfig) -> None:
        super().__init__()
        self.config = config
        self.refiner_blocks = nn.ModuleList(
            [
                RefinerBlock(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.refiner_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.temporal_residual_block = TemporalResidualBlock(
            hidden_dim=config.hidden_dim,
            kernel_size=config.temporal_residual_kernel_size,
            dropout=config.dropout,
        )
        self.gate_head = GateHead(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            head_hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.residual_head = ResidualHead(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            residual_scale=config.residual_scale,
            head_hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        canonical_nominal_action: Tensor,
        trajectory_tokens: Tensor,
        geometry_tokens: Tensor,
        region_tokens: Tensor,
        action_padding_mask: Tensor | None = None,
        geometry_padding_mask: Tensor | None = None,
        region_padding_mask: Tensor | None = None,
    ) -> GeometryConditionedResidualRefinerOutput:
        self._validate_action(canonical_nominal_action)
        self._validate_tokens("trajectory_tokens", trajectory_tokens)
        self._validate_tokens("geometry_tokens", geometry_tokens)
        self._validate_tokens("region_tokens", region_tokens)
        if trajectory_tokens.shape[:2] != canonical_nominal_action.shape[:2]:
            raise ValueError(
                "trajectory_tokens must share [B, T] with canonical_nominal_action; "
                f"got {tuple(trajectory_tokens.shape)} and "
                f"{tuple(canonical_nominal_action.shape)}."
            )
        if geometry_tokens.shape[0] != trajectory_tokens.shape[0]:
            raise ValueError("geometry_tokens batch size must match trajectory_tokens.")
        if region_tokens.shape[0] != trajectory_tokens.shape[0]:
            raise ValueError("region_tokens batch size must match trajectory_tokens.")

        geometry_context = torch.cat([geometry_tokens, region_tokens], dim=1)
        context_padding_mask = self._concat_context_masks(
            geometry_tokens,
            region_tokens,
            geometry_padding_mask,
            region_padding_mask,
        )
        self._validate_mask("action_padding_mask", action_padding_mask, trajectory_tokens)

        fused_action_tokens = trajectory_tokens
        for block in self.refiner_blocks:
            fused_action_tokens = block(
                fused_action_tokens,
                geometry_context,
                action_padding_mask=action_padding_mask,
                context_padding_mask=context_padding_mask,
            )
        fused_action_tokens = self.output_norm(fused_action_tokens)
        gate = self.gate_head(fused_action_tokens)
        temporal_tokens = self.temporal_residual_block(fused_action_tokens)
        residual = self.residual_head(temporal_tokens)
        applied_correction = gate * residual
        refined_action = canonical_nominal_action + applied_correction
        return GeometryConditionedResidualRefinerOutput(
            refined_action=refined_action,
            gate=gate,
            residual=residual,
            applied_correction=applied_correction,
            fused_action_tokens=fused_action_tokens,
            geometry_context=geometry_context,
        )

    def _concat_context_masks(
        self,
        geometry_tokens: Tensor,
        region_tokens: Tensor,
        geometry_padding_mask: Tensor | None,
        region_padding_mask: Tensor | None,
    ) -> Tensor | None:
        if geometry_padding_mask is None and region_padding_mask is None:
            return None

        batch_size = geometry_tokens.shape[0]
        device = geometry_tokens.device
        if geometry_padding_mask is None:
            geometry_padding_mask = torch.zeros(
                batch_size, geometry_tokens.shape[1], dtype=torch.bool, device=device
            )
        if region_padding_mask is None:
            region_padding_mask = torch.zeros(
                batch_size, region_tokens.shape[1], dtype=torch.bool, device=device
            )
        self._validate_mask("geometry_padding_mask", geometry_padding_mask, geometry_tokens)
        self._validate_mask("region_padding_mask", region_padding_mask, region_tokens)
        return sanitize_padding_mask(
            torch.cat(
                [
                    geometry_padding_mask.to(device=device, dtype=torch.bool),
                    region_padding_mask.to(device=device, dtype=torch.bool),
                ],
                dim=1,
            )
        )

    def _validate_action(self, action: Tensor) -> None:
        if action.ndim != 3:
            raise ValueError(
                "canonical_nominal_action must have shape [B, T, action_dim]; "
                f"got {tuple(action.shape)}."
            )
        if action.shape[-1] != self.config.action_dim:
            raise ValueError(
                "canonical_nominal_action last dimension must match action_dim="
                f"{self.config.action_dim}; got {tuple(action.shape)}."
            )

    def _validate_tokens(self, name: str, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B, N, D]; got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"{name} last dimension must match hidden_dim={self.config.hidden_dim}; "
                f"got {tuple(tokens.shape)}."
            )

    @staticmethod
    def _validate_mask(name: str, mask: Tensor | None, tokens: Tensor) -> None:
        if mask is None:
            return
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"{name} must have shape [B, N] matching tokens; "
                f"got mask {tuple(mask.shape)} and tokens {tuple(tokens.shape)}."
            )
