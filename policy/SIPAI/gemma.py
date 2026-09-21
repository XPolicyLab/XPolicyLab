"""Inference layers; parameters are populated by strict checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

_MASKED_LOGIT = -2.3819763e38


@dataclass(frozen=True)
class GemmaConfig:
    """Geometry for one Gemma transformer stream."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    adaptive_norm: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.width,
                self.depth,
                self.mlp_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            <= 0
        ):
            raise ValueError("Gemma dimensions must be positive.")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")


@dataclass(frozen=True)
class LayerKVCache:
    """RoPE keys and values: [batch, tokens, kv_heads, head_dim]."""

    key: torch.Tensor
    value: torch.Tensor


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm using the ``1 + scale`` parameterization."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if condition is not None:
            raise ValueError("GemmaRMSNorm does not accept a condition.")
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        variance = torch.mean(hidden_float**2, dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + 1e-6)
        normalized = normalized * (1.0 + self.scale.float())
        return normalized.to(input_dtype), None


class AdaptiveGemmaRMSNorm(nn.Module):
    """Gemma adaptive RMSNorm with scale, shift, and residual-gate modulation."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.ada_modulation = nn.Linear(dim, dim * 3, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition is None:
            raise ValueError("AdaptiveGemmaRMSNorm requires a condition.")
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        variance = torch.mean(hidden_float**2, dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + 1e-6)

        modulation = self.ada_modulation(condition.to(self.ada_modulation.weight.dtype))
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        if hidden_states.ndim == 3:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
            gate = gate.unsqueeze(-2)
        normalized = normalized * (1.0 + scale.float()) + shift.float()
        return normalized.to(input_dtype), gate.to(input_dtype)


class GemmaGatedMLP(nn.Module):
    """Bias-free GELU-gated Gemma feed-forward layer."""

    def __init__(
        self,
        width: int,
        mlp_dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.w_gating = nn.Parameter(torch.empty(2, width, mlp_dim, dtype=dtype, device=device))
        self.w_linear = nn.Parameter(torch.empty(mlp_dim, width, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        gate = torch.matmul(hidden_states, self.w_gating[0].to(input_dtype))
        value = torch.matmul(hidden_states, self.w_gating[1].to(input_dtype))
        activated = F.gelu(gate, approximate="tanh") * value
        return torch.matmul(activated, self.w_linear.to(input_dtype))


def apply_split_half_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    max_wavelength: float = 10_000.0,
) -> torch.Tensor:
    """Apply Gemma's split-half rotary position embedding to queries or keys."""

    head_dim = tensor.shape[-1]
    frequency_exponents = (2.0 / head_dim) * torch.arange(
        head_dim // 2,
        dtype=torch.float32,
        device=tensor.device,
    )
    timescale = max_wavelength**frequency_exponents
    radians = positions[..., None].float() / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    first_half, second_half = torch.chunk(tensor, 2, dim=-1)
    rotated = torch.cat(
        [first_half * cos - second_half * sin, second_half * cos + first_half * sin],
        dim=-1,
    )
    return rotated.to(tensor.dtype)


class GemmaGQAttention(nn.Module):
    """Single-stream grouped-query attention with optional prefix K/V reuse.

    Projection layers intentionally remain one-element ``ModuleList`` objects.
    Besides keeping the forward API single-stream, this preserves legacy keys
    such as ``attn.q_proj.0.weight`` when the module is used by an outer model.
    """

    def __init__(
        self,
        config: GemmaConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        projection_width = config.num_heads * config.head_dim
        kv_width = config.num_kv_heads * config.head_dim
        self.q_proj = nn.ModuleList(
            [
                nn.Linear(
                    config.width,
                    projection_width,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
            ]
        )
        self.k_proj = nn.ModuleList([nn.Linear(config.width, kv_width, bias=False, dtype=dtype, device=device)])
        self.v_proj = nn.ModuleList([nn.Linear(config.width, kv_width, bias=False, dtype=dtype, device=device)])
        self.o_proj = nn.ModuleList(
            [
                nn.Linear(
                    projection_width,
                    config.width,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        config = self.config
        batch_size, token_count, _ = hidden_states.shape
        input_dtype = hidden_states.dtype

        query = self.q_proj[0](hidden_states).reshape(
            batch_size,
            token_count,
            config.num_heads,
            config.head_dim,
        )
        key = self.k_proj[0](hidden_states).reshape(
            batch_size,
            token_count,
            config.num_kv_heads,
            config.head_dim,
        )
        value = self.v_proj[0](hidden_states).reshape(
            batch_size,
            token_count,
            config.num_kv_heads,
            config.head_dim,
        )

        query = apply_split_half_rope(query, positions)
        key = apply_split_half_rope(key, positions)
        if kv_cache is not None:
            key = torch.cat([kv_cache.key, key], dim=1)
            value = torch.cat([kv_cache.value, value], dim=1)
        new_cache = LayerKVCache(key=key, value=value)

        query = query * (config.head_dim**-0.5)
        query_groups = config.num_heads // config.num_kv_heads
        query = query.reshape(
            batch_size,
            token_count,
            config.num_kv_heads,
            query_groups,
            config.head_dim,
        )
        logits = torch.einsum("BTKGH,BSKH->BKGTS", query.float(), key.float())
        broadcast_mask = attention_mask[:, None, None, :, :].expand_as(logits)
        masked_logits = torch.where(
            broadcast_mask,
            logits,
            torch.tensor(_MASKED_LOGIT, dtype=logits.dtype, device=logits.device),
        )
        probabilities = F.softmax(masked_logits, dim=-1).to(input_dtype)
        encoded = torch.einsum("BKGTS,BSKH->BTKGH", probabilities, value.to(input_dtype))
        encoded = encoded.reshape(batch_size, token_count, config.num_heads * config.head_dim)
        return self.o_proj[0](encoded), new_cache


class GemmaDecoderLayer(nn.Module):
    """One single-stream Gemma decoder layer."""

    def __init__(
        self,
        config: GemmaConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.attn = GemmaGQAttention(config, dtype=dtype, device=device)
        norm_cls = AdaptiveGemmaRMSNorm if config.adaptive_norm else GemmaRMSNorm
        # The one-element containers preserve the existing single-stream
        # checkpoint paths: pre_attention_norms.0, pre_ffw_norms.0, mlps.0.
        self.pre_attention_norms = nn.ModuleList([norm_cls(config.width, dtype=dtype, device=device)])
        self.pre_ffw_norms = nn.ModuleList([norm_cls(config.width, dtype=dtype, device=device)])
        self.mlps = nn.ModuleList([GemmaGatedMLP(config.width, config.mlp_dim, dtype=dtype, device=device)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: LayerKVCache | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        normalized, gate = self.pre_attention_norms[0](hidden_states, condition)
        branch, new_cache = self.attn(normalized, positions, attention_mask, kv_cache)
        hidden_states = _gated_residual(hidden_states, branch, gate)

        normalized, gate = self.pre_ffw_norms[0](hidden_states, condition)
        branch = self.mlps[0](normalized)
        return _gated_residual(hidden_states, branch, gate), new_cache


def _gated_residual(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    if gate is None:
        return residual + branch
    return residual + branch * gate
