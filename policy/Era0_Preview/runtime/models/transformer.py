# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .architecture_contract import (
    PREFIX_KV_ACTION_CONTEXT_MODE,
    PREFIX_KV_CACHE_TOPOLOGY,
    PREFIX_KV_CACHE_TOPOLOGY_ALIASES,
    PREFIX_KV_ROPE_PARTIAL_FACTOR,
    PREFIX_KV_ROPE_THETA,
    canonicalize_action_context_mode,
    is_prefix_kv_action_context_mode,
)

# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


class RMSNorm(nn.Module):
    """Qwen-style RMSNorm used by the G0.5 prefix-KV GQA attention path.

    Only instantiated when the checkpoint declares the prefix-KV expert, so
    legacy fused-MHA state dicts never gain or lose a key because of it.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        return (
            x
            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)

# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C] for Linear variant; caller is responsible for shapes.
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional fused SDPA fallback.

    If PyTorch provides `scaled_dot_product_attention`, it will be used
    (usually faster and more stable); otherwise we use a manual implementation.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        gated_output: bool = False,
        rope_partial_factor: float = 0.0,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.gated_output = bool(gated_output)
        self.fused_attn = _has_sdp_attention()
        if num_kv_heads is not None or head_dim is not None:
            self._init_gqa(
                dim,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                rope_partial_factor=rope_partial_factor,
                rope_theta=rope_theta,
            )
            return
        self.q_proj = self.k_proj = self.v_proj = None
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_gqa(
        self,
        dim: int,
        *,
        num_kv_heads: int | None,
        head_dim: int | None,
        qkv_bias: bool,
        qk_norm: bool,
        attn_drop: float,
        proj_drop: float,
        norm_layer: type[nn.Module],
        rope_partial_factor: float,
        rope_theta: float,
    ) -> None:
        """Build independent q/k/v projections with grouped K/V heads.

        Checkpoint layout (Qwen3.5-4B expert, hidden=1024, Hq=16, Dh=256):
        ``q_proj [2*Hq*Dh, dim]`` (output gate), ``k_proj/v_proj [Hkv*Dh, dim]``
        and ``proj [dim, Hq*Dh]``, all bias-free except ``proj``.
        """
        if num_kv_heads is None or head_dim is None:
            raise ValueError(
                "GQA attention requires both num_kv_heads and head_dim; "
                "omit both to keep fused MHA."
            )
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError(
                "num_heads must be a positive multiple of num_kv_heads, got "
                f"Hq={self.num_heads} Hkv={self.num_kv_heads}."
            )
        self.scale = self.head_dim ** -0.5
        inner_q = self.num_heads * self.head_dim
        inner_kv = self.num_kv_heads * self.head_dim
        self.qkv = None
        self.q_proj = nn.Linear(
            dim, inner_q * (2 if self.gated_output else 1), bias=qkv_bias
        )
        self.k_proj = nn.Linear(dim, inner_kv, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, inner_kv, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(inner_q, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        factor = float(rope_partial_factor)
        if factor < 0.0:
            raise ValueError(
                f"rope_partial_factor must be >= 0, got {rope_partial_factor}."
            )
        rotary_dim = int(self.head_dim * factor)
        if rotary_dim % 2:
            rotary_dim -= 1
        self.rotary_dim = max(0, rotary_dim)
        self.rope_theta = float(rope_theta)
        # ``inv_freq`` is fully derived from (rotary_dim, rope_theta) and is
        # never trained, so it must NOT be a buffer: ``from_pretrained`` builds
        # the module on meta and calls ``to_empty()``, which leaves
        # non-persistent buffers pointing at uninitialised memory (they are
        # absent from the checkpoint and ``_init_weights`` does not touch
        # buffers). Reading that garbage poisons q/k with NaN. Compute it
        # lazily instead, cached per device.
        self._rope_inv_freq_cache = None

    @property
    def uses_fused_qkv(self) -> bool:
        return self.qkv is not None

    def _rope_inv_freq(self, device: torch.device) -> torch.Tensor:
        cache = self._rope_inv_freq_cache
        if cache is None or cache.device != device:
            cache = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0, self.rotary_dim, 2, dtype=torch.float32, device=device
                    )
                    / self.rotary_dim
                )
            )
            self._rope_inv_freq_cache = cache
        return cache

    def _apply_partial_rope(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate the first ``rotary_dim`` channels with 1D RoPE."""
        if self.rotary_dim <= 0:
            return states
        if positions.dim() not in {1, 2}:
            raise ValueError(
                f"RoPE positions must be [T] or [B,T], got {tuple(positions.shape)}"
            )
        if positions.shape[-1] != states.shape[-2]:
            raise ValueError(
                "RoPE position length must match the token axis, got "
                f"positions={int(positions.shape[-1])} "
                f"tokens={int(states.shape[-2])}."
            )
        if positions.dim() == 2 and positions.shape[0] != states.shape[0]:
            raise ValueError("RoPE position batch size must match states.")
        orig_dtype = states.dtype
        inv_freq = self._rope_inv_freq(states.device)
        freqs = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
        if positions.dim() == 2:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            while cos.dim() < states.dim():
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        rotary = states[..., : self.rotary_dim].float()
        passed = states[..., self.rotary_dim :]
        rotated = rotary * cos + _rotate_half(rotary) * sin
        return torch.cat((rotated.to(dtype=orig_dtype), passed), dim=-1)

    def _project_gqa(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Project GQA Q/K/V. Query may carry a Qwen-style output gate."""
        if self.uses_fused_qkv:
            raise RuntimeError(
                "_project_gqa is only valid on the GQA attention layout."
            )
        batch_size, token_count, _hidden = tokens.shape
        query = self.q_proj(tokens)
        gate = None
        if self.gated_output:
            query, gate = query.chunk(2, dim=-1)
        query = query.view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(tokens).view(
            batch_size, token_count, self.num_kv_heads, self.head_dim
        )
        key = self.k_norm(key).transpose(1, 2)
        value = self.v_proj(tokens).view(
            batch_size, token_count, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        query = self.q_norm(query)
        return query, key, value, gate

    @staticmethod
    def _expand_cached_kv(
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_query_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat grouped K/V heads so SDPA sees one head per query head."""
        num_kv_heads = int(key.shape[1])
        if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
            raise ValueError(
                "query heads must be a positive multiple of K/V heads, got "
                f"Hq={num_query_heads} Hkv={num_kv_heads}"
            )
        if num_kv_heads == num_query_heads:
            return key, value
        repeat = num_query_heads // num_kv_heads
        return (
            key.repeat_interleave(repeat, dim=1),
            value.repeat_interleave(repeat, dim=1),
        )

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        key, value = self._expand_cached_kv(
            key, value, num_query_heads=self.num_heads
        )
        if self.fused_attn:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        scores = (query * self.scale) @ key.transpose(-2, -1)
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))
        return self.attn_drop(scores.softmax(dim=-1)) @ value

    def forward_with_prefix_kv(
        self,
        query: torch.Tensor,
        prefix_k: torch.Tensor | None,
        prefix_v: torch.Tensor | None,
        *,
        prefix_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        prefix_len: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        """G0.5 action attention: query attends to native VLM prefix K/V.

        Prefix K/V are already in Qwen head space ``[B, Hkv, T_prefix, Dh]``
        and are not re-projected. Linear-attention VLM layers publish no K/V,
        so they pass ``prefix_k is None`` and the expert layer degenerates to
        suffix self-attention.
        """
        if self.uses_fused_qkv:
            raise RuntimeError(
                "forward_with_prefix_kv requires the GQA attention layout."
            )
        if query.dim() != 3:
            raise ValueError(f"query must be [B,T,C], got {tuple(query.shape)}")
        batch_size, query_len, _hidden = query.shape
        q_states, k_states, v_states, gate = self._project_gqa(query)
        if self.rotary_dim > 0:
            positions = torch.arange(
                query_len, device=query.device, dtype=torch.long
            )
            if isinstance(prefix_len, torch.Tensor):
                if prefix_len.shape != (batch_size,):
                    raise ValueError("prefix_len tensor must be [B].")
                positions = positions[None, :] + prefix_len.to(
                    device=query.device, dtype=torch.long
                )[:, None]
            else:
                positions = positions + int(prefix_len)
            q_states = self._apply_partial_rope(q_states, positions)
            k_states = self._apply_partial_rope(k_states, positions)
        if (prefix_k is None) ^ (prefix_v is None):
            raise ValueError(
                "prefix_k and prefix_v must both be set or both None."
            )
        if prefix_k is not None:
            if prefix_k.shape != prefix_v.shape:
                raise ValueError(
                    "prefix K/V shapes must match, got "
                    f"k={tuple(prefix_k.shape)} v={tuple(prefix_v.shape)}"
                )
            if prefix_k.shape[0] != batch_size:
                raise ValueError("prefix K/V batch size must match query.")
            if (
                prefix_k.shape[1] != self.num_kv_heads
                or prefix_k.shape[-1] != self.head_dim
            ):
                raise ValueError(
                    "prefix K/V geometry must match GQA attention, got "
                    f"{tuple(prefix_k.shape)} expected "
                    f"[B,{self.num_kv_heads},T,{self.head_dim}]."
                )
            k_states = torch.cat((prefix_k, k_states), dim=2)
            v_states = torch.cat((prefix_v, v_states), dim=2)
        if attn_mask is None and prefix_k is not None:
            prefix_tokens = int(prefix_k.shape[2])
            keep = query.new_ones(
                (batch_size, 1, query_len, prefix_tokens + query_len),
                dtype=torch.bool,
            )
            if prefix_mask is not None:
                mask = torch.as_tensor(
                    prefix_mask, device=query.device, dtype=torch.bool
                )
                if mask.shape != (batch_size, prefix_tokens):
                    raise ValueError(
                        "prefix_mask must be [B,T_prefix], got "
                        f"{tuple(mask.shape)} for "
                        f"B={batch_size} T={prefix_tokens}."
                    )
                keep[:, :, :, :prefix_tokens] = mask[:, None, None, :]
            attn_mask = keep
        elif attn_mask is not None and attn_mask.dim() == 4:
            key_len = int(k_states.shape[2])
            if attn_mask.shape[-1] != key_len:
                attn_mask = attn_mask[..., -key_len:]
            if attn_mask.shape[-2] != query_len:
                attn_mask = attn_mask[:, :, -query_len:, :]
        output = self._attend(q_states, k_states, v_states, attn_mask)
        inner_q = self.num_heads * self.head_dim
        output = output.transpose(1, 2).reshape(batch_size, query_len, inner_q)
        if gate is not None:
            output = output * torch.sigmoid(gate)
        return self.proj_drop(self.proj(output))

    def reset_parameters(self) -> None:
        """Restore learnable QK-norm parameters after HF from_pretrained.

        Only runs when ``qk_norm_mode='unit_v1_learnable_ln'``; other modes
        keep the module's default parameter initialization. This matches the
        contract of ``nn.Linear.reset_parameters``/``nn.LayerNorm.reset_parameters``
        so HuggingFace's ``_initialize_missing_keys`` -> ``initialize_weights``
        traversal picks it up automatically.
        """
        if self.qk_norm_mode != "unit_v1_learnable_ln":
            return
        with torch.no_grad():
            self.qk_norm_learnable_log_T.data.fill_(
                self._qk_norm_learnable_log_init
            )
            # The two LayerNorm(head_dim) modules have their own reset_parameters
            # (weight=1, bias=0). ``nn.LayerNorm`` implements it in PyTorch and
            # HF's traversal already handles it, but call explicitly so this
            # reset is self-contained and safe under any load path.
            for norm in (self.qk_norm_learnable_q_ln, self.qk_norm_learnable_k_ln):
                if isinstance(norm, nn.LayerNorm):
                    norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            Input sequence.
        attn_mask : Tensor, optional
            Bool mask [1, 1, T, T] for SDPA (True = keep, False = mask out).

        Returns
        -------
        Tensor, shape [B, T, C]
            Output sequence after MHSA + projection.
        """
        if not self.uses_fused_qkv:
            return self.forward_with_prefix_kv(x, None, None, attn_mask=attn_mask)
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        )
        q, k, v = qkv.unbind(0)  # each: [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, T, Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)        # [B, H, T, T]
            if attn_mask is not None:
                mask = attn_mask
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                           # [B, H, T, Dh]

        x = x.transpose(1, 2).reshape(B, T, C)     # [B, T, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """
    Apply a basic initialization scheme to Linear layers.

    - Weight: Xavier uniform initialization.
    - Bias: Set to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 100,
    input_scale: float = 1.0,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B]. Each element is a timestep index, may be fractional.
    dim : int
        Dimensionality of the output embedding.
    max_period : int, default=100
        Controls the minimum frequency of the sinusoids.
    input_scale : float, default=1.0
        Multiplier applied to ``t`` before building the sinusoid arguments.

    Notes
    -----
    The defaults (``max_period=100``, ``input_scale=1.0``) reproduce the
    original X-VLA parameterization and are kept so existing checkpoints stay
    bit-exact. They are, however, badly conditioned for flow matching, where
    ``t`` lives in ``[0.001, 1.0]``: every sinusoid argument ``t * freq`` then
    stays below 1.0, so ``cos`` and ``sin`` never leave their monotone region
    and the embedding degenerates to roughly the 2-D curve ``(t, t**2)``.
    Measured on ``dim=32``, the centered singular values are
    ``1.465, 0.151, 0.006, 0.0002`` and adjacent timesteps satisfy
    ``||e(0.1) - e(0.2)|| / ||e|| = 0.038``.

    The standard diffusion parameterization (``max_period=10000`` with
    ``input_scale=1000``) yields ``5.18, 4.25, 3.64, 3.21`` and a ratio of
    ``1.50`` instead, i.e. roughly 40x more discriminative per unit t. That
    matters here because the analytic flow-matching optimum
    ``v = (x_t - x1) / t`` requires the network to modulate its input gain by
    ``1/t`` across two orders of magnitude, and the timestep embedding is the
    only signal available to do it.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = (t * float(input_scale))[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- Core Layers ----------------------------------

class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(nn.Module):
    """
    Standard Transformer block (pre-LN): LN → MHSA → residual, LN → MLP → residual.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
        attn_num_kv_heads: int | None = None,
        attn_head_dim: int | None = None,
        attn_qk_norm: bool = False,
        attn_gated_output: bool = False,
        attn_rope_partial_factor: float = 0.0,
        attn_rope_theta: float = 1_000_000.0,
        attn_qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        if attn_num_kv_heads is not None or attn_head_dim is not None:
            self.attn = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=attn_qkv_bias,
                attn_drop=0.1,
                num_kv_heads=attn_num_kv_heads,
                head_dim=attn_head_dim,
                qk_norm=attn_qk_norm,
                gated_output=attn_gated_output,
                rope_partial_factor=attn_rope_partial_factor,
                rope_theta=attn_rope_theta,
                norm_layer=RMSNorm if attn_qk_norm else nn.LayerNorm,
            )
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    @staticmethod
    def _zero_initialized_adalayernorm(
        norm: nn.LayerNorm,
        x: torch.Tensor,
        *,
        scale: torch.Tensor | None,
        action_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply a zero-initialized, action-only AdaLayerNorm scale."""
        baseline = norm(x)
        if scale is None or action_token_mask is None:
            return baseline
        if action_token_mask.shape != x.shape[:2]:
            raise ValueError(
                "action_token_mask must be [B,T] matching x, got "
                f"{tuple(action_token_mask.shape)} for {tuple(x.shape)}"
            )
        adaptive = baseline * (1.0 + scale.to(dtype=x.dtype).unsqueeze(1))
        return torch.where(action_token_mask.unsqueeze(-1), adaptive, baseline)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        time_adalayernorm_scale: torch.Tensor | None = None,
        action_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, [B, T, H]
        attn_mask : optional bool [1, 1, T, T] passed to MHSA.

        Returns
        -------
        Tensor, [B, T, H]
        """
        x = x + self.attn(
            self._zero_initialized_adalayernorm(
                self.norm1,
                x,
                scale=time_adalayernorm_scale,
                action_token_mask=action_token_mask,
            ),
            attn_mask=attn_mask,
        )
        x = x + self.mlp(
            self._zero_initialized_adalayernorm(
                self.norm2,
                x,
                scale=time_adalayernorm_scale,
                action_token_mask=action_token_mask,
            )
        )
        return x


class ReadOnlyContextCrossAttention(nn.Module):
    """Update policy queries from immutable semantic context tokens."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.gate_init = float(gate_init)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must divide num_heads={num_heads}"
            )
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = 0.1
        # Shape (1,) rather than a 0-d scalar: FSDP2's `fully_shard` rejects scalar
        # parameters outright ("Change raw_gate to a 1D tensor with numel equal to
        # 1"), which is how the qwen3_5 default strategy reaches this module. The
        # broadcast against [B, L, H] in forward is unchanged.
        self.raw_gate = nn.Parameter(
            torch.tensor([math.atanh(self.gate_init)], dtype=torch.float32)
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints written before the 1-D change hold raw_gate as a 0-d tensor.
        key = f"{prefix}raw_gate"
        saved = state_dict.get(key)
        if saved is not None and saved.shape != self.raw_gate.shape:
            state_dict[key] = saved.reshape(self.raw_gate.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_parameters(self) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            basic_init(module)
        with torch.no_grad():
            self.raw_gate.fill_(math.atanh(self.gate_init))

    def project_context(
        self,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = context_tokens.shape
        normalized = self.context_norm(context_tokens)
        key = self.k_proj(normalized).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(normalized).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        return key, value

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        context_tokens: torch.Tensor | None = None,
        projected_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if projected_context is None:
            if context_tokens is None:
                raise ValueError(
                    "context_tokens is required without projected_context."
                )
            key, value = self.project_context(context_tokens)
        else:
            key, value = projected_context
        batch_size, query_len, hidden_size = query_tokens.shape
        query = self.q_proj(self.query_norm(query_tokens)).view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        # A history bank can be empty for early timesteps. SDPA handles an
        # all-false row inconsistently across PyTorch backends, so expose one
        # placeholder key and explicitly zero that sample's update afterwards.
        has_context = context_mask.any(dim=1)
        safe_context_mask = context_mask.clone()
        if safe_context_mask.shape[1] == 0:
            raise ValueError("context attention requires at least one key slot.")
        safe_context_mask[~has_context, 0] = True
        attn_mask = safe_context_mask[:, None, None, :]
        if _has_sdp_attention():
            update = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
        else:
            weights = (query * self.scale) @ key.transpose(-2, -1)
            weights = weights.masked_fill(~attn_mask, float("-inf"))
            weights = F.softmax(weights, dim=-1)
            weights = F.dropout(
                weights,
                p=self.attn_drop,
                training=self.training,
            )
            update = weights @ value
        update = update.transpose(1, 2).reshape(
            batch_size,
            query_len,
            hidden_size,
        )
        update = self.out_proj(update)
        update = update * has_context.to(dtype=update.dtype).view(-1, 1, 1)
        return query_tokens + torch.tanh(self.raw_gate).to(
            dtype=update.dtype
        ) * update


class DorsalTokenRouter(nn.Module):
    """Route action queries over separate semantic and visual-dynamics streams."""

    def __init__(
        self,
        *,
        multi_modal_input_size: int,
        hidden_size: int,
        num_heads: int,
        num_domains: int,
        use_hetero_proj: bool,
        init_dorsal_prob: float,
    ) -> None:
        super().__init__()
        if not 0.0 < float(init_dorsal_prob) < 1.0:
            raise ValueError(
                f"init_dorsal_prob must be in (0,1), got {init_dorsal_prob}"
            )
        self.hidden_size = int(hidden_size)
        self.use_hetero_proj = bool(use_hetero_proj)
        self.init_dorsal_prob = float(init_dorsal_prob)
        if self.use_hetero_proj:
            self.dorsal_proj = DomainAwareLinear(
                multi_modal_input_size,
                hidden_size,
                num_domains=num_domains,
            )
        else:
            self.dorsal_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.stream_embeddings = nn.Parameter(torch.zeros(3, hidden_size))
        self.action_norm = nn.LayerNorm(hidden_size)
        self.semantic_norm = nn.LayerNorm(hidden_size)
        self.dorsal_norm = nn.LayerNorm(hidden_size)
        self.semantic_cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.dorsal_cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.route_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.route_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.route_bias = nn.Parameter(
            torch.tensor(
                [
                    math.log(1.0 - self.init_dorsal_prob),
                    math.log(self.init_dorsal_prob),
                ],
                dtype=torch.float32,
            )
        )
        self.route_out = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def reset_parameters(self) -> None:
        if self.use_hetero_proj:
            nn.init.xavier_uniform_(self.dorsal_proj.fc.weight)
            nn.init.zeros_(self.dorsal_proj.bias.weight)
        else:
            basic_init(self.dorsal_proj)
        nn.init.zeros_(self.stream_embeddings)
        nn.init.normal_(self.stream_embeddings[2], std=0.02)
        for module in (
            self.route_query,
            self.route_key,
        ):
            basic_init(module)
        for cross_attn in (
            self.semantic_cross_attn,
            self.dorsal_cross_attn,
        ):
            nn.init.xavier_uniform_(cross_attn.in_proj_weight)
            if cross_attn.in_proj_bias is not None:
                nn.init.zeros_(cross_attn.in_proj_bias)
            nn.init.xavier_uniform_(cross_attn.out_proj.weight)
            if cross_attn.out_proj.bias is not None:
                nn.init.zeros_(cross_attn.out_proj.bias)
        with torch.no_grad():
            self.route_bias.copy_(
                self.route_bias.new_tensor(
                    [
                        math.log(1.0 - self.init_dorsal_prob),
                        math.log(self.init_dorsal_prob),
                    ]
                )
            )
        nn.init.zeros_(self.route_out.weight)
        nn.init.zeros_(self.route_out.bias)

    def forward(
        self,
        *,
        action_tokens: torch.Tensor,
        vlm_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
        semantic_mask: torch.Tensor,
        dorsal_features: torch.Tensor,
        dorsal_mask: torch.Tensor,
        domain_id: torch.LongTensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if not bool(dorsal_mask.any(dim=1).all()):
            raise ValueError(
                "dorsal_feature_mask requires at least one valid token per sample."
            )
        if not bool(semantic_mask.any(dim=1).all()):
            raise ValueError(
                "semantic context requires at least one valid token per sample."
            )

        if self.use_hetero_proj:
            dorsal_tokens = self.dorsal_proj(dorsal_features, domain_id)
        else:
            dorsal_tokens = self.dorsal_proj(dorsal_features)

        # Type identity is router-local. Returning the unmodified semantic
        # tokens keeps the shared Transformer path identical when route_out is
        # closed, so Dorsal can affect actions only through this router.
        semantic_tokens = torch.cat(
            [
                vlm_tokens + self.stream_embeddings[0].view(1, 1, -1),
                aux_tokens + self.stream_embeddings[1].view(1, 1, -1),
            ],
            dim=1,
        )
        dorsal_tokens = (
            dorsal_tokens + self.stream_embeddings[2].view(1, 1, -1)
        )

        query = self.action_norm(action_tokens)
        semantic_value = self.semantic_norm(semantic_tokens)
        dorsal_value = self.dorsal_norm(dorsal_tokens)
        semantic_update, _ = self.semantic_cross_attn(
            query,
            semantic_value,
            semantic_value,
            key_padding_mask=~semantic_mask,
            need_weights=False,
        )
        dorsal_update, _ = self.dorsal_cross_attn(
            query,
            dorsal_value,
            dorsal_value,
            key_padding_mask=~dorsal_mask,
            need_weights=False,
        )

        summaries = torch.stack(
            [
                self._masked_mean(semantic_value, semantic_mask),
                self._masked_mean(dorsal_value, dorsal_mask),
            ],
            dim=1,
        )
        route_logits = torch.einsum(
            "btc,bsc->bts",
            self.route_query(query),
            self.route_key(summaries),
        ) * (self.hidden_size ** -0.5)
        route_weights = torch.softmax(
            route_logits + self.route_bias.view(1, 1, 2),
            dim=-1,
        )
        routed_update = (
            route_weights[..., 0:1] * semantic_update
            + route_weights[..., 1:2] * dorsal_update
        )
        action_tokens = action_tokens + self.route_out(routed_update)

        entropy = -(
            route_weights
            * route_weights.clamp_min(torch.finfo(route_weights.dtype).tiny).log()
        ).sum(dim=-1)
        metrics = {
            "dorsal_route_weights": route_weights.detach(),
            "metric_route_semantic": route_weights[..., 0].mean().detach(),
            "metric_route_dorsal": route_weights[..., 1].mean().detach(),
            "metric_route_entropy": entropy.mean().detach(),
        }
        return action_tokens, vlm_tokens, aux_tokens, metrics


# --------------------------- Main Model ---------------------------------------

class SoftPromptedTransformer(nn.Module):
    """
    Multi-modal, domain-aware Transformer with optional soft prompts.

    See parameter and forward I/O descriptions inside the docstrings.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        time_embedding_max_period: int = 100,
        time_embedding_input_scale: float = 1.0,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        dino_enabled: bool = False,
        dino_target_dim: int = 1024,
        dino_num_latent_tokens: int = 4,
        dino_detach_action: bool = False,
        dino_pos_emb_decoupled: bool = False,
        dorsal_token_routing: bool = False,
        dorsal_router_init_prob: float = 0.1,
        action_context_mode: str = "legacy_joint_v1",
        context_cross_attn_every_n_layers: int = 4,
        context_cross_attn_gate_init: float = 0.1,
        context_semantic_route: list[int] | tuple[int, ...] | None = None,
        history_memory_enabled: bool = False,
        history_memory_gate_init: float = 0.0,
        history_memory_num_views: int = 8,
        time_adalayernorm_enabled: bool = False,
        num_kv_heads: int | None = None,
        attn_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        # Defaults reproduce the original degenerate parameterization; see the
        # note in ``timestep_embedding``.
        self.time_embedding_max_period = int(time_embedding_max_period)
        self.time_embedding_input_scale = float(time_embedding_input_scale)
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj
        self.dino_enabled = dino_enabled
        self.dino_num_latent_tokens = dino_num_latent_tokens
        self.dino_detach_action = bool(dino_detach_action)
        self.dino_pos_emb_decoupled = bool(dino_pos_emb_decoupled)
        self.dorsal_token_routing = bool(dorsal_token_routing)
        self.action_context_mode = canonicalize_action_context_mode(action_context_mode, default="legacy_joint_v1")
        if self.action_context_mode not in {
            "legacy_joint_v1",
            "readonly_cross_attention_v1",
            "readonly_cross_attention_v2",
            PREFIX_KV_ACTION_CONTEXT_MODE,
        }:
            raise ValueError(
                "action_context_mode must be legacy_joint_v1, "
                "readonly_cross_attention_v1, readonly_cross_attention_v2, or prefix_kv; "
                f"got {self.action_context_mode!r}"
            )
        # v2 reuses v1's read-only cross-attention wiring and changes two things:
        # the soft prompts join the cross-attn query (they already sit in every
        # self-attn/FFN, so reading the context lets them relay it back to the
        # action tokens on the next layer), and each cross-attn layer reads its
        # own semantic depth instead of one globally mixed tensor.
        self.context_readonly = self.action_context_mode.startswith(
            "readonly_cross_attention_v"
        )
        self.context_deep_fusion = (
            self.action_context_mode == "readonly_cross_attention_v2"
        )
        self.prefix_kv = is_prefix_kv_action_context_mode(
            self.action_context_mode
        )
        self.num_kv_heads = None if num_kv_heads is None else int(num_kv_heads)
        self.attn_head_dim = (
            None if attn_head_dim is None else int(attn_head_dim)
        )
        if self.prefix_kv:
            if self.num_kv_heads is None or self.attn_head_dim is None:
                raise ValueError(
                    "prefix_kv requires num_kv_heads and attn_head_dim "
                    "matching the VLM full-attention geometry."
                )
            if use_hetero_proj:
                raise ValueError(
                    "prefix_kv does not use DomainAwareLinear projections; "
                    "the expert consumes native VLM prefix K/V."
                )
            if dorsal_token_routing:
                raise ValueError(
                    "prefix_kv is incompatible with dorsal token routing."
                )
        elif self.num_kv_heads is not None or self.attn_head_dim is not None:
            raise ValueError(
                "num_kv_heads/attn_head_dim are only valid with "
                "action_context_mode=prefix_kv so the fused Florence/Qwen "
                "attention stays unchanged."
            )
        if self.context_deep_fusion and use_hetero_proj:
            raise ValueError(
                "readonly_cross_attention_v2 needs a plain vlm_proj: the "
                "per-depth semantic stream is 4-D and DomainAwareLinear only "
                "handles [B, L, H]."
            )
        if self.context_deep_fusion and dorsal_token_routing:
            raise ValueError(
                "readonly_cross_attention_v2 is incompatible with dorsal token "
                "routing; the router assumes a single 3-D semantic stream."
            )
        self.context_cross_attn_every_n_layers = int(
            context_cross_attn_every_n_layers
        )
        if self.context_cross_attn_every_n_layers <= 0:
            raise ValueError(
                "context_cross_attn_every_n_layers must be positive, got "
                f"{self.context_cross_attn_every_n_layers}"
            )
        self.context_cross_attn_gate_init = float(
            context_cross_attn_gate_init
        )
        if not 0.0 < self.context_cross_attn_gate_init < 1.0:
            raise ValueError(
                "context_cross_attn_gate_init must be in (0,1), got "
                f"{self.context_cross_attn_gate_init}"
            )
        self.history_memory_enabled = bool(history_memory_enabled)
        self.history_memory_gate_init = float(history_memory_gate_init)
        if not 0.0 <= self.history_memory_gate_init < 1.0:
            raise ValueError(
                "history_memory_gate_init must be in [0,1), got "
                f"{self.history_memory_gate_init}"
            )
        self.history_memory_num_views = int(history_memory_num_views)
        if self.history_memory_enabled and not self.context_readonly:
            raise ValueError(
                "history_memory_enabled requires read-only action context."
            )
        if self.history_memory_num_views <= 0:
            raise ValueError(
                "history_memory_num_views must be positive, got "
                f"{self.history_memory_num_views}"
            )
        self.time_adalayernorm_enabled = bool(time_adalayernorm_enabled)
        # Structural dino (latent_tokens) can stay on while dino_head supervision is
        # off — e.g. LIBERO finetune from a dino-pretrained cross-emb checkpoint.
        self.dino_supervision = dino_enabled

        if dino_enabled:
            self.latent_tokens = nn.Parameter(
                torch.randn(1, dino_num_latent_tokens, hidden_size) * 0.02
            )
            self.dino_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, dino_target_dim),
            )

        expert_attn = {}
        if self.prefix_kv:
            if self.time_adalayernorm_enabled:
                raise ValueError("prefix_kv does not support action AdaLayerNorm")
            expert_attn = dict(
                attn_num_kv_heads=self.num_kv_heads, attn_head_dim=self.attn_head_dim,
                attn_qk_norm=True, attn_gated_output=True, attn_qkv_bias=False,
                attn_rope_partial_factor=PREFIX_KV_ROPE_PARTIAL_FACTOR,
                attn_rope_theta=PREFIX_KV_ROPE_THETA,
            )
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **expert_attn)
            for _ in range(depth)
        ])
        if self.time_adalayernorm_enabled:
            # One scale-only adapter is shared by every block. Zero init makes
            # it exactly identity at checkpoint load while retaining a direct
            # first-step gradient into the time projection.
            self.time_adalayernorm_modulator = nn.Linear(
                dim_time,
                hidden_size,
            )
        else:
            self.time_adalayernorm_modulator = None
        if self.context_readonly:
            selected_layers = tuple(
                range(0, depth, self.context_cross_attn_every_n_layers)
            )
            self.context_cross_attn = nn.ModuleDict(
                {
                    str(layer_idx): ReadOnlyContextCrossAttention(
                        hidden_size,
                        num_heads=num_heads,
                        gate_init=self.context_cross_attn_gate_init,
                    )
                    for layer_idx in selected_layers
                }
            )
            self.context_stream_embeddings = nn.Parameter(
                torch.zeros(2, hidden_size)
            )
            # Explicit ordering: the semantic route is indexed by cross-attn
            # position, so never lean on ModuleDict iteration order for it.
            self.context_cross_attn_order = tuple(
                str(layer_idx) for layer_idx in selected_layers
            )
            self.context_cross_attn_position = {
                layer_key: position
                for position, layer_key in enumerate(
                    self.context_cross_attn_order
                )
            }
        else:
            self.context_cross_attn = nn.ModuleDict()
            self.register_parameter("context_stream_embeddings", None)
            self.context_cross_attn_order = ()
            self.context_cross_attn_position = {}
        if self.history_memory_enabled:
            self.history_memory_proj = nn.Linear(
                multi_modal_input_size,
                hidden_size,
            )
            self.history_memory_cross_attn = nn.ModuleDict(
                {
                    layer_key: ReadOnlyContextCrossAttention(
                        hidden_size,
                        num_heads=num_heads,
                        gate_init=self.history_memory_gate_init,
                    )
                    for layer_key in self.context_cross_attn_order
                }
            )
            self.history_memory_stream_embedding = nn.Parameter(
                torch.zeros(1, hidden_size)
            )
            self.history_memory_view_embeddings = nn.Embedding(
                self.history_memory_num_views,
                hidden_size,
            )
        else:
            self.history_memory_proj = None
            self.history_memory_cross_attn = nn.ModuleDict()
            self.register_parameter("history_memory_stream_embedding", None)
            self.history_memory_view_embeddings = None
        self._context_semantic_route = (
            None
            if context_semantic_route is None
            else tuple(int(index) for index in context_semantic_route)
        )
        if self._context_semantic_route is not None:
            if not self.context_deep_fusion:
                raise ValueError(
                    "context_semantic_route is only valid with "
                    "readonly_cross_attention_v2."
                )
            if len(self._context_semantic_route) != len(
                self.context_cross_attn_order
            ):
                raise ValueError(
                    "context_semantic_route needs one entry per cross-attn "
                    f"layer ({len(self.context_cross_attn_order)}), got "
                    f"{len(self._context_semantic_route)}."
                )
        self._semantic_route_cache: dict[int, tuple[int, ...]] = {}

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)
        self.dorsal_router = (
            DorsalTokenRouter(
                multi_modal_input_size=multi_modal_input_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_domains=num_domains,
                use_hetero_proj=use_hetero_proj,
                init_dorsal_prob=dorsal_router_init_prob,
            )
            if self.dorsal_token_routing
            else None
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)
        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        self.apply(basic_init)
        if self.time_adalayernorm_modulator is not None:
            nn.init.zeros_(self.time_adalayernorm_modulator.weight)
            nn.init.zeros_(self.time_adalayernorm_modulator.bias)
        for cross_attn in self.context_cross_attn.values():
            cross_attn.reset_parameters()
        self.reset_history_memory_parameters()
        if self.dorsal_router is not None:
            self.dorsal_router.reset_parameters()

    def reset_history_memory_parameters(self) -> None:
        """Re-initialize every history-memory branch parameter deterministically.

        The history memory branch is opt-in and is typically absent from any
        pretrained checkpoint. Hugging Face `from_pretrained` reserves
        uninitialized storage for missing parameters and defers to
        ``_init_weights``; a custom ``nn.Parameter`` like ``raw_gate`` is not
        covered by the default rule, so the gate can be materialized with
        garbage (observed |raw_gate| ~ 1e30, driving ``tanh`` to +/-1 before
        any training step). Callers can invoke this method after
        ``from_pretrained`` to restore the deterministic zero-gated identity
        contract on the new memory branch without touching pretrained weights.
        """

        if not self.history_memory_enabled:
            return
        if self.history_memory_proj is not None:
            basic_init(self.history_memory_proj)
        for cross_attn in self.history_memory_cross_attn.values():
            cross_attn.reset_parameters()
        if self.history_memory_stream_embedding is not None:
            with torch.no_grad():
                self.history_memory_stream_embedding.zero_()
        if self.history_memory_view_embeddings is not None:
            nn.init.normal_(self.history_memory_view_embeddings.weight, std=0.02)

    @staticmethod
    def _build_dino_detach_attn_mask(
        seq_len: int,
        latent_offset: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Block action/visual/soft queries from attending to latent key positions."""
        if latent_offset <= 0:
            return None
        keep = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        keep[latent_offset:, :latent_offset] = False
        return keep.view(1, 1, seq_len, seq_len)

    def _semantic_route(self, num_depths: int) -> tuple[int, ...]:
        """Map each cross-attn layer to the semantic depth it reads."""
        cached = self._semantic_route_cache.get(num_depths)
        if cached is not None:
            return cached
        if num_depths <= 0:
            raise ValueError(
                f"semantic depth count must be positive, got {num_depths}."
            )
        num_layers = len(self.context_cross_attn_order)
        if self._context_semantic_route is not None:
            route = self._context_semantic_route
            if len(route) != num_layers:
                raise ValueError(
                    "action_context_semantic_route needs one entry per "
                    f"cross-attn layer ({num_layers}), got {len(route)}."
                )
        elif num_layers <= 1:
            route = (num_depths - 1,) * num_layers
        elif num_layers >= num_depths:
            # Keep adjacent expert stages on the same VLM depth and assign any
            # extra readers to the shallower depths first. For the formal
            # 24-layer expert (cross-attn at 0/4/8/12/16/20) and four sampled
            # Qwen depths this is (0,0,1,1,2,3), matching the v5 contract.
            readers_per_depth, extra_readers = divmod(
                num_layers,
                num_depths,
            )
            route = tuple(
                depth_index
                for depth_index in range(num_depths)
                for _ in range(
                    readers_per_depth
                    + int(depth_index < extra_readers)
                )
            )
        else:
            # There are fewer readers than sampled depths. Preserve both ends
            # and spread the readers monotonically through the available states.
            route = tuple(
                round(position * (num_depths - 1) / (num_layers - 1))
                for position in range(num_layers)
            )
        if any(index < 0 or index >= num_depths for index in route):
            raise ValueError(
                f"semantic route {route} is out of range for {num_depths} "
                "semantic depths."
            )
        self._semantic_route_cache[num_depths] = route
        return route

    def _compose_context_tokens(
        self,
        vlm_tokens: torch.Tensor,
        aux_tokens: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Tag the semantic/geometry streams and concatenate them.

        v1 yields one [B, L, H] context; v2 yields one per semantic depth,
        because each cross-attn layer reads its own depth.
        """
        semantic_bias = self.context_stream_embeddings[0].view(1, 1, -1)
        geometry = aux_tokens + self.context_stream_embeddings[1].view(1, 1, -1)
        if not self.context_deep_fusion:
            return torch.cat([vlm_tokens + semantic_bias, geometry], dim=1)
        if vlm_tokens.dim() != 4:
            raise ValueError(
                "readonly_cross_attention_v2 expects a per-depth semantic "
                f"stream [B, depths, L, H], got {tuple(vlm_tokens.shape)}."
            )
        return [
            torch.cat([vlm_tokens[:, depth] + semantic_bias, geometry], dim=1)
            for depth in range(vlm_tokens.shape[1])
        ]

    def _select_layer_context(
        self,
        context_tokens: torch.Tensor | list[torch.Tensor],
        layer_key: str,
    ) -> torch.Tensor:
        if not self.context_deep_fusion:
            return context_tokens
        route = self._semantic_route(len(context_tokens))
        return context_tokens[
            route[self.context_cross_attn_position[layer_key]]
        ]

    def _project_context_by_layer(
        self,
        context_tokens: torch.Tensor | list[torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            layer_key: self.context_cross_attn[layer_key].project_context(
                self._select_layer_context(context_tokens, layer_key)
            )
            for layer_key in self.context_cross_attn_order
        }

    @staticmethod
    def _history_time_embedding(
        offsets: torch.Tensor,
        hidden_size: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fixed relative-time identity for independent history frames."""
        half = hidden_size // 2
        if half == 0:
            return offsets.new_zeros((*offsets.shape, hidden_size), dtype=dtype)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=offsets.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = offsets.to(dtype=torch.float32).unsqueeze(-1) * frequencies
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if hidden_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding.to(dtype=dtype)

    def _prepare_history_memory(
        self,
        *,
        memory_features: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        memory_time_offsets: torch.Tensor | None,
        memory_view_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.history_memory_enabled:
            if any(value is not None for value in (
                memory_features,
                memory_mask,
                memory_time_offsets,
                memory_view_ids,
            )):
                raise ValueError(
                    "history memory inputs were provided but the feature is disabled."
                )
            return None, None
        if memory_features is None:
            return None, None
        if memory_features.dim() != 3 or memory_features.shape[0] != batch_size:
            raise ValueError(
                "memory_features must be [B,T_memory,D], got "
                f"{tuple(memory_features.shape)}"
            )
        if memory_features.shape[-1] != self.history_memory_proj.in_features:
            raise ValueError(
                "memory_features last dimension must match visual projection: "
                f"got {memory_features.shape[-1]} expected "
                f"{self.history_memory_proj.in_features}"
            )
        memory_length = memory_features.shape[1]
        if memory_length == 0:
            return None, None
        if memory_mask is None:
            memory_mask = torch.ones(
                (batch_size, memory_length), device=device, dtype=torch.bool
            )
        else:
            memory_mask = torch.as_tensor(
                memory_mask, device=device, dtype=torch.bool
            )
            if memory_mask.shape != (batch_size, memory_length):
                raise ValueError(
                    "memory_mask must be [B,T_memory], got "
                    f"{tuple(memory_mask.shape)}"
                )
        if memory_time_offsets is None or memory_view_ids is None:
            raise ValueError(
                "history memory requires memory_time_offsets and memory_view_ids."
            )
        memory_time_offsets = torch.as_tensor(
            memory_time_offsets, device=device
        )
        memory_view_ids = torch.as_tensor(memory_view_ids, device=device)
        if memory_time_offsets.shape != (batch_size, memory_length):
            raise ValueError("memory_time_offsets must be [B,T_memory].")
        if memory_view_ids.shape != (batch_size, memory_length):
            raise ValueError("memory_view_ids must be [B,T_memory].")
        if memory_view_ids.numel() and (
            int(memory_view_ids.min()) < 0
            or int(memory_view_ids.max()) >= self.history_memory_num_views
        ):
            raise ValueError(
                "memory_view_ids exceed history_memory_num_views="
                f"{self.history_memory_num_views}."
            )
        projection_dtype = self._projection_dtype(self.history_memory_proj)
        if projection_dtype is not None:
            memory_features = memory_features.to(dtype=projection_dtype)
        tokens = self.history_memory_proj(memory_features)
        tokens = tokens + self.history_memory_stream_embedding.to(
            dtype=tokens.dtype
        ).view(1, 1, -1)
        tokens = tokens + self.history_memory_view_embeddings(
            memory_view_ids.to(dtype=torch.long)
        ).to(dtype=tokens.dtype)
        tokens = tokens + self._history_time_embedding(
            memory_time_offsets,
            self.hidden_size,
            dtype=tokens.dtype,
        )
        return tokens, memory_mask

    @staticmethod
    def _projection_dtype(proj: nn.Module) -> torch.dtype | None:
        if isinstance(proj, DomainAwareLinear):
            return proj.fc.weight.dtype
        parameter = next(proj.parameters(), None)
        return None if parameter is None else parameter.dtype

    def prepare_readonly_context(
        self,
        *,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        vlm_feature_mask: torch.Tensor | None,
        aux_visual_mask: torch.Tensor | None,
        memory_features: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_time_offsets: torch.Tensor | None = None,
        memory_view_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Precompute immutable context projections for iterative inference."""
        if not self.context_readonly:
            raise RuntimeError(
                "prepare_readonly_context requires read-only context mode."
            )
        if self.training:
            raise RuntimeError(
                "Read-only context caching is inference-only; call model.eval()."
            )
        batch_size = int(vlm_features.shape[0])
        device = vlm_features.device

        def _mask(
            value: torch.Tensor | None,
            *,
            seq_len: int,
            name: str,
        ) -> torch.Tensor:
            if value is None:
                return torch.ones(
                    (batch_size, seq_len),
                    device=device,
                    dtype=torch.bool,
                )
            value = torch.as_tensor(value, device=device, dtype=torch.bool)
            if value.shape != (batch_size, seq_len):
                raise ValueError(
                    f"{name} must be {(batch_size, seq_len)}, "
                    f"got {tuple(value.shape)}"
                )
            return value

        semantic_mask = torch.cat(
            [
                _mask(
                    vlm_feature_mask,
                    # ``-2`` so v2's per-depth [B, depths, L, H] stream reports
                    # the token count, not the number of depths.
                    seq_len=vlm_features.shape[-2],
                    name="vlm_feature_mask",
                ),
                _mask(
                    aux_visual_mask,
                    seq_len=aux_visual_inputs.shape[1],
                    name="aux_visual_mask",
                ),
            ],
            dim=1,
        )
        if not bool(semantic_mask.any(dim=1).all()):
            raise ValueError(
                "read-only action context requires at least one valid "
                "semantic token per sample."
            )
        vlm_proj_dtype = self._projection_dtype(self.vlm_proj)
        aux_proj_dtype = self._projection_dtype(self.aux_visual_proj)
        if vlm_proj_dtype is not None:
            vlm_features = vlm_features.to(dtype=vlm_proj_dtype)
        if aux_proj_dtype is not None:
            aux_visual_inputs = aux_visual_inputs.to(dtype=aux_proj_dtype)
        if self.use_hetero_proj:
            vlm_tokens = self.vlm_proj(vlm_features, domain_id)
            aux_tokens = self.aux_visual_proj(
                aux_visual_inputs,
                domain_id,
            )
        else:
            vlm_tokens = self.vlm_proj(vlm_features)
            aux_tokens = self.aux_visual_proj(aux_visual_inputs)
        context_tokens = self._compose_context_tokens(vlm_tokens, aux_tokens)
        projected_by_layer = self._project_context_by_layer(context_tokens)
        memory_tokens, normalized_memory_mask = self._prepare_history_memory(
            memory_features=memory_features,
            memory_mask=memory_mask,
            memory_time_offsets=memory_time_offsets,
            memory_view_ids=memory_view_ids,
            batch_size=batch_size,
            device=device,
        )
        projected_memory_by_layer = (
            {
                layer_key: self.history_memory_cross_attn[layer_key].project_context(
                    memory_tokens
                )
                for layer_key in self.context_cross_attn_order
            }
            if memory_tokens is not None
            else {}
        )
        return {
            "vlm_tokens": vlm_tokens,
            "aux_tokens": aux_tokens,
            "context_tokens": context_tokens,
            "semantic_mask": semantic_mask,
            "projected_by_layer": projected_by_layer,
            "memory_tokens": memory_tokens,
            "memory_mask": normalized_memory_mask,
            "projected_memory_by_layer": projected_memory_by_layer,
        }

    def _forward_prefix_kv(
        self,
        *,
        domain_id: torch.LongTensor,
        action_hidden: torch.Tensor,
        num_actions: int,
        prefix_kv_by_layer: list | tuple | None,
        prefix_kv_mask: torch.Tensor | None,
        prefix_kv_position_offset: torch.Tensor | None = None,
        prepared_context: dict[str, object] | None = None,
    ) -> torch.Tensor | dict:
        """G0.5 expert: native VLM prefix K/V, suffix-only action self-attn.

        The suffix is ``[latent?, action, soft_prompts]``. Every expert layer
        concatenates its VLM layer's prefix K/V in front of the suffix K/V,
        so action queries read the frozen multimodal prefix without ever
        re-projecting it through ``vlm_proj``.
        """
        if not self.prefix_kv:
            raise RuntimeError(
                "_forward_prefix_kv requires action_context_mode=prefix_kv."
            )
        if prepared_context is not None:
            if self.training:
                raise RuntimeError("prefix_kv cache is inference-only.")
            version = prepared_context.get("version")
            if version not in PREFIX_KV_CACHE_TOPOLOGY_ALIASES:
                raise RuntimeError(
                    f"Unsupported prefix_kv cache version: {version!r}"
                )
            prefix_kv_by_layer = prepared_context["prefix_kv_by_layer"]
            prefix_kv_mask = prepared_context["prefix_mask"]
            prefix_kv_position_offset = prepared_context.get("prefix_position_offset")
        if prefix_kv_by_layer is None or prefix_kv_mask is None:
            raise ValueError(
                "prefix_kv requires prefix_kv_by_layer and prefix_kv_mask."
            )
        if len(prefix_kv_by_layer) != len(self.blocks):
            raise ValueError(
                "prefix_kv must have one entry per expert layer "
                f"({len(self.blocks)}), got {len(prefix_kv_by_layer)}."
            )
        prefix_mask = torch.as_tensor(
            prefix_kv_mask,
            device=action_hidden.device,
            dtype=torch.bool,
        )
        if (
            prefix_mask.dim() != 2
            or prefix_mask.shape[0] != action_hidden.shape[0]
        ):
            raise ValueError(
                "prefix_kv_mask must be [B,T_prefix], got "
                f"{tuple(prefix_mask.shape)}."
            )
        if not bool(prefix_mask.any(dim=1).all()):
            raise ValueError(
                "prefix_kv requires at least one valid prefix token per sample."
            )
        batch_size = action_hidden.shape[0]
        latent_offset = 0
        expert = action_hidden
        if self.dino_enabled:
            latent = self.latent_tokens.expand(batch_size, -1, -1)
            expert = torch.cat([latent, expert], dim=1)
            latent_offset = self.dino_num_latent_tokens
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(
                batch_size,
                self.len_soft_prompts,
                self.hidden_size,
            )
            expert = torch.cat([expert, soft_prompts], dim=1)
        suffix_len = expert.shape[1]
        if suffix_len > self.pos_emb.shape[1]:
            raise ValueError(
                "prefix_kv expert length "
                f"{suffix_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        expert = expert + self.pos_emb[:, :suffix_len, :]
        # Padding belongs to the batch layout, not a sample's rotary position,
        # so the suffix RoPE offset is each sample's true prefix length.
        prefix_len = (prefix_mask.sum(dim=1) if prefix_kv_position_offset is None else
                      torch.as_tensor(prefix_kv_position_offset, device=expert.device, dtype=torch.long))
        if prefix_len.shape != (batch_size,):
            raise ValueError("prefix_kv_position_offset must be [B].")
        for layer_idx, block in enumerate(self.blocks):
            kv = prefix_kv_by_layer[layer_idx]
            prefix_k = prefix_v = None
            if kv is not None:
                prefix_k, prefix_v = kv
                prefix_k = prefix_k.to(device=expert.device, dtype=expert.dtype)
                prefix_v = prefix_v.to(device=expert.device, dtype=expert.dtype)
            normed = block.norm1(expert)
            expert = expert + block.attn.forward_with_prefix_kv(
                normed,
                prefix_k,
                prefix_v,
                prefix_mask=prefix_mask if prefix_k is not None else None,
                prefix_len=prefix_len,
            )
            expert = expert + block.mlp(block.norm2(expert))
        normed = self.norm(expert)
        action_pred = self.action_decoder(
            normed[:, latent_offset : latent_offset + num_actions],
            domain_id,
        )
        if self.dino_enabled and self.dino_supervision:
            latent_out = normed[:, :latent_offset]
            return {
                "action_pred": action_pred,
                "dino_pred": self.dino_head(latent_out.mean(dim=1)),
            }
        return action_pred

    @torch.no_grad()
    def prepare_prefix_kv_context(
        self,
        *,
        prefix_kv_by_layer: list | tuple,
        prefix_kv_mask: torch.Tensor,
        prefix_kv_position_offset: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Cache native VLM prefix K/V for iterative flow-matching sampling."""
        if not self.prefix_kv:
            raise RuntimeError(
                "prepare_prefix_kv_context requires action_context_mode=prefix_kv."
            )
        if self.training:
            raise RuntimeError("prefix_kv cache is inference-only.")
        cached = []
        for kv in prefix_kv_by_layer:
            if kv is None:
                cached.append(None)
                continue
            key, value = kv
            cached.append((key.detach().clone(), value.detach().clone()))
        return {
            "version": PREFIX_KV_CACHE_TOPOLOGY,
            "prefix_kv_by_layer": tuple(cached),
            "prefix_mask": prefix_kv_mask.detach().clone(),
            "prefix_position_offset": (None if prefix_kv_position_offset is None else
                                       prefix_kv_position_offset.detach().clone()),
        }

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        vlm_feature_mask: torch.Tensor | None = None,
        aux_visual_mask: torch.Tensor | None = None,
        dorsal_features: torch.Tensor | None = None,
        dorsal_feature_mask: torch.Tensor | None = None,
        memory_features: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_time_offsets: torch.Tensor | None = None,
        memory_view_ids: torch.Tensor | None = None,
        prepared_context: dict[str, object] | None = None,
        prefix_kv_by_layer: list | tuple | None = None,
        prefix_kv_mask: torch.Tensor | None = None,
        prefix_kv_position_offset: torch.Tensor | None = None,
    ) -> torch.Tensor | dict:
        """
        Forward pass.

        Inputs
        ------
        domain_id : [B]
        vlm_features : [B, T_vlm, D]
        aux_visual_inputs : [B, T_aux, D]
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_propio]
        t : [B]
        vlm_feature_mask : optional [B, T_vlm] bool/0-1
            Valid positions in `vlm_features` (primary visual tokens and
            non-padding text). Invalid positions are excluded as attention keys.
        aux_visual_mask : optional [B, T_aux] bool/0-1
            Valid auxiliary-view tokens. Invalid/padded views are excluded as
            attention keys.
        dorsal_features : optional [B, T_dorsal, D]
            Independent Dorsal visual-dynamics tokens.
        dorsal_feature_mask : optional [B, T_dorsal] bool/0-1
            Valid Dorsal token positions.

        Returns
        -------
        Tensor or dict
            Plain action tensor when neither DINO supervision nor Dorsal routing
            is active. Otherwise a dict containing ``action_pred`` plus optional
            ``dino_pred`` and observable ``metric_route_*`` scalars.
        """
        if action_with_noise.dim() != 3:
            raise ValueError(
                "action_with_noise must be [B,T_action,dim_action], "
                f"got {tuple(action_with_noise.shape)}"
            )
        B, num_actions, action_dim = action_with_noise.shape
        if action_dim != self.dim_action:
            raise ValueError(
                f"action_with_noise last dim must be {self.dim_action}, got {action_dim}"
            )
        if domain_id.shape != (B,):
            raise ValueError(f"domain_id must be [B]={B}, got {tuple(domain_id.shape)}")
        if domain_id.dtype != torch.long:
            raise ValueError(f"domain_id must be torch.long, got {domain_id.dtype}")
        num_domains = self.action_encoder.fc.num_embeddings
        if domain_id.numel() > 0 and (
            int(domain_id.min().item()) < 0
            or int(domain_id.max().item()) >= num_domains
        ):
            raise ValueError(
                f"domain_id must be in [0,{num_domains - 1}], "
                f"got min={int(domain_id.min().item())} max={int(domain_id.max().item())}"
            )
        if t.shape != (B,):
            raise ValueError(f"t must be [B]={B}, got {tuple(t.shape)}")
        expected_proprio_dim = (
            self.action_encoder.input_size - self.dim_action - self.dim_time
        )
        if proprio.shape != (B, expected_proprio_dim):
            raise ValueError(
                "proprio must be [B,dim_proprio]="
                f"{(B, expected_proprio_dim)}, got {tuple(proprio.shape)}"
            )
        if self.prefix_kv:
            # The G0.5 expert never assembles a projected context sequence;
            # it consumes native VLM prefix K/V per layer instead. The
            # ``vlm_features``/``aux_visual_inputs`` arguments are the
            # backend's placeholder tensors and are deliberately unused.
            action_dtype = self.action_encoder.fc.weight.dtype
            time_emb = timestep_embedding(
                t,
                self.dim_time,
                max_period=self.time_embedding_max_period,
                input_scale=self.time_embedding_input_scale,
            ).to(dtype=action_dtype)
            time_tokens = time_emb.unsqueeze(1).expand(
                B, num_actions, self.dim_time
            )
            proprio_tokens = proprio.to(dtype=action_dtype).unsqueeze(1).expand(
                B, num_actions, proprio.shape[-1]
            )
            action_tokens = torch.cat(
                [
                    action_with_noise.to(dtype=action_dtype),
                    proprio_tokens,
                    time_tokens,
                ],
                dim=-1,
            )
            return self._forward_prefix_kv(
                domain_id=domain_id,
                action_hidden=self.action_encoder(action_tokens, domain_id),
                num_actions=num_actions,
                prefix_kv_by_layer=prefix_kv_by_layer,
                prefix_kv_mask=prefix_kv_mask,
                prefix_kv_position_offset=prefix_kv_position_offset,
                prepared_context=prepared_context,
            )
        if prefix_kv_by_layer is not None or prefix_kv_mask is not None:
            raise ValueError(
                "prefix_kv_by_layer/prefix_kv_mask require "
                "action_context_mode=prefix_kv."
            )
        # v2 carries one semantic state per sampled VLM depth, so its
        # vlm_features is [B,depths,T_vlm,D] instead of [B,T_vlm,D].
        expected_vlm_dim = 4 if self.context_deep_fusion else 3
        if (
            vlm_features.dim() != expected_vlm_dim
            or vlm_features.shape[0] != B
        ):
            raise ValueError(
                "vlm_features must be "
                + (
                    "[B,depths,T_vlm,D]"
                    if self.context_deep_fusion
                    else "[B,T_vlm,D]"
                )
                + f", got {tuple(vlm_features.shape)}"
            )
        if aux_visual_inputs.dim() != 3 or aux_visual_inputs.shape[0] != B:
            raise ValueError(
                "aux_visual_inputs must be [B,T_aux,D], "
                f"got {tuple(aux_visual_inputs.shape)}"
            )
        expected_visual_dim = (
            self.vlm_proj.input_size
            if self.use_hetero_proj
            else self.vlm_proj.in_features
        )
        if (
            vlm_features.shape[-1] != aux_visual_inputs.shape[-1]
            or vlm_features.shape[-1] != expected_visual_dim
        ):
            raise ValueError(
                "visual feature dimensions must match the projection input: "
                f"vlm={vlm_features.shape[-1]} aux={aux_visual_inputs.shape[-1]} "
                f"expected={expected_visual_dim}"
            )

        def _token_mask(
            mask: torch.Tensor | None,
            *,
            seq_len: int,
            name: str,
        ) -> torch.Tensor:
            if mask is None:
                return torch.ones((B, seq_len), device=action_with_noise.device, dtype=torch.bool)
            mask = torch.as_tensor(mask, device=action_with_noise.device, dtype=torch.bool)
            if mask.shape != (B, seq_len):
                raise ValueError(f"{name} must be {(B, seq_len)}, got {tuple(mask.shape)}")
            return mask

        vlm_feature_mask = _token_mask(
            vlm_feature_mask,
            seq_len=vlm_features.shape[-2],
            name="vlm_feature_mask",
        )
        aux_visual_mask = _token_mask(
            aux_visual_mask,
            seq_len=aux_visual_inputs.shape[1],
            name="aux_visual_mask",
        )
        if prepared_context is not None:
            memory_tokens = prepared_context.get("memory_tokens")
            normalized_memory_mask = prepared_context.get("memory_mask")
            projected_memory_by_layer = prepared_context.get(
                "projected_memory_by_layer", {}
            )
        else:
            memory_tokens, normalized_memory_mask = self._prepare_history_memory(
                memory_features=memory_features,
                memory_mask=memory_mask,
                memory_time_offsets=memory_time_offsets,
                memory_view_ids=memory_view_ids,
                batch_size=B,
                device=action_with_noise.device,
            )
            projected_memory_by_layer = {}
        if self.dorsal_token_routing:
            if dorsal_features is None:
                raise ValueError(
                    "dorsal_features is required when dorsal_token_routing is enabled."
                )
            if dorsal_features.dim() != 3 or dorsal_features.shape[0] != B:
                raise ValueError(
                    "dorsal_features must be [B,T_dorsal,D], "
                    f"got {tuple(dorsal_features.shape)}"
                )
            if dorsal_features.shape[-1] != expected_visual_dim:
                raise ValueError(
                    "dorsal_features last dim must match visual projection input: "
                    f"got {dorsal_features.shape[-1]} expected {expected_visual_dim}"
                )
            dorsal_feature_mask = _token_mask(
                dorsal_feature_mask,
                seq_len=dorsal_features.shape[1],
                name="dorsal_feature_mask",
            )
        elif dorsal_features is not None or dorsal_feature_mask is not None:
            raise ValueError(
                "Dorsal token inputs were provided but dorsal_token_routing is disabled."
            )

        # Encode (action + proprio + time) -> tokens. Keep the concatenated
        # tensor in the action encoder dtype for bf16/fp16 training.
        action_dtype = self.action_encoder.fc.weight.dtype
        time_emb = timestep_embedding(
            t,
            self.dim_time,
            max_period=self.time_embedding_max_period,
            input_scale=self.time_embedding_input_scale,
        ).to(dtype=action_dtype)  # [B, dim_time]
        time_adalayernorm_scale = None
        if self.time_adalayernorm_modulator is not None:
            time_adalayernorm_scale = self.time_adalayernorm_modulator(
                time_emb.to(
                    dtype=self.time_adalayernorm_modulator.weight.dtype
                )
            )
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.to(dtype=action_dtype).unsqueeze(1).expand(
            B, num_actions, proprio.shape[-1]
        )
        action_tokens = torch.cat(
            [action_with_noise.to(dtype=action_dtype), proprio_tokens, time_tokens],
            dim=-1,
        )
        action_hidden = self.action_encoder(action_tokens, domain_id)       # [B, T_action, H]

        if prepared_context is not None:
            if not self.context_readonly:
                raise ValueError(
                    "prepared_context is only valid in read-only context mode."
                )
            vlm_tokens = prepared_context["vlm_tokens"]
            aux_tokens = prepared_context["aux_tokens"]
        else:
            vlm_proj_dtype = self._projection_dtype(self.vlm_proj)
            aux_proj_dtype = self._projection_dtype(self.aux_visual_proj)
            if vlm_proj_dtype is not None:
                vlm_features = vlm_features.to(dtype=vlm_proj_dtype)
            if aux_proj_dtype is not None:
                aux_visual_inputs = aux_visual_inputs.to(dtype=aux_proj_dtype)
            if self.use_hetero_proj:
                vlm_tokens = self.vlm_proj(vlm_features, domain_id)
                aux_tokens = self.aux_visual_proj(aux_visual_inputs, domain_id)
            else:
                vlm_tokens = self.vlm_proj(vlm_features)
                aux_tokens = self.aux_visual_proj(aux_visual_inputs)

        route_metrics: dict[str, torch.Tensor] = {}
        if self.dorsal_router is not None:
            semantic_mask = torch.cat(
                [vlm_feature_mask, aux_visual_mask],
                dim=1,
            )
            (
                action_hidden,
                vlm_tokens,
                aux_tokens,
                route_metrics,
            ) = self.dorsal_router(
                action_tokens=action_hidden,
                vlm_tokens=vlm_tokens,
                aux_tokens=aux_tokens,
                semantic_mask=semantic_mask,
                dorsal_features=dorsal_features,
                dorsal_mask=dorsal_feature_mask,
                domain_id=domain_id,
            )

        if self.context_readonly:
            semantic_mask = (
                prepared_context["semantic_mask"]
                if prepared_context is not None
                else torch.cat(
                    [vlm_feature_mask, aux_visual_mask],
                    dim=1,
                )
            )
            if not bool(semantic_mask.any(dim=1).all()):
                raise ValueError(
                    "read-only action context requires at least one valid "
                    "semantic token per sample."
                )
            context_tokens = (
                prepared_context["context_tokens"]
                if prepared_context is not None
                else self._compose_context_tokens(vlm_tokens, aux_tokens)
            )
            projected_by_layer = (
                prepared_context["projected_by_layer"]
                if prepared_context is not None
                else {}
            )

            latent_offset = 0
            x = action_hidden
            if self.dino_enabled:
                latent = self.latent_tokens.expand(B, -1, -1)
                x = torch.cat([latent, x], dim=1)
                latent_offset = self.dino_num_latent_tokens
            if self.len_soft_prompts > 0:
                soft_prompts = self.soft_prompt_hub(domain_id).view(
                    B,
                    self.len_soft_prompts,
                    self.hidden_size,
                )
                x = torch.cat([x, soft_prompts], dim=1)

            action_token_mask = torch.zeros(
                (B, x.shape[1]), device=x.device, dtype=torch.bool
            )
            action_token_mask[
                :, latent_offset:latent_offset + num_actions
            ] = True

            seq_len = x.shape[1]
            if seq_len > self.pos_emb.shape[1]:
                raise ValueError(
                    "Action-query sequence length "
                    f"{seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
                )
            if latent_offset > 0 and self.dino_pos_emb_decoupled:
                x_lat = x[:, :latent_offset] + self.pos_emb[:, :latent_offset, :]
                rest_len = seq_len - latent_offset
                x_rest = x[:, latent_offset:] + self.pos_emb[:, :rest_len, :]
                x = torch.cat([x_lat, x_rest], dim=1)
            else:
                x = x + self.pos_emb[:, :seq_len, :]

            self_attn_mask = None
            if self.dino_enabled and self.dino_detach_action and latent_offset > 0:
                self_attn_mask = self._build_dino_detach_attn_mask(
                    seq_len,
                    latent_offset,
                    device=x.device,
                )
            # v1 lets only the action tokens query the context. v2 widens the
            # query to the soft prompts as well; the dino latent tokens stay out
            # so ``dino_detach_action`` semantics are untouched.
            query_start = latent_offset
            query_stop = (
                seq_len
                if self.context_deep_fusion
                else latent_offset + num_actions
            )
            for layer_idx, block in enumerate(self.blocks):
                x = block(
                    x,
                    attn_mask=self_attn_mask,
                    time_adalayernorm_scale=time_adalayernorm_scale,
                    action_token_mask=action_token_mask,
                )
                layer_key = str(layer_idx)
                if layer_key in self.context_cross_attn:
                    query_update = self.context_cross_attn[layer_key](
                        x[:, query_start:query_stop],
                        semantic_mask,
                        context_tokens=self._select_layer_context(
                            context_tokens,
                            layer_key,
                        ),
                        projected_context=projected_by_layer.get(layer_key),
                    )
                    parts = []
                    if query_start > 0:
                        parts.append(x[:, :query_start])
                    parts.append(query_update)
                    if query_stop < seq_len:
                        parts.append(x[:, query_stop:])
                    x = (
                        parts[0]
                        if len(parts) == 1
                        else torch.cat(parts, dim=1)
                    )
                if memory_tokens is not None and layer_key in self.history_memory_cross_attn:
                    query_update = self.history_memory_cross_attn[layer_key](
                        x[:, query_start:query_stop],
                        normalized_memory_mask,
                        context_tokens=memory_tokens,
                        projected_context=projected_memory_by_layer.get(layer_key),
                    )
                    parts = []
                    if query_start > 0:
                        parts.append(x[:, :query_start])
                    parts.append(query_update)
                    if query_stop < seq_len:
                        parts.append(x[:, query_stop:])
                    x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

            normed = self.norm(x)
            action_pred = self.action_decoder(
                normed[:, latent_offset:latent_offset + num_actions],
                domain_id,
            )
            if self.dino_enabled and self.dino_supervision:
                latent_out = normed[:, :latent_offset]
                dino_pred = self.dino_head(latent_out.mean(dim=1))
                return {
                    "action_pred": action_pred,
                    "dino_pred": dino_pred,
                    **route_metrics,
                }
            if route_metrics:
                return {"action_pred": action_pred, **route_metrics}
            return action_pred

        # Prepend legacy latent tokens before action tokens.
        latent_offset = 0
        x = action_hidden
        if self.dino_enabled:
            latent = self.latent_tokens.expand(B, -1, -1)
            x = torch.cat([latent, x], dim=1)
            latent_offset = self.dino_num_latent_tokens

        x = torch.cat([x, vlm_tokens], dim=1)
        x = torch.cat([x, aux_tokens], dim=1)

        # Add positional embeddings (truncate if needed).
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        if latent_offset > 0 and self.dino_pos_emb_decoupled:
            # Latent prefix uses pos_emb[0:offset]; action/visual keep the same
            # pos_emb indices they would use without latent (no shift). Required
            # when enabling DINO from a non-dino checkpoint.
            x_lat = x[:, :latent_offset] + self.pos_emb[:, :latent_offset, :]
            rest_len = seq_len - latent_offset
            x_rest = x[:, latent_offset:] + self.pos_emb[:, :rest_len, :]
            x = torch.cat([x_lat, x_rest], dim=1)
        else:
            x = x + self.pos_emb[:, :seq_len, :]

        # Append soft prompts
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        action_token_mask = torch.zeros(
            (B, x.shape[1]), device=x.device, dtype=torch.bool
        )
        action_token_mask[:, latent_offset:latent_offset + num_actions] = True

        # Preserve Florence's legacy joint-transformer behavior exactly: visual
        # padding remained ordinary self-attention keys. Token-key masking is
        # only part of the new Dorsal routing path; Qwen's read-only context
        # returns above and masks its context there.
        attn_mask = None
        if self.dorsal_router is not None:
            action_key_mask = torch.ones(
                (B, num_actions),
                device=x.device,
                dtype=torch.bool,
            )
            prefix_token_mask = action_key_mask
            if latent_offset > 0:
                latent_token_mask = torch.ones(
                    (B, latent_offset),
                    device=x.device,
                    dtype=torch.bool,
                )
                prefix_token_mask = torch.cat(
                    [latent_token_mask, action_key_mask],
                    dim=1,
                )
            token_key_mask = torch.cat(
                [
                    prefix_token_mask,
                    vlm_feature_mask,
                    aux_visual_mask,
                ],
                dim=1,
            )
            if self.len_soft_prompts > 0:
                prompt_token_mask = torch.ones(
                    (B, self.len_soft_prompts),
                    device=x.device,
                    dtype=torch.bool,
                )
                token_key_mask = torch.cat([token_key_mask, prompt_token_mask], dim=1)
            attn_mask = token_key_mask[:, None, None, :].expand(
                -1,
                1,
                x.shape[1],
                -1,
            )
        if self.dino_enabled and self.dino_detach_action and latent_offset > 0:
            dino_attn_mask = self._build_dino_detach_attn_mask(
                x.shape[1],
                latent_offset,
                device=x.device,
            )
            attn_mask = (
                dino_attn_mask
                if attn_mask is None
                else attn_mask & dino_attn_mask
            )
        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                attn_mask=attn_mask,
                time_adalayernorm_scale=time_adalayernorm_scale,
                action_token_mask=action_token_mask,
            )

        # Extract outputs
        normed = self.norm(x)
        action_pred = self.action_decoder(
            normed[:, latent_offset:latent_offset + num_actions], domain_id
        )

        if self.dino_enabled and self.dino_supervision:
            latent_out = normed[:, :latent_offset]
            dino_pred = self.dino_head(latent_out.mean(dim=1))
            return {
                "action_pred": action_pred,
                "dino_pred": dino_pred,
                **route_metrics,
            }

        if route_metrics:
            return {"action_pred": action_pred, **route_metrics}
        return action_pred
