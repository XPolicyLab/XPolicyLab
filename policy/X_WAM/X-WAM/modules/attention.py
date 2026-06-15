# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings
from typing import Optional, Callable

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    warnings.warn("flash_attn not found, falling back to torch scaled_dot_product_attention.")
    HAS_FLASH_ATTN = False

__all__ = [
    "attention",
]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    attn_mask_func: Optional[Callable] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
        )

    if attn_mask_func is not None:
        # flex_attention expects (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        block_mask = create_block_mask(attn_mask_func, q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous()
    elif HAS_FLASH_ATTN:
        # flash_attn_func expects (batch, seq_len, heads, head_dim)
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        out = flash_attn_func(q, k, v, causal=causal)
        out = out.contiguous()
    else:
        # fallback: torch SDPA expects (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)
        out = out.transpose(1, 2).contiguous()

    return out
