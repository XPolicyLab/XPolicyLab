# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for ME-Dex-1.0.
import os
import warnings

import torch


# FlashAttention is optional; PyTorch SDPA is used when it is unavailable.
_DISABLE_FLASH_ATTN = os.environ.get('WAN_DISABLE_FLASH_ATTN', '0') == '1'

if _DISABLE_FLASH_ATTN:
    FLASH_ATTN_3_AVAILABLE = False
    FLASH_ATTN_2_AVAILABLE = False
    warnings.warn(
        'WAN_DISABLE_FLASH_ATTN=1: using PyTorch scaled-dot-product attention.')
else:
    try:
        import flash_attn_interface
        FLASH_ATTN_3_AVAILABLE = True
    except Exception as error:
        FLASH_ATTN_3_AVAILABLE = False
        warnings.warn(
            f'FlashAttention 3 unavailable; falling back to PyTorch attention: {error}')

    try:
        import flash_attn
        FLASH_ATTN_2_AVAILABLE = True
    except Exception as error:
        FLASH_ATTN_2_AVAILABLE = False
        warnings.warn(
            f'FlashAttention 2 unavailable; falling back to PyTorch attention: {error}')

__all__ = [
    'flash_attention',
]


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    q_scale=None,
    causal=False,
    attn_mask=None,
    dtype=torch.bfloat16,
):
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    q = q if q.dtype in half_dtypes else q.to(dtype)
    k = k if k.dtype in half_dtypes else k.to(dtype)
    v = v if v.dtype in half_dtypes else v.to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if attn_mask is not None:
        if causal:
            raise ValueError('Custom attention mask and causal=True are mutually exclusive')
        expected_shape = (q.size(1), k.size(1))
        if tuple(attn_mask.shape[-2:]) != expected_shape:
            raise ValueError(
                f'Attention mask trailing shape must be {expected_shape}, '
                f'got {tuple(attn_mask.shape)}'
            )

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    attn_mask=None,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if attn_mask is not None:
        if q_lens is not None or k_lens is not None:
            raise ValueError('Custom attention mask currently requires fixed-length Q/K/V')
        return sdpa_attention(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            q_scale=q_scale,
            causal=False,
            attn_mask=attn_mask,
            dtype=dtype)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        x = sdpa_attention(
            q=q.unflatten(0, (b, lq)),
            k=k.unflatten(0, (b, lk)),
            v=v.unflatten(0, (b, lk)),
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            q_scale=None,
            causal=causal,
            attn_mask=None,
            dtype=dtype)

    # output
    return x.type(out_dtype)
