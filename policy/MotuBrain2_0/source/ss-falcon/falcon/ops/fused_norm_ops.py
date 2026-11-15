"""Fused RMSNorm + modulate/broadcast_mul: varlen wrappers for Falcon.

Delegates to the portable operators in ``soma.inference.models.operators``.

Both operators fold the AdaLN RMSNorm (``eps=1e-6``) into the scaling step:
``modulate`` multiplies the normalized tensor by ``1 + scale`` and ``gate``
multiplies it by ``scale``.  They are deployment operators for the
single-sample inference path this bundle ships, so they ignore the packed
sequence metadata and require a single AdaLN row (``scale`` of shape
``[1, hidden_size]``); the training stack's CUDA kernels instead broadcast one
scale row per packed sequence segment.  The per-modality wrappers below are
kept as distinct names because Falcon's layers document which cu_seqlens
layout each modality would use.
"""

import torch

from falcon.utils.common import nvtx_profiled_function
from soma.inference.models.operators import gate as _gate
from soma.inference.models.operators import modulate as _modulate

# ---- Context modality (batch-wise cu_seqlens) ----


@nvtx_profiled_function
def fused_varlen_c_modulate(c: torch.Tensor, scale: torch.Tensor, packed_seq_params: dict) -> torch.Tensor:
    r"""Fused RMSNorm + batch-wise modulate for context modality.

    Args:
        c (torch.Tensor): Context tensor in [totalC_local, hidden_size].
        scale (torch.Tensor): Scale tensor in [B, hidden_size].
        packed_seq_params (dict): Packed sequence parameters.
    """
    return _modulate(c, scale, packed_seq_params)


@nvtx_profiled_function
def fused_varlen_c_broadcast_mul(c: torch.Tensor, scale: torch.Tensor, packed_seq_params: dict) -> torch.Tensor:
    r"""Fused RMSNorm + batch-wise broadcast_mul for context modality.

    Args:
        c (torch.Tensor): Context tensor in [totalC_local, hidden_size].
        scale (torch.Tensor): Gate tensor in [B, hidden_size].
        packed_seq_params (dict): Packed sequence parameters.
    """
    return _gate(c, scale, packed_seq_params)


# ---- Video modality (batch-wise cu_seqlens) — single-t AdaLN ----
#
# This model keeps batch-wise AdaLN (single ``t`` -> [B, H] scale).  These
# batch-wise v/a wrappers mirror ``fused_varlen_c_*``: global cu_seqlens +
# unpadded ``valid_seqlens`` + [B, H] scale.


@nvtx_profiled_function
def fused_varlen_batch_v_modulate(v: torch.Tensor, scale: torch.Tensor, packed_seq_params: dict) -> torch.Tensor:
    r"""Fused RMSNorm + batch-wise modulate for video modality (single-t AdaLN).

    Args:
        v (torch.Tensor): Video tensor in [totalV_local, hidden_size].
        scale (torch.Tensor): Scale tensor in [B, hidden_size]. Batch-wise.
        packed_seq_params (dict): Packed sequence parameters.
    """
    return _modulate(v, scale, packed_seq_params)


@nvtx_profiled_function
def fused_varlen_batch_v_broadcast_mul(v: torch.Tensor, scale: torch.Tensor, packed_seq_params: dict) -> torch.Tensor:
    r"""Fused RMSNorm + batch-wise broadcast_mul for video modality (single-t AdaLN).

    Args:
        v (torch.Tensor): Video tensor in [totalV_local, hidden_size].
        scale (torch.Tensor): Gate tensor in [B, hidden_size]. Batch-wise.
        packed_seq_params (dict): Packed sequence parameters.
    """
    return _gate(v, scale, packed_seq_params)


# ---- Audio modality (batch-wise cu_seqlens) — single-t AdaLN ----


@nvtx_profiled_function
def fused_varlen_batch_a_modulate(a: torch.Tensor, scale: torch.Tensor, packed_seq_params: dict) -> torch.Tensor:
    r"""Fused RMSNorm + batch-wise modulate for audio modality (single-t AdaLN).

    Args:
        a (torch.Tensor): Audio tensor in [totalA_padded_local, hidden_size].
        scale (torch.Tensor): Scale tensor in [B, hidden_size]. Batch-wise.
        packed_seq_params (dict): Packed sequence parameters.
    """
    return _modulate(a, scale, packed_seq_params)


@nvtx_profiled_function
def fused_varlen_batch_a_broadcast_mul(a: torch.Tensor, scale: torch.Tensor, packed_seq_params: dict) -> torch.Tensor:
    r"""Fused RMSNorm + batch-wise broadcast_mul for audio modality (single-t AdaLN).

    Args:
        a (torch.Tensor): Audio tensor in [totalA_padded_local, hidden_size].
        scale (torch.Tensor): Gate tensor in [B, hidden_size]. Batch-wise.
        packed_seq_params (dict): Packed sequence parameters.
    """
    return _gate(a, scale, packed_seq_params)
