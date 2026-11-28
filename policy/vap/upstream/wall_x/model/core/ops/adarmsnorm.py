"""Adaptive RMS normalization operator (fused variance+scale+shift+gate).

Used by pi05's expert post_attention_layernorm where modulation comes from
``self.dense(cond)`` and gets chunk'd into (scale, shift, gate).
"""

import logging

import torch
from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)


class AdaRMSNormOp(OpsProxy):
    """Adaptive RMSNorm: adarmsnorm(x, modulation, eps) -> (out, gate).

    Args:
        x: input [..., H]
        modulation: [..., 3H] or [B, 3H] when x is [B, S, H]. Will be
            chunked into (scale, shift, gate) along the last dim.
        eps: epsilon for variance.

    Returns:
        out: x normalized then ``(1+scale)*x + shift`` matching x shape.
        gate: extracted from modulation, broadcast-shape to align with x
            (so subsequent gated residual ops work directly).
    """

    @property
    def _xcompute_name(self):
        return "adarmsnorm"

    def _pytorch_fallback(self, x, modulation, eps=1e-6):
        """Pure PyTorch reference matching gemma adaptive RMSNorm math."""
        dtype = x.dtype
        # Variance + rsqrt in fp32 (mirrors GemmaRMSNorm._norm).
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + eps)
        if x.dim() == 3 and modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        out = normed * (1.0 + scale.to(torch.float32)) + shift.to(torch.float32)
        return out.to(dtype), gate.to(dtype)


adarmsnorm = AdaRMSNormOp()
