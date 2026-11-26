"""Fused pointwise inference operations with explicit IEEE rounding."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rotate_kernel(
    value,
    cosine,
    sine,
    partner,
    output,
    elements: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    column = offsets % width
    token = offsets // (heads * width)
    paired_column = tl.load(partner + column)
    left = tl.load(value + offsets, mask, other=0).to(tl.float32)
    right = tl.load(value + offsets - column + paired_column, mask, other=0).to(tl.float32)
    cos = tl.load(cosine + token * width + column, mask, other=0)
    sin = tl.load(sine + token * width + column, mask, other=0)
    result = libdevice.add_rn(libdevice.mul_rn(left, cos), libdevice.mul_rn(right, sin))
    tl.store(output + offsets, result, mask)


@triton.jit
def _scale_kernel(
    value, scale, output, elements: tl.constexpr, width: tl.constexpr, constant: tl.constexpr, block: tl.constexpr
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    factor = tl.load(scale + offsets % width).to(tl.float32)
    factor = libdevice.add_rn(factor, constant).to(scale.dtype.element_ty).to(tl.float32)
    item = tl.load(value + offsets, mask, other=0).to(tl.float32)
    tl.store(output + offsets, libdevice.mul_rn(item, factor), mask)


def rotate(
    normalized: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, partner: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Match the separate FP32 multiply/add and final cast in portable RoPE."""
    value = normalized.contiguous()
    output = torch.empty(value.shape, device=value.device, dtype=dtype)
    if value.numel():
        _rotate_kernel[(triton.cdiv(value.numel(), 1024),)](
            value,
            cosine.contiguous(),
            sine.contiguous(),
            partner.contiguous(),
            output,
            value.numel(),
            value.shape[-2],
            value.shape[-1],
            1024,
            enable_fp_fusion=False,
        )
    return output


def scale_normalized(
    normalized: torch.Tensor, scale: torch.Tensor, constant: float, dtype: torch.dtype
) -> torch.Tensor:
    """Round the scale addition before multiplication, as PyTorch does."""
    value = normalized.contiguous()
    output = torch.empty(value.shape, device=value.device, dtype=dtype)
    if value.numel():
        _scale_kernel[(triton.cdiv(value.numel(), 1024),)](
            value,
            scale.contiguous(),
            output,
            value.numel(),
            value.shape[-1],
            constant,
            1024,
            enable_fp_fusion=False,
        )
    return output
