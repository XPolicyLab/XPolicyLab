"""Shape-explicit initialization helpers for the VAP action expert."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_matrix(weight: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    """Resize a 2D weight and preserve variance when changing input width."""

    old_out, old_in = weight.shape
    resized = weight.detach().to(dtype=torch.float32, device="cpu")
    if old_in != in_dim:
        resized = F.interpolate(
            resized.unsqueeze(0), size=in_dim, mode="linear", align_corners=False
        ).squeeze(0)
    if old_out != out_dim:
        resized = F.interpolate(
            resized.t().unsqueeze(0),
            size=out_dim,
            mode="linear",
            align_corners=False,
        ).squeeze(0).t().contiguous()
    if old_in != in_dim:
        resized = resized * (old_in / in_dim) ** 0.5
    return resized


def resize_vector(vector: torch.Tensor, dim: int) -> torch.Tensor:
    source = vector.detach().to(dtype=torch.float32, device="cpu")
    if source.numel() == dim:
        return source.clone()
    return F.interpolate(
        source.reshape(1, 1, -1), size=dim, mode="linear", align_corners=False
    ).reshape(-1)


@torch.no_grad()
def copy_linear(dst: nn.Linear, src: nn.Linear, *, allow_resize: bool) -> None:
    if not allow_resize and dst.weight.shape != src.weight.shape:
        raise ValueError(
            f"linear weight mismatch: dst={tuple(dst.weight.shape)}, "
            f"src={tuple(src.weight.shape)}"
        )
    weight = resize_matrix(src.weight, *dst.weight.shape)
    dst.weight.copy_(weight.to(device=dst.weight.device, dtype=dst.weight.dtype))
    if dst.bias is not None:
        if src.bias is None:
            dst.bias.zero_()
        else:
            bias = resize_vector(src.bias, dst.bias.numel())
            dst.bias.copy_(bias.to(device=dst.bias.device, dtype=dst.bias.dtype))


@torch.no_grad()
def copy_norm(dst: nn.Module, src: nn.Module, *, allow_resize: bool) -> None:
    for name in ("weight", "bias"):
        dst_value = getattr(dst, name, None)
        src_value = getattr(src, name, None)
        if dst_value is None or src_value is None:
            continue
        if not allow_resize and dst_value.shape != src_value.shape:
            raise ValueError(
                f"norm {name} mismatch: dst={tuple(dst_value.shape)}, "
                f"src={tuple(src_value.shape)}"
            )
        value = resize_vector(src_value, dst_value.numel()).reshape(dst_value.shape)
        dst_value.copy_(value.to(device=dst_value.device, dtype=dst_value.dtype))


__all__ = ["copy_linear", "copy_norm", "resize_matrix", "resize_vector"]
