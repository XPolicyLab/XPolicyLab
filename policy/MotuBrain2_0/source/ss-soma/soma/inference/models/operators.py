"""Portable leaf operators used by Motubrain inference models.

Every operator here is plain PyTorch (plus the optional Triton kernels in
``soma.inference.models.pointwise``), so the deployment bundle runs without the
fused CUDA kernel packages the model is trained against.  Falcon's model code
imports these directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from soma.inference.types import RopeAxisConfig


def build_rope_cache(
    total_local_rows: int,
    head_dim: int,
    cu_tokens: torch.Tensor,
    axes_shapes: Sequence[torch.Tensor],
    axes_config: Sequence[RopeAxisConfig],
    axes_seqlen_offsets: Sequence[torch.Tensor | None] | None = None,
    cp_rank: int = 0,
    total_local: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build portable cosine, sine, and partner caches for ND RoPE."""
    device = cu_tokens.device
    num_axes = len(axes_config)
    if num_axes > 4:
        raise ValueError(
            f"build_rope_cache supports up to 4 axes, got {num_axes}",
        )
    if len(axes_shapes) != num_axes:
        raise ValueError("axes_shapes and axes_config must have equal length")
    if total_local <= 0:
        total_local = total_local_rows

    cumulative_tokens = cu_tokens.to(
        dtype=torch.int64,
        device=device,
    ).contiguous()
    batch_size = cumulative_tokens.numel() - 1
    global_start = cp_rank * total_local
    global_end = global_start + total_local_rows
    if (
        global_start < 0
        or global_end > int(cumulative_tokens[-1].item())
    ):
        raise ValueError(
            "build_rope_cache local range is outside cu_tokens: "
            f"[{global_start}, {global_end}) vs "
            f"total={int(cumulative_tokens[-1].item())}",
        )

    partner = torch.arange(head_dim, dtype=torch.int32, device=device)
    for axis in axes_config:
        start, end = axis["dim_range"]
        half = (end - start) // 2
        indices = torch.arange(half, dtype=torch.int32, device=device)
        partner[start : start + half] = indices + start + half
        partner[start + half : end] = indices + start

    covered_dims = sum(
        axis["dim_range"][1] - axis["dim_range"][0]
        for axis in axes_config
    )
    if covered_dims >= head_dim:
        cosine = torch.empty(
            total_local_rows,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        sine = torch.empty_like(cosine)
    else:
        cosine = torch.ones(
            total_local_rows,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        sine = torch.zeros_like(cosine)

    stacked_shapes = torch.stack(
        [shape.to(device=device, dtype=torch.int64) for shape in axes_shapes],
        dim=0,
    ).contiguous()
    stacked_offsets = torch.zeros(
        num_axes,
        batch_size,
        dtype=torch.int64,
        device=device,
    )
    if axes_seqlen_offsets is not None:
        if len(axes_seqlen_offsets) != num_axes:
            raise ValueError(
                "axes_seqlen_offsets and axes_config must have equal length",
            )
        for axis_index, offset in enumerate(axes_seqlen_offsets):
            if offset is not None:
                stacked_offsets[axis_index] = offset.to(
                    device=device,
                    dtype=torch.int64,
                )
    stacked_offsets = stacked_offsets.contiguous()

    global_positions = torch.arange(
        global_start,
        global_end,
        dtype=torch.int64,
        device=device,
    )
    batch_indices = torch.searchsorted(
        cumulative_tokens[1:],
        global_positions,
        right=True,
    )
    remainder = global_positions - cumulative_tokens[batch_indices]
    coordinates = [torch.empty_like(remainder) for _ in range(num_axes)]
    for axis_index in range(num_axes - 1, 0, -1):
        axis_size = stacked_shapes[axis_index, batch_indices]
        coordinates[axis_index] = remainder % axis_size
        remainder = remainder // axis_size
    coordinates[0] = remainder

    for axis_index, axis in enumerate(axes_config):
        coordinate = coordinates[axis_index]
        coordinate = coordinate + stacked_offsets[axis_index, batch_indices]
        start, end = axis["dim_range"]
        rotary_dim = end - start
        half = rotary_dim // 2
        dimension = torch.arange(half, dtype=torch.float32, device=device)
        position = coordinate.to(torch.float32).unsqueeze(1)
        position = position / float(axis.get("interpolate_factor", 1.0))
        frequency = position * torch.exp(
            -2.0
            * dimension.unsqueeze(0)
            / float(rotary_dim)
            * math.log(float(axis["base"])),
        )
        axis_cosine = torch.cos(frequency)
        axis_sine = torch.sin(frequency)
        cosine[:, start : start + half] = axis_cosine
        sine[:, start : start + half] = -axis_sine
        cosine[:, start + half : end] = axis_cosine
        sine[:, start + half : end] = axis_sine
    return cosine, sine, partner


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(value, (value.shape[-1],), eps=1e-6)


def _scaled_norm(
    value: torch.Tensor,
    scale: torch.Tensor,
    constant: float,
) -> torch.Tensor:
    if scale.ndim != 2 or scale.shape[0] != 1:
        raise ValueError(
            "inference normalization requires a single AdaLN sample",
        )
    normalized = _normalized(value)
    if (
        value.is_cuda
        and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        from soma.inference.models import pointwise

        return pointwise.scale_normalized(normalized, scale, constant, value.dtype)
    return (normalized * (constant + scale)).to(value.dtype)


def modulate(
    value: torch.Tensor,
    scale: torch.Tensor,
    metadata: dict[str, object],
) -> torch.Tensor:
    """Apply inference RMSNorm and AdaLN scaling."""
    del metadata
    return _scaled_norm(value, scale, 1.0)


def gate(
    value: torch.Tensor,
    scale: torch.Tensor,
    metadata: dict[str, object],
) -> torch.Tensor:
    """Apply inference RMSNorm and residual gating."""
    del metadata
    return _scaled_norm(value, scale, 0.0)


def chunk_contiguous_tanh(
    value: torch.Tensor,
    tanh_mask: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[N, 4H]`` into four contiguous lanes, tanh-ing masked lanes.

    Lane ``i`` is ``value[:, i * H:(i + 1) * H]`` and bit ``i`` of ``tanh_mask``
    — least significant bit first — applies ``tanh`` to that lane.  AdaLN calls
    this with ``tanh_mask=0b1010``, i.e. tanh on lanes 1 and 3, the two gates.

    This is the portable form of the fused CUDA chunk+tanh the training stack
    uses.  It drops that kernel's hardware preconditions (CUDA tensor, last
    dimension divisible by 32) and keeps only the shape contract, so the same
    code runs on CPU.
    """
    if value.ndim != 2:
        raise ValueError(
            f"chunk_contiguous_tanh expects a 2D tensor, got shape {tuple(value.shape)}",
        )
    if value.shape[1] % 4 != 0:
        raise ValueError(
            "chunk_contiguous_tanh expects a last dimension divisible by 4, "
            f"got {value.shape[1]}",
        )
    lanes = tuple(lane.contiguous() for lane in value.chunk(4, dim=1))
    return tuple(
        lane.tanh() if (tanh_mask >> index) & 1 else lane
        for index, lane in enumerate(lanes)
    )


def chunk_qkv(
    projections: Sequence[torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, ...]:
    """Split projected modalities into contiguous query, key, and value."""
    chunks: list[torch.Tensor] = []
    for projection in projections:
        reshaped = projection.reshape(
            projection.shape[0],
            num_heads,
            head_dim * 3,
        )
        chunks.extend(item.contiguous() for item in reshaped.chunk(3, dim=-1))
    return tuple(chunks)


def rmsnorm_cached_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    partner: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm and a precomputed RoPE cache using PyTorch operators."""

    def apply(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(
            value,
            (value.shape[-1],),
            weight=weight,
            eps=epsilon,
        )
        if (
            value.is_cuda
            and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and cosine.dtype == torch.float32
            and sine.dtype == torch.float32
        ):
            from soma.inference.models import pointwise

            return pointwise.rotate(normalized, cosine, sine, partner, value.dtype)
        paired = torch.index_select(normalized, -1, partner.long())
        rotated = (
            normalized.float() * cosine[:, None, :]
            + paired.float() * sine[:, None, :]
        )
        return rotated.to(value.dtype)

    return apply(query, query_weight), apply(key, key_weight)


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    **kwargs: object,
) -> torch.Tensor:
    """Run portable dense SDPA for the single-sequence inference setting."""
    del max_seqlen_q, max_seqlen_k, kwargs
    if cu_seqlens_q.numel() != 2 or cu_seqlens_k.numel() != 2:
        raise ValueError("inference attention requires exactly one sequence")
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    output = F.scaled_dot_product_attention(query, key, value)
    return output.squeeze(0).transpose(0, 1)


def unite_vac(
    video: torch.Tensor,
    action: torch.Tensor,
    caption: torch.Tensor,
    **metadata: object,
) -> torch.Tensor:
    """Pack video, action, and caption streams for one inference sample."""
    del metadata
    return torch.cat((video, action, caption), dim=0)


def split_vac(
    value: torch.Tensor,
    **metadata: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack video, action, and caption streams for one inference sample."""
    return value.split(
        (
            int(metadata["totalV"]),
            int(metadata["totalA_padded"]),
            int(metadata["totalC_padded"]),
        ),
        dim=0,
    )


def unite_vc(
    video: torch.Tensor,
    caption: torch.Tensor,
    **metadata: object,
) -> torch.Tensor:
    """Pack video and caption streams for one inference sample."""
    del metadata
    return torch.cat((video, caption), dim=0)


def split_vc(
    value: torch.Tensor,
    **metadata: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack video and caption streams for one inference sample."""
    return value.split(
        (
            int(metadata["totalV"]),
            int(metadata["totalC_padded"]),
        ),
        dim=0,
    )


def _ensure_dtype(value: Any, dtype: torch.dtype) -> Any:
    """Cast every tensor in a nested list/tuple/dict to ``dtype``."""
    if isinstance(value, list):
        return [_ensure_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_ensure_dtype(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _ensure_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value if value.dtype == dtype else value.to(dtype)
    raise ValueError(f"Unsupported type {type(value)}")


def ensure_bf16(value: Any) -> Any:
    """Return ``value`` with every nested tensor in bfloat16."""
    return _ensure_dtype(value, torch.bfloat16)


def ensure_fp32(value: Any) -> Any:
    """Return ``value`` with every nested tensor in float32."""
    return _ensure_dtype(value, torch.float32)
