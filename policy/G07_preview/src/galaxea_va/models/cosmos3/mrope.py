# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# Modified by Galaxea AI, 2026, for inference-only use in G07-preview.
# Licensed under the Apache License, Version 2.0; see LICENSE and THIRD_PARTY_NOTICES.md
# in the G07_preview directory.
"""Independent 3D mRoPE position-id builders (from pipeline_cosmos3_omni.py).

Cosmos3 uses interleaved 3D (T/H/W) mRoPE. Text tokens share one increasing id on
all 3 axes; VAE/action tokens use (frame, patch_h, patch_w). This module builds
positions for individual segments only. FastWAM variants own segment ordering,
offset relationships, and concatenation.
"""

from __future__ import annotations

import math

import torch


def build_text_position_ids(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    """All 3 axes share one increasing id starting at ``temporal_offset``."""
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()  # [3, num_tokens]
    return mrope_ids, temporal_offset + num_tokens


def build_vae_position_ids(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """3D mRoPE ids for VAE vision / action tokens (image/video/action latents)."""
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_temporal_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_temporal_offset


def build_und_position_ids(
    *,
    text_len: int,
    n_img: int = 0,
    image_grid_thw=None,
    spatial_merge_size: int = 1,
    use_float_positions: bool = False,
    device,
) -> tuple[torch.Tensor, int | float]:
    """3D mRoPE ids for a text-only or first-frame-vision understanding segment.

    The multimodal branch ports Cosmos3/Qwen3-VL ``get_rope_index`` for the layout
    ``[vision_start(1) | image(n_img) | vision_end(1) | text(text_len)]``:
    text runs advance all 3 axes diagonally; the image block gets a compact 3D grid
    ``(t=cur, h=cur+row, w=cur+col)`` over the LLM grid ``(gh//m, gw//m)``, and after
    the image the running position advances by ``max(gh, gw)//m`` (NOT by token count).

    Args:
        text_len: number of trailing text tokens (padded prompt length).
        n_img: number of merged vision tokens; zero selects the text-only layout.
        image_grid_thw: ``(t, gh, gw)`` in patch units, required when ``n_img > 0``.
        spatial_merge_size: vision merge size (2 for Cosmos3-Nano).
        use_float_positions: return float positions for FPS-modulated joint sequences.

    Returns:
        ``(position_ids, next_offset)`` for this segment only.
    """
    if n_img == 0:
        position_ids, next_offset = build_text_position_ids(
            text_len,
            0,
            use_float_positions=use_float_positions,
        )
        return position_ids.to(device), next_offset
    if image_grid_thw is None:
        raise ValueError("`image_grid_thw` is required when `n_img > 0`.")

    m = int(spatial_merge_size)
    t, gh, gw = (int(x) for x in image_grid_thw)
    lt, lh, lw = t, gh // m, gw // m
    if lt * lh * lw != int(n_img):
        raise ValueError(f"n_img={n_img} != llm grid {lt}*{lh}*{lw}={lt * lh * lw}")

    parts = []
    cur = 0
    # vision_start (single text token)
    parts.append(torch.arange(1, device=device).view(1, -1).expand(3, -1) + cur)
    cur += 1
    # image block: compact 3D grid starting at `cur`
    pw = (torch.arange(lw, device=device) + cur).repeat(lh * lt)
    ph = (torch.arange(lh, device=device) + cur).repeat_interleave(lw).repeat(lt)
    pt = torch.arange(lt, device=device).repeat_interleave(lh * lw) + cur
    parts.append(torch.stack([pt, ph, pw], dim=0))
    cur += max(gh, gw) // m
    # vision_end + text (contiguous text run)
    tail = 1 + int(text_len)
    parts.append(torch.arange(tail, device=device).view(1, -1).expand(3, -1) + cur)
    cur += tail

    position_ids = torch.cat(parts, dim=1).to(torch.long)
    if use_float_positions:
        position_ids = position_ids.to(torch.float32)
    return position_ids, cur
