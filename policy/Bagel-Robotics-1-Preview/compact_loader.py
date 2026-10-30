"""Instantiate the dense compact video architecture before strict loading.

Adapted from Bagel-Robotics-1-Preview's inference-only checkpoint and structure helpers.
The hook is scoped to one model construction and never changes another policy.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from vpp2.models.wan21_14b.wan_video_dit import I2VCrossAttention


def _video_blocks(video):
    blocks = getattr(video, "blocks", None)
    if blocks is None or len(blocks) != 40:
        raise ValueError("Expected the 40-block VPP2 video expert")
    for block in blocks:
        if not isinstance(block.ffn, nn.Sequential) or len(block.ffn) != 3:
            raise ValueError("Unexpected video FFN")
        if not isinstance(block.cross_attn, I2VCrossAttention):
            raise ValueError("Unexpected video cross-attention")
    return blocks


def _reshape_video(video, metadata):
    blocks = _video_blocks(video)
    layers = metadata["layers"]
    if len(layers) != len(blocks):
        raise ValueError("Pruning metadata must describe all 40 blocks")
    with torch.device("meta"):
        for block, spec in zip(blocks, layers):
            width, heads = int(spec["ffn_width"]), int(spec["cross_heads"])
            if not 1 <= width <= 13824 or not 1 <= heads <= 40:
                raise ValueError("Invalid compact video layer shape")
            up, gelu, down = block.ffn
            block.ffn = nn.Sequential(
                nn.Linear(up.in_features, width, bias=up.bias is not None),
                deepcopy(gelu),
                nn.Linear(width, down.out_features, bias=down.bias is not None),
            )
            block.ffn_dim = width
            old = block.cross_attn
            block.cross_attn = I2VCrossAttention(
                hidden_dim=old.hidden_dim, attn_head_dim=old.attn_head_dim,
                num_heads=heads, eps=old.norm_q.eps, qk_norm=True,
            )
    count = sum(parameter.numel() for parameter in video.parameters())
    if count != metadata["compact_video_parameters"]:
        raise ValueError("Compact video parameter count differs from metadata")


@contextmanager
def compact_video_loading(video_path: Path, architecture_path: Path):
    """Reshape only this video expert before the upstream strict state load."""
    import vpp2.models.wan21_14b.helpers.loader as loader

    video_path = Path(video_path).resolve()
    metadata = json.loads(Path(architecture_path).read_text())
    original = loader._load_state_into_model
    used = False

    def patched(model, path, *args, **kwargs):
        nonlocal used
        paths = path if isinstance(path, (tuple, list)) else [path]
        if any(Path(item).resolve() == video_path for item in paths):
            if used:
                raise RuntimeError("Compact video checkpoint loaded twice")
            _reshape_video(model, metadata)
            used = True
        return original(model, path, *args, **kwargs)

    loader._load_state_into_model = patched
    try:
        yield
        if not used:
            raise RuntimeError("Compact video checkpoint was never loaded")
    finally:
        loader._load_state_into_model = original
