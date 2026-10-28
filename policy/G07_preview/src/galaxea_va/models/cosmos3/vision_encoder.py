# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modified by Galaxea AI, 2026, for inference-only use in G07-preview.
# Licensed under the Apache License, Version 2.0; see LICENSE and THIRD_PARTY_NOTICES.md
# in the G07_preview directory.
"""Vendored ``Qwen3VLVisionModel`` (Cosmos3-Nano understanding-tower vision encoder).

Behaviour-preserving copy of ``transformers.models.qwen3_vl.modeling_qwen3_vl``
(the ``Qwen3VLVisionModel`` half only) + the three ``transformers.vision_utils``
helpers it needs, so fastwam can run the encoder **without** installing/upgrading
``transformers`` (the repo pins 4.49.0, which predates ``qwen3_vl``).

Submodule / parameter names are kept identical to the published Cosmos3-Nano
``vision_encoder/model.safetensors`` (351 keys: ``patch_embed.proj``, ``pos_embed``,
``blocks.N.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}``,
``merger.{norm,linear_fc1,linear_fc2}``, ``deepstack_merger_list.N.*``) so the
state_dict loads directly.

Given ``(pixel_values [seq, C*tps*ps*ps], grid_thw [N,3])`` the model returns
``(merged_embeds [num_merged, out_hidden], deepstack_features list)``.
``merged_embeds`` is what gets scattered into the reasoner (und) sequence at
image-placeholder positions; ``deepstack_features`` are the optional early-layer
injections (unused by the first reasoner-integration cut).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn



@dataclass
class Qwen3VLVisionConfig:
    hidden_size: int = 1152
    depth: int = 27
    num_heads: int = 16
    intermediate_size: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (8, 16, 24)
    hidden_act: str = "gelu_pytorch_tanh"

    @classmethod
    def from_json(cls, path: str) -> Qwen3VLVisionConfig:
        with open(path) as f:
            cfg = json.load(f)
        return cls(
            hidden_size=cfg["hidden_size"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            intermediate_size=cfg["intermediate_size"],
            in_channels=cfg.get("in_channels", 3),
            patch_size=cfg["patch_size"],
            temporal_patch_size=cfg["temporal_patch_size"],
            spatial_merge_size=cfg["spatial_merge_size"],
            out_hidden_size=cfg["out_hidden_size"],
            num_position_embeddings=cfg["num_position_embeddings"],
            deepstack_visual_indexes=tuple(cfg["deepstack_visual_indexes"]),
            hidden_act=cfg.get("hidden_act", "gelu_pytorch_tanh"),
        )


# ----------------------------------------------------------------- vision_utils
# (standalone tensor builders, copied from transformers.vision_utils; no weights)


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """`(num_images,3)` -> `(total_patches+1,)` int32 cumulative segment boundaries."""
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """`(num_images,3)` -> `(total_tokens,2)` long (row,col) ids for vision RoPE."""
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        t, h, w, merge = int(t), int(h), int(w), int(spatial_merge_size)
        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear interpolation corner indices/weights into the `pos_embed` table."""
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])
    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights


# --------------------------------------------------------------------- modules


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        # config hidden_act == "gelu_pytorch_tanh"
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.in_channels = cfg.in_channels
        self.temporal_patch_size = cfg.temporal_patch_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.hidden_size
        kernel = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = cfg.hidden_size * (cfg.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else cfg.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()  # NOTE: merger uses exact GELU (not tanh)
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def forward(self, x):
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )  # each [seq, heads, head_dim]
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # [seq, heads, hd] -> [1, heads, seq, hd]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        # Full (non-causal) attention within each image segment (split by cu_seqlens).
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(q, lengths, dim=2)
        k_splits = torch.split(k, lengths, dim=2)
        v_splits = torch.split(v, lengths, dim=2)
        outs = [
            F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None, scale=self.scaling)
            for qs, ks, vs in zip(q_splits, k_splits, v_splits)
        ]
        attn_output = torch.cat(outs, dim=2)  # [1, heads, seq, hd]
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(cfg)
        self.mlp = Qwen3VLVisionMLP(cfg)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    """Vendored ``Qwen3VLVisionModel``: (pixel_values, grid_thw) -> (merged, deepstack)."""

    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes]
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """Args: hidden_states `[seq, C*tps*ps*ps]`, grid_thw `[N,3]` (t,h,w in patch units).

        Returns: (merged `[num_merged, out_hidden]`, deepstack_features list).
        """
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side, spatial_merge_size=self.spatial_merge_size
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self.deepstack_merger_list[idx](hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack_feature_lists


# ---------------------------------------------------------------- preprocessing
# Port of Qwen2VLImageProcessor (the image processor Cosmos3-Nano uses:
# preprocessor_config.json -> patch 16 / temporal_patch 2 / merge 2 / mean=std=0.5,
# size {shortest_edge: 65536, longest_edge: 16777216}). Resize matches the real
# TorchvisionBackend (tvF.resize BICUBIC, antialias=True).


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int = 65536, max_pixels: int = 16777216
) -> tuple[int, int]:
    """Round H,W to multiples of ``factor`` keeping pixels in [min,max] and aspect ~constant."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = (height * width / max_pixels) ** 0.5
        h_bar = max(factor, int((height / beta) // factor) * factor)
        w_bar = max(factor, int((width / beta) // factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = -(-int(height * beta) // factor) * factor  # ceil-div
        w_bar = -(-int(width * beta) // factor) * factor
    return h_bar, w_bar


def preprocess_image(
    image: torch.Tensor,
    cfg: Qwen3VLVisionConfig,
    *,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
    image_mean: float = 0.5,
    image_std: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw image ``[3,H,W]`` in ``[0,1]`` -> ``(pixel_values [N, C*tp*p*p], grid_thw [[1,gh,gw]])``.

    Reproduces Qwen2VLImageProcessor: smart_resize (factor=patch*merge) -> BICUBIC resize
    (antialias) -> normalize ``(x-mean)/std`` -> merge-block-major patch flatten with the
    single image duplicated across ``temporal_patch_size``. The output patch order matches
    what ``Qwen3VLVisionPatchMerger`` expects (4 consecutive rows = one 2x2 merge block).
    """
    import torchvision.transforms.functional as tvF

    if image.ndim != 3 or image.shape[0] != cfg.in_channels:
        raise ValueError(f"`image` must be [{cfg.in_channels},H,W], got {tuple(image.shape)}")
    p, m, tp, c = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size, cfg.in_channels
    factor = p * m
    _, h, w = image.shape
    rh, rw = smart_resize(h, w, factor, min_pixels, max_pixels)
    img = image.to(torch.float32)
    if (rh, rw) != (h, w):
        img = tvF.resize(img, [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    img = (img - image_mean) / image_std  # [C, rh, rw]

    grid_h, grid_w = rh // p, rw // p
    x = img.reshape(1, c, grid_h // m, m, p, grid_w // m, m, p)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)  # [1, gh/m, gw/m, m, m, C, p, p]
    x = (
        x.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, tp, -1, -1)
        .reshape(grid_h * grid_w, c * tp * p * p)
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return x.contiguous(), grid_thw


def load_cosmos3_vision_encoder(
    root: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Qwen3VLVisionModel:
    """Build the vendored vision model and load ``vision_encoder/model.safetensors`` (351 keys)."""
    ve_dir = os.path.join(root, "vision_encoder")
    cfg = Qwen3VLVisionConfig.from_json(os.path.join(ve_dir, "config.json"))
    with torch.device("meta"):
        model = Qwen3VLVisionModel(cfg)
    model = model.to(dtype)
    model = model.to_empty(device=device)
    # rotary inv_freq is non-persistent; recompute after to_empty.
    hd = cfg.hidden_size // cfg.num_heads
    dim = hd // 2
    model.rotary_pos_emb.inv_freq = (
        1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    ).to(device=device)

    state = load_file(os.path.join(ve_dir, "model.safetensors"), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=False)
    missing = {m for m in missing if not m.endswith("rotary_pos_emb.inv_freq")}
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Cosmos3 vision_encoder load mismatch.\n  missing={sorted(missing)[:8]}\n  unexpected={sorted(unexpected)[:8]}"
        )
    return model.to(device=device, dtype=dtype)
