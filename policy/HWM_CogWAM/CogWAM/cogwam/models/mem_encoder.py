"""Opt-in MEM-style space-time attention for the RynnBrain vision tower.

The adapter borrows HY-Embodied's low-overhead MEM idea without changing the
Qwen3.5 checkpoint ABI: selected native vision blocks keep their existing
Q/K/V, output projection, norms, and MLP, while their forward method performs
causal temporal mixing followed by spatial attention.  The only temporal
embedding is a fixed sinusoid, so enabling this module adds no parameters or
state-dict entries.  After the vision tower, past-frame tokens are discarded;
only the temporally enriched current-frame tokens enter Qwen's language model.

Nothing is patched unless ``framework.qwenvl.mem_vision_encoder.enabled`` is
true.  This is deliberately strict: an enabled model must receive consecutive,
same-resolution image groups of exactly ``num_frames`` frames.
"""

from __future__ import annotations

import importlib
import math
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


def _config_value(config: Any, name: str, default: Any) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _fixed_time_embedding(
    num_frames: int,
    hidden_size: int,
    *,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """HY-style fixed sinusoid with e(0)=0 and no persistent buffer."""

    steps = torch.arange(num_frames, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
        * (-math.log(base) / hidden_size)
    )
    embedding = torch.empty(
        num_frames,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    embedding[:, 0::2] = torch.sin(steps * frequencies)
    embedding[:, 1::2] = torch.cos(steps * frequencies) - 1.0
    return embedding.to(dtype=dtype)


def _mem_vision_block_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **_kwargs,
) -> torch.Tensor:
    """Parameter-free temporal-then-spatial attention on packed image groups."""

    del rotary_pos_emb  # Qwen3.5 publishes the already materialized cos/sin pair.
    if hidden_states.ndim != 2:
        raise ValueError(
            "Rynn MEM expects flattened Qwen vision tokens [S,D], got "
            f"{tuple(hidden_states.shape)}"
        )
    if position_embeddings is None:
        raise ValueError("Rynn MEM requires Qwen3.5 vision position_embeddings")
    if not torch.is_tensor(cu_seqlens) or cu_seqlens.ndim != 1:
        raise ValueError("Rynn MEM requires packed vision cu_seqlens")

    num_frames = int(self._cogwam_mem_num_frames)
    segment_count = int(cu_seqlens.numel()) - 1
    total_tokens, hidden_size = hidden_states.shape
    if segment_count <= 0 or segment_count % num_frames:
        raise ValueError(
            "Rynn MEM image packing must contain complete temporal groups: "
            f"segments={segment_count}, num_frames={num_frames}"
        )
    if total_tokens % segment_count:
        raise ValueError(
            "Rynn MEM requires the same vision-token count in every frame: "
            f"tokens={total_tokens}, segments={segment_count}"
        )
    tokens_per_frame = total_tokens // segment_count
    expected_cu = torch.arange(
        segment_count + 1,
        device=cu_seqlens.device,
        dtype=cu_seqlens.dtype,
    ) * tokens_per_frame
    if not torch.equal(cu_seqlens, expected_cu):
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
        raise ValueError(
            "Rynn MEM requires same-resolution consecutive frames; packed token "
            f"lengths are {lengths}"
        )

    batch_size = segment_count // num_frames
    time_embedding = _fixed_time_embedding(
        num_frames,
        hidden_size,
        base=float(self._cogwam_mem_time_embed_base),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    normalized = self.norm1(
        hidden_states.view(
            batch_size,
            num_frames,
            tokens_per_frame,
            hidden_size,
        )
        + time_embedding.view(1, num_frames, 1, hidden_size)
    ).view(total_tokens, hidden_size)

    attention = self.attn
    num_heads = int(attention.num_heads)
    query, key, value = (
        attention.qkv(normalized)
        .reshape(total_tokens, 3, num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query, key = self._cogwam_mem_apply_rotary(query, key, cos, sin)
    head_dim = query.shape[-1]

    def to_temporal(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.view(
                batch_size,
                num_frames,
                tokens_per_frame,
                num_heads,
                head_dim,
            )
            .permute(0, 2, 3, 1, 4)
            .reshape(batch_size * tokens_per_frame, num_heads, num_frames, head_dim)
        )

    dropout = (
        float(getattr(attention, "attention_dropout", 0.0))
        if self.training
        else 0.0
    )
    mixed_value = F.scaled_dot_product_attention(
        to_temporal(query),
        to_temporal(key),
        to_temporal(value),
        dropout_p=dropout,
        is_causal=True,
        scale=float(getattr(attention, "scaling", head_dim**-0.5)),
    )
    mixed_value = (
        mixed_value.view(
            batch_size,
            tokens_per_frame,
            num_heads,
            num_frames,
            head_dim,
        )
        .permute(0, 3, 2, 1, 4)
        .reshape(segment_count, num_heads, tokens_per_frame, head_dim)
    )

    query = query.view(
        segment_count, tokens_per_frame, num_heads, head_dim
    ).permute(0, 2, 1, 3)
    key = key.view(
        segment_count, tokens_per_frame, num_heads, head_dim
    ).permute(0, 2, 1, 3)
    spatial_output = F.scaled_dot_product_attention(
        query,
        key,
        mixed_value,
        dropout_p=dropout,
        is_causal=False,
        scale=float(getattr(attention, "scaling", head_dim**-0.5)),
    )
    spatial_output = (
        spatial_output.permute(0, 2, 1, 3)
        .reshape(total_tokens, hidden_size)
        .contiguous()
    )
    hidden_states = hidden_states + attention.proj(spatial_output)
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


def _qwen35_visual(model):
    backbone = getattr(model, "model", None)
    candidates = (
        getattr(backbone, "visual", None),
        getattr(model, "visual", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "blocks"):
            return candidate
    raise RuntimeError(
        "Rynn MEM could not locate Qwen3.5 vision blocks at model.model.visual"
    )


def _mem_vision_model_forward(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
):
    """Encode K frames but expose only current-frame tokens to the language model.

    ``grid_thw`` remains the processor's current-image grid so Qwen's language
    RoPE and image placeholders keep their native one-image contract.  The
    framework supplies ``_cogwam_mem_active_grid_thw`` only for this vision
    call; the original vision forward sees the full K-frame packed grid.
    """

    mem_grid = getattr(self, "_cogwam_mem_active_grid_thw", None)
    if mem_grid is None:
        raise RuntimeError(
            "Rynn MEM vision forward is missing its full-frame grid context"
        )
    num_frames = int(self._cogwam_mem_num_frames)
    if mem_grid.ndim != 2 or mem_grid.shape[-1] != 3:
        raise ValueError(
            f"Rynn MEM grid_thw must be [B*K,3], got {tuple(mem_grid.shape)}"
        )
    segment_count = int(mem_grid.shape[0])
    if segment_count <= 0 or segment_count % num_frames:
        raise ValueError(
            "Rynn MEM full-frame grid must contain complete groups: "
            f"segments={segment_count}, frames={num_frames}"
        )
    if not bool((mem_grid[:, 0] == 1).all()):
        raise ValueError(
            "Rynn MEM expects separately packed still images with temporal grid 1"
        )
    batch_size = segment_count // num_frames
    if grid_thw.ndim != 2 or tuple(grid_thw.shape) != (batch_size, 3):
        raise ValueError(
            "Rynn MEM language-side current grid must be [B,3], got "
            f"{tuple(grid_thw.shape)} for batch={batch_size}"
        )

    frame_hw = mem_grid[:, 1:]
    first_hw = frame_hw[::num_frames]
    expected_hw = first_hw.repeat_interleave(num_frames, dim=0)
    if not torch.equal(frame_hw, expected_hw):
        raise ValueError(
            "Rynn MEM requires identical processed H/W grids within each frame group"
        )
    if not torch.equal(grid_thw[:, 1:], first_hw):
        raise ValueError(
            "Rynn MEM current-image placeholder grid does not match the frame groups"
        )

    outputs = self._cogwam_mem_original_forward(
        hidden_states,
        grid_thw=mem_grid,
        **kwargs,
    )
    raw_tokens_per_frame = int(first_hw[0].prod().item())
    if not bool((first_hw == first_hw[0]).all()):
        raise ValueError(
            "Rynn MEM currently requires one common processed resolution per batch"
        )
    merge_size = int(getattr(self, "spatial_merge_size", 1))
    merge_unit = merge_size * merge_size
    if raw_tokens_per_frame % merge_unit:
        raise ValueError(
            "Rynn MEM raw token count must be divisible by the spatial merge unit"
        )
    merged_tokens_per_frame = raw_tokens_per_frame // merge_unit

    pooled = outputs.pooler_output
    expected_pooled = batch_size * num_frames * merged_tokens_per_frame
    if pooled.ndim != 2 or int(pooled.shape[0]) != expected_pooled:
        raise ValueError(
            "Rynn MEM vision pooler output does not match its full-frame grid: "
            f"shape={tuple(pooled.shape)}, expected_tokens={expected_pooled}"
        )
    outputs.pooler_output = (
        pooled.view(
            batch_size,
            num_frames,
            merged_tokens_per_frame,
            pooled.shape[-1],
        )[:, -1]
        .reshape(batch_size * merged_tokens_per_frame, pooled.shape[-1])
        .contiguous()
    )

    raw = getattr(outputs, "last_hidden_state", None)
    expected_raw = batch_size * num_frames * raw_tokens_per_frame
    if torch.is_tensor(raw):
        if raw.ndim != 2 or int(raw.shape[0]) != expected_raw:
            raise ValueError(
                "Rynn MEM raw vision output does not match its full-frame grid: "
                f"shape={tuple(raw.shape)}, expected_tokens={expected_raw}"
            )
        outputs.last_hidden_state = (
            raw.view(
                batch_size,
                num_frames,
                raw_tokens_per_frame,
                raw.shape[-1],
            )[:, -1]
            .reshape(batch_size * raw_tokens_per_frame, raw.shape[-1])
            .contiguous()
        )
    return outputs


def apply_rynn_mem_encoder_patch(model, mem_config: Any) -> dict[str, Any]:
    """Install the opt-in zero-parameter MEM forward on selected blocks."""

    enabled = bool(_config_value(mem_config, "enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "num_frames": 1,
            "spacetime_layer_stride": 0,
            "patched_blocks": 0,
        }

    num_frames = int(_config_value(mem_config, "num_frames", 6))
    stride = int(_config_value(mem_config, "spacetime_layer_stride", 4))
    time_embed_base = float(_config_value(mem_config, "time_embed_base", 100.0))
    if num_frames < 2:
        raise ValueError(f"Rynn MEM num_frames must be >=2, got {num_frames}")
    if stride <= 0:
        raise ValueError(
            f"Rynn MEM spacetime_layer_stride must be positive, got {stride}"
        )
    if time_embed_base <= 1.0:
        raise ValueError(
            f"Rynn MEM time_embed_base must be >1, got {time_embed_base}"
        )

    visual = _qwen35_visual(model)
    blocks = list(visual.blocks)
    if not blocks:
        raise RuntimeError("Rynn MEM found an empty Qwen3.5 vision tower")
    patched_indices = list(range(stride - 1, len(blocks), stride))
    if not patched_indices:
        raise ValueError(
            "Rynn MEM stride does not select any vision block: "
            f"depth={len(blocks)}, stride={stride}"
        )

    for index in patched_indices:
        block = blocks[index]
        if bool(getattr(block, "_cogwam_mem_enabled", False)):
            if (
                int(block._cogwam_mem_num_frames) != num_frames
                or float(block._cogwam_mem_time_embed_base) != time_embed_base
            ):
                raise RuntimeError(
                    f"Qwen vision block {index} already has a different MEM contract"
                )
            continue
        modeling_module = importlib.import_module(block.__class__.__module__)
        apply_rotary = getattr(
            modeling_module,
            "apply_rotary_pos_emb_vision",
            None,
        )
        if not callable(apply_rotary):
            raise RuntimeError(
                "Rynn MEM requires Qwen3.5 apply_rotary_pos_emb_vision"
            )
        # Plain Python attributes/methods only: no nn.Module/Parameter/Buffer is
        # registered, so the checkpoint tensor layout remains byte-for-byte ABI
        # compatible with the native Rynn vision tower.
        block._cogwam_mem_enabled = True
        block._cogwam_mem_num_frames = num_frames
        block._cogwam_mem_time_embed_base = time_embed_base
        block._cogwam_mem_apply_rotary = apply_rotary
        block.forward = MethodType(_mem_vision_block_forward, block)

    visual._cogwam_mem_enabled = True
    visual._cogwam_mem_num_frames = num_frames
    visual._cogwam_mem_spacetime_layer_stride = stride
    if not hasattr(visual, "_cogwam_mem_original_forward"):
        visual._cogwam_mem_original_forward = visual.forward
        visual.forward = MethodType(_mem_vision_model_forward, visual)
    return {
        "enabled": True,
        "num_frames": num_frames,
        "spacetime_layer_stride": stride,
        "time_embed_base": time_embed_base,
        "patched_blocks": len(patched_indices),
        "patched_indices": patched_indices,
        "added_parameters": 0,
        "language_frame_tokens": "current_only",
        "output_token_policy": "current_only",
    }
