"""Fixed observed RGB contexts, visibility masks, and per-plan KV prefill."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class RGBMemoryConfig:
    anchor_frames: int = 2
    recent_frames: int = 4
    raw_frame_stride: int = 4
    vae_temporal_stride: int = 4
    sparse_context_mode: str = "anchor_recent_current"
    sparse_recent_action_offset: int = 32
    sparse_frame_encoding: str = "independent"
    rgb_proprio_context: bool = False
    conditioning_latent_noise_prob: float = 0.0
    conditioning_latent_noise_stream: str = "rgb_depth"
    rgb_conditioning_latent_noise_prob: float | None = None
    depth_conditioning_latent_noise_prob: float | None = None

    def __post_init__(self) -> None:
        if (
            self.sparse_frame_encoding != "independent"
            or self.sparse_context_mode != "anchor_recent_current"
        ):
            raise ValueError("Sparse memory uses three independent observed frames.")
        for name in (
            "anchor_frames",
            "recent_frames",
            "raw_frame_stride",
            "vae_temporal_stride",
            "sparse_recent_action_offset",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.recent_frames < 3 or self.rgb_proprio_context:
            raise ValueError(
                "All three RGB observations must be visible, without state conditioning."
            )
        for name in (
            "conditioning_latent_noise_prob",
            "rgb_conditioning_latent_noise_prob",
            "depth_conditioning_latent_noise_prob",
        ):
            value = getattr(self, name)
            if value is not None and (not 0 <= float(value) <= 1):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.conditioning_latent_noise_stream not in {
            "rgb",
            "depth",
            "rgb_depth",
            "none",
        }:
            raise ValueError("Unknown conditioning noise stream")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def conditioning_noise_probability(self, stream: str) -> float:
        """Return the configured train-only corruption probability for a stream."""
        if stream == "rgb" and self.rgb_conditioning_latent_noise_prob is not None:
            return float(self.rgb_conditioning_latent_noise_prob)
        if stream == "depth" and self.depth_conditioning_latent_noise_prob is not None:
            return float(self.depth_conditioning_latent_noise_prob)
        return float(self.conditioning_latent_noise_prob)


def memory_mask(
    rgb_mask: torch.Tensor,
    latest: torch.Tensor,
    depth_mask: torch.Tensor,
    action_tokens: int,
) -> torch.Tensor:
    """PrismWAM chain: RGB observed memory -> Depth (including forecast) -> Action."""
    (batch, rgb_tokens, _) = rgb_mask.shape
    depth_tokens = int(depth_mask.shape[-1])
    prefix = int(latest.shape[-1])
    if depth_mask.dtype != torch.bool or depth_mask.shape != (
        depth_tokens,
        depth_tokens,
    ):
        raise ValueError("Depth self-attention must be a square boolean matrix")
    if not 0 < prefix <= rgb_tokens or action_tokens <= 0:
        raise ValueError("Invalid PrismWAM memory token counts")
    depth_end = rgb_tokens + depth_tokens
    result = torch.zeros(
        (batch, depth_end + action_tokens, depth_end + action_tokens),
        dtype=torch.bool,
        device=rgb_mask.device,
    )
    result[:, :rgb_tokens, :rgb_tokens] = rgb_mask
    result[:, rgb_tokens:depth_end, rgb_tokens:depth_end] = depth_mask
    result[:, rgb_tokens:depth_end, :prefix] = latest[:, None]
    result[:, depth_end:, rgb_tokens:] = True
    return result


def memory_suffix_mask(
    *,
    rgb_tokens: int,
    depth_mask: torch.Tensor,
    action_tokens: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Return the compact rectangular suffix mask ``[Q_suffix, K_total]``.

    The query rows are ``[Depth, Action]`` and key columns are
    ``[retained-RGB, Depth, Action]``.  It is exactly the suffix slice of
    :func:`memory_mask` when every retained RGB cache unit is visible to
    the current Depth state, but it avoids allocating the unused RGB-query
    rows and the square ``S x S`` container.
    """
    if depth_mask.ndim != 2 or depth_mask.shape[0] != depth_mask.shape[1]:
        raise ValueError("Depth mask must be square")
    (rgb_tokens, depth_tokens, action_tokens) = map(
        int, (rgb_tokens, depth_mask.shape[0], action_tokens)
    )
    if min(rgb_tokens, depth_tokens, action_tokens) <= 0:
        raise ValueError("Compact suffix mask dimensions must be positive")
    total_keys = rgb_tokens + depth_tokens + action_tokens
    suffix_queries = depth_tokens + action_tokens
    shape = (
        (batch_size, suffix_queries, total_keys)
        if batch_size is not None
        else (suffix_queries, total_keys)
    )
    result = torch.zeros(shape, dtype=torch.bool, device=depth_mask.device)
    result[..., :depth_tokens, :rgb_tokens] = True
    result[..., :depth_tokens, rgb_tokens : rgb_tokens + depth_tokens] = depth_mask
    action_start = depth_tokens
    result[..., action_start:, rgb_tokens : rgb_tokens + depth_tokens] = True
    result[..., action_start:, rgb_tokens + depth_tokens :] = True
    return result


def visible_history(query_frames, key_frames, config):
    (q, k) = (query_frames.unsqueeze(-1), key_frames.unsqueeze(-2))
    return (
        (k <= q)
        & (k >= 0)
        & (q >= 0)
        & ((k < config.anchor_frames) | (k >= q - config.recent_frames + 1))
    )


def pack_rgb_memory(pre, *, condition_frames, lengths, expert, future_frame_offset=2):
    """Pack observed visual tokens and an optional supervised endpoint."""
    tokens = pre["tokens"]
    (batch, seq_len, hidden) = tokens.shape
    per_frame = int(pre["meta"]["tokens_per_frame"])
    total_frames = int(pre["meta"]["grid_size"][0])
    if total_frames not in (condition_frames, condition_frames + 1):
        raise ValueError("Expected observed frames and at most one endpoint.")
    if seq_len != total_frames * per_frame or tuple(lengths.shape) != (batch,):
        raise ValueError("Invalid sparse RGB tensor geometry.")
    if bool(((lengths < 1) | (lengths > condition_frames)).any()):
        raise ValueError("Invalid observed lengths.")
    positions = pre["meta"]["temporal_rope_positions"][:condition_frames]
    if positions != list(range(positions[0], positions[0] + condition_frames)):
        raise ValueError("Sparse latent positions must be consecutive.")
    ids = torch.arange(condition_frames, device=tokens.device).repeat_interleave(
        per_frame
    ) + int(positions[0])
    valid = ids[None] < lengths[:, None] + int(positions[0])
    freqs = pre["freqs"]
    if freqs.ndim == 3:
        freqs = freqs[None].expand(batch, -1, -1, -1)
    if total_frames > condition_frames:
        freqs = freqs.clone()
        future = lengths - 1 + int(positions[0]) + future_frame_offset
        width = int(expert.freqs[0].shape[-1])
        freqs[:, -per_frame:, :, :width] = expert.freqs[0].to(tokens.device)[future][
            :, None, None, :
        ]
    return dict(
        tokens=tokens,
        freqs=freqs,
        t_mod=pre["t_mod"],
        context=pre["context"],
        context_mask=pre["context_mask"],
        frame_ids=ids,
        valid=valid,
        visual_indices=torch.arange(seq_len, device=tokens.device),
        condition_seq_len=condition_frames * per_frame,
        future_seq_len=(total_frames - condition_frames) * per_frame,
    )


def training_rgb_mask(packed, lengths, config):
    (frames, valid) = (packed["frame_ids"], packed["valid"])
    (batch, prefix) = valid.shape
    total = int(packed["tokens"].shape[1])
    mask = torch.zeros((batch, total, total), dtype=torch.bool, device=frames.device)
    clean = (
        visible_history(frames, frames, config)[None]
        & valid[:, None, :]
        & valid[:, :, None]
    )
    clean |= (
        torch.eye(prefix, dtype=torch.bool, device=frames.device)[None]
        & ~valid[:, :, None]
    )
    mask[:, :prefix, :prefix] = clean
    current = lengths - 1 + int(frames[0])
    latest = visible_history(current[:, None], frames[None], config).squeeze(1) & valid
    if total > prefix:
        mask[:, prefix:, :prefix] = latest[:, None, :]
        mask[:, prefix:, prefix:] = True
    return (mask, latest)


def append_rgb_kv(mot, packed, config, previous=None):
    """Build one fixed observed context; discard it at the next replanning call."""
    if previous is not None:
        raise ValueError(
            "Sparse memory is rebuilt per plan; persistent prefixes are unsupported."
        )
    if packed["future_seq_len"] or not bool(packed["valid"].all()):
        raise ValueError("Cache prefill requires valid observed tokens only.")
    ids = packed["frame_ids"]
    mask = visible_history(ids, ids, config)
    keep = visible_history(ids.new_tensor([int(ids[-1])]), ids, config).squeeze(0)
    (x, expert, layers) = (packed["tokens"], mot.mixtures["rgb"], [])
    context = {
        "cross_attention": {
            "text": {
                "context": packed["context"],
                "mask": packed["context_mask"],
                "scale": 1.0,
            }
        }
    }
    for layer in range(mot.num_layers):
        block = expert.blocks[layer]
        (q, k, v, residual, gate, shift, scale, gate_mlp, gc) = (
            mot._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=packed["freqs"],
                t_mod=packed["t_mod"],
            )
        )
        mixed = mot._mixed_attention(q_cat=q, k_cat=k, v_cat=v, attention_mask=mask)
        x = mot._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual,
            gate_msa=gate,
            shift_mlp=shift,
            scale_mlp=scale,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=gc,
            mixed_slice=mixed,
            context_payload=context,
        )
        layers.append(
            {
                "k": k[:, keep],
                "v": v[:, keep],
                "prefix_expert_name": "rgb",
                "prefix_seq_len": int(keep.sum()),
            }
        )
    return {"layers": layers, "frame_ids": ids[keep], "latest": int(ids[-1])}
