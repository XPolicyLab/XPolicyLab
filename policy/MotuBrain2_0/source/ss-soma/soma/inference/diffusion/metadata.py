"""Varlen metadata construction for embodied inference."""

from itertools import accumulate
from collections.abc import Sequence

import torch


def _to_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(values, dtype=torch.int32)


def _cumsum(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        list(accumulate(values, initial=0)),
        dtype=torch.int32,
    )


def get_caption_meta_dict(
    padded_captions: Sequence[torch.Tensor],
    caption_lengths: Sequence[int],
) -> dict[str, object]:
    """Build caption sequence metadata for one microbatch."""
    padded_lengths = [caption.shape[0] for caption in padded_captions]
    lengths = _to_tensor(caption_lengths)
    padded_lengths_tensor = _to_tensor(padded_lengths)
    cumulative_lengths = _cumsum(lengths)
    cumulative_padded_lengths = _cumsum(padded_lengths_tensor)
    return {
        "Cs": lengths,
        "maxC": lengths.max().item(),
        "cuCs": cumulative_lengths,
        "totalC": cumulative_lengths[-1].item(),
        "Cs_padded": padded_lengths_tensor,
        "maxC_padded": padded_lengths_tensor.max().item(),
        "cuCs_padded": cumulative_padded_lengths,
        "totalC_padded": cumulative_padded_lengths[-1].item(),
    }


def get_multiview_video_meta_dict(
    flat_view_latents: Sequence[torch.Tensor],
    view_counts: Sequence[int],
    rope_mode: str = "hw_offset",
) -> dict[str, object]:
    """Build video metadata for multiview: both per-view (RoPE) and per-sample (attention).

    Args:
        flat_view_latents: list of [C, F, H, W] tensors (all views, flattened)
        view_counts: list of int, views per sample, e.g. [3]
        rope_mode: ``hw_offset`` or ``per_view``, matching Motubrain training.
    """
    if rope_mode not in ("hw_offset", "per_view"):
        raise ValueError("multiview rope_mode must be 'hw_offset' or 'per_view'")
    frames = [latent.shape[1] for latent in flat_view_latents]
    heights = [latent.shape[2] for latent in flat_view_latents]
    widths = [latent.shape[3] for latent in flat_view_latents]
    view_tokens = [frame * height * width for frame, height, width in zip(frames, heights, widths)]
    frame_tensor = _to_tensor(frames)
    height_tensor = _to_tensor(heights)
    width_tensor = _to_tensor(widths)
    view_token_tensor = _to_tensor(view_tokens)

    metadata: dict[str, object] = {}
    # Per-view (for RoPE: each view has its own F/H/W)
    metadata["Fs_view"] = frame_tensor
    metadata["Hs_view"] = height_tensor
    metadata["Ws_view"] = width_tensor
    metadata["Vs_view"] = view_token_tensor
    metadata["Fs"] = frame_tensor
    metadata["Hs"] = height_tensor
    metadata["Ws"] = width_tensor
    metadata["cuVs_rope"] = _cumsum(view_tokens)

    # Per-sample (for attention, loss)
    metadata["NVs"] = _to_tensor(view_counts)
    sample_tokens = []
    offset = 0
    for num_views in view_counts:
        sample_tokens.append(sum(view_tokens[offset : offset + num_views]))
        offset += num_views
    sample_token_tensor = _to_tensor(sample_tokens)
    cumulative_sample_tokens = _cumsum(sample_tokens)
    metadata["Vs_sample"] = sample_token_tensor
    metadata["Vs"] = sample_token_tensor
    metadata["cuVs"] = cumulative_sample_tokens
    metadata["maxV"] = sample_token_tensor.max().item()
    metadata["totalV"] = cumulative_sample_tokens[-1].item()
    return metadata


def get_audio_meta_dict(
    padded_audio: Sequence[torch.Tensor],
    audio_lengths: Sequence[int],
) -> dict[str, object]:
    """Build action-token sequence metadata for one microbatch."""
    padded_lengths = [audio.shape[1] for audio in padded_audio]
    lengths = _to_tensor(audio_lengths)
    padded_lengths_tensor = _to_tensor(padded_lengths)
    cumulative_lengths = _cumsum(lengths)
    cumulative_padded_lengths = _cumsum(padded_lengths_tensor)
    return {
        "hasA": True,
        "As": lengths,
        "maxA": lengths.max().item(),
        "cuAs": cumulative_lengths,
        "totalA": cumulative_lengths[-1].item(),
        "As_padded": padded_lengths_tensor,
        "maxA_padded": padded_lengths_tensor.max().item(),
        "cuAs_padded": cumulative_padded_lengths,
        "totalA_padded": cumulative_padded_lengths[-1].item(),
    }
