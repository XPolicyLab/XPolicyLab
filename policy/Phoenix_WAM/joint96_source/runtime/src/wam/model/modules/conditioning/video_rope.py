"""Temporal RoPE helpers shared by video and memory streams."""

import torch

from ...backbones.wan22.wan_video_dit import build_3d_rope_freqs


def build_video_time_only_freqs(
    model, t_ids: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    t_ids = normalize_video_rope_time_ids(t_ids, device=device)
    neutral_h = torch.zeros_like(t_ids)
    neutral_w = torch.zeros_like(t_ids)
    return build_3d_rope_freqs(
        model.video_expert.rope_freq_bases_3d,
        t_ids,
        neutral_h,
        neutral_w,
        device=device,
    )


def normalize_video_rope_time_ids(
    t_ids: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Move video positions without imposing a finite lookup-table limit."""
    return t_ids.to(device=device, dtype=torch.long)


def video_rope_time_ids_from_chunk_ids(
    chunk_ids: torch.Tensor,
    latent_frames_per_chunk: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map real episode chunk ids onto the latent-video time axis."""
    latent_frames_per_chunk = int(latent_frames_per_chunk)
    if latent_frames_per_chunk <= 0:
        raise ValueError(
            "`latent_frames_per_chunk` must be positive, got "
            f"{latent_frames_per_chunk}."
        )
    return (
        normalize_video_rope_time_ids(chunk_ids, device=device)
        * latent_frames_per_chunk
    )


def sequence_start_video_rope_time_ids(
    batch_size: int, num_frames: int, *, device: torch.device
) -> torch.Tensor:
    """Return the real video timeline for frames at the start of an episode."""
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
    if num_frames < 0:
        raise ValueError(f"`num_frames` must be non-negative, got {num_frames}.")
    return torch.arange(num_frames, device=device, dtype=torch.long).unsqueeze(
        0
    ).expand(batch_size, -1)
