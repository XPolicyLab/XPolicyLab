from typing import Tuple

import torch


def build_joint_denoising_timesteps(
    num_inference_steps: int,
    future_video_denoise_fraction: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build independent video and action flow-matching timesteps."""
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}"
        )
    if not 0.0 <= future_video_denoise_fraction <= 1.0:
        raise ValueError(
            "future_video_denoise_fraction must be in [0, 1], got "
            f"{future_video_denoise_fraction}"
        )

    action_timesteps = torch.linspace(
        1.0,
        0.0,
        num_inference_steps + 1,
        device=device,
        dtype=dtype,
    )
    video_timesteps = torch.linspace(
        1.0,
        1.0 - future_video_denoise_fraction,
        num_inference_steps + 1,
        device=device,
        dtype=dtype,
    )
    return video_timesteps, action_timesteps
