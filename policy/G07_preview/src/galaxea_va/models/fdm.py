"""Action conditions and temporal routing shared by the two FDM backbones."""

import torch


def action_video_group_mask(action_len, future_latents, tokens_per_latent, device):
    """Future latent j reads its action interval, independently of RoPE IDs."""
    if future_latents <= 0 or action_len % future_latents:
        raise ValueError("FDM action length must be divisible by the positive number of future latents.")
    video_groups = torch.arange(future_latents, device=device).repeat_interleave(tokens_per_latent)
    action_groups = torch.arange(action_len, device=device) // (action_len // future_latents)
    return video_groups[:, None] == action_groups[None, :]
