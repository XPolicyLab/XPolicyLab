"""Load the Wan2.2 video VAE, which is Cosmos3-Nano's video tokenizer bit for bit.

``proj_in``/``proj_out`` were trained against this latent space, so the exact Wan2.2 weights are required.
"""

from __future__ import annotations

import os

import torch
from safetensors import safe_open

from ..wan22.wan_video_vae import WanVideoVAE38

# Relative to $DIFFSYNTH_MODEL_BASE_PATH (default ./checkpoints/).
VAE_FILE = "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"


def load_wan22_vae(*, torch_dtype: torch.dtype, device: str) -> WanVideoVAE38:
    """Load the Wan2.2 VAE (48-ch, spatial/16, temporal/4)."""
    path = os.path.join(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./checkpoints/"), VAE_FILE)
    state_dict = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            state_dict[f"model.{key}"] = f.get_tensor(key).to(torch_dtype)
    vae = WanVideoVAE38()
    vae.load_state_dict(state_dict, strict=False)
    return vae.to(device=device, dtype=torch_dtype)
