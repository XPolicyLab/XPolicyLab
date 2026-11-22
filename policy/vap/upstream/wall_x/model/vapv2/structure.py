"""Structure-only VAPv2 serving construction; no Wan weight files are read.

Training still initializes through ``load_wan22_5b_video_base``.  A complete
serving checkpoint instead supplies every persistent core/VAE tensor, so only
the pinned Wan2.2 architecture and its non-persistent constants are needed here.
"""

from __future__ import annotations

import torch

from wall_x.model.vap.backbone import WAN22_TI2V_5B_ID
from wall_x.model.vap.text_conditioning import validate_text_config
from wall_x.utils.io import init_weights_on_device


# These are the architecture fields previously inferred from the official
# Wan2.2-TI2V-5B DiT state-dict shape hash, not learned serving weights.
WAN22_TI2V_5B_STRUCTURE = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


def _structure_pipeline(config, *, torch_dtype):
    from wall_x.model.vap.qwen35_online import Qwen35OnlinePrompter
    from wall_x.model.vga.modeling.wan_dit import WanModel
    from wall_x.model.vga.modeling.wan_vae import WanVideoVAE38
    from wall_x.model.vga.modeling.wan_video import WanVideoPipeline

    if config.base_model_id != WAN22_TI2V_5B_ID:
        raise ValueError(f"Unsupported VAPv2 serving base: {config.base_model_id}")
    if config.text_encoder_backend != "qwen35_online":
        raise ValueError("Wan-free VAPv2 serving requires qwen35_online")
    validate_text_config(config)
    pipeline = WanVideoPipeline(
        device="cpu",
        torch_dtype=torch_dtype,
        enable_action=False,
        num_views=1,
        scheduler_shift=float(config.video_flow["shift"]),
        timestep_sample_mode=str(config.video_flow["sample_mode"]),
        skip_text_encoder=True,
    )
    # DiT parameters are uninitialized storage until the complete checkpoint is
    # strictly loaded. Its plain (non-buffer) RoPE tables remain real CPU values.
    with init_weights_on_device():
        pipeline.dit = WanModel(**WAN22_TI2V_5B_STRUCTURE)
    pipeline.dit = pipeline.dit.to_empty(device="cpu").to(dtype=torch_dtype)
    # VAE mean/std are non-persistent buffers. Match the reference ModelManager
    # dtype conversion for both registered buffers and persistent parameters.
    # Its plain scale list keeps the original tensors; VAE encode moves those
    # values to the activation dtype/device, so compare it independently.
    pipeline.vae = WanVideoVAE38().to(dtype=torch_dtype)
    pipeline.height_division_factor = pipeline.vae.upsampling_factor * 2
    pipeline.width_division_factor = pipeline.vae.upsampling_factor * 2
    pipeline.prompter = Qwen35OnlinePrompter(config)
    return pipeline


def build_vapv2_model_structure(config, *, action_dim, state_dim):
    """Build the exact serving module tree without opening Wan model assets."""
    from .training import VAPv2TrainingModel

    # Scaffolding initialization is irrelevant after strict restoration, but
    # must not perturb a caller's CPU RNG stream or leak random non-persistent
    # state. Qwen weights are loaded lazily and are not part of this operation.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        pipeline = _structure_pipeline(config, torch_dtype=torch.bfloat16)
        model = VAPv2TrainingModel.from_wan_pipeline(
            pipeline,
            config,
            action_dim=action_dim,
            state_dim=state_dim,
        )
    if model.core.video_dim != 3072 or len(model.core.blocks) != 30:
        raise RuntimeError("Unexpected Wan2.2-TI2V-5B VAPv2 structure")
    return model
