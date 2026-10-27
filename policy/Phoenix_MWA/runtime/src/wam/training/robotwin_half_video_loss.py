"""Isolated RoboTwin posttraining objective; shared pretraining stays unchanged."""

from contextlib import contextmanager

import torch
from omegaconf import OmegaConf

from wam.model.modules.conditioning.embodiment_prompt import action_prompt_kwargs
from wam.model.modules.conditioning.frame_attention import build_joint96_attention_mask
from wam.model.modules.conditioning.input_builder import (
    build_wam_inputs,
    joint96_camera_valid_mask,
)
from wam.model.modules.conditioning.video_rope import sequence_start_video_rope_time_ids
from wam.training.losses import _joint96_masked_mse, _joint96_temporal_valid_mask
from wam.training.module import WAMTrainingModule


LOSS_WEIGHTS = {"vae": 0.25, "stvae": 0.25, "action": 1.0}


def validate_robotwin_half_video_loss(cfg):
    """Reject use outside the explicitly selected RoboTwin posttraining recipe."""
    required = {
        "joint96_posttrain.enabled": True,
        "joint96_posttrain.online_robotwin": True,
        "joint96_posttrain.robotwin_softprompt.enabled": True,
        "local_robot_joint96.enabled": False,
        "model.joint96_three_view": True,
        "data.train.domain": "robotwin",
        "data.val.domain": "robotwin",
        "joint96_posttrain.loss_recipe": "robotwin_half_video_v1",
    }
    for key, expected in required.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"RoboTwin half-video loss requires {key}={expected!r}")
    weights = OmegaConf.select(cfg, "joint96_posttrain.loss_weights")
    if weights is None or dict(weights) != LOSS_WEIGHTS:
        raise ValueError(f"Expected posttraining loss weights {LOSS_WEIGHTS}")


def robotwin_half_video_training_loss(model, sample, *, tiled=False):
    """Joint96 posttraining: 0.25 Wan + 0.25 STVAE + 1.0 action.

    Noise, masking, timestep weighting and prompt handling follow the original
    Joint96 objective. Only the final video coefficients change. This separate
    implementation leaves the original pretraining objective byte-for-byte intact.
    """
    if not bool(getattr(model, "joint96_three_view", False)):
        raise ValueError("RoboTwin half-video loss requires Joint96")
    inputs = build_wam_inputs(model, sample, tiled=tiled)
    if float(model.loss_lambda_video) != 1.0 or float(model.loss_lambda_action) != 1.0:
        raise ValueError("Keep model lambdas at 1; the posttraining recipe owns weights")
    if bool(getattr(model.action_expert, "training_time_rtc_enabled", False)):
        raise ValueError("Joint96 does not enable training-time RTC prefixes")
    if bool(getattr(model.video_expert, "action_conditioned", False)):
        raise ValueError("Joint96 does not use GT-action video conditioning")
    for name in ("action_dim_loss_weight", "action_loss_weight"):
        value = inputs.get(name)
        if value is not None and not bool((value == 1).all()):
            raise ValueError(f"Joint96 does not support non-unit {name}")
    for name in ("action_loss_weighted_valid_cells", "action_loss_weighted_valid_cells_pool"):
        value = inputs.get(name)
        if value is not None and bool(value.any()):
            raise ValueError(f"Joint96 does not support {name}")

    input_latents = inputs["input_latents"]
    batch_size, _, frames, height, width = input_latents.shape
    patch_t, patch_h, patch_w = map(int, model.video_expert.patch_size)
    if patch_t != 1 or height % patch_h or (width // 3) % patch_w:
        raise ValueError("Joint96 patches must align to each camera and frame")
    spatial_valid = joint96_camera_valid_mask(
        inputs["camera_is_pad"],
        batch_size=batch_size,
        height=height,
        width=width,
        device=input_latents.device,
    )
    frame_valid = _joint96_temporal_valid_mask(model, inputs)
    latent_valid = spatial_valid[:, None, None] & frame_valid[:, None, :, None, None]
    input_latents = input_latents.masked_fill(~latent_valid, 0.0)
    noise_video = torch.randn_like(input_latents).masked_fill(~latent_valid, 0.0)
    timestep_video = model.train_video_scheduler.sample_training_t(
        batch_size=batch_size, device=model.device, dtype=input_latents.dtype
    )
    noisy_video = model.train_video_scheduler.add_noise(
        input_latents, noise_video, timestep_video
    )
    noisy_video = noisy_video.masked_fill(~latent_valid, 0.0)
    noisy_video[:, :, :1] = input_latents[:, :, :1]
    target_video = model.train_video_scheduler.training_target(
        input_latents, noise_video, timestep_video
    )
    target_video = target_video.masked_fill(~latent_valid, 0.0)
    target_video[:, :, :1] = 0.0

    action = inputs["action"]
    action_valid = torch.ones_like(action, dtype=torch.bool)
    if inputs.get("action_mask") is not None:
        action_valid &= inputs["action_mask"]
    if inputs.get("action_is_pad") is not None:
        action_valid &= ~inputs["action_is_pad"].unsqueeze(-1)
    if inputs.get("action_dim_is_pad") is not None:
        action_valid &= ~inputs["action_dim_is_pad"].unsqueeze(1)
    if bool((~action_valid.flatten(1).any(dim=1)).any()):
        raise ValueError("Every sample requires valid action supervision")
    action = action.masked_fill(~action_valid, 0.0)
    noise_action = torch.randn_like(action).masked_fill(~action_valid, 0.0)
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch_size=batch_size, device=model.device, dtype=action.dtype
    )
    noisy_action = model.train_action_scheduler.add_noise(
        action, noise_action, timestep_action
    )
    noisy_action = noisy_action.masked_fill(~action_valid, 0.0)
    target_action = model.train_action_scheduler.training_target(
        action, noise_action, timestep_action
    )
    target_action = target_action.masked_fill(~action_valid, 0.0)

    video_pre = model.video_expert.pre_dit(
        x=noisy_video,
        timestep=timestep_video[:, None].expand(-1, frames - 1),
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        fuse_vae_embedding_in_latents=True,
        frame_ids=sequence_start_video_rope_time_ids(
            batch_size, frames, device=input_latents.device
        ),
    )
    action_pre = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        **action_prompt_kwargs(model.action_expert, sample),
    )
    grid = (frames, height // patch_h, width // patch_w)
    if tuple(video_pre["meta"]["grid_size"]) != grid:
        raise ValueError("Joint96 video grid differs from its latent canvas")
    spatial_token_valid = joint96_camera_valid_mask(
        inputs["camera_is_pad"],
        batch_size=batch_size,
        height=grid[1],
        width=grid[2],
        device=input_latents.device,
    ).flatten(1)
    tokens_per_frame = int(spatial_token_valid.shape[1])
    video_valid = (frame_valid[:, :, None] & spatial_token_valid[:, None, :]).flatten(1)
    action_token_valid = action_valid.any(dim=-1)
    if tuple(action_pre["tokens"].shape[:2]) != tuple(action_token_valid.shape):
        raise ValueError("Joint96 requires one action token per step")
    if tuple(video_pre["tokens"].shape[:2]) != tuple(video_valid.shape):
        raise ValueError("Joint96 video tokens differ from their validity mask")
    attention_mask = build_joint96_attention_mask(
        video_seq_len=int(video_valid.shape[1]),
        action_seq_len=int(action_token_valid.shape[1]),
        video_tokens_per_frame=tokens_per_frame,
        device=input_latents.device,
    )
    validity = {"video": video_valid, "action": action_token_valid}
    states = {"video": video_pre, "action": action_pre}
    contexts = {}
    for name, pre in states.items():
        mask = pre["context_mask"]
        if mask.ndim == 3:
            mask = mask & validity[name][:, :, None]
        elif mask.ndim == 4:
            mask = mask & validity[name][:, None, :, None]
        else:
            raise ValueError("Joint96 context masks must have rank 3 or 4")
        contexts[name] = {"context": pre["context"], "mask": mask}
    tokens_out = model.mot(
        embeds_all={name: pre["tokens"] for name, pre in states.items()},
        attention_mask=attention_mask,
        freqs_all={name: pre["freqs"] for name, pre in states.items()},
        context_all=contexts,
        t_mod_all={name: pre["t_mod"] for name, pre in states.items()},
        key_masks_all=validity,
        query_masks_all=validity,
    )
    pred_video = model.video_expert.post_dit(tokens_out["video"], video_pre)[:, :, 1:]
    pred_action = model.action_expert.post_dit(tokens_out["action"], action_pre)
    future_valid = latent_valid[:, :, 1:]
    video_weight = model.train_video_scheduler.training_weight(timestep_video)
    loss_vae, mse_vae = _joint96_masked_mse(
        pred_video[:, :48], target_video[:, :48, 1:], future_valid,
        video_weight, name="VAE",
    )
    loss_stvae, mse_stvae = _joint96_masked_mse(
        pred_video[:, 48:], target_video[:, 48:, 1:], future_valid,
        video_weight, name="STVAE",
    )
    loss_action, mse_action = _joint96_masked_mse(
        pred_action, target_action, action_valid,
        model.train_action_scheduler.training_weight(timestep_action), name="action",
    )
    weighted_vae = LOSS_WEIGHTS["vae"] * loss_vae
    weighted_stvae = LOSS_WEIGHTS["stvae"] * loss_stvae
    loss_video = weighted_vae + weighted_stvae
    return loss_video + loss_action, {
        "loss_video": loss_video.detach(),
        "loss_action": loss_action.detach(),
        "loss_vae": weighted_vae.detach(),
        "loss_stvae": weighted_stvae.detach(),
        "loss_vae_raw": loss_vae.detach(),
        "loss_stvae_raw": loss_stvae.detach(),
        "mse_vae": mse_vae,
        "mse_stvae": mse_stvae,
        "mse_action": mse_action,
    }


class RobotwinHalfVideoTrainingModule(WAMTrainingModule):
    """Keep state-dict keys and non-loss operations identical to the base module."""

    def forward(self, sample=None, *, operation="training_loss", tiled=False, **kwargs):
        if operation == "training_loss":
            return robotwin_half_video_training_loss(self.wam, sample, tiled=tiled)
        return super().forward(sample, operation=operation, tiled=tiled, **kwargs)


@contextmanager
def robotwin_half_video_objective(cfg):
    """Select the wrapper only inside this standalone posttraining process.

    The existing trainer constructs its wrapper via a module-level class. Bind
    that class during this entrypoint's run and restore it afterward; no shared
    trainer, runtime, model or pretraining loss source file is modified.
    """
    validate_robotwin_half_video_loss(cfg)
    import wam.trainer as trainer_module

    previous = trainer_module.WAMTrainingModule
    if previous is not WAMTrainingModule:
        raise RuntimeError("Another training-wrapper override is already active")
    trainer_module.WAMTrainingModule = RobotwinHalfVideoTrainingModule
    try:
        yield
    finally:
        trainer_module.WAMTrainingModule = previous
