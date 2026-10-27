# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
import torch
import torch.nn.functional as F
from awomo.awomo.training.gripper_loss import weighted_action_per_sample

def gripper_train_step(self, batch: dict[str, Any], step: int, micro_idx: int = 0) -> dict[str, torch.Tensor]:
    del step, micro_idx
    encoded = self._encode_batch(batch)
    target, action = encoded["target"], encoded["action"]
    indices = torch.randint(
        0,
        self.scheduler.num_train_timesteps,
        (target.shape[0],),
        device=self.device,
    )
    timestep = self.scheduler.timesteps.to(self.device)[indices]
    image_noise, action_noise = torch.randn_like(target), torch.randn_like(action)
    noisy_image = self.scheduler.add_noise(target, image_noise, timestep)
    noisy_action = self.scheduler.add_noise(action, action_noise, timestep)
    prediction_image, prediction_action = self.model(
        reference=encoded["reference"],
        noisy_image=noisy_image,
        noisy_action=noisy_action,
        context=encoded["context"],
        text_mask=encoded["text_mask"],
        state=encoded["state"],
        timestep=timestep,
        reference_ids=encoded["reference_ids"],
        target_ids=encoded["target_ids"],
        condition_mask=encoded["condition_mask"],
    )
    video_loss_per_sample = (
        F.mse_loss(
            prediction_image.float(),
            (image_noise - target).float(),
            reduction="none",
        )
        .flatten(1)
        .mean(1)
    )
    element_loss = F.mse_loss(prediction_action.float(), (action_noise - action).float(), reduction="none")
    # Normalize each sample over valid dimensions and time steps before averaging samples.
    dims = (~batch["action_dim_is_pad"]).to(device=self.device, dtype=element_loss.dtype)
    steps = (~batch["action_is_pad"]).to(device=self.device, dtype=element_loss.dtype)
    assert torch.all(dims.sum(1) == 14)
    multiplier = float(self.cfg.gripper_experiment.loss_multiplier)
    action_loss_per_sample = weighted_action_per_sample(element_loss, dims, steps, multiplier)
    video_loss_unweighted = video_loss_per_sample.mean()
    action_loss_unweighted = action_loss_per_sample.mean()
    weight = torch.ones_like(video_loss_per_sample)
    if self._model_cfg.bsmntw_enabled:
        weight = self.scheduler.training_weight(timestep).to(video_loss_per_sample)
    video_loss = (video_loss_per_sample * weight).mean()
    action_loss = (action_loss_per_sample * weight).mean()
    video_loss = self.loss_lambda_video * video_loss
    action_loss = self.loss_lambda_action * action_loss
    total = video_loss + action_loss
    assert torch.isfinite(total), "Non-finite loss; refusing optimizer update"
    losses = {
        "total_loss": total,
        "dynamics_loss": video_loss,
        "action_loss": action_loss,
    }
    if self._model_cfg.bsmntw_enabled:
        self._bsmntw_log_scalars = {
            "action_loss_unweighted": float((self.loss_lambda_action * action_loss_unweighted).detach().item()),
            "dynamics_loss_unweighted": float((self.loss_lambda_video * video_loss_unweighted).detach().item()),
            "bsmntw_mean_weight": float(weight.mean().item()),
        }
    gripper_raw = (element_loss[:, :, [16, 45]].mean(2) * steps).sum(1) / steps.sum(1).clamp_min(1)
    joint_mask = dims.clone()
    joint_mask[:, [16, 45]] = 0
    joint_raw = ((element_loss * joint_mask[:, None]).sum(2) / 12 * steps).sum(1) / steps.sum(1).clamp_min(1)
    self._gripper_log_scalars = dict(
        gripper_loss_raw=float(gripper_raw.detach().mean().item()),
        joint_loss_raw=float(joint_raw.detach().mean().item()),
        gripper_loss_bsmntw=float((gripper_raw.detach() * weight).mean().item()),
        joint_loss_bsmntw=float((joint_raw.detach() * weight).mean().item()),
        gripper_loss_multiplier=multiplier,
    )
    return losses
