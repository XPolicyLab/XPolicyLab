"""Training forward for the released video-action-tactile model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .config import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "runtime") not in sys.path:
    sys.path.insert(0, str(ROOT / "runtime"))

from models.me_dex import MEDexConfig, MEDexModel  # noqa: E402


class MEDEXTrainingModel(nn.Module):
    """Clean Uni training wrapper around the runtime MoT model."""

    def __init__(self, config: TrainingConfig, wan_config: Path, vae_path: Path):
        super().__init__()
        self.model = MEDexModel(
            MEDexConfig(
                vae_path=str(vae_path),
                wan_config_path=str(wan_config),
                video_precision="bfloat16",
                batch_size=config.batch_size,
                tactile_ae_checkpoint_path=str(config.tactile.checkpoint),
                tactile_expert_config={
                    "latent_dim": config.tactile.latent_dim,
                    "latent_slices": config.tactile.frame_count,
                    "queries_per_slice": config.tactile.queries_per_frame,
                    "condition_slices": config.tactile.observed_frames,
                    "hidden_size": config.tactile.hidden_size,
                },
                attention_topology=config.topology,
                h_bridge_joint_start_layer=config.h_bridge_joint_start_layer,
                h_bridge_joint_end_layer=config.h_bridge_joint_end_layer,
            )
        )

    @staticmethod
    def _sigma(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand(batch, device=device, dtype=dtype).clamp_(1.0e-4, 0.9999)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.model.device
        dtype = self.model.dtype
        first_frame = batch["first_frame"].to(device=device, dtype=dtype)
        video_frames = batch["video_frames"].to(device=device, dtype=dtype)
        state = batch["state"].to(device=device, dtype=dtype)
        actions = batch["actions"].to(device=device, dtype=dtype)
        language_embeddings = [item.to(device=device, dtype=dtype) for item in batch["language_embeddings"]]

        with torch.no_grad():
            first_latent = self.model.video_model.encode_video(
                (first_frame * 2.0 - 1.0).unsqueeze(2)
            )
            clean_video = self.model.video_model.encode_video(
                torch.cat(
                    (
                        (first_frame * 2.0 - 1.0).unsqueeze(2),
                        (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4),
                    ),
                    dim=2,
                )
            )
            clean_tactile, _ = self.model.tactile_codec.encode_raw(
                batch["tactile_observed_source"].to(device=device),
                batch["tactile_future_source"].to(device=device),
                observed_support_source=batch["tactile_observed_support_source"].to(device=device),
                future_support_source=batch["tactile_future_support_source"].to(device=device),
            )

        batch_size = first_frame.shape[0]
        video_noise = torch.randn_like(clean_video)
        action_noise = torch.randn_like(actions)
        tactile_noise = torch.randn_like(clean_tactile[:, 2:])
        video_sigma = self._sigma(batch_size, device, dtype).view(batch_size, 1, 1, 1, 1)
        action_sigma = video_sigma.flatten().view(batch_size, 1, 1)
        tactile_sigma = video_sigma.flatten().view(batch_size, 1, 1, 1)
        noisy_video = clean_video * (1 - video_sigma) + video_noise * video_sigma
        noisy_video[:, :, :1] = first_latent
        noisy_actions = actions * (1 - action_sigma) + action_noise * action_sigma
        noisy_tactile = clean_tactile[:, 2:] * (1 - tactile_sigma) + tactile_noise * tactile_sigma
        tactile_latent = torch.cat((clean_tactile[:, :2], noisy_tactile), dim=1)

        t5_context = self.model.video_module.preprocess_t5_embeddings(language_embeddings)
        video_velocity, action_velocity, tactile_velocity = self.model._joint_video_action_tactile_velocity(
            video_latent=noisy_video,
            noisy_actions=noisy_actions,
            tactile_latent=tactile_latent,
            state=state,
            processed_t5_context=t5_context,
            video_timestep=video_sigma.flatten() * 1000,
            action_timestep=action_sigma.flatten() * 1000,
            tactile_timestep=tactile_sigma.flatten() * 1000,
        )
        video_target = video_noise - clean_video
        video_target[:, :, :1] = 0
        action_target = action_noise - actions
        tactile_target = tactile_noise - clean_tactile[:, 2:]
        video_loss = torch.nn.functional.mse_loss(video_velocity, video_target)
        action_loss = torch.nn.functional.mse_loss(action_velocity, action_target)
        tactile_loss = torch.nn.functional.mse_loss(
            tactile_velocity[:, 2:], tactile_target
        )
        return {
            "loss": video_loss + action_loss + tactile_loss,
            "video_loss": video_loss,
            "action_loss": action_loss,
            "tactile_loss": tactile_loss,
        }
