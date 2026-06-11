import functools
import os

import imageio
import numpy as np
import torch

from .wa_casual_trainer import CasualWATrainer
from .wa_trainer_pretrain import _as_dim_mask, masked_mse


class CasualWATrainerPretrain(CasualWATrainer):
    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        bs = images.shape[0]
        prompt_embeds = batch_dict["prompt_embeds"]
        timestep, sigma = self.get_timestep_and_sigma(images.shape[0], images.ndim)
        action = batch_dict["action"]
        state = batch_dict["state"]

        # Independent sigma for action branch (with its own flow_shift)
        # action shape: (bs, T, D), need sigma (bs, 1, 1) for broadcasting
        _, action_sigma_5d = self.get_timestep_and_sigma(bs, images.ndim, flow_shift=self.action_flow_shift)  # (bs, 1, 1, 1, 1)
        action_sigma = action_sigma_5d.squeeze(-1).squeeze(-1)  # (bs, 1, 1)
        action_noise_t = torch.round(action_sigma[:, 0, 0].unsqueeze(-1) * 1000).to(dtype=sigma.dtype, device=sigma.device)  # (bs, 1)

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * sigma + visual_latents * (1 - sigma)

        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)

        # Use gt action (clean) as input to transformer for video conditioning
        if self.use_gt_action_for_video:
            input_action = action.to(self.dtype)
        else:
            input_action = noisy_action.to(self.dtype)

        prompt_embeds = prompt_embeds.to(self.dtype)
        if "ref_images" in batch_dict:
            if not self.expand_timesteps:
                ref_images = batch_dict["ref_images"]
                ref_latents = self.forward_vae(ref_images)
                num_frames = images.shape[1]
                batch_size = ref_latents.shape[0]
                latent_height = ref_latents.shape[-2]
                latent_width = ref_latents.shape[-1]
                mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
                mask_lat_size[:, :, list(range(1, num_frames))] = 0
                first_frame_mask = mask_lat_size[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
                mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
                mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
                mask_lat_size = mask_lat_size.transpose(1, 2)
                mask_lat_size = mask_lat_size.to(ref_latents.device)
                condition = torch.concat([mask_lat_size, ref_latents], dim=1)
                insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
            else:
                num_latent_frames = visual_latents.shape[2]
                latent_height = visual_latents.shape[-2]
                latent_width = visual_latents.shape[-1]
                ref_images = batch_dict["ref_images"][:, :1]
                ref_latents = self.forward_vae(ref_images)
                first_frame_mask = torch.ones(
                    bs, 1, num_latent_frames, latent_height, latent_width, dtype=visual_latents.dtype, device=visual_latents.device
                )
                first_frame_mask[:, :, 0] = 0
                insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents
                temp_ts = (first_frame_mask[:, :, :, ::2, ::2] * timestep[:, None, None, None, None]).reshape(bs, -1)
                timestep = temp_ts
        else:
            raise ValueError("CasualWATrainerPretrain requires ref_images in batch_dict")

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noise_t = timestep[:, -2:-1]

        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)

        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens

        # Build per-token timestep aligned with model's token order:
        # Model reorders to: [state(1) | ref(ref_tokens) | action(action_tokens) | noisy(noisy_tokens)]
        # But timestep is NOT reordered by model, so we must match this layout directly.
        timestep = torch.zeros(bs, num_state_tokens + num_action_tokens + num_latent_tokens, device=noisy_latents.device, dtype=noisy_latents.dtype)
        # Positions in timestep map to model's reordered hidden_states:
        #   pos 0                              → state = 0
        #   pos 1 .. num_clean_latent_tokens   → ref_video = 0
        #   pos num_clean_latent_tokens+1 .. +num_action_tokens → action = action_noise_t
        #   pos after that                     → noisy_video = visual noise_t
        action_start = num_state_tokens + num_clean_latent_tokens
        action_end = action_start + num_action_tokens
        if self.use_gt_action_for_video:
            timestep[:, action_start:action_end] = 0  # gt action = clean, timestep=0
        else:
            timestep[:, action_start:action_end] = action_noise_t
        timestep[:, action_end:] = noise_t

        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=input_action,
            state=state,
        )

        # --- Visualization ---
        if self.if_visualize():
            with torch.no_grad():
                # Video: decode GT and pred, stitch into one side-by-side mp4
                pred_x0 = noisy_latents - visual_pred * sigma
                if self.expand_timesteps:
                    pred_x0 = (1 - first_frame_mask) * ref_latents + first_frame_mask * pred_x0
                gt_video = self.vae_decode(latents=visual_latents, sign="_tmp_gt", return_tensor=True)
                pred_video = self.vae_decode(latents=pred_x0, sign="_tmp_pred", return_tensor=True)
                if gt_video is not None and pred_video is not None:
                    # gt_video, pred_video: (B, C, T, H, W) tensors
                    concat_video = torch.cat([gt_video, pred_video], dim=-1)  # side-by-side along width
                    concat_video = self.video_processor.postprocess_video(concat_video, output_type="pil")[0]
                    save_dir = os.path.join(self.view_dir, "images", str(self.cur_step))
                    os.makedirs(save_dir, exist_ok=True)
                    imageio.mimsave(os.path.join(save_dir, "visual_gt_vs_pred.mp4"), concat_video, fps=16)
                    # Clean up temp files
                    for tmp in ("_tmp_gt.mp4", "_tmp_pred.mp4"):
                        tmp_path = os.path.join(save_dir, tmp)
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                # Action: pred vs gt comparison plot
                pred_action = noisy_action - action_pred * action_sigma
                if self.action_repeats > 1:
                    pred_action = pred_action.reshape(bs, self.action_repeats, -1, pred_action.shape[-1])
                    pred_action = pred_action.mean(1)
                self.vae_decode(action=pred_action, sign="action_visual", gt_action=action)

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        dim_mask = None
        if "action_dim_mask" in batch_dict:
            dim_mask = _as_dim_mask(batch_dict["action_dim_mask"], batch_size=bs, seq_len=action.shape[1], dim=action.shape[2], device=action_pred.device)

        action_loss = masked_mse(action_pred.float(), action_target.float(), dim_mask=dim_mask, time_mask=None)

        return {
            "visual_loss": visual_loss * self.visual_loss_weight,
            "action_loss": action_loss * self.action_loss_weight,
        }
