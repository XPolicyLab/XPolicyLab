import copy
import functools
import os

import torch
from diffusers.models import AutoencoderKLWan
from ..models.transformer_wa_casual import CasualWorldActionTransformer, WanRotaryPosEmbed1D
from einops import rearrange
from ..trainer import Trainer, ModuleDict, DictConfig
import torch.nn as nn
from PIL import Image
import imageio
import numpy as np
import matplotlib.pyplot as plt
from diffusers.video_processor import VideoProcessor

from .wa_trainer import get_model_path, process_transformer


class CasualWATrainer(Trainer):
    def get_models(self, model_config):
        pretrained = get_model_path(model_config.pretrained)
        self.flow_shift = model_config.flow_shift
        self.action_flow_shift = float(model_config.get("action_flow_shift", self.flow_shift))
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_loss_weight = float(model_config.get("action_loss_weight", 1.0))
        self.visual_loss_weight = float(model_config.get("visual_loss_weight", 1.0))
        self.use_gt_action_for_video = model_config.get("use_gt_action_for_video", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.state_dim = int(model_config.get("state_dim", self.action_dim))
        self.view_interval = int(model_config.get("view_interval", 50))
        self.view_dir = model_config.view_dir
        model = dict()
        # vae
        vae_pretrained = model_config.get('vae_pretrained', os.path.join(pretrained, 'vae'))
        vae_dtype = model.get('vae_dtype', self.dtype)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        # transformer
        transformer_pretrained = model_config.get('transformer_pretrained', os.path.join(pretrained, 'transformer'))
        if model_config.get("unpretrain", False):
            print("Load unet from config only.")
            transformer = CasualWorldActionTransformer.from_config(transformer_pretrained, torch_dtype=self.dtype)
        else:
            transformer = CasualWorldActionTransformer.from_pretrained(transformer_pretrained, torch_dtype=self.dtype)

        encoder = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 3072),
        )
        decoder = nn.Sequential(
            nn.Linear(3072, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.action_dim),
        )
        state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 3072),
        )
        transformer.action_encoder = copy.deepcopy(encoder)
        transformer.action_decoder = copy.deepcopy(decoder)
        transformer.state_encoder = copy.deepcopy(state_encoder)
        transformer.action_rope = WanRotaryPosEmbed1D(128, 1024)
        # Initialize condition_embedder_action from the pretrained condition_embedder
        # (the __init__ version is on meta device; deepcopy gives real weights)
        transformer.condition_embedder_action = copy.deepcopy(transformer.condition_embedder)
        # Remove image_embedder from action embedder (action doesn't need it)
        transformer.condition_embedder_action.image_embedder = None
        transformer_cfg = model_config.get('transformer', dict())
        transformer = process_transformer(transformer, transformer_cfg)
        transformer.to(self.device, dtype=self.dtype)
        model.update(transformer=transformer)
        # model
        checkpoint = model_config.get('checkpoint', None)
        strict = model_config.get('strict', True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()

        # Freeze backbone: only train action_encoder/decoder/state_encoder/action_rope/condition_embedder_action
        if model_config.get("freeze_backbone", False):
            action_keywords = ("action_encoder", "action_decoder", "state_encoder", "action_rope", "condition_embedder_action")
            frozen_count, trainable_count = 0, 0
            for name, param in transformer.named_parameters():
                if any(kw in name for kw in action_keywords):
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
            if self.process_index == 0:
                print(f"Freeze backbone: {frozen_count} params frozen, {trainable_count} params trainable")

        # Freeze action: only train backbone, freeze action_encoder/decoder/state_encoder/action_rope/condition_embedder_action
        if model_config.get("freeze_action", False):
            action_keywords = ("action_encoder", "action_decoder", "state_encoder", "action_rope", "condition_embedder_action")
            frozen_count, trainable_count = 0, 0
            for name, param in transformer.named_parameters():
                if any(kw in name for kw in action_keywords):
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    trainable_count += 1
            if self.process_index == 0:
                print(f"Freeze action: {frozen_count} action params frozen, {trainable_count} backbone params trainable")

        return model

    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, 'transformer')
        images = batch_dict['images']
        bs = images.shape[0]
        prompt_embeds = batch_dict['prompt_embeds']
        timestep, sigma = self.get_timestep_and_sigma(images.shape[0], images.ndim)
        action = batch_dict['action']
        state = batch_dict['state']
        self.vae_decode(action=action, sign='input_action')
        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)
        # inputs
        visual_latents = self.forward_vae(images)
        self.vae_decode(latents=visual_latents, sign='input_visual')
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * sigma + visual_latents * (1 - sigma)
        action_sigma = sigma.squeeze(-1).squeeze(-1)
        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)
        # loss
        prompt_embeds = prompt_embeds.to(self.dtype)
        if 'ref_images' in batch_dict:
            if not self.expand_timesteps:
                ref_images = batch_dict['ref_images']
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
                mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height,
                                                   latent_width)
                mask_lat_size = mask_lat_size.transpose(1, 2)
                mask_lat_size = mask_lat_size.to(ref_latents.device)
                condition = torch.concat([mask_lat_size, ref_latents], dim=1)
                noisy_latents = torch.concat([noisy_latents, condition], dim=1)
            else:
                num_latent_frames = visual_latents.shape[2]
                latent_height = visual_latents.shape[-2]
                latent_width = visual_latents.shape[-1]
                ref_images = batch_dict['ref_images'][:, :1]
                ref_latents = self.forward_vae(ref_images)
                first_frame_mask = torch.ones(
                    bs, 1, num_latent_frames, latent_height, latent_width, dtype=visual_latents.dtype, device=visual_latents.device
                )
                first_frame_mask[:, :, 0] = 0
                insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents
                temp_ts = (first_frame_mask[:, :, :, ::2, ::2] * timestep[:, None, None, None, None]).reshape(bs, -1)
                timestep = temp_ts
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
        timestep = torch.zeros(bs, num_state_tokens + num_action_tokens + num_latent_tokens, device=noisy_latents.device, dtype=noisy_latents.dtype)
        num_clean_latent_tokens = frame_per_tokens
        num_noisy_latent_tokens = num_latent_tokens - num_clean_latent_tokens
        timestep[:, num_state_tokens + num_clean_latent_tokens:] = noise_t
        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
        )
        if self.if_visualize():
            with torch.no_grad():
                pred_x0 = noisy_latents - visual_pred * sigma
                if self.expand_timesteps:
                    pred_x0 = (1 - first_frame_mask) * ref_latents + first_frame_mask * pred_x0
                self.vae_decode(latents=pred_x0, sign='pred_visual')
                pred_action = noisy_action - action_pred * action_sigma
                if self.action_repeats > 1:
                    pred_action = pred_action.reshape(bs, self.action_repeats, -1, 14)
                    pred_action = pred_action.mean(1)
                self.vae_decode(action=pred_action, sign='action_visual', gt_action=action)
        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask) ** 2
        visual_loss = visual_loss.mean()
        action_loss = (action_pred.float() - action_target.float()) ** 2
        # Apply action_dim_mask to exclude zero-std / padding dims
        if 'action_dim_mask' in batch_dict:
            mask = batch_dict['action_dim_mask']  # (bs, action_dim) or (action_dim,)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
            elif mask.ndim == 2:
                mask = mask.unsqueeze(1)  # (bs, 1, action_dim)
            action_loss = (action_loss * mask.float()).sum() / (mask.float().sum() * action_loss.shape[1])
        else:
            action_loss = action_loss.mean()
        loss = {
            'visual_loss': visual_loss * self.visual_loss_weight,
            'action_loss': action_loss,
        }
        return loss

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim, flow_shift=None):
        if flow_shift is None:
            flow_shift = self.flow_shift
        sigma = torch.rand(batch_size).to(self.device)
        sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

    def if_visualize(self):
        return self.process_index == 0 and (self.cur_step % self.view_interval == 0 or self.cur_step == 1) and len(self._outputs) == 0

    def vae_decode(self, latents=None, action=None, gt_action=None, images=None, sign=None, return_tensor=False):
        if self.if_visualize():
            try:
                return self._vae_decode_impl(latents, action, gt_action, images, sign, return_tensor)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[Step {self.cur_step}] VAE decode OOM for '{sign}', skipping visualization")
                return None

    def _vae_decode_impl(self, latents=None, action=None, gt_action=None, images=None, sign=None, return_tensor=False):
            save_dir = os.path.join(self.view_dir, "images", "{}".format(self.cur_step))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.mp4".format(sign))
            if latents is not None:
                latents = latents.to(self.vae.dtype)
                latents = latents / self.latents_std + self.latents_mean
                # Only decode a single-view-sized chunk to save memory
                lh, lw = latents.shape[-2], latents.shape[-1]
                if lw > lh:  # multi-view concatenated along width
                    latents = latents[..., :lh]  # crop to square = 1 view
                with torch.no_grad():
                    tensor_video = self.vae.decode(latents, return_dict=False)[0].detach()
                video = self.video_processor.postprocess_video(tensor_video, output_type='pil')[0]
                vis_images = video
                imageio.mimsave(save_path, vis_images, fps=16)
                if return_tensor:
                    return tensor_video
                return vis_images
            if images is not None:
                image_tensor = images
                images = (images + 1.0) / 2.0 * 255
                images = images.astype(np.uint8)
                images = [Image.fromarray(images[i]) for i in range(images.shape[0])]
                imageio.mimsave(save_path, images, fps=16)
                return image_tensor
            if action is not None:
                action = action.float().detach().cpu().numpy()
                T = action.shape[1]
                plot_dims = min(int(action.shape[2]), 32)
                cols = 4
                rows = (plot_dims + cols - 1) // cols
                fig = plt.figure(figsize=(cols * 3, max(1, rows) * 2.5))
                has_gt = gt_action is not None
                if has_gt:
                    gt_np = gt_action.float().detach().cpu().numpy()
                for i in range(plot_dims):
                    plt.subplot(rows, cols, i + 1)
                    if has_gt:
                        plt.plot(range(T), gt_np[0, :, i], label="gt", color="tab:blue", linewidth=1.2)
                        plt.plot(range(T), action[0, :, i], label="pred", color="tab:orange", linewidth=1.2, alpha=0.8)
                        if i == 0:
                            plt.legend(fontsize=6)
                    else:
                        plt.plot(range(T), action[0, :, i])
                    plt.title("Dim {}".format(i), fontsize=9)
                plt.tight_layout()
                save_path = os.path.join(save_dir, "{}.png".format(sign))
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                return action
