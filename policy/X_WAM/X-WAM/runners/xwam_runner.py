import os
import logging
import imageio.v2 as imageio

import torch
import torch.nn.functional as F
import lightning as L
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from modules.wan_model import XWAMModel
from modules.t5 import T5EncoderModel
from modules.vae2_2 import Wan2_2_VAE
from utils.utils import sample_beta


class XWAMRunner(L.LightningModule):
    def __init__(self, config, run_depth=True):
        super().__init__()
        self.config = config
        self.run_depth = run_depth

        # TODO: remove hard-coded views and modalities
        self.num_views = 3
        self.num_modalities = 2 if config.use_depth else 1
        self.num_frames_per_latent = config.vae_stride[0]

        logging.info(f"Loading Wan2_2_VAE from {config.wan_checkpoint_dir}...")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=eval(config.t5_dtype),
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(config.wan_checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(config.wan_checkpoint_dir, config.t5_tokenizer),
        )
        self.text_encoder.eval()
        self.vae = Wan2_2_VAE(vae_pth=os.path.join(config.wan_checkpoint_dir, config.vae_checkpoint))
        self.vae.eval()
        self.vae.requires_grad_(False)

        logging.info(f"Creating WanModel from {config.wan_checkpoint_dir}...")
        self.model = XWAMModel.from_pretrained(
            config.wan_checkpoint_dir,
            num_modalities=self.num_modalities,
            num_views=self.num_views,
            action_dim=config.action_dim,
            action_num=config.action_num,
            proprio_dim=config.proprio_dim,
            num_extra_layers=config.num_extra_layers,
            low_cpu_mem_usage=False,
        )
        self.model.init_new_weights()
        print("Copying weights to extra blocks...")
        self.model.copy_weights_to_extra_blocks()

        if getattr(self.config, "use_gradient_checkpointing", False):
            print("Enabling gradient checkpointing...")
            self.model.gradient_checkpointing = True

        self.model.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def on_fit_start(self):
        print(f"Setting generator for rank {self.global_rank} with seed {self.config.seed + self.global_rank}")
        self.generator_per_rank = torch.Generator(device="cpu").manual_seed(self.config.seed + self.global_rank)

    def training_step(self, batch, batch_idx):
        # 1. prepare condition
        context_embeddings, gt_latents, gt_depth_latents = self._prepare_condition(batch)
        B, C, T, MV, H, W = gt_latents.shape

        # Apply text dropout for CFG training
        text_dropout_prob = getattr(self.config, "text_dropout_prob", 0.1)
        if text_dropout_prob > 0:
            drop_mask = torch.rand(B, generator=self.generator_per_rank).to(self.device) < text_dropout_prob
            if drop_mask.any():
                null_context = self.text_encoder([""] * B)
                context_embeddings = torch.where(drop_mask.view(B, 1, 1), null_context, context_embeddings)

        gt_actions = batch["actions"].float()
        gt_proprios = batch["proprios"].float()
        action_valid_mask = batch["action_mask"].float()
        proprio_valid_mask = batch["proprio_mask"].float()
        action_proprio_loss_sample_mask = torch.ones((B,))

        # 2. sample timesteps and add noise to video
        if self.config.use_decoupled_sampling:
            if self.config.use_joint_distribution:
                # Case discussion:
                # 1. Action already decoded, uniformly sample video timesteps
                uniform_video_timesteps = torch.rand((B,), generator=self.generator_per_rank)
                clean_action_timesteps = torch.zeros((B,))
                # 2. Action not yet decoded: uniform action + high-noise video, avoiding low-noise video with high-noise action
                uniform_action_timesteps = torch.rand((B,), generator=self.generator_per_rank)
                high_noise_video_timesteps = sample_beta((B,), generator=self.generator_per_rank, alpha=1.5, beta=1.0)
                high_noise_video_timesteps = uniform_action_timesteps + high_noise_video_timesteps * (
                    1 - uniform_action_timesteps
                )
                # Randomly assign the ratio between the two cases
                clean_action_mask = (
                    torch.rand((B,), generator=self.generator_per_rank) < self.config.clean_action_ratio
                ).float()
                timesteps = (
                    clean_action_mask * uniform_video_timesteps + (1 - clean_action_mask) * high_noise_video_timesteps
                )
                timesteps = self.config.time_shifting * timesteps / (1 + (self.config.time_shifting - 1) * timesteps)
                timesteps = timesteps.to(self.device)
                timesteps_a = (
                    clean_action_mask * clean_action_timesteps + (1 - clean_action_mask) * uniform_action_timesteps
                )
                timesteps_a = (
                    self.config.time_shifting * timesteps_a / (1 + (self.config.time_shifting - 1) * timesteps_a)
                )
                timesteps_a = timesteps_a.to(self.device)
                action_proprio_loss_sample_mask = 1 - clean_action_mask
            else:
                # DreamZero-style decoupled sampling
                # video use beta distribution with more weight on high noise
                # timesteps = sample_beta((B,), generator=self.generator_per_rank, alpha=3.0, beta=1.0)
                timesteps = torch.rand((B,), generator=self.generator_per_rank)
                timesteps = self.config.time_shifting * timesteps / (1 + (self.config.time_shifting - 1) * timesteps)
                timesteps = timesteps.to(self.device)

                # action use uniform distribution
                timesteps_a = torch.rand((B,), generator=self.generator_per_rank)
                timesteps_a = (
                    self.config.time_shifting * timesteps_a / (1 + (self.config.time_shifting - 1) * timesteps_a)
                )
                timesteps_a = timesteps_a.to(self.device)
        else:
            # video and action use the same timesteps
            if self.config.rf_distribution == "uniform":
                timesteps = torch.rand((B,), generator=self.generator_per_rank)
            elif self.config.rf_distribution == "beta":
                timesteps = sample_beta((B,), generator=self.generator_per_rank, alpha=1.5, beta=1.0)
            else:
                raise ValueError(f"Unsupported RF distribution: {self.config.rf_distribution}")
            timesteps = self.config.time_shifting * timesteps / (1 + (self.config.time_shifting - 1) * timesteps)
            timesteps = timesteps.to(self.device)
            timesteps_a = timesteps

        # create random noise, get xt and vt
        noise_latents = torch.randn(gt_latents.shape, generator=self.generator_per_rank, dtype=gt_latents.dtype).to(
            self.device
        )
        t = timesteps.view(noise_latents.shape[0], *([1] * (len(noise_latents.shape) - 1)))
        xt_latents = (1 - t) * gt_latents + t * noise_latents
        vt_latents = noise_latents - gt_latents

        noise_actions = torch.randn(gt_actions.shape, generator=self.generator_per_rank, dtype=gt_actions.dtype).to(
            self.device
        )
        t_actions = timesteps_a.view(noise_actions.shape[0], *([1] * (len(noise_actions.shape) - 1)))
        xt_actions = (1 - t_actions) * gt_actions + t_actions * noise_actions
        vt_actions = noise_actions - gt_actions

        noise_proprios = torch.randn(gt_proprios.shape, generator=self.generator_per_rank, dtype=gt_proprios.dtype).to(
            self.device
        )
        t_proprios = timesteps_a.view(noise_proprios.shape[0], *([1] * (len(noise_proprios.shape) - 1)))
        xt_proprios = (1 - t_proprios) * gt_proprios + t_proprios * noise_proprios
        vt_proprios = noise_proprios - gt_proprios

        # apply conditional mask
        latent_mask = torch.zeros((B, 1, T, 1, 1, 1), dtype=torch.long, device=self.device)
        latent_mask[:, :, 0] = 1
        xt_latents = gt_latents * latent_mask + xt_latents * (1 - latent_mask)

        action_mask = torch.zeros((B, gt_actions.shape[1], 1), dtype=torch.long, device=self.device)
        xt_actions = gt_actions * action_mask + xt_actions * (1 - action_mask)

        proprio_mask = torch.zeros((B, gt_proprios.shape[1], 1), dtype=torch.long, device=self.device)
        proprio_mask[:, 0] = 1
        xt_proprios = gt_proprios * proprio_mask + xt_proprios * (1 - proprio_mask)

        # 3. denoise video latents
        latent_timesteps = (
            timesteps.view(B, 1) * (1 - latent_mask).view(B, T) * self.config.flow_matching_num_train_timesteps
        )
        action_timesteps = (
            timesteps_a.view(B, 1)
            * (1 - action_mask).view(B, gt_actions.shape[1])
            * self.config.flow_matching_num_train_timesteps
        )
        proprio_timesteps = (
            timesteps_a.view(B, 1)
            * (1 - proprio_mask).view(B, gt_proprios.shape[1])
            * self.config.flow_matching_num_train_timesteps
        )
        vt_latents_pred, vt_actions_pred, vt_proprios_pred, depth_latents_pred = self.model(
            xt_latents,
            latent_timesteps,
            context_embeddings,
            actions=xt_actions,
            t_actions=action_timesteps,
            proprios=xt_proprios,
            t_proprios=proprio_timesteps,
            run_depth=self.run_depth,
        )

        # 4. compute loss
        video_mask = (1 - latent_mask).float().expand_as(vt_latents)
        # When actions are treated as clean conditions, do not train action denoising on those samples.
        action_proprio_loss_sample_mask = action_proprio_loss_sample_mask.to(self.device).view(B, 1, 1)
        action_mask_f = (
            (action_proprio_loss_sample_mask * (1 - action_mask) * action_valid_mask).float().expand_as(vt_actions)
        )
        proprio_mask_f = (
            (action_proprio_loss_sample_mask * (1 - proprio_mask) * proprio_valid_mask).float().expand_as(vt_proprios)
        )

        video_loss = ((vt_latents_pred - vt_latents) * video_mask).pow(2).sum() / (video_mask.sum() + 1e-8)
        action_loss = ((vt_actions_pred - vt_actions) * action_mask_f).pow(2).sum() / (action_mask_f.sum() + 1e-8)
        proprio_loss = ((vt_proprios_pred - vt_proprios) * proprio_mask_f).pow(2).sum() / (proprio_mask_f.sum() + 1e-8)
        if self.config.use_depth and self.run_depth:
            depth_loss = (depth_latents_pred[0] - gt_depth_latents).pow(2).mean()
        else:
            depth_loss = 0.0

        # frequency-domain loss for action smoothness
        dct_loss_weight = getattr(self.config, "dct_loss_weight", 0.0)
        if dct_loss_weight > 0:
            # Only compute for samples with fully valid action sequences
            batch_mask = action_mask_f.any(dim=-1).all(dim=1)
            if batch_mask.any():
                pred_valid = vt_actions_pred[batch_mask]
                target_valid = vt_actions[batch_mask]
                dct_loss = (torch.fft.rfft(pred_valid, dim=1) - torch.fft.rfft(target_valid, dim=1)).abs().mean()
            else:
                dct_loss = torch.zeros(1, device=self.device).squeeze()
        else:
            dct_loss = 0.0

        loss = (
            video_loss
            + self.config.action_loss_weight * action_loss
            + self.config.proprio_loss_weight * proprio_loss
            + self.config.depth_loss_weight * depth_loss
            + dct_loss_weight * dct_loss
        )

        # Log info tensors
        self.log("train/video_loss", video_loss, prog_bar=True, sync_dist=True)
        self.log("train/action_loss", action_loss, prog_bar=True, sync_dist=True)
        self.log("train/proprio_loss", proprio_loss, prog_bar=True, sync_dist=True)
        self.log("train/depth_loss", depth_loss, prog_bar=True, sync_dist=True)
        if dct_loss_weight > 0:
            self.log("train/dct_loss", dct_loss, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # First, center crop the video and depths
        if hasattr(self.config, "datasets"):
            crop_ratio = list(self.config.datasets.values())[0].crop_ratio
        else:
            crop_ratio = self.config.dataset.crop_ratio

        H, W = batch["video"].shape[-2:]
        crop_h = max(1, int(H * crop_ratio))
        crop_w = max(1, int(W * crop_ratio))

        if crop_h != H or crop_w != W:
            top = torch.randint(0, H - crop_h + 1, ()).item()
            left = torch.randint(0, W - crop_w + 1, ()).item()

            rgb_crop = batch["video"][..., top : top + crop_h, left : left + crop_w]
            depth_crop = batch["depths"][..., top : top + crop_h, left : left + crop_w]

            ori_shape = rgb_crop.shape[:-3]
            batch["video"] = F.interpolate(
                rgb_crop.flatten(0, -4),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).unflatten(0, ori_shape)

            ori_shape = depth_crop.shape[:-3]
            batch["depths"] = F.interpolate(
                depth_crop.flatten(0, -4),
                size=(H, W),
                mode="nearest",
            ).unflatten(0, ori_shape)

        cfg_list = list(getattr(self.config, "cfg_list", [0, 3, 7]))

        # Run generation for each CFG value and collect decoded videos
        # pred_videos_per_cfg: list of tensors shaped [t, (m h), (b v w), c]
        pred_videos_per_cfg = []
        psnr_logged = False
        for cfg_val in cfg_list:
            xt_latents, xt_actions, xt_proprios, xt_depth_latents = self.forward(batch, early_stop=False, cfg=cfg_val)
            if self.config.use_depth and self.run_depth:
                xt_latents_mv = torch.cat([xt_latents, xt_depth_latents], dim=3)
            else:
                xt_latents_mv = xt_latents
            xt_latents_mv = rearrange(xt_latents_mv, "b c t (m v) h w -> (b m v) c t h w", v=self.num_views)
            pred_videos = self.vae.decode(xt_latents_mv)

            # Log PSNR / SSIM only once (for cfg=0 or first entry)
            if not psnr_logged:
                B = batch["video"].shape[0]
                pred_all = rearrange(
                    pred_videos,
                    "(b m v) c t h w -> m (b v t) c h w",
                    b=B,
                    v=self.num_views,
                )
                pred_rgb = torch.clamp((pred_all[0] + 1) / 2, 0, 1)
                gt_rgb = rearrange(batch["video"], "b v t c h w -> (b v t) c h w")
                gt_rgb = torch.clamp((gt_rgb + 1) / 2, 0, 1)
                rgb_psnr = self._psnr(pred_rgb, gt_rgb)
                rgb_ssim = self._ssim(pred_rgb, gt_rgb)
                self.log("val/rgb_psnr", rgb_psnr, prog_bar=True, sync_dist=True)
                self.log("val/rgb_ssim", rgb_ssim, prog_bar=True, sync_dist=True)
                if self.config.use_depth and self.run_depth:
                    pred_depth = torch.clamp((pred_all[1] + 1) / 2, 0, 1)
                    gt_depth = rearrange(batch["depths"], "b v t c h w -> (b v t) c h w")
                    gt_depth = torch.clamp((gt_depth + 1) / 2, 0, 1)
                    depth_psnr = self._psnr(pred_depth, gt_depth)
                    depth_ssim = self._ssim(pred_depth, gt_depth)
                    self.log("val/depth_psnr", depth_psnr, prog_bar=True, sync_dist=True)
                    self.log("val/depth_ssim", depth_ssim, prog_bar=True, sync_dist=True)

                gt_actions = batch["actions"].float()
                action_valid_mask = batch["action_mask"].float()
                action_mse = ((xt_actions - gt_actions) * action_valid_mask).pow(2).sum() / (
                    action_valid_mask.sum() + 1e-8
                )
                self.log("val/action_mse", action_mse, prog_bar=True, sync_dist=True)

                gt_proprios = batch["proprios"].float()
                proprio_valid_mask = batch["proprio_mask"].float()
                proprio_mse = ((xt_proprios - gt_proprios) * proprio_valid_mask).pow(2).sum() / (
                    proprio_valid_mask.sum() + 1e-8
                )
                self.log("val/proprio_mse", proprio_mse, prog_bar=True, sync_dist=True)

                psnr_logged = True

            # Rearrange to [t, (m h), (b v w), c] for video writing
            pred_videos_per_cfg.append(
                rearrange(
                    pred_videos, "(b m v) c t h w -> t (m h) (b v w) c", b=batch["video"].shape[0], v=self.num_views
                )
            )

        # Save videos on rank 0 only
        if self.global_rank == 0:
            if self.config.use_depth and self.run_depth:
                gt_videos = torch.cat([batch["video"], batch["depths"]], dim=0)
            else:
                gt_videos = batch["video"]
            gt_videos = rearrange(gt_videos, "(m b) v t c h w -> t (m h) (b v w) c", b=batch["video"].shape[0])

            # Stack all CFG results + GT vertically along height axis
            all_videos = torch.cat(pred_videos_per_cfg + [gt_videos], dim=1)
            all_videos = torch.clamp((all_videos + 1) * 127.5, 0, 255).byte().cpu().numpy()

            video_path = os.path.join(self.trainer.default_root_dir, f"videos/{self.trainer.global_step}")
            os.makedirs(video_path, exist_ok=True)

            writer = imageio.get_writer(
                os.path.join(video_path, f"{batch_idx}.mp4"), fps=self.config.sample_fps, codec="libx264", quality=8
            )
            for frame in all_videos:
                writer.append_data(frame)
            writer.close()

        return None

    def forward(self, batch, seeds=None, early_stop=False, cfg=0.0):
        # 1. prepare condition
        context_embeddings, gt_latents, _ = self._prepare_condition(batch)
        B, C, T, MV, H, W = gt_latents.shape
        gt_actions = batch["actions"].float()
        gt_proprios = batch["proprios"].float()

        # Prepare uncond embeddings for CFG (encode empty strings once)
        if cfg > 0.0:
            uncond_embeddings = self.text_encoder([""] * B)
            context_for_model = torch.cat([context_embeddings, uncond_embeddings], dim=0)
        else:
            context_for_model = context_embeddings

        # 2. sample timesteps and add noise to video
        if seeds is not None:
            # Per-element reproducible noise using hashed seeds
            noise_latents_list, noise_actions_list, noise_proprios_list = [], [], []
            for i, seed in enumerate(seeds):
                gen = torch.Generator(device=self.device).manual_seed(int(seed))
                noise_latents_list.append(
                    torch.randn(gt_latents[i : i + 1].shape, generator=gen, device=self.device, dtype=gt_latents.dtype)
                )
                noise_actions_list.append(
                    torch.randn(gt_actions[i : i + 1].shape, generator=gen, device=self.device, dtype=gt_actions.dtype)
                )
                noise_proprios_list.append(
                    torch.randn(
                        gt_proprios[i : i + 1].shape, generator=gen, device=self.device, dtype=gt_proprios.dtype
                    )
                )
            noise_latents = torch.cat(noise_latents_list, dim=0)
            noise_actions = torch.cat(noise_actions_list, dim=0)
            noise_proprios = torch.cat(noise_proprios_list, dim=0)
        else:
            noise_latents = torch.randn(gt_latents.shape, generator=self.generator_per_rank, dtype=gt_latents.dtype).to(
                self.device
            )
            noise_actions = torch.randn(gt_actions.shape, generator=self.generator_per_rank, dtype=gt_actions.dtype).to(
                self.device
            )
            noise_proprios = torch.randn(
                gt_proprios.shape, generator=self.generator_per_rank, dtype=gt_proprios.dtype
            ).to(self.device)

        # apply conditional mask
        latent_mask = torch.zeros((B, 1, T, 1, 1, 1), dtype=torch.long, device=self.device)
        latent_mask[:, :, 0] = 1
        xt_latents = gt_latents * latent_mask + noise_latents * (1 - latent_mask)

        action_mask = torch.zeros((B, gt_actions.shape[1], 1), dtype=torch.long, device=self.device)
        xt_actions = gt_actions * action_mask + noise_actions * (1 - action_mask)

        proprio_mask = torch.zeros((B, gt_proprios.shape[1], 1), dtype=torch.long, device=self.device)
        proprio_mask[:, 0] = 1
        xt_proprios = gt_proprios * proprio_mask + noise_proprios * (1 - proprio_mask)

        if self.config.use_decoupled_inference:
            action_denoise_steps = self.config.action_denoise_steps
        else:
            action_denoise_steps = self.config.sample_steps

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.flow_matching_num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            self.config.sample_steps,
            device=self.device,
            shift=self.config.time_shifting,
        )

        sample_scheduler_actions = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.flow_matching_num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_actions.set_timesteps(
            action_denoise_steps,
            device=self.device,
            shift=self.config.time_shifting,
        )

        sample_scheduler_proprios = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.flow_matching_num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_proprios.set_timesteps(
            action_denoise_steps,
            device=self.device,
            shift=self.config.time_shifting,
        )

        video_timesteps = sample_scheduler.timesteps
        action_timesteps = sample_scheduler_actions.timesteps
        proprio_timesteps = sample_scheduler_proprios.timesteps

        # 3. denoise video latents
        for ti in range(self.config.sample_steps):
            video_t = video_timesteps[ti]
            if self.config.use_decoupled_inference and ti >= action_denoise_steps:
                if early_stop:
                    break
                action_t = 0
                proprio_t = 0
            else:
                action_t = action_timesteps[ti]
                proprio_t = proprio_timesteps[ti]

            latent_ts = video_t * (1 - latent_mask).view(B, T)
            action_ts = action_t * (1 - action_mask).view(B, gt_actions.shape[1])
            proprio_ts = proprio_t * (1 - proprio_mask).view(B, gt_proprios.shape[1])
            vt_latents_pred, vt_actions_pred, vt_proprios_pred, depth_latents_pred = self.model(
                x=xt_latents,
                t=latent_ts,
                context=context_for_model,
                actions=xt_actions,
                t_actions=action_ts,
                proprios=xt_proprios,
                t_proprios=proprio_ts,
                cfg=cfg,
                run_depth=self.run_depth,
            )

            xt_latents = sample_scheduler.step(vt_latents_pred, video_t, xt_latents, return_dict=False)[0]
            xt_latents = gt_latents * latent_mask + xt_latents * (1 - latent_mask)

            if not self.config.use_decoupled_inference or ti < action_denoise_steps:
                xt_actions = sample_scheduler_actions.step(vt_actions_pred, action_t, xt_actions, return_dict=False)[0]
                xt_actions = gt_actions * action_mask + xt_actions * (1 - action_mask)

                xt_proprios = sample_scheduler_proprios.step(
                    vt_proprios_pred, proprio_t, xt_proprios, return_dict=False
                )[0]
                xt_proprios = gt_proprios * proprio_mask + xt_proprios * (1 - proprio_mask)

        xt_depth_latents = depth_latents_pred[0] if self.config.use_depth and self.run_depth else None
        return xt_latents, xt_actions, xt_proprios, xt_depth_latents

    def generate(self, rgb, proprio, prompt, seeds=None, early_stop=True, cfg=0.0, run_depth=True):
        """
        Args:
            rgb: [B, V, C, H, W]
            proprio: [B, Dp]
            prompt: list[str] with length B
            prompt_embedding: [B, L, text_dim]
            seeds: optional list[int] of length B for reproducible noise generation
            cfg: classifier-free guidance scale
        """
        B = rgb.shape[0]
        T = self.config.frame_num  # frame_num is the number of RGB frames in the video
        Ta = (self.config.frame_num - 1) * self.model.action_num
        Tp = self.config.frame_num

        batch = {
            "video": rgb.unsqueeze(2).repeat(1, 1, T, 1, 1, 1),
            "proprios": proprio.unsqueeze(1).repeat(1, Tp, 1),
            "actions": torch.zeros((B, Ta, self.config.action_dim)).to(proprio.device, dtype=proprio.dtype),
            "prompt": prompt,
        }

        self.run_depth = run_depth
        xt_latents, xt_actions, xt_proprios, xt_depth_latents = self.forward(
            batch, seeds=seeds, early_stop=early_stop, cfg=cfg
        )

        if early_stop:
            return None, xt_actions, xt_proprios, None

        if self.config.use_depth and self.run_depth:
            xt_latents = torch.cat([xt_latents, xt_depth_latents], dim=3)
        xt_latents = rearrange(xt_latents, "b c t (m v) h w -> (b m v) c t h w", v=self.num_views)
        pred_videos = self.vae.decode(xt_latents)
        pred_videos = rearrange(
            pred_videos, "(b m v) c t h w -> t (m h) (b v w) c", b=batch["video"].shape[0], v=self.num_views
        )

        pred_videos = torch.clamp((pred_videos + 1) * 127.5, 0, 255).byte().cpu().numpy()

        return pred_videos, xt_actions, xt_proprios, xt_depth_latents

    @torch.no_grad()
    def _psnr(self, pred, gt):
        """pred, gt: (N, C, H, W) in [0, 1]. Returns mean PSNR over the batch."""
        mse = (pred - gt).pow(2).mean(dim=[1, 2, 3])
        return (10.0 * torch.log10(1.0 / (mse + 1e-8))).mean()

    @torch.no_grad()
    def _ssim(self, pred, gt, window_size=11):
        """pred, gt: (N, C, H, W) in [0, 1]. Returns mean SSIM over the batch."""
        C = pred.shape[1]
        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(C, 1, -1, -1).contiguous()

        c1, c2 = 0.01**2, 0.03**2

        with torch.amp.autocast("cuda", dtype=torch.float32):
            mu1 = F.conv2d(pred, kernel, groups=C)
            mu2 = F.conv2d(gt, kernel, groups=C)
            mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2

            sigma1_sq = F.conv2d(pred.pow(2), kernel, groups=C) - mu1_sq
            sigma2_sq = F.conv2d(gt.pow(2), kernel, groups=C) - mu2_sq
            sigma12 = F.conv2d(pred * gt, kernel, groups=C) - mu12

            numer = (2 * mu12 + c1) * (2 * sigma12 + c2)
            denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        return (numer / denom).mean()

    def _prepare_condition(self, batch):
        # context embeddings: [B, L, C]
        context_embeddings = self.text_encoder(batch["prompt"])

        # rgbd: [B, C, MV, T, H, W]
        B = batch["video"].shape[0]
        gt_rgb = rearrange(batch["video"], "b v t c h w -> (b v) c t h w")
        if self.config.use_depth and self.run_depth and "depths" in batch:
            gt_depth = rearrange(batch["depths"], "b v t c h w -> (b v) c t h w")
            gt_video = torch.cat([gt_rgb, gt_depth], dim=0)
            gt_latents = self.vae.encode(gt_video)
            gt_latents = rearrange(gt_latents, "(m b v) c t h w -> m b c t v h w", b=B, v=self.num_views)
            return context_embeddings, gt_latents[0], gt_latents[1]

        gt_latents = self.vae.encode(gt_rgb)
        gt_latents = rearrange(gt_latents, "(b v) c t h w -> b c t v h w", b=B, v=self.num_views)
        return context_embeddings, gt_latents, None
