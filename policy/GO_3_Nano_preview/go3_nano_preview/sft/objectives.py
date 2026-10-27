"""Reference three-objective SFT, with eager candidate scoring and noise replay."""
from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from policy_core.utils.WAM_policy.WAM_policy_config import as_dataset_kwargs
from go3_nano_preview.sft.task_model import SFTTaskModel
from go3_nano_preview.sft.loss import compute_action_loss
from go3_nano_preview.sft.video_loss import compute_video_loss
from go3_nano_preview.sft.vlm import compute_vlm_hidden_states
from go3_nano_preview.sft.helpers.ae_judge import ae_judge_fixed_t, make_noise_replay_hook
from go3_nano_preview.sft.helpers.two_step_cotrain import (
    setup_two_step_inputs, repeat_members, select_topk_members,
    confidence_weights, slice_two_step_ctx, gather_member_noise,
    build_two_step_diagnostics, apply_filter,
)


class Cotrain(torch.nn.Module):
    def __init__(self, flow, cfg):
        super().__init__()
        self.model = SFTTaskModel(vgen=flow.video_diffusion_model,
                                 action_expert=flow.action_diffusion_model)
        self.cfg = cfg
        object.__setattr__(self, 'vae', flow.vae.requires_grad_(False).eval())
        object.__setattr__(self, 'stats', flow.stats_tracker.requires_grad_(False).eval())
        self.scheduler = FlowMatchEulerDiscreteScheduler(**cfg.model.diffusion_scheduler_args)
        self.scheduler_sigmas = self.scheduler.sigmas.to(next(self.model.parameters()).device)
        self.frame_indices = flow.video_frame_indices
        self.scale = flow.ae_video_scaling_factor
        self.spatial = flow.spatial_down_ratio
        self.temporal = flow.temporal_down_ratio
        self.data = as_dataset_kwargs(cfg.data.train)
        self.cache = {}
        self.diagnostics = {}
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.model.vgen.vlm_interface.requires_grad_(False).eval()
        return self

    def prepare_vgen_text_conditioning(self, captions):
        return dict(prompt_embeds=None, prompt_attention_mask=None)

    def clear_window(self):
        self.cache.clear()

    def _features(self, batch):
        if not self.cache:
            device = next(self.model.parameters()).device
            video = batch['video'].to(device, dtype=torch.bfloat16)[:, :, :, self.frame_indices]
            b, _, views, _, _, _ = video.shape
            inputs, hidden, mask = compute_vlm_hidden_states(
                self.model.vgen, video, batch['caption'], use_vlm=True,
                vlm_text_only=False, batch_size=b, n_view=views,
                mem_size=self.data['n_previous'], with_grad=False)
            self.cache.update(video=video, vlm_inputs=inputs, hidden=hidden, mask=mask)
        return self.cache

    def _vgen(self, batch, video, hidden, mask, generator, step, *, one_step,
              vae_latents=None, noise_hook=None):
        return self.model.train_vgen_forward(
            batch=batch, video=video, batch_size=video.shape[0], n_view=video.shape[2],
            weight_dtype=torch.bfloat16, generator=generator,
            prompt_embeds=None, prompt_attention_mask=None,
            vlm_inputs=self.cache['vlm_inputs'], precomputed_vlm_hidden_states=hidden,
            gradient_vgen2ae=one_step, cfg=self.cfg, vae=self.vae,
            stats_tracker=self.stats, scheduler=self.scheduler, lpips_model=None,
            use_video_mixed_path=False, use_video_sparse_mode=False,
            video_sparse_downsample_factor=1, video_sparse_spatial_rope_mode='center',
            video_frame_indices=self.frame_indices, spatial_down_ratio=self.spatial,
            temporal_down_ratio=self.temporal, timestep_scale=self.cfg.cotrain.timestep_scale,
            expert_dual_train=False, phase_role='cotrain' if one_step else 'vgen',
            global_step=step, data_config=self.data, vlm_attention_mask=mask,
            one_step_gen_only=one_step, precomputed_vae_latents=vae_latents,
            noise_hook=noise_hook)

    def _ae(self, batch, video, latents, weights=None):
        outputs = self.model.train_ae_forward(
            batch=batch, video=video, n_view=video.shape[2], weight_dtype=torch.bfloat16,
            cfg=self.cfg, vae=self.vae, stats_tracker=self.stats,
            scheduler=self.scheduler, scheduler_sigmas=self.scheduler_sigmas,
            ae_video_scaling_factor=self.scale, video_latents_for_ae=latents,
            data_config=self.data, sample_weights=weights)
        return compute_action_loss(outputs)['loss_action'] * self.cfg.cotrain.action_loss_scale

    def forward(self, batch, generator, step, objective):
        cache = self._features(batch)
        video, hidden, mask = cache['video'], cache['hidden'], cache['mask']
        self.diagnostics = {}
        if objective == 'vgen':
            outputs = self._vgen(batch, video, hidden, mask, generator, step, one_step=False)
            cache['vae_raw'] = {k: v.detach() if v is not None else None
                                for k, v in outputs['vae_latents_raw'].items()}
            loss = compute_video_loss(self.cfg, **outputs['_vgen_loss_inputs'])['loss_video']
            loss = loss * self.cfg.cotrain.gen_loss_scale
        elif objective == 'ae':
            raw = cache['vae_raw']
            latents = torch.cat([raw['mem_latents'], raw['future_video_latents']], dim=2)
            loss = self._ae(batch, video, latents)
        elif objective == 'cotrain':
            loss = self._two_step(batch, video, hidden, mask, generator, step)
        else:
            raise ValueError(f'Unsupported objective: {objective}')
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite {objective} loss')
        return loss

    def _two_step(self, batch, video, hidden, mask, generator, step):
        views = video.shape[2]
        ctx = setup_two_step_inputs(self, self.cfg, batch, video, batch['caption'],
                                   hidden, video.shape[0], precomputed_vlm_mask=mask)
        config = ctx.config
        raw = {k: repeat_members(v, config.group_size, views) if v is not None else None
               for k, v in self.cache['vae_raw'].items()}
        with torch.no_grad():
            candidates = self._vgen(ctx.vg_batch, ctx.vg_video, ctx.vg_precomputed_vlm,
                ctx.vg_vlm_attention_mask, generator, step, one_step=True,
                vae_latents=raw, noise_hook=ctx.capture_hook)
            judge = SimpleNamespace(action_diffusion_model=self.model.ae, cfg=self.cfg,
                                    stats_tracker=self.stats, vae=self.vae,
                                    ae_video_scaling_factor=self.scale)
            rewards, energy = ae_judge_fixed_t(judge,
                gen_latents=candidates['video_latents_for_ae'],
                ref_latents=candidates['gt_video_latents_for_ae'], batch=batch,
                n_view=views, sparse_length=int(self.data['sparse_length']),
                data_config=self.data, config=config, weight_dtype=torch.bfloat16,
                return_ref_energy=True, generator=generator)
            valid, extra, _ = apply_filter(rewards, config)
            selected = select_topk_members(rewards, config.backward_topk)
            weights = confidence_weights(rewards, energy, valid, selected, config)
            self.diagnostics = build_two_step_diagnostics(rewards, valid)
            self.diagnostics.update(extra)
            self.diagnostics['ts_conf_weight_mean'] = weights.mean()
            self.diagnostics['ts_v_ref_energy_mean'] = energy.mean()
            selected_ctx = slice_two_step_ctx(ctx, selected, views)
            noise = gather_member_noise(ctx.noise_holder['noise'], selected, ctx.vg_batch_size, views)
            selected_raw = {k: gather_member_noise(v, selected, ctx.vg_batch_size, views)
                            if v is not None else None for k, v in raw.items()}
        del candidates, raw
        generated = self._vgen(selected_ctx.vg_batch, selected_ctx.vg_video,
            selected_ctx.vg_precomputed_vlm, selected_ctx.vg_vlm_attention_mask,
            generator, step, one_step=True, vae_latents=selected_raw,
            noise_hook=make_noise_replay_hook(noise))
        original = self.cfg.data.aug_vision_token
        self.cfg.data.aug_vision_token = False
        try:
            # Rejected winners retain a zero-weight graph, including on empty ranks.
            return self._ae(selected_ctx.vg_batch, selected_ctx.vg_video,
                            generated['video_latents_for_ae'], weights)
        finally:
            self.cfg.data.aug_vision_token = original
