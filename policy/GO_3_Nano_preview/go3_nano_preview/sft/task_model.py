"""SFT forwards extracted from the reference training implementation."""
from __future__ import annotations
import contextlib
import logging
import math
import random
from typing import Any
import kornia
import torch
import torch.nn as nn
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import rearrange
from torch.func import jvp
from go3_nano_preview.sft.jvp_math import run_imf_jvp_fast, ltx_dit_jvp
from policy_core.utils.WAM_policy.data_utils import apply_color_jitter_to_video, gen_noise_from_condition_frame_latent, get_latents, randn_tensor
from policy_core.utils.WAM_policy.WAM_policy_config import as_dataset_kwargs
from go3_nano_preview.sft.meanflow import inverse_shift, sample_improved_meanflow_times
from policy_core.utils.common_utils import get_device
from policy_core.utils.WAM_policy.model_utils import forward_pass
from policy_core.utils.WAM_policy.video_sparse_utils import build_frame_latent_spec, build_mixed_conditioning_tokens, build_mixed_video_layout, _encode_2dvae_frames, encode_mixed_video_2dvae, expand_frame_indices_for_views, normalize_sparse_downsample_factor, pack_frame_latent_groups

def _apply_rotary_emb(x: torch.Tensor, freqs):
    (cos, sin) = freqs
    (x_real, x_imag) = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

def _random_temporal_token_keep(x: torch.Tensor, keep_ratio: float, mask_cur_obs: bool=False):
    (batch_size, num_frames, num_tokens, channels) = x.shape
    assert 0 < keep_ratio <= 1.0, 'keep_ratio must be in (0, 1]'
    kept_len = int(num_tokens * keep_ratio)
    if not mask_cur_obs:
        x_first = x[:, 0]
        x_rest = x[:, 1:]
        scores = torch.rand(batch_size, num_frames - 1, num_tokens, device=x.device)
        indices = scores.topk(kept_len, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
        x_rest_selected = torch.gather(x_rest, dim=2, index=indices)
        x_rest_selected = x_rest_selected.reshape(batch_size, (num_frames - 1) * kept_len, channels)
    else:
        scores = torch.rand(batch_size, num_frames, num_tokens, device=x.device)
        indices = scores.topk(kept_len, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
        x_selected = torch.gather(x, dim=2, index=indices)
        x_first = x_selected[:, 0]
        x_rest_selected = x_selected[:, 1:].reshape(batch_size, (num_frames - 1) * kept_len, channels)
    return (x_first, x_rest_selected)

def _augment_vision_latents(x: torch.Tensor, n_frame: int, h: int, w: int, noise_std: float=0.05, blur_prob: float=0.5) -> torch.Tensor:
    (batch_views, length, channels) = x.shape
    if random.random() < blur_prob:
        x_spatial = x.view(batch_views, n_frame, h, w, channels).permute(0, 1, 4, 2, 3)
        x_spatial = x_spatial.reshape(-1, channels, h, w)
        max_kernel = min(h, w) - 1
        if max_kernel >= 3:
            kernel_size = min(random.randint(3, 5), max_kernel)
            if kernel_size % 2 == 0:
                kernel_size -= 1
            sigma = random.uniform(0.1, 1.3)
            x_spatial = kornia.filters.gaussian_blur2d(x_spatial, (kernel_size, kernel_size), (sigma, sigma))
            x = x_spatial.view(batch_views, n_frame, channels, h, w).permute(0, 1, 3, 4, 2).reshape(batch_views, length, channels)
    return x + torch.randn_like(x) * noise_std
from policy_core.model.WAM_policy.task.WAM_policy_task_model import WAM_policyTaskModel as InferenceTaskModel
logger = logging.getLogger(__name__)

class SFTTaskModel(InferenceTaskModel):

    @staticmethod
    def _profile_start(profiler, name: str) -> None:
        if profiler is not None:
            profiler.start_region(name)

    @staticmethod
    def _profile_end(profiler, name: str) -> None:
        if profiler is not None:
            profiler.end_region(name)

    @staticmethod
    def rescale_for_ae(latents, *, cfg: Any, stats_tracker, vae, ae_video_scaling_factor: float):
        if latents is None or stats_tracker is not None:
            return latents
        if not cfg.validation.ae_rescale:
            return latents
        ratio = cfg.validation.ae_rescale_ratio
        if ratio is None:
            ratio = float(ae_video_scaling_factor) / float(vae.config.scaling_factor)
        return latents * float(ratio)

    def _prepare_ae_latents(self, *, video, video_latents_for_ae, sparse_length: int, n_view: int, cfg: Any, vae, stats_tracker, ae_video_scaling_factor: float, apply_color_jitter: bool=False, profiler=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return canonical dense/sparse AE latents from pixels or latents."""
        if video_latents_for_ae is not None:
            if isinstance(video_latents_for_ae, tuple):
                (dense_latents, sparse_latents) = video_latents_for_ae
            else:
                n_frame = video_latents_for_ae.shape[2]
                if sparse_length <= 0 or sparse_length >= n_frame:
                    raise ValueError(f'Invalid sparse_length={sparse_length} for AE latent frames={n_frame}.')
                dense_latents = video_latents_for_ae[:, :, :-sparse_length]
                sparse_latents = video_latents_for_ae[:, :, -sparse_length:]
            dense_latents = self.rescale_for_ae(dense_latents, cfg=cfg, stats_tracker=stats_tracker, vae=vae, ae_video_scaling_factor=ae_video_scaling_factor)
            sparse_latents = self.rescale_for_ae(sparse_latents, cfg=cfg, stats_tracker=stats_tracker, vae=vae, ae_video_scaling_factor=ae_video_scaling_factor)
            return (dense_latents, sparse_latents)
        if video is None:
            raise ValueError('AE requires either video or video_latents_for_ae.')
        n_frame = video.shape[3]
        if sparse_length <= 0 or sparse_length >= n_frame:
            raise ValueError(f'Invalid sparse_length={sparse_length} for AE video frames={n_frame}.')
        raw_video = video
        if apply_color_jitter and cfg.data.use_color_jitter:
            raw_video = rearrange(raw_video, 'b c v t h w -> (b v) c t h w')
            raw_video = apply_color_jitter_to_video(raw_video, n_view=n_view if getattr(cfg.data, 'color_jitter_same_across_views', False) else 1, jitter_kwargs=getattr(cfg.data, 'color_jitter_kwargs', None))
            raw_video = rearrange(raw_video, '(b v) c t h w -> b c v t h w', v=n_view)
        n_frame_dense = n_frame - sparse_length
        dense_video = raw_video[:, :, :, :-sparse_length]
        sparse_video = raw_video[:, :, :, -sparse_length:]
        dense_frames = rearrange(dense_video, 'b c v t h w -> (b v t) c h w')
        sparse_frames = rearrange(sparse_video, 'b c v t h w -> (b v t) c h w')
        with torch.no_grad():
            self._profile_start(profiler, 'vae_encode')
            dense_latents = _encode_2dvae_frames(vae, dense_frames, cfg.model.batch_size_2dvae, stats_tracker)
            sparse_latents = _encode_2dvae_frames(vae, sparse_frames, cfg.model.batch_size_2dvae, stats_tracker)
            self._profile_end(profiler, 'vae_encode')
        dense_latents = rearrange(dense_latents, '(bv t) c h w -> bv c t h w', t=n_frame_dense)
        sparse_latents = rearrange(sparse_latents, '(bv t) c h w -> bv c t h w', t=sparse_length)
        return (dense_latents, sparse_latents)

    @staticmethod
    def ae_sparse_rope_args(cfg: Any) -> dict[str, Any]:
        factor = int(normalize_sparse_downsample_factor(cfg.cotrain.video_sparse_downsample_factor))
        return {'is_sparse': factor > 1, 'factor': factor}

    def _build_ae_vision_condition(self, *, ae_latents_dense: torch.Tensor, ae_latents_sparse: torch.Tensor, n_view: int, cfg: Any, augment_vision: bool=False, repeat_in_batch: int=1) -> tuple[torch.Tensor, list[int] | None]:
        """Build the shared AE RoPE and temporal-token vision condition."""
        n_frame_dense = ae_latents_dense.shape[2]
        sparse_length = ae_latents_sparse.shape[2]
        (_, _, _, l_h_dense, l_w_dense) = ae_latents_dense.shape
        (_, _, _, l_h_sparse, l_w_sparse) = ae_latents_sparse.shape
        dense_tokens = rearrange(ae_latents_dense, 'bv c t h w -> bv (t h w) c')
        sparse_tokens = rearrange(ae_latents_sparse, 'bv c t h w -> bv (t h w) c')
        if augment_vision:
            dense_tokens = _augment_vision_latents(dense_tokens, n_frame=n_frame_dense, h=l_h_dense, w=l_w_dense, noise_std=random.uniform(0.01, 0.1))
            sparse_tokens = _augment_vision_latents(sparse_tokens, n_frame=sparse_length, h=l_h_sparse, w=l_w_sparse, noise_std=random.uniform(0.01, 0.15))
        dense_rope = self.action_expert.get_vision_rope(dense_tokens, n_frame_dense, l_h_dense, l_w_dense, is_sparse=False, factor=1)
        dense_tokens = _apply_rotary_emb(dense_tokens, dense_rope)
        dense_tokens = rearrange(dense_tokens, '(b v) l c -> b v l c', v=n_view)
        sparse_offset = torch.zeros((sparse_tokens.shape[0], n_frame_dense * l_h_sparse * l_w_sparse, sparse_tokens.shape[-1]), device=sparse_tokens.device, dtype=sparse_tokens.dtype)
        sparse_with_offset = torch.cat([sparse_offset, sparse_tokens], dim=1)
        sparse_rope = self.action_expert.get_vision_rope(sparse_with_offset, n_frame_dense + sparse_length, l_h_sparse, l_w_sparse, **self.ae_sparse_rope_args(cfg))
        sparse_tokens = _apply_rotary_emb(sparse_with_offset, sparse_rope)[:, sparse_offset.shape[1]:]
        sparse_tokens = rearrange(sparse_tokens, '(b v) l c -> b v l c', v=n_view)
        permutation = None
        if repeat_in_batch > 1:
            dense_tokens = dense_tokens.repeat(repeat_in_batch, 1, 1, 1)
            sparse_tokens = sparse_tokens.repeat(repeat_in_batch, 1, 1, 1)
            permutation = torch.randperm(dense_tokens.shape[0]).tolist()
            dense_tokens = dense_tokens[permutation]
            sparse_tokens = sparse_tokens[permutation]
        dense_tokens = rearrange(dense_tokens, 'b v (t h w) c -> (b v) t (h w) c', t=n_frame_dense, h=l_h_dense, w=l_w_dense)
        sparse_tokens = rearrange(sparse_tokens, 'b v (t h w) c -> (b v) t (h w) c', t=sparse_length, h=l_h_sparse, w=l_w_sparse)
        (dense_current, dense_history) = _random_temporal_token_keep(dense_tokens, keep_ratio=1 / float(cfg.data.mask_ratio), mask_cur_obs=False)
        (sparse_current, sparse_history) = _random_temporal_token_keep(sparse_tokens, keep_ratio=1 / float(cfg.data.mask_ratio), mask_cur_obs=True)
        dense_condition = dense_history if cfg.data.drop_cur_obs else torch.cat((dense_current, dense_history), dim=1)
        sparse_condition = torch.cat((sparse_current, sparse_history), dim=1)
        return (rearrange(torch.cat((dense_condition, sparse_condition), dim=1), '(b v) l c -> b v l c', v=n_view), permutation)

    def train_ae_forward(self, *, batch, video, n_view: int, weight_dtype: torch.dtype, device=None, cfg: Any, vae, stats_tracker, scheduler, scheduler_sigmas: torch.Tensor, ae_video_scaling_factor: float, video_latents_for_ae=None, data_config=None, sample_weights=None, profiler=None) -> dict[str, Any]:
        """Build Action Expert inputs and run its training forward.

        Loss reduction intentionally remains in ``WAM_policyTaskFlow``.
        """
        if device is None:
            device = get_device()
        if self.action_expert is None:
            raise RuntimeError('train_ae_forward requires an Action Expert.')
        data_config = as_dataset_kwargs(data_config or cfg.data.train)
        sparse_length = int(data_config['sparse_length'])
        if sample_weights is not None and cfg.data.repeat_in_batch > 1:
            raise ValueError('sample_weights is incompatible with repeat_in_batch > 1 (shuffled rows would misalign with the weight vector).')
        (ae_latents_dense, ae_latents_sparse) = self._prepare_ae_latents(video=video, video_latents_for_ae=video_latents_for_ae, sparse_length=sparse_length, n_view=n_view, cfg=cfg, vae=vae, stats_tracker=stats_tracker, ae_video_scaling_factor=ae_video_scaling_factor, apply_color_jitter=True, profiler=profiler)
        repeat_in_batch = cfg.data.repeat_in_batch
        (vision_cond, permutation) = self._build_ae_vision_condition(ae_latents_dense=ae_latents_dense, ae_latents_sparse=ae_latents_sparse, n_view=n_view, cfg=cfg, augment_vision=bool(cfg.data.aug_vision_token), repeat_in_batch=repeat_in_batch)
        actions = batch['actions'].to(device, dtype=weight_dtype).contiguous()[:, :-1]
        state = batch['state'].to(device, dtype=weight_dtype).contiguous()
        action_padding_mask = batch['action_mask'].to(device, dtype=weight_dtype).unsqueeze(1)
        video_slices = batch['video_slices'].to(device=device)
        if repeat_in_batch > 1:
            actions = actions.repeat(repeat_in_batch, 1, 1)
            state = state.repeat(repeat_in_batch, 1, 1)
            action_padding_mask = action_padding_mask.repeat(repeat_in_batch, 1, 1)
            video_slices = video_slices.repeat(repeat_in_batch, *[1] * (video_slices.ndim - 1))
            if 'view_ids' in batch:
                batch['view_ids'] = batch['view_ids'].repeat(repeat_in_batch, *[1] * (batch['view_ids'].ndim - 1))
            else:
                batch['view_ids'] = None
            actions = actions[permutation]
            state = state[permutation]
            action_padding_mask = action_padding_mask[permutation]
            video_slices = video_slices[permutation]
            if batch.get('view_ids') is not None:
                batch['view_ids'] = batch['view_ids'][permutation]
        action_weights_smpl = compute_density_for_timestep_sampling(weighting_scheme=cfg.cotrain.ae_flow_weighting_scheme, batch_size=actions.shape[0], logit_mean=cfg.cotrain.ae_flow_logit_mean, logit_std=cfg.cotrain.ae_flow_logit_std, mode_scale=cfg.cotrain.ae_flow_mode_scale)
        action_indices = (action_weights_smpl * scheduler.config.num_train_timesteps).long()
        action_sigmas = scheduler_sigmas[action_indices]
        action_timesteps = (action_sigmas * 1000.0).long().unsqueeze(-1).repeat(1, actions.shape[1])
        noise_actions = randn_tensor(actions.shape, device=device, dtype=weight_dtype)
        action_ss = action_sigmas.to(weight_dtype).reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])
        noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions
        noisy_actions = noisy_actions * action_padding_mask
        body_emb_id = batch.get('body_emb_id')
        if body_emb_id is not None:
            body_emb_id = body_emb_id.to(device, dtype=torch.long).contiguous()
        # Remove the two gripper values from the padded Robotwin state.
        joint_indices = [index for index in range(32) if index not in (6, 13)]
        if body_emb_id is not None and not torch.all(body_emb_id == 11):
            raise ValueError('Only robotwin body embedding ID 11 is supported')
        new_state = state[..., joint_indices]
        if not (torch.isfinite(vision_cond).all() and torch.isfinite(new_state).all() and torch.isfinite(noisy_actions).all()):
            logger.error('Non-finite AE model INPUT before forward (garbage in): vision_cond_finite=%s new_state_finite=%s noisy_actions_finite=%s path=%s', torch.isfinite(vision_cond).all().item(), torch.isfinite(new_state).all().item(), torch.isfinite(noisy_actions).all().item(), batch.get('path'))
        self._profile_start(profiler, 'ae_forward')
        pred_all = self.action_expert(vision_velocity=vision_cond, action_states=noisy_actions, action_timestep=action_timesteps, state_cond=new_state, view_ids=batch['view_ids'], body_emb_id=body_emb_id)
        self._profile_end(profiler, 'ae_forward')
        return {'actions': actions, 'target_action': (noise_actions - actions).float(), 'pred_action': pred_all['action'].float(), 'action_sigmas': action_sigmas, 'action_padding_mask': action_padding_mask, 'sparse_length': sparse_length, 'sample_weights': sample_weights, 'paths': batch.get('path'), 'video_slices': video_slices}

    def _decode_latents_for_pixel_loss(self, latents_5d, *, vae, stats_tracker, batch_size_decode=4):
        """Decode latents to pixel space WITH gradient flow (for LPIPS loss).

        Args:
            latents_5d: (bv, c, f, h, w) — any spatial resolution (dense or sparse).
            batch_size_decode: chunk size for VAE decode to control memory.

        Returns:
            pixels_5d: (bv, 3, f, H, W) clamped to [-1, 1].
        """
        (bv, c, f, h, w) = latents_5d.shape
        frames_4d = rearrange(latents_5d, 'bv c f h w -> (bv f) c h w')
        _st = stats_tracker
        frames_4d = _st.denormalize(frames_4d) if _st is not None else frames_4d / vae.config.scaling_factor
        decoded_chunks = []
        for i in range(0, frames_4d.shape[0], batch_size_decode):
            chunk = frames_4d[i:i + batch_size_decode]
            decoded_chunks.append(vae.decode(chunk, return_dict=False)[0])
        pixels_4d = torch.cat(decoded_chunks, dim=0)
        pixels_5d = rearrange(pixels_4d, '(bv f) c h w -> bv c f h w', bv=bv)
        return pixels_5d.clamp(-1, 1)

    def _vae_encode_mem_future(self, mem, future_video, batch_size, mem_size, latent_height, latent_width, n_dense_frames, *, cfg, vae, stats_tracker, use_video_mixed_path, video_sparse_downsample_factor, profiler=None):
        """VAE-encode (no_grad) pixel memory + future frames into canonical latents.

        The deterministic, dedup-able front-end of the VGen encode:
          - mixed path -> {mem_latents, dense_video_latents, sparse_video_latents,
            sparse_init_latents}
          - non-mixed  -> {mem_latents, future_video_latents}

        Called by the VGen TaskFlow path (B*N) and by the cotrain orchestration at B,
        where the result is repeat_interleave'd to B*N: a rejection-sampling
        group's members share identical encoded latents (only the generation
        noise differs). The 2D VAE uses the deterministic latent mode, so
        encode-at-B-then-repeat is bit-identical to encoding B*N copies.
        """
        with torch.no_grad():
            self._profile_start(profiler, 'vae_encode')
            if use_video_mixed_path:
                mixed_latents = encode_mixed_video_2dvae(vae, mem, future_video, dense_frame_count=n_dense_frames, sparse_downsample_factor=video_sparse_downsample_factor, batch_size_2dvae=cfg.model.batch_size_2dvae, stats_tracker=stats_tracker)
                out = {'mem_latents': mixed_latents['mem_latents'], 'dense_video_latents': mixed_latents['dense_latents'], 'sparse_video_latents': mixed_latents['sparse_latents'], 'sparse_init_latents': mixed_latents['sparse_init_latents']}
            else:
                (mem_latents, future_video_latents) = get_latents(vae, mem, future_video, batch_size_2dvae=cfg.model.batch_size_2dvae if vae.vae_encoding_mode == '2dvae' else None, stats_tracker=stats_tracker)
                mem_latents = rearrange(mem_latents, '(b v m) (h w) c -> (b v) c m h w', b=batch_size, m=mem_size, h=latent_height)
                future_video_latents = rearrange(future_video_latents, '(b v) (f h w) c -> (b v) c f h w', b=batch_size, h=latent_height, w=latent_width)
                out = {'mem_latents': mem_latents, 'future_video_latents': future_video_latents}
            self._profile_end(profiler, 'vae_encode')
            return out

    @staticmethod
    def _goal_gt_swap_prob_at(cfg, global_step):
        """Scheduled GT-goal swap probability.

        goal_gt_swap_curriculum (overrides the fixed goal_gt_swap_prob when set):
            p_start:     prob for the first hold_steps
            hold_steps:  steps to hold p_start
            decay_steps: decay window p_start -> p_end after the hold
                         (0 = hard drop at hold_steps)
            decay:       "cosine" (default) | "linear"
            p_end:       prob held forever afterwards
        No curriculum -> constant goal_gt_swap_prob (0 = feature off).
        """
        cur = cfg.goal_mode.goal_gt_swap_curriculum or {}
        if not cur:
            return float(cfg.goal_mode.goal_gt_swap_prob or 0.0)
        p_start = float(cur.get('p_start', 0.9))
        p_end = float(cur.get('p_end', 0.1))
        hold_steps = int(cur.get('hold_steps', 0))
        decay_steps = int(cur.get('decay_steps', 0))
        if global_step < hold_steps:
            return p_start
        if decay_steps <= 0 or global_step >= hold_steps + decay_steps:
            return p_end
        frac = (global_step - hold_steps) / float(decay_steps)
        decay = str(cur.get('decay', 'cosine'))
        if decay == 'cosine':
            scale = 0.5 * (1.0 + math.cos(math.pi * frac))
        elif decay == 'linear':
            scale = 1.0 - frac
        else:
            raise ValueError(f'unknown goal_gt_swap_curriculum.decay: {decay!r}')
        return p_end + (p_start - p_end) * scale

    def prepare_cotrain_goal_batch(self, batch, *, cfg, batch_size, device, generator, global_step):
        """Choose goal conditioning before candidate expansion and replay."""
        if cfg.goal_mode is None or not cfg.goal_mode.enabled:
            return batch
        goal_cond = batch['goal_cond']
        if not torch.is_tensor(goal_cond) or tuple(goal_cond.shape) != (batch_size,):
            raise ValueError(f'goal_cond must be a ({batch_size},) tensor')
        goal_cond = goal_cond.to(device=device, dtype=torch.bool)
        swap_probability = self._goal_gt_swap_prob_at(cfg, global_step)
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError('goal_gt_swap probability must be between 0 and 1')
        swap_mask = torch.rand(batch_size, generator=generator, device=generator.device if generator is not None else device) < swap_probability
        return {**batch, 'goal_cond': goal_cond, 'cotrain_goal_swap_mask': swap_mask.to(device) | goal_cond}

    @staticmethod
    def _index_select_first_dim(value, index, expected_size, name):
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise TypeError(f'{name} must be a Tensor, got {type(value).__name__}')
        if value.ndim == 0 or value.shape[0] != expected_size:
            raise ValueError(f'{name}: leading dim must be {expected_size}, got shape={tuple(value.shape)}')
        return value.index_select(0, index.to(value.device))

    @classmethod
    def _tile_b_or_bv(cls, value, *, idx_b, idx_bv, base_b, n_view, name):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] == 1:
                return value
            if value.shape[0] == base_b:
                return value.index_select(0, idx_b.to(value.device))
            if value.shape[0] == base_b * n_view:
                return value.index_select(0, idx_bv.to(value.device))
            raise ValueError(f'{name}: unsupported leading dim {value.shape[0]}, expected B={base_b}, B*V={base_b * n_view}, or 1')
        if isinstance(value, (list, tuple)):
            expanded = [cls._tile_b_or_bv(item, idx_b=idx_b, idx_bv=idx_bv, base_b=base_b, n_view=n_view, name=f'{name}[{i}]') for (i, item) in enumerate(value)]
            return type(value)(expanded)
        raise TypeError(f'{name}: unsupported type {type(value).__name__}')

    def train_vgen_forward(self, batch, video, batch_size, n_view, *, weight_dtype, generator, prompt_embeds, prompt_attention_mask, vlm_inputs, precomputed_vlm_hidden_states, gradient_vgen2ae, cfg, vae, stats_tracker, scheduler, lpips_model, use_video_mixed_path, use_video_sparse_mode, video_sparse_downsample_factor, video_sparse_spatial_rope_mode, video_frame_indices, spatial_down_ratio, temporal_down_ratio, timestep_scale, expert_dual_train, phase_role, global_step, data_config=None, vlm_attention_mask=None, one_step_gen_only=False, noise_hook=None, start_sigma_override=None, model_override=None, precomputed_vae_latents=None, device=None, profiler=None, record_time_histogram=None, _prepare_only=False):
        if device is None:
            device = get_device()
        vgen_model = model_override if model_override is not None else getattr(self, 'vgen', None)
        vgen_unwrapped = vgen_model.module if hasattr(vgen_model, 'module') else vgen_model
        data_config = as_dataset_kwargs(data_config or cfg.data.train)
        video_reshaped = rearrange(video, 'b c v t h w -> (b v) c t h w')
        if cfg.data.use_color_jitter and precomputed_vae_latents is None:
            video_reshaped = apply_color_jitter_to_video(video_reshaped, n_view=n_view if getattr(cfg.data, 'color_jitter_same_across_views', False) else 1, jitter_kwargs=getattr(cfg.data, 'color_jitter_kwargs', None))
        mem_size = data_config['n_previous']
        mem = video_reshaped[:, :, :mem_size]
        future_video = video_reshaped[:, :, mem_size:]
        concat_clean_mem = None
        concat_clean_future = None
        (_, _, raw_frames, raw_height, raw_width) = future_video.shape
        latent_height = raw_height // spatial_down_ratio
        latent_width = raw_width // spatial_down_ratio
        mixed_mode_active = bool(use_video_mixed_path)
        sparse_mode_active = bool(use_video_sparse_mode)
        if mixed_mode_active:
            assert int(temporal_down_ratio) == 1, f'当前 dense/sparse loss 仅支持 TEMPORAL_DOWN_RATIO=1，got {temporal_down_ratio}'
        n_dense_frames = min(data_config['continious_vid_length'], raw_frames)
        n_sparse_frames = raw_frames - n_dense_frames
        video_layout = None
        video_rope_coords = None
        sparse_token_stats = None
        with torch.no_grad():
            if precomputed_vae_latents is not None:
                _vlat = precomputed_vae_latents
            else:
                _vlat = self._vae_encode_mem_future(mem, future_video, batch_size, mem_size, latent_height, latent_width, n_dense_frames, cfg=cfg, vae=vae, stats_tracker=stats_tracker, use_video_mixed_path=use_video_mixed_path, video_sparse_downsample_factor=video_sparse_downsample_factor, profiler=profiler)
            video_repeat = int(getattr(cfg.data, 'video_repeat_in_batch', 1))
            base_batch_size = batch_size
            _vlat_base = _vlat
            idx_b = None
            idx_bv = None
            if video_repeat > 1:
                if one_step_gen_only:
                    raise ValueError('video_repeat_in_batch currently supports normal VGen only')
                if gradient_vgen2ae:
                    raise ValueError('video_repeat_in_batch is not yet supported with gradient_vgen2ae')
                if precomputed_vae_latents is not None:
                    raise ValueError('video_repeat_in_batch is not yet supported with precomputed/group-expanded VAE latents')
                idx_b = torch.arange(base_batch_size, device=device).repeat(video_repeat)
                view_idx = torch.arange(n_view, device=device)
                idx_bv = (idx_b[:, None] * n_view + view_idx[None, :]).reshape(-1)
                _vlat = {key: None if value is None else self._index_select_first_dim(value, idx_bv, base_batch_size * n_view, f'_vlat[{key!r}]') for (key, value) in _vlat.items()}
                flow_batch = dict(batch)
                for key in ('view_ids', 'video_slices', 'fps', 'goal_cond'):
                    value = flow_batch.get(key)
                    if value is not None:
                        flow_batch[key] = self._index_select_first_dim(value, idx_b, base_batch_size, f'batch[{key!r}]')
                paths = flow_batch.get('path')
                if paths is not None:
                    if len(paths) != base_batch_size:
                        raise ValueError(f"batch['path'] length {len(paths)} != {base_batch_size}")
                    source_rows = list(range(base_batch_size)) * video_repeat
                    flow_batch['path'] = type(paths)((paths[i] for i in source_rows))
                batch = flow_batch
                batch_size = base_batch_size * video_repeat
            mem_latents = _vlat['mem_latents']
            if mixed_mode_active:
                if 'video_slices' not in batch:
                    raise KeyError("Mixed sparse video mode requires batch['video_slices'] for explicit frame indices.")
                frame_indices = batch['video_slices'].to(device)[:, video_frame_indices]
                frame_indices_bv = expand_frame_indices_for_views(frame_indices, batch_size=batch_size, n_view=n_view, device=device)
                dense_video_latents = _vlat['dense_video_latents']
                sparse_video_latents = _vlat['sparse_video_latents']
                sparse_init_latents = _vlat['sparse_init_latents']
                (frame_latent_hw, frame_is_sparse) = build_frame_latent_spec(mem_frame_count=mem_size, dense_frame_count=n_dense_frames, sparse_frame_count=n_sparse_frames, full_latent_hw=mem_latents.shape[-2:], sparse_latent_hw=None if sparse_video_latents is None else sparse_video_latents.shape[-2:], device=device, sparse_downsample_factor=video_sparse_downsample_factor)
                video_layout = build_mixed_video_layout(frame_indices=frame_indices_bv, frame_latent_hw=frame_latent_hw, frame_is_sparse=frame_is_sparse, sparse_downsample_factor=video_sparse_downsample_factor, sparse_spatial_rope_mode=video_sparse_spatial_rope_mode)
                video_rope_coords = video_layout.video_rope_coords
                dense_token_count_total = int(video_layout.frame_offsets[mem_size + n_dense_frames].item() - video_layout.frame_offsets[mem_size].item())
                sparse_token_count_total = int(video_layout.frame_offsets[-1].item() - video_layout.frame_offsets[mem_size + n_dense_frames].item())
                fullres_token_count_per_frame = int(mem_latents.shape[-2] * mem_latents.shape[-1])
                sparse_token_count_per_frame = 0 if sparse_video_latents is None else int(sparse_video_latents.shape[-2] * sparse_video_latents.shape[-1])
                memory_token_count_total = int(video_layout.frame_offsets[mem_size].item())
                mixed_video_token_count_total = int(video_layout.frame_offsets[-1].item())
                baseline_video_token_count_total = int(video_layout.total_frames * fullres_token_count_per_frame)
                sparse_token_stats = {'memory_token_count_total': memory_token_count_total, 'dense_token_count_total': dense_token_count_total, 'sparse_token_count_total': sparse_token_count_total, 'baseline_video_token_count_total': baseline_video_token_count_total, 'mixed_video_token_count_total': mixed_video_token_count_total, 'video_token_reduction_ratio': 1.0 - float(mixed_video_token_count_total) / float(baseline_video_token_count_total) if baseline_video_token_count_total > 0 else 0.0, 'fullres_token_count_per_frame': fullres_token_count_per_frame, 'sparse_token_count_per_frame': sparse_token_count_per_frame, 'sparse_vs_fullres_token_ratio': float(sparse_token_count_per_frame) / float(fullres_token_count_per_frame) if fullres_token_count_per_frame > 0 and sparse_token_count_per_frame > 0 else 0.0}
                latents_flat = pack_frame_latent_groups(mem_latents, dense_video_latents, sparse_video_latents)
                (noise, conditioning_mask, cond_indicator) = build_mixed_conditioning_tokens(mem_latents=mem_latents, dense_frame_count=n_dense_frames, sparse_frame_count=n_sparse_frames, sparse_init_latents=sparse_init_latents, generator=generator, noise_to_condition_frames=cfg.train.noise_to_first_frame)
                latent_frames = video_layout.total_frames
                future_token_start = int(video_layout.frame_offsets[mem_size].item())
                sparse_tok_start = int(video_layout.frame_offsets[mem_size + n_dense_frames].item())
                future_token_count = int(video_layout.frame_offsets[-1].item() - video_layout.frame_offsets[mem_size].item())
                latents_5d = None
            else:
                if vae.vae_encoding_mode == '3dvae':
                    latent_frames = raw_frames // temporal_down_ratio + 1 + mem_size
                elif vae.vae_encoding_mode == '2dvae':
                    latent_frames = raw_frames // temporal_down_ratio + mem_size
                else:
                    raise NotImplementedError
                future_video_latents = _vlat['future_video_latents']
                latents_5d = torch.cat((mem_latents, future_video_latents), dim=2)
                latents_flat = rearrange(latents_5d, 'bv c f h w -> bv (f h w) c')
                (noise, conditioning_mask, cond_indicator) = gen_noise_from_condition_frame_latent(mem_latents, latent_frames, latent_height, latent_width, generator=generator, noise_to_condition_frames=cfg.train.noise_to_first_frame)
                future_token_start = mem_size * latent_height * latent_width
                sparse_tok_start = (mem_size + n_dense_frames) * latent_height * latent_width
                future_token_count = future_video_latents.shape[2] * latent_height * latent_width
        if video_repeat > 1:
            prompt_embeds = self._tile_b_or_bv(prompt_embeds, idx_b=idx_b, idx_bv=idx_bv, base_b=base_batch_size, n_view=n_view, name='prompt_embeds')
            prompt_attention_mask = self._tile_b_or_bv(prompt_attention_mask, idx_b=idx_b, idx_bv=idx_bv, base_b=base_batch_size, n_view=n_view, name='prompt_attention_mask')
            precomputed_vlm_hidden_states = self._tile_b_or_bv(precomputed_vlm_hidden_states, idx_b=idx_b, idx_bv=idx_bv, base_b=base_batch_size, n_view=n_view, name='precomputed_vlm_hidden_states')
            vlm_attention_mask = self._tile_b_or_bv(vlm_attention_mask, idx_b=idx_b, idx_bv=idx_bv, base_b=base_batch_size, n_view=n_view, name='vlm_attention_mask')
            if vlm_inputs is not None:
                if precomputed_vlm_hidden_states is None:
                    raise ValueError('video_repeat requires precomputed VLM hidden states')
                vlm_inputs = None
        goal_cond_b = batch.get('goal_cond') if isinstance(batch, dict) else None
        if goal_cond_b is not None:
            goal_cond_b = goal_cond_b.to(device).bool()
            if goal_cond_b.shape[0] != batch_size:
                raise ValueError(f'goal_cond batch dim {goal_cond_b.shape[0]} != batch_size {batch_size}')
        _has_goal_rows = goal_cond_b is not None and bool(goal_cond_b.any())
        _goal_swap_p = self._goal_gt_swap_prob_at(cfg, global_step)
        goal_swap_mask_bv = None
        cotrain_goal_swap_mask = None
        if one_step_gen_only and cfg.goal_mode.enabled:
            cotrain_goal_swap_mask = batch['cotrain_goal_swap_mask']
            if not torch.is_tensor(cotrain_goal_swap_mask) or tuple(cotrain_goal_swap_mask.shape) != (batch_size,):
                raise ValueError(f'cotrain_goal_swap_mask must be a ({batch_size},) tensor')
            cotrain_goal_swap_mask = cotrain_goal_swap_mask.to(device=device, dtype=torch.bool)
            if goal_cond_b is None or bool((goal_cond_b & ~cotrain_goal_swap_mask).any()):
                raise ValueError('goal_cond rows require cotrain_goal_swap_mask=True')
        if cotrain_goal_swap_mask is not None or ((_goal_swap_p > 0.0 or _has_goal_rows) and (not one_step_gen_only)):
            if latents_5d is None or (one_step_gen_only and cfg.model.frame_concat_conditioning):
                raise NotImplementedError('goal_gt_swap_prob is only wired for the dense non-concat latent path; frame_concat / mixed-sparse modes are unsupported.')
            if cotrain_goal_swap_mask is None:
                _swap_b = torch.rand(batch_size, generator=generator, device=generator.device if generator is not None else device) < _goal_swap_p
                _swap_b = _swap_b.to(device)
                if goal_cond_b is not None:
                    _swap_b = _swap_b | goal_cond_b
            else:
                _swap_b = cotrain_goal_swap_mask
            _swap_bv = _swap_b.repeat_interleave(n_view)
            _goal_tok_start = (latent_frames - 1) * latent_height * latent_width
            conditioning_mask[_swap_bv, _goal_tok_start:] = 1.0
            noise[_swap_bv, _goal_tok_start:, :] = latents_flat[_swap_bv, _goal_tok_start:, :]
            cond_indicator = cond_indicator.expand(noise.shape[0], -1).clone()
            cond_indicator[_swap_bv, -1] = 1.0
            goal_swap_mask_bv = _swap_bv
        if one_step_gen_only:
            if noise_hook is not None:
                _orig_shape = noise.shape
                noise = noise_hook(noise, conditioning_mask)
                if noise.shape != _orig_shape:
                    raise ValueError(f'noise_hook returned shape {tuple(noise.shape)} != expected {tuple(_orig_shape)}')
            cond_mask_expanded = conditioning_mask.unsqueeze(-1)
            noise = noise * (1 - cond_mask_expanded) + latents_flat * cond_mask_expanded
            if start_sigma_override is not None:
                t_sigma_val = torch.full((batch_size,), float(start_sigma_override), device=device, dtype=weight_dtype)
            elif cfg.cotrain.one_step_start_t['enabled']:
                t_sigma_val = torch.empty(batch_size, device=device, dtype=weight_dtype).uniform_(cfg.cotrain.one_step_start_t['noise_min'], cfg.cotrain.one_step_start_t['noise_max'], generator=generator)
            else:
                t_sigma_val = torch.ones(batch_size, device=device, dtype=weight_dtype)
            r_sigma_val = torch.zeros(batch_size, device=device, dtype=weight_dtype)
            h_sigma = rearrange((t_sigma_val - r_sigma_val).unsqueeze(1).repeat(1, n_view), 'b v -> (b v)')
            sigmas = rearrange(t_sigma_val.unsqueeze(1).repeat(1, n_view), 'b v -> (b v)')
            ss = sigmas.reshape(-1, 1, 1).repeat(1, latents_flat.size(1), 1)
            noisy_latents = (1.0 - ss) * latents_flat + ss * noise
            if cfg.cotrain.pixel_wise_timestep:
                ts = h_sigma.unsqueeze(-1) * (1 - conditioning_mask)
            else:
                ts = h_sigma.unsqueeze(-1) * (1 - cond_indicator)
            ts = ts * timestep_scale
            ts_r_kwargs = {}
            if bool(getattr((vgen_model.module if hasattr(vgen_model, 'module') else vgen_model).config, 'use_t_and_r_embedding', False)):
                _keep = 1 - conditioning_mask if cfg.cotrain.pixel_wise_timestep else 1 - cond_indicator
                ts = sigmas.unsqueeze(-1) * _keep * timestep_scale
                _r_sig = rearrange(r_sigma_val.unsqueeze(1).repeat(1, n_view), 'b v -> (b v)')
                ts_r = _r_sig.unsqueeze(-1) * _keep * timestep_scale
                ts_r_kwargs = {'timestep_r': ts_r}
                ts_r_kwargs['timestep_r_is_constant'] = True
            keep = 1 - conditioning_mask if cfg.cotrain.pixel_wise_timestep else 1 - cond_indicator
            time_base = sigmas if ts_r_kwargs else h_sigma
            r_base = _r_sig * timestep_scale if ts_r_kwargs else None
            ts_r_kwargs['_compact_timestep_inputs'] = (time_base * timestep_scale, r_base, keep)
            _concat_kwargs = {}
            self._profile_start(profiler, 'flow_forward_one_step')
            out = forward_pass(model=vgen_model, compute_v_head=False, timesteps=ts, noisy_latents=noisy_latents, prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_attention_mask, num_frames=latent_frames, height=latent_height, width=latent_width, n_view=n_view, view_ids=batch['view_ids'], return_video=True, return_action=False, video_attention_mask=None, history_action_state=None, condition_mask=conditioning_mask, vlm_inputs=vlm_inputs, num_frames_wo_dropout=cfg.model.num_frames_wo_dropout if cfg.model.num_frames_wo_dropout is not None else mem_size, precomputed_vlm_hidden_states=precomputed_vlm_hidden_states, vlm_attention_mask=vlm_attention_mask, goal_mode_mask=goal_cond_b, video_rope_coords=video_rope_coords, **ts_r_kwargs, **_concat_kwargs)['latents']
            self._profile_end(profiler, 'flow_forward_one_step')
            u = out['video']
            latents_hat = noisy_latents - ss * u
            if goal_swap_mask_bv is not None:
                latents_hat = torch.where(conditioning_mask.bool().unsqueeze(-1), latents_flat.detach(), latents_hat)
            mem_latents_det = mem_latents.detach()
            if mixed_mode_active:
                mem_tok = mem_size * latent_height * latent_width
                dense_tok = n_dense_frames * latent_height * latent_width
                dense_hat = latents_hat[:, mem_tok:mem_tok + dense_tok]
                sparse_hat = latents_hat[:, mem_tok + dense_tok:]
                (s_h, s_w) = (sparse_video_latents.shape[-2], sparse_video_latents.shape[-1])
                dense_hat_5d = rearrange(dense_hat, 'bv (f h w) c -> bv c f h w', f=n_dense_frames, h=latent_height, w=latent_width)
                sparse_hat_5d = rearrange(sparse_hat, 'bv (f h w) c -> bv c f h w', f=n_sparse_frames, h=s_h, w=s_w)
                dense_with_mem = torch.cat([mem_latents_det, dense_hat_5d], dim=2)
                ae_latents = (dense_with_mem, sparse_hat_5d)
                gt_ae_latents = (torch.cat([mem_latents_det, dense_video_latents.detach()], dim=2), sparse_video_latents.detach())
            else:
                latents_hat_5d = rearrange(latents_hat, 'bv (f h w) c -> bv c f h w', f=latent_frames, h=latent_height, w=latent_width)
                latents_hat_5d = torch.cat([mem_latents_det, latents_hat_5d[:, :, mem_size:]], dim=2)
                ae_latents = latents_hat_5d
                gt_ae_latents = latents_5d.detach()
            zero = torch.tensor(0.0, device=device)
            return {'_vgen_loss_kind': 'one_step', '_vgen_loss_inputs': {'device': device, 'generator': generator, 'lpips_model': lpips_model, 'vae': vae, 'stats_tracker': stats_tracker, 'mixed_mode_active': mixed_mode_active, 'dense_hat_5d': dense_hat_5d if mixed_mode_active else None, 'dense_video_latents': dense_video_latents if mixed_mode_active else None, 'sparse_hat_5d': sparse_hat_5d if mixed_mode_active else None, 'sparse_video_latents': sparse_video_latents if mixed_mode_active else None, 'n_dense_frames': n_dense_frames, 'n_sparse_frames': n_sparse_frames, 'future_video': future_video, 'latents_hat_5d': latents_hat_5d if not mixed_mode_active else None, 'mem_size': mem_size, 'future_video_latents': future_video_latents if not mixed_mode_active else None}, 'has_loss_metrics': False, 'video_latents_for_ae': ae_latents, 'gt_video_latents_for_ae': gt_ae_latents, 'noise_shape': tuple(noise.shape), 'mixed_mode_active': mixed_mode_active, 'sparse_mode_active': sparse_mode_active, 'sparse_token_stats': None, 'w_sparse_bv': zero, 'loss_u_dense': zero, 'loss_u_sparse': zero, 'loss_v_dense': zero, 'loss_v_sparse': zero, 'fps': batch['fps'].to(device).float(), 'sparse_stride': zero, 'latents_flat': latents_flat.detach(), 'v_t': zero, 'u': u.detach(), 'v_tangent': zero, 'du_dt': zero, 'V': zero, 'h_sigma': h_sigma, 'conditioning_mask': conditioning_mask, 'n_dense_frames': n_dense_frames}
        fm_shift = getattr(scheduler.config, 'shift', 1.0)
        imf_mu = cfg.cotrain.imf_mu
        imf_sigma = cfg.cotrain.imf_sigma
        (imf_raw_lo, imf_raw_hi) = (0.0, 1.0)
        expert_cfg = cfg.cotrain.expert
        _expert_enabled = expert_cfg.get('enabled', False)
        _dual_train = expert_dual_train
        role = None
        if _dual_train:
            _active_phase = phase_role
            if _active_phase == 'vgen_high':
                role = 'high'
            elif _active_phase in ('vgen_low', 'cotrain'):
                role = 'low'
            else:
                raise ValueError(f'dual_train: VGen TaskFlow reached with unexpected phase {_active_phase!r}; expected one of vgen_high / vgen_low / cotrain.')
        elif _expert_enabled:
            role = expert_cfg.get('role')
            if role not in ('high', 'low'):
                raise ValueError(f"expert.role must be 'high' or 'low', got {role!r}")
        if role is not None:
            role_cfg = expert_cfg.get(role) or {}
            for _k in ('band', 'mu', 'sigma'):
                if _k not in role_cfg:
                    raise ValueError(f"expert.{role} must define '{_k}'")
            imf_mu = float(role_cfg['mu'])
            imf_sigma = float(role_cfg['sigma'])
            (band_lo, band_hi) = role_cfg['band']
            imf_raw_lo = inverse_shift(float(band_lo), fm_shift)
            imf_raw_hi = inverse_shift(float(band_hi), fm_shift)
            _logged_attr = f'_expert_band_logged_{role}'
            if not getattr(self, _logged_attr, False):
                logger.info('IMF expert sampler: role=%s  t-band=[%.3f, %.3f]  raw-band=[%.4f, %.4f]  mu=%.3f  sigma=%.3f  (shift=%.2f)', role, band_lo, band_hi, imf_raw_lo, imf_raw_hi, imf_mu, imf_sigma, fm_shift)
                setattr(self, _logged_attr, True)
        flow_ratio = cfg.cotrain.imf_flow_ratio
        (t_sigma, r_sigma) = sample_improved_meanflow_times(batch_size=batch_size, device=device, dist=cfg.cotrain.imf_dist, mu=imf_mu, sigma=imf_sigma, flow_ratio=flow_ratio, raw_lo=imf_raw_lo, raw_hi=imf_raw_hi, r_mode=cfg.cotrain.imf_r_mode, dtype=weight_dtype, generator=generator)
        if fm_shift != 1.0:
            t_sigma = fm_shift * t_sigma / (1.0 + (fm_shift - 1.0) * t_sigma)
            r_sigma = fm_shift * r_sigma / (1.0 + (fm_shift - 1.0) * r_sigma)
        h_sigma = rearrange((t_sigma - r_sigma).unsqueeze(1).repeat(1, n_view), 'b v -> (b v)')
        sigmas = rearrange(t_sigma.unsqueeze(1).repeat(1, n_view), 'b v -> (b v)')
        r_sigma_bv = rearrange(r_sigma.unsqueeze(1).repeat(1, n_view), 'b v -> (b v)')
        use_t_and_r = bool(getattr((vgen_model.module if hasattr(vgen_model, 'module') else vgen_model).config, 'use_t_and_r_embedding', False))
        if record_time_histogram is not None:
            record_time_histogram(t_sigma, r_sigma, h_sigma)
        if cfg.cotrain.pixel_wise_timestep:
            timesteps = h_sigma.unsqueeze(-1) * (1 - conditioning_mask)
        else:
            timesteps = h_sigma.unsqueeze(-1) * (1 - cond_indicator)
        timesteps = timesteps * timestep_scale
        ss = sigmas.reshape(-1, 1, 1).repeat(1, latents_flat.size(1), 1)
        noisy_latents = (1.0 - ss) * latents_flat + ss * noise
        v_t = noise - latents_flat
        cond_latents = None
        cond_mode_ids = None
        prepared = {'_goal_swap_p': _goal_swap_p, '_vlat': _vlat, 'batch': batch, 'batch_size': batch_size, 'cfg': cfg, 'cond_indicator': cond_indicator, 'cond_latents': cond_latents, 'cond_mode_ids': cond_mode_ids, 'conditioning_mask': conditioning_mask, 'data_config': data_config, 'device': device, 'flow_ratio': flow_ratio, 'future_token_count': future_token_count, 'future_token_start': future_token_start, 'goal_cond_b': goal_cond_b, 'goal_swap_mask_bv': goal_swap_mask_bv, 'gradient_vgen2ae': gradient_vgen2ae, 'h_sigma': h_sigma, 'latent_frames': latent_frames, 'latent_height': latent_height, 'latent_width': latent_width, 'latents_5d': latents_5d, 'latents_flat': latents_flat, 'mem_size': mem_size, 'mixed_mode_active': mixed_mode_active, 'n_dense_frames': n_dense_frames, 'n_sparse_frames': n_sparse_frames, 'n_view': n_view, 'noisy_latents': noisy_latents, 'precomputed_vlm_hidden_states': precomputed_vlm_hidden_states, 'profiler': profiler, 'prompt_attention_mask': prompt_attention_mask, 'prompt_embeds': prompt_embeds, 'r_sigma_bv': r_sigma_bv, 'sigmas': sigmas, 'sparse_mode_active': sparse_mode_active, 'sparse_tok_start': sparse_tok_start, 'sparse_token_stats': sparse_token_stats, 'timestep_scale': timestep_scale, 'use_t_and_r': use_t_and_r, 'v_t': v_t, 'vgen_model': vgen_model, 'video_frame_indices': video_frame_indices, 'video_rope_coords': video_rope_coords, 'vlm_attention_mask': vlm_attention_mask, 'vlm_inputs': vlm_inputs}
        if _prepare_only:
            return prepared
        return self._run_prepared_vgen(**prepared)

    def _run_prepared_vgen(self, *, _goal_swap_p, _vlat, batch, batch_size, cfg, cond_indicator, cond_latents, cond_mode_ids, conditioning_mask, data_config, device, flow_ratio, future_token_count, future_token_start, goal_cond_b, goal_swap_mask_bv, gradient_vgen2ae, h_sigma, latent_frames, latent_height, latent_width, latents_5d, latents_flat, mem_size, mixed_mode_active, n_dense_frames, n_sparse_frames, n_view, noisy_latents, precomputed_vlm_hidden_states, profiler, prompt_attention_mask, prompt_embeds, r_sigma_bv, sigmas, sparse_mode_active, sparse_tok_start, sparse_token_stats, timestep_scale, use_t_and_r, v_t, vgen_model, video_frame_indices, video_rope_coords, vlm_attention_mask, vlm_inputs, before_vgen_loss=None):
        num_flow_b = int(flow_ratio * batch_size)
        num_mf_b = batch_size - num_flow_b
        num_mf_bv = num_mf_b * n_view
        flow_size = num_flow_b * n_view
        mf_slice_b = slice(num_flow_b, batch_size)
        mf_slice_bv = slice(num_flow_b * n_view, batch_size * n_view)

        def _slice_first_dim(value, name):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                size = value.shape[0]
                if size == 1:
                    return value
                if size == batch_size * n_view:
                    return value[mf_slice_bv]
                if size == batch_size:
                    return value[mf_slice_b]
                raise ValueError(f'Cannot slice {name} for partial-batch IMF: leading dimension {size} must equal 1 (broadcast), B={batch_size} or B*V={batch_size * n_view}.')
            if isinstance(value, (list, tuple)):
                sliced = [_slice_first_dim(item, f'{name}[{idx}]') for (idx, item) in enumerate(value)]
                return type(value)(sliced)
            raise TypeError(f'Partial-batch slicing of {name} (type {type(value).__name__}) is not supported.')

        def _make_fwd_fn(model, slice_to_mf=False, compute_v_head=True, use_eager_forward=False, r_source=None, fp32=False, u_tail_batch_size=None):
            if slice_to_mf:
                if vlm_inputs is not None and precomputed_vlm_hidden_states is None:
                    raise ValueError('Partial-batch IMF with raw vlm_inputs is unsupported because Qwen pixel/image metadata does not have a simple batch axis. Provide precomputed_vlm_hidden_states instead.')
                fwd_conditioning_mask = _slice_first_dim(conditioning_mask, 'conditioning_mask')
                fwd_cond_indicator = _slice_first_dim(cond_indicator, 'cond_indicator')
                fwd_prompt_embeds = _slice_first_dim(prompt_embeds, 'prompt_embeds')
                fwd_prompt_attention_mask = _slice_first_dim(prompt_attention_mask, 'prompt_attention_mask')
                fwd_view_ids = _slice_first_dim(batch['view_ids'], 'view_ids')
                fwd_vlm_inputs = None
                fwd_vlm_states = _slice_first_dim(precomputed_vlm_hidden_states, 'precomputed_vlm_hidden_states')
                fwd_vlm_mask = _slice_first_dim(vlm_attention_mask, 'vlm_attention_mask') if vlm_attention_mask is not None else None
                fwd_goal_mask = _slice_first_dim(goal_cond_b, 'goal_mode_mask')
                fwd_rope_coords = _slice_first_dim(video_rope_coords, 'video_rope_coords')
                fwd_r_source = _slice_first_dim(r_source, 'r_source')
                fwd_cond_latents = _slice_first_dim(cond_latents, 'cond_latents')
                fwd_cond_mode_ids = _slice_first_dim(cond_mode_ids, 'cond_mode_ids')
            else:
                fwd_conditioning_mask = conditioning_mask
                fwd_cond_indicator = cond_indicator
                fwd_prompt_embeds = prompt_embeds
                fwd_prompt_attention_mask = prompt_attention_mask
                fwd_view_ids = batch['view_ids']
                fwd_vlm_inputs = vlm_inputs
                fwd_vlm_states = precomputed_vlm_hidden_states
                fwd_vlm_mask = vlm_attention_mask
                fwd_goal_mask = goal_cond_b
                fwd_rope_coords = video_rope_coords
                fwd_r_source = r_source
                fwd_cond_latents = cond_latents
                fwd_cond_mode_ids = cond_mode_ids
            model_forward = getattr(model, '_vgen_eager_forward', None) if use_eager_forward else None
            base_model_kwargs = {'encoder_attention_mask': fwd_prompt_attention_mask, 'n_view': n_view, 'view_ids': fwd_view_ids, 'rope_interpolation_scale': [1.0, 1.0, 1.0], 'return_dict': False, 'return_video': True, 'return_action': False, 'video_attention_mask': None, 'history_action_state': None, 'condition_mask': fwd_conditioning_mask, 'vlm_inputs': fwd_vlm_inputs, 'num_frames': latent_frames, 'height': latent_height, 'width': latent_width, 'num_frames_wo_dropout': cfg.model.num_frames_wo_dropout if cfg.model.num_frames_wo_dropout is not None else mem_size, 'vlm_attention_mask': fwd_vlm_mask, 'goal_mode_mask': fwd_goal_mask, 'video_rope_coords': fwd_rope_coords, 'compute_v_head': compute_v_head, 'u_tail_batch_size': u_tail_batch_size, 'cond_mode_ids': fwd_cond_mode_ids}

            def fwd_fn(z_t, time_val, *, _stage1_payload_only=False, _stage1_ordinary_pair=None):
                if use_t_and_r:
                    keep = 1 - fwd_conditioning_mask if cfg.cotrain.pixel_wise_timestep else 1 - fwd_cond_indicator
                    ts = time_val.unsqueeze(-1) * keep * timestep_scale
                    ts_r = fwd_r_source.unsqueeze(-1) * keep * timestep_scale
                    extra_ts_kwargs = {'timestep_r': ts_r}
                else:
                    if cfg.cotrain.pixel_wise_timestep:
                        ts = time_val.unsqueeze(-1) * (1 - fwd_conditioning_mask)
                    else:
                        ts = time_val.unsqueeze(-1) * (1 - fwd_cond_indicator)
                    ts = ts * timestep_scale
                    extra_ts_kwargs = {}
                keep = 1 - fwd_conditioning_mask if cfg.cotrain.pixel_wise_timestep else 1 - fwd_cond_indicator
                extra_ts_kwargs['_compact_timestep_inputs'] = (time_val * timestep_scale, fwd_r_source * timestep_scale if use_t_and_r else None, keep)
                (p_embeds, vlm_states, cond_lat) = (fwd_prompt_embeds, fwd_vlm_states, fwd_cond_latents)
                if fp32:

                    def _f32(x):
                        if torch.is_tensor(x):
                            return x.float() if x.is_floating_point() else x
                        if isinstance(x, (list, tuple)):
                            return type(x)((_f32(v) for v in x))
                        if isinstance(x, dict):
                            return {k: _f32(v) for (k, v) in x.items()}
                        return x
                    (z_t, ts) = (_f32(z_t), _f32(ts))
                    extra_ts_kwargs = {k: _f32(v) for (k, v) in extra_ts_kwargs.items()}
                    (p_embeds, vlm_states, cond_lat) = (_f32(p_embeds), _f32(vlm_states), _f32(cond_lat))
                call_kwargs = dict(base_model_kwargs, hidden_states=z_t, encoder_hidden_states=p_embeds, timestep=ts, precomputed_vlm_hidden_states=vlm_states, cond_latents=cond_lat, **extra_ts_kwargs)
                if _stage1_payload_only:
                    return call_kwargs
                if _stage1_ordinary_pair is not None:
                    call_kwargs['_stage1_ordinary_pair'] = _stage1_ordinary_pair
                out = (model_forward or model)(**call_kwargs)[0]
                if _stage1_ordinary_pair is not None:
                    if compute_v_head:
                        return (out['video'], out['v_video'], out['_stage1_ordinary_pair_video'])
                    return (out['video'], out['_stage1_ordinary_pair_video'])
                if compute_v_head:
                    return (out['video'], out['v_video'])
                return out['video']
            return fwd_fn
        vgen_unwrapped = vgen_model.module if hasattr(vgen_model, 'module') else vgen_model
        _use_partial_batch_mf = cfg.cotrain.use_partial_batch_mf
        skip_v_head = num_mf_b == 0 and _use_partial_batch_mf
        use_stage1_ordinary_pair = bool(getattr(cfg.model, 'vgen_ordinary_pair', False)) and _use_partial_batch_mf and (n_view in (1, 2, 3)) and (num_flow_b > 0) and (num_mf_b > 0)
        u_tail_batch_size = num_flow_b if getattr(cfg.model, 'vgen_prune_u_tail', False) and _use_partial_batch_mf and (num_mf_bv > 0) else None
        fwd_fn_ddp = _make_fwd_fn(vgen_model, compute_v_head=not skip_v_head, r_source=sigmas, u_tail_batch_size=u_tail_batch_size)
        h_zero = torch.zeros_like(h_sigma)
        time_h0 = sigmas if use_t_and_r else h_zero
        self._profile_start(profiler, 'flow_forward_h0')
        paired_u_mf = None
        if use_stage1_ordinary_pair:
            fwd_fn_ddp_mf = _make_fwd_fn(vgen_model, slice_to_mf=True, compute_v_head=False, r_source=r_sigma_bv)
            z_mf_pair = noisy_latents[mf_slice_bv]
            time_mf_pair = sigmas[mf_slice_bv] if use_t_and_r else h_sigma[mf_slice_bv]
            mf_payload = fwd_fn_ddp_mf(z_mf_pair, time_mf_pair, _stage1_payload_only=True)
            (u_at_h0, v_tangent, paired_u_mf) = fwd_fn_ddp(noisy_latents, time_h0, _stage1_ordinary_pair=mf_payload)
        else:
            _out_h0 = fwd_fn_ddp(noisy_latents, time_h0)
        if skip_v_head:
            (u_at_h0, v_tangent) = (_out_h0, None)
        elif paired_u_mf is None:
            (u_at_h0, v_tangent) = _out_h0
        self._profile_end(profiler, 'flow_forward_h0')
        _jvp_mode = cfg.cotrain.imf_jvp_term_mode

        def _imf_jvp(fn, primals, v_tan, h_tan):
            """Return (du_dt, adv, tim) per _jvp_mode; adv/tim are None unless computed."""
            if _jvp_mode == 'combined':
                (_, d) = jvp(fn, primals, (v_tan, h_tan))
                return (d, None, None)
            (zero_h, zero_v) = (torch.zeros_like(h_tan), torch.zeros_like(v_tan))
            adv = tim = None
            if _jvp_mode in ('advection', 'both'):
                (_, adv) = jvp(fn, primals, (v_tan, zero_h))
            if _jvp_mode in ('time', 'both'):
                (_, tim) = jvp(fn, primals, (zero_v, h_tan))
            d = adv if _jvp_mode == 'advection' else tim if _jvp_mode == 'time' else adv + tim
            return (d, adv, tim)
        _jvp_fp32 = cfg.cotrain.imf_jvp_fp32

        def _run_imf_jvp(fn, z, t, v_tan, h_tan):
            if not _jvp_fp32:
                return _imf_jvp(fn, (z, t), v_tan, h_tan)
            out_dtype = z.dtype
            with self._module_params_fp32(vgen_unwrapped):
                (d, a, ti) = _imf_jvp(fn, (z.float(), t.float()), v_tan.float(), h_tan.float())
            _back = lambda x: None if x is None else x.to(out_dtype)
            return (_back(d), _back(a), _back(ti))
        _imf_jvp_impl = cfg.cotrain.imf_jvp_impl
        if _imf_jvp_impl in ('fast', 'strict'):
            assert not _jvp_fp32, f'imf_jvp_fp32=True is not supported with imf_jvp_impl={_imf_jvp_impl}'
            _jvp_caller = None
        else:
            _jvp_caller = _run_imf_jvp

        def _run_fast_jvp(z, time_val, v_tan, h_tan, slice_to_mf=False, r_source=None):
            s = _slice_first_dim if slice_to_mf else lambda v, n: v
            cm = s(conditioning_mask, 'conditioning_mask')
            ci = s(cond_indicator, 'cond_indicator')
            keep = 1 - cm if cfg.cotrain.pixel_wise_timestep else 1 - ci
            ts = time_val.unsqueeze(-1) * keep * timestep_scale
            d_ts = h_tan.unsqueeze(-1) * keep * timestep_scale
            cond = dict(encoder_hidden_states=s(prompt_embeds, 'prompt_embeds'), encoder_attention_mask=s(prompt_attention_mask, 'prompt_attention_mask'), video_attention_mask=None, video_rope_coords=s(video_rope_coords, 'video_rope_coords'), n_view=n_view, num_frames=latent_frames, height=latent_height, width=latent_width, view_ids=s(batch['view_ids'], 'view_ids'), precomputed_vlm_hidden_states=s(precomputed_vlm_hidden_states, 'vlm_hs'), vlm_attention_mask=s(vlm_attention_mask, 'vlm_mask') if vlm_attention_mask is not None else None, goal_mode_mask=s(goal_cond_b, 'goal_mode_mask'), num_frames_wo_dropout=cfg.model.num_frames_wo_dropout if cfg.model.num_frames_wo_dropout is not None else mem_size, cond_latents=s(cond_latents, 'cond_latents'), cond_mode_ids=s(cond_mode_ids, 'cond_mode_ids'), compute_v_head=False)
            if use_t_and_r and r_source is not None:
                rs = s(r_source, 'r_source')
                cond['timestep_r'] = rs.unsqueeze(-1) * keep * timestep_scale if rs is not None else None
            return run_imf_jvp_fast(vgen_unwrapped, z, ts, v_tan, d_ts, _jvp_mode, backend=_imf_jvp_impl, conditioning_kwargs=cond)
        du_dt_adv = du_dt_time = None
        use_partial_batch_mf = cfg.cotrain.use_partial_batch_mf
        if use_partial_batch_mf and num_mf_bv == 0:
            u = u_at_h0
            du_dt = torch.zeros_like(u)
        elif use_partial_batch_mf and num_flow_b > 0:
            fwd_u_fn_ddp_mf = None
            if paired_u_mf is None:
                fwd_u_fn_ddp_mf = _make_fwd_fn(vgen_model, slice_to_mf=True, compute_v_head=False, r_source=r_sigma_bv)
            z_mf = noisy_latents[mf_slice_bv]
            h_mf = h_sigma[mf_slice_bv]
            time_mf = sigmas[mf_slice_bv] if use_t_and_r else h_mf
            v_tangent_mf = (v_tangent[mf_slice_bv] * (1 - conditioning_mask[mf_slice_bv].unsqueeze(-1))).detach()
            h_tangent_mf = torch.ones_like(h_mf)
            with torch.no_grad():
                self._profile_start(profiler, 'imf_jvp')
                if _imf_jvp_impl in ('fast', 'strict'):
                    (du_dt_mf, adv_mf, tim_mf) = _run_fast_jvp(z_mf, time_mf, v_tangent_mf, h_tangent_mf, slice_to_mf=True, r_source=r_sigma_bv)
                else:
                    fwd_u_fn_unwrapped_mf = _make_fwd_fn(vgen_unwrapped, slice_to_mf=True, compute_v_head=False, use_eager_forward=True, r_source=r_sigma_bv, fp32=_jvp_fp32)
                    (du_dt_mf, adv_mf, tim_mf) = _jvp_caller(fwd_u_fn_unwrapped_mf, z_mf, time_mf, v_tangent_mf, h_tangent_mf)
                du_dt_mf = du_dt_mf
                adv_mf = adv_mf
                tim_mf = tim_mf
                self._profile_end(profiler, 'imf_jvp')
            if paired_u_mf is None:
                self._profile_start(profiler, 'flow_forward_h')
                u_mf = fwd_u_fn_ddp_mf(z_mf, time_mf)
                self._profile_end(profiler, 'flow_forward_h')
            else:
                u_mf = paired_u_mf
            _zero_flow = torch.zeros_like(u_at_h0[:flow_size])
            u = torch.cat([u_at_h0[:flow_size], u_mf], dim=0)
            du_dt = torch.cat([_zero_flow, du_dt_mf], dim=0)
            if adv_mf is not None:
                du_dt_adv = torch.cat([_zero_flow, adv_mf], dim=0)
            if tim_mf is not None:
                du_dt_time = torch.cat([_zero_flow, tim_mf], dim=0)
        else:
            fwd_u_fn_ddp = _make_fwd_fn(vgen_model, compute_v_head=False, r_source=r_sigma_bv)
            time_full = sigmas if use_t_and_r else h_sigma
            h_tangent = torch.ones_like(h_sigma)
            with torch.no_grad():
                self._profile_start(profiler, 'imf_jvp')
                if _imf_jvp_impl in ('fast', 'strict'):
                    (du_dt, du_dt_adv, du_dt_time) = _run_fast_jvp(noisy_latents, time_full, (v_tangent * (1 - conditioning_mask.unsqueeze(-1))).detach(), h_tangent, r_source=r_sigma_bv)
                else:
                    fwd_u_fn_unwrapped = _make_fwd_fn(vgen_unwrapped, compute_v_head=False, use_eager_forward=True, r_source=r_sigma_bv, fp32=_jvp_fp32)
                    (du_dt, du_dt_adv, du_dt_time) = _jvp_caller(fwd_u_fn_unwrapped, noisy_latents, time_full, (v_tangent * (1 - conditioning_mask.unsqueeze(-1))).detach(), h_tangent)
                du_dt = du_dt
                du_dt_adv = du_dt_adv
                du_dt_time = du_dt_time
                self._profile_end(profiler, 'imf_jvp')
            self._profile_start(profiler, 'flow_forward_h')
            u = fwd_u_fn_ddp(noisy_latents, time_full)
            self._profile_end(profiler, 'flow_forward_h')
        if before_vgen_loss is not None:
            before_vgen_loss()
        h_du_dt_raw = h_sigma.reshape(-1, 1, 1) * du_dt.detach()
        du_dt_limit_active_frac = None
        _h_du_dt_capped = None
        _dudt_limit_on = cfg.cotrain.imf_dudt_limit
        if _dudt_limit_on or cfg.cotrain.imf_dudt_limit_log_only:
            (_h_du_dt_capped, _laf) = self._limit_h_du_dt(h_du_dt_raw, u, mode=cfg.cotrain.imf_dudt_limit_mode, budget=cfg.cotrain.imf_dudt_limit_budget, clamp=cfg.cotrain.imf_dudt_limit_clamp, kappa=cfg.cotrain.imf_dudt_limit_kappa, c=cfg.cotrain.imf_dudt_limit_c, beta=cfg.cotrain.imf_dudt_limit_beta)
            du_dt_limit_active_frac = float(_laf)
        h_du_dt = _h_du_dt_capped if _dudt_limit_on else h_du_dt_raw
        V = u + h_du_dt
        target = v_t.detach()
        video_latents_for_ae = None
        if latents_5d is not None:
            video_latents_for_ae = latents_5d if gradient_vgen2ae else latents_5d.detach()
        return {'_vgen_loss_kind': 'normal', '_vgen_loss_inputs': {'batch': batch, 'device': device, 'batch_size': batch_size, 'n_view': n_view, 'n_sparse_frames': n_sparse_frames, 'n_dense_frames': n_dense_frames, 'mem_size': mem_size, 'video_frame_indices': video_frame_indices, 'data_config': data_config, 'conditioning_mask': conditioning_mask, 'V': V, 'target': target, 'v_tangent': v_tangent, 'skip_v_head': skip_v_head, 'future_token_count': future_token_count, 'future_token_start': future_token_start, 'sparse_tok_start': sparse_tok_start, 'u': u, 'h_du_dt_raw': h_du_dt_raw, 'h_du_dt_capped': _h_du_dt_capped, 'dudt_limit_on': _dudt_limit_on}, 'has_loss_metrics': True, 'video_latents_for_ae': video_latents_for_ae, 'vae_latents_raw': _vlat, 'mixed_mode_active': mixed_mode_active, 'sparse_mode_active': sparse_mode_active, 'sparse_token_stats': sparse_token_stats, 'latents_flat': latents_flat, 'v_t': v_t, 'u': u, 'v_tangent': v_tangent, 'du_dt': du_dt, 'du_dt_adv': du_dt_adv, 'du_dt_time': du_dt_time, 'du_dt_limit_active_frac': du_dt_limit_active_frac, 'V': V, 'h_sigma': h_sigma, 'sigmas_bv': sigmas, 'conditioning_mask': conditioning_mask, 'n_dense_frames': n_dense_frames, 'goal_swap_mask_bv': goal_swap_mask_bv, 'goal_swap_p': _goal_swap_p, 'goal_cond_mask_b': goal_cond_b}

    @staticmethod
    @contextlib.contextmanager
    def _module_params_fp32(module):
        """Temporarily upcast a module's floating params+buffers to fp32, then
        restore the originals on exit.

        Used to compute the du/dt JVP in fp32 (imf_jvp_fp32). The eager JVP runs
        in pure bf16 (the unwrap dodges accelerate's autocast), so du_dt suffers
        bf16 cancellation between its advection/time terms and rounding through
        depth — this isolates that. Safe under ZeRO-2: params are replicated and
        materialized in bf16, the fp32 masters live in the optimizer, and the
        swap is reversed before any backward/step, so DeepSpeed never sees it.
        The original bf16 tensors are kept by reference, so restore is exact.
        """
        saved = []
        try:
            for t in list(module.parameters()) + list(module.buffers()):
                if t.is_floating_point():
                    saved.append((t, t.data))
                    t.data = t.data.float()
            yield
        finally:
            for (t, data) in saved:
                t.data = data

    @staticmethod
    def _limit_h_du_dt(h_du_dt, u, mode='hard', budget='token', clamp='element', kappa=0.7, c=5.0, beta=0.3, eps=1e-06):
        """Limit the detached correction h·du/dt in V = u + h·du/dt.

        Tames the concentrated high-noise JVP spikes — a CFL / PD-control
        instability: the bootstrap target u <- v_t - h·du/dt amplifies sharp
        modes with loop gain h·v·ξ, worst at high t (large h). Caps that gain to
        ~kappa without touching the stable bulk. All-detached, so it bounds the
        target only (no gradient distortion). Returns (limited, active_frac).

        budget — field-amplitude scale the cap is relative to:
          'token'  : kappa·rms_channels(u), floored by beta·rms_sample(u) — the
                     per-token envelope (zero-safe, phase-independent; NOT |u_i|,
                     which is 90° out of phase with the slope).
          'sample' : kappa·rms(u) over the whole sample.
          'none'   : fixed constant c.
        clamp — where the cut lands:
          'element': per-component (surgical when a spike is a few channels —
                     our case, since absmax >> absmean).
          'token'  : per-token L2 rescale (direction-preserving).
        mode — 'hard' (clamp) or 'tanh' (smooth knee / derivative filter).
        """
        x = h_du_dt
        uf = u.detach().float()
        red = tuple(range(1, x.dim()))
        if budget == 'none':
            C = x.new_tensor(float(c)).clamp_min(eps)
        else:
            rms_s = uf.pow(2).mean(dim=red, keepdim=True).sqrt()
            if budget == 'token':
                rms_t = uf.pow(2).mean(dim=-1, keepdim=True).sqrt()
                rms = (rms_t.pow(2) + (beta * rms_s).pow(2)).sqrt()
            else:
                rms = rms_s
            C = (kappa * rms).clamp_min(eps)
        Cf = C.float()
        if clamp == 'token':
            n_ch = x.shape[-1]
            tok_norm = x.float().pow(2).sum(dim=-1, keepdim=True).sqrt()
            tok_budget = (Cf * n_ch ** 0.5).clamp_min(eps)
            if mode == 'tanh':
                new_norm = tok_budget * torch.tanh(tok_norm / tok_budget)
            else:
                new_norm = torch.minimum(tok_norm, tok_budget)
            active_frac = (tok_norm > tok_budget).float().mean()
            limited = (x.float() * (new_norm / tok_norm.clamp_min(eps))).to(x.dtype)
        else:
            active_frac = (x.float().abs() > Cf).float().mean()
            if mode == 'tanh':
                limited = (Cf * torch.tanh(x.float() / Cf)).to(x.dtype)
            else:
                limited = torch.minimum(torch.maximum(x.float(), -Cf), Cf).to(x.dtype)
        return (limited, active_frac)
