from policy_core.utils.WAM_policy.WAM_policy_config import as_dataset_kwargs
import os
import time
import logging
import torch
from einops import rearrange

from policy_core.utils.WAM_policy import save_video
from policy_core.utils.WAM_policy.data_utils import gen_noise_from_condition_frame_latent, get_latents, get_text_conditions
from policy_core.utils.WAM_policy.model_utils import forward_pass
from policy_core.model.WAM_policy.vlm_patches.vlm_token_select import slice_vlm_text_part
from policy_core.utils.WAM_policy.profile_utils import StageProfiler
from policy_core.utils.WAM_policy.video_sparse_utils import (
    build_frame_latent_spec,
    build_mixed_conditioning_tokens,
    build_mixed_video_layout,
    encode_mixed_video_2dvae,
    expand_frame_indices_for_views,
    pack_frame_latent_groups,
)

logger = logging.getLogger(__name__)

VIDEO_FRAME_INDICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
VIDEO_PROFILE_STAGES = (
    "input_preparation",
    "latent_encoding",
    "noise_conditioning",
    "text_conditioning",
    "vlm_input_build",
    "vlm_forward",
    "vlm_postprocess",
    "dit_input_preparation",
    "video_dit_forward",
    "euler_update",
    "output_postprocess",
)
VLM_PROFILE_STAGES = ("vlm_input_build", "vlm_forward", "vlm_postprocess")


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def random_temporal_token_keep(
    x: torch.Tensor,
    keep_ratio: float,
    mask_cur_obs: bool = False,
    topk_idx: torch.Tensor | None = None,
):
    batch_size, num_frames, num_tokens, channels = x.shape
    assert 0 < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
    kept_len = int(num_tokens * keep_ratio)

    if not mask_cur_obs:
        x_first = x[:, 0]
        x_rest = x[:, 1:]
        if topk_idx is None:
            scores = torch.rand(
                batch_size, num_frames - 1, num_tokens, device=x.device
            )
            indices = scores.topk(kept_len, dim=-1).indices
        else:
            indices = topk_idx.to(device=x.device, dtype=torch.long)
            if indices.shape != (batch_size, num_frames - 1, kept_len):
                raise ValueError(
                    "dense topk_idx shape "
                    f"{tuple(indices.shape)} != "
                    f"{(batch_size, num_frames - 1, kept_len)}"
                )
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
        x_rest_selected = torch.gather(x_rest, dim=2, index=indices)
        x_rest_selected = x_rest_selected.reshape(
            batch_size, (num_frames - 1) * kept_len, channels
        )
    else:
        if topk_idx is None:
            scores = torch.rand(
                batch_size, num_frames, num_tokens, device=x.device
            )
            indices = scores.topk(kept_len, dim=-1).indices
        else:
            indices = topk_idx.to(device=x.device, dtype=torch.long)
            if indices.shape != (batch_size, num_frames, kept_len):
                raise ValueError(
                    "sparse topk_idx shape "
                    f"{tuple(indices.shape)} != "
                    f"{(batch_size, num_frames, kept_len)}"
                )
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
        x_selected = torch.gather(x, dim=2, index=indices)
        x_first = x_selected[:, 0]
        x_rest_selected = x_selected[:, 1:].reshape(
            batch_size, (num_frames - 1) * kept_len, channels
        )
    return x_first, x_rest_selected

@torch.no_grad()
def one_step_vgen_from_obs(
    self,
    obs_frame,
    caption,
    view_ids=None,
    raw_t=None,
    t_sigma_val=None,
    return_at_t=0.0,
    precomputed_noisy_latents=None,
    vgen_model_override=None,
    noise_seed=None,
    future_noise=None,
    bboxes=None,
    bbox_reference_images=None,
):
    if obs_frame.dim() == 5:
        obs_frame = obs_frame.unsqueeze(3)
    elif obs_frame.dim() != 6:
        raise ValueError(f"obs_frame must be 5D or 6D, got shape {tuple(obs_frame.shape)}")

    if obs_frame.shape[3] != 1:
        raise ValueError(f"obs_frame time dimension must be 1, got {obs_frame.shape[3]}")

    if raw_t is None:
        data_cfg = as_dataset_kwargs(
            self.cfg.data.val if self.cfg.data.val is not None else self.cfg.data.train
        )
        raw_t = (
            data_cfg["n_previous"]
            + data_cfg["continious_vid_length"]
            + data_cfg["sparse_length"]
        )

    b, c, v, _, h, w = obs_frame.shape

    video_segment = obs_frame.repeat(1, 1, 1, raw_t, 1, 1)

    if isinstance(caption, str):
        caption = [caption] * b

    if view_ids is None:
        view_ids = torch.arange(v, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1)

    batch_segment = {
        "caption": caption,
        "view_ids": view_ids,
        "video_slices": torch.arange(raw_t, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1),
    }

    if bboxes is not None:
        batch_segment["bbox"] = bboxes
    if bbox_reference_images is not None:
        batch_segment["bbox_reference_images"] = bbox_reference_images

    return _one_step_vgen_for_ae(
        self, video_segment, batch_segment, v, self.device, self.dtype,
        t_sigma_val=t_sigma_val,
        return_at_t=return_at_t,
        precomputed_noisy_latents=precomputed_noisy_latents,
        vgen_model_override=vgen_model_override,
        noise_seed=noise_seed,
        future_noise=future_noise,
    )


@torch.no_grad()
def two_step_vgen_from_obs(
    self,
    obs_frame,
    caption,
    val_sigma,
    view_ids=None,
    raw_t=9,
    vgen_model_step1=None,
    noise_seed=None,
):
    if not (0.0 < float(val_sigma) < 1.0):
        raise ValueError(f"val_sigma must be in (0, 1), got {val_sigma}")

    _timing = getattr(self, "_timing", None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _s1_t0 = time.perf_counter()
    x_mid = one_step_vgen_from_obs(
        self, obs_frame, caption,
        view_ids=view_ids, raw_t=raw_t,
        t_sigma_val=None,
        return_at_t=float(val_sigma),
        vgen_model_override=vgen_model_step1,
        noise_seed=noise_seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if _timing is not None:
        _timing["vgen_step1"] = (time.perf_counter() - _s1_t0) * 1000.0

    _s2_t0 = time.perf_counter()
    out = one_step_vgen_from_obs(
        self, obs_frame, caption,
        view_ids=view_ids, raw_t=raw_t,
        t_sigma_val=float(val_sigma),
        precomputed_noisy_latents=x_mid,
        noise_seed=noise_seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if _timing is not None:
        _timing["vgen_step2"] = (time.perf_counter() - _s2_t0) * 1000.0
    return out


@torch.no_grad()
def _one_step_vgen_for_ae(
    self, video_segment, batch_segment, n_view, device, weight_dtype,
    t_sigma_val=None,
    return_at_t=0.0,
    precomputed_noisy_latents=None,
    vgen_model_override=None,
    noise_seed=None,
    future_noise=None,
):
    compute_v_head = self.cfg.infer.compute_v_head
    profiler = StageProfiler("VIDEO_PROFILE", device)
    batch_size = video_segment.shape[0]
    data_config = as_dataset_kwargs(
        self.cfg.data.val if self.cfg.data.val is not None else self.cfg.data.train
    )
    video_reshaped = rearrange(video_segment, "b c v t h w -> (b v) c t h w")

    mem_size = data_config["n_previous"]
    mem = video_reshaped[:, :, :mem_size]
    future_video = video_reshaped[:, :, mem_size:]

    _, _, raw_frames, raw_height, raw_width = future_video.shape
    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
    latent_width = raw_width // self.SPATIAL_DOWN_RATIO
    mixed_mode_active = bool(getattr(self, "use_video_mixed_path", False))
    n_dense_frames = min(data_config.get("continious_vid_length", raw_frames), raw_frames)
    n_sparse_frames = raw_frames - n_dense_frames
    video_rope_coords = None

    vgen_model_active = (
        vgen_model_override
        if vgen_model_override is not None
        else self.video_diffusion_model
    )
    vgen_model_unwrapped = (
        vgen_model_active.module
        if hasattr(vgen_model_active, "module")
        else vgen_model_active
    )
    noise_generator = (
        torch.Generator(device=device).manual_seed(int(noise_seed))
        if noise_seed is not None
        else None
    )

    profiler.mark("input_preparation")

    if mixed_mode_active:
        if future_noise is not None:
            raise ValueError(
                "future_noise replay is only supported on the non-mixed VGen path"
            )
        frame_indices = batch_segment["video_slices"].to(device) if "video_slices" in batch_segment else None
        if frame_indices is None:
            raise KeyError("Mixed sparse video mode requires batch_segment['video_slices'].")

        frame_indices_bv = expand_frame_indices_for_views(
            frame_indices,
            batch_size=batch_size,
            n_view=n_view,
            device=device,
        )
        mixed_latents = encode_mixed_video_2dvae(
            self.vae,
            mem,
            future_video,
            dense_frame_count=n_dense_frames,
            sparse_downsample_factor=self.video_sparse_downsample_factor,
            batch_size_2dvae=self.cfg.model.batch_size_2dvae,
            stats_tracker=getattr(self, "stats_tracker", None),
        )
        mem_latents = mixed_latents["mem_latents"]
        dense_video_latents = mixed_latents["dense_latents"]
        sparse_video_latents = mixed_latents["sparse_latents"]
        sparse_init_latents = mixed_latents["sparse_init_latents"]

        frame_latent_hw, frame_is_sparse = build_frame_latent_spec(
            mem_frame_count=mem_size,
            dense_frame_count=n_dense_frames,
            sparse_frame_count=n_sparse_frames,
            full_latent_hw=mem_latents.shape[-2:],
            sparse_latent_hw=None if sparse_video_latents is None else sparse_video_latents.shape[-2:],
            device=device,
            sparse_downsample_factor=self.video_sparse_downsample_factor,
        )
        video_layout = build_mixed_video_layout(
            frame_indices=frame_indices_bv,
            frame_latent_hw=frame_latent_hw,
            frame_is_sparse=frame_is_sparse,
            sparse_downsample_factor=self.video_sparse_downsample_factor,
            sparse_spatial_rope_mode=self.video_sparse_spatial_rope_mode,
        )
        video_rope_coords = video_layout.video_rope_coords
        latents_flat = pack_frame_latent_groups(mem_latents, dense_video_latents, sparse_video_latents)
        noise, conditioning_mask, cond_indicator = build_mixed_conditioning_tokens(
            mem_latents=mem_latents,
            dense_frame_count=n_dense_frames,
            sparse_frame_count=n_sparse_frames,
            sparse_init_latents=sparse_init_latents,
            generator=noise_generator,
            noise_to_condition_frames=self.cfg.train.noise_to_first_frame,
        )
        latent_frames = video_layout.total_frames
    else:
        if self.vae.vae_encoding_mode == "3dvae":
            latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
        elif self.vae.vae_encoding_mode == "2dvae":
            latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + mem_size
        else:
            raise NotImplementedError

        skip_future_vae_encoding = (
            self.vae.vae_encoding_mode == "2dvae"
            and self.cfg.infer.skip_future_vae_encoding
            and (t_sigma_val is None or precomputed_noisy_latents is not None)
        )
        mem_latents, future_video_latents = get_latents(
            self.vae,
            mem,
            future_video,
            batch_size_2dvae=self.cfg.model.batch_size_2dvae if self.vae.vae_encoding_mode == "2dvae" else None,
            stats_tracker=getattr(self, "stats_tracker", None),
            encode_future_video=not skip_future_vae_encoding,
        )
        mem_latents = rearrange(
            mem_latents,
            "(b v m) (h w) c -> (b v) c m h w",
            b=batch_size,
            m=mem_size,
            h=latent_height,
        )
        if skip_future_vae_encoding:
            future_video_latents = mem_latents[:, :, :1].expand(
                -1, -1, latent_frames - mem_size, -1, -1
            )
        else:
            future_video_latents = rearrange(
                future_video_latents,
                "(b v) (f h w) c -> (b v) c f h w",
                b=batch_size,
                h=latent_height,
                w=latent_width,
            )
        latents_5d = torch.cat((mem_latents, future_video_latents), dim=2)
        latents_flat = rearrange(latents_5d, "bv c f h w -> bv (f h w) c")
        noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
            mem_latents,
            latent_frames,
            latent_height,
            latent_width,
            generator=noise_generator,
            noise_to_condition_frames=self.cfg.train.noise_to_first_frame,
        )
        if future_noise is not None:
            mem_tok = mem_size * latent_height * latent_width
            fn = future_noise.to(device=noise.device, dtype=noise.dtype)
            expected = noise[:, mem_tok:].shape
            if tuple(fn.shape) != tuple(expected):
                raise ValueError(
                    f"future_noise shape {tuple(fn.shape)} != expected {tuple(expected)}"
                )
            noise = noise.clone()
            noise[:, mem_tok:] = fn
        sparse_video_latents = None

    profiler.mark("latent_encoding")

    cond_mask_expanded = conditioning_mask.unsqueeze(-1)
    noise = noise * (1 - cond_mask_expanded) + latents_flat * cond_mask_expanded
    if t_sigma_val is None:
        t_sigma_val = torch.ones(batch_size, device=device, dtype=weight_dtype)
    else:
        t_sigma_val = torch.full((batch_size,), float(t_sigma_val), device=device, dtype=weight_dtype)
    r_sigma_val = torch.zeros(batch_size, device=device, dtype=weight_dtype)
    h_sigma = rearrange((t_sigma_val - r_sigma_val).unsqueeze(1).repeat(1, n_view), "b v -> (b v)")
    sigmas = rearrange(t_sigma_val.unsqueeze(1).repeat(1, n_view), "b v -> (b v)")

    ss = sigmas.reshape(-1, 1, 1).repeat(1, latents_flat.size(1), 1)
    noisy_latents = (1.0 - ss) * latents_flat + ss * noise
    if precomputed_noisy_latents is not None:
        mem_tok = mem_size * latent_height * latent_width
        noisy_latents = noisy_latents.clone()
        noisy_latents[:, mem_tok:] = precomputed_noisy_latents[:, mem_tok:]

    if self.cfg.cotrain.pixel_wise_timestep:
        ts = h_sigma.unsqueeze(-1) * (1 - conditioning_mask)
    else:
        ts = h_sigma.unsqueeze(-1) * (1 - cond_indicator)

    profiler.mark("noise_conditioning")

    captions = batch_segment["caption"]
    prompt_embeds = None
    prompt_attention_mask = None
    vlm_inputs = None
    precomputed_vlm_hidden_states = None
    vlm_attention_mask = None

    tokenizer = getattr(self, "tokenizer", None)
    text_encoder = getattr(self, "text_encoder", None)
    if tokenizer is None:
        task_model = getattr(self, "_task_model", None)
        if task_model is not None:
            tokenizer = task_model.tokenizer
            text_encoder = task_model.text_encoder
    if tokenizer is not None:
        text_conds = get_text_conditions(tokenizer, text_encoder, captions)
        prompt_embeds = text_conds["prompt_embeds"]
        prompt_attention_mask = text_conds["prompt_attention_mask"]

    profiler.mark("text_conditioning")

    use_vlm = self.cfg.model.use_vlm
    if use_vlm and vgen_model_active is not None:
        vgen_model = vgen_model_active
        if hasattr(vgen_model, "module"):
            vgen_model = vgen_model.module
        if getattr(vgen_model, "vlm_interface", None) is not None:
            if getattr(vgen_model, "use_bbox_condition", False):
                from policy_core.trainer.WAM_policy_helpers.vlm_bbox_inputs import build_bbox_vlm_inputs
                if "bbox" not in batch_segment:
                    raise ValueError("bbox-enabled model requires bbox in the request")
                vlm_inputs = build_bbox_vlm_inputs(
                    vgen_model.vlm_interface, video=video_segment,
                    bboxes=batch_segment["bbox"], frame_index=mem_size - 1,
                    head_view_index=vgen_model.vlm_head_view_index, instructions=captions,
                    bbox_reference_images=batch_segment.get("bbox_reference_images"),
                    bbox_add_image_description=vgen_model.bbox_add_image_description,
                )
                if vlm_inputs["input_ids"].shape[-1] != 384:
                    raise ValueError("bbox VLM input must have exactly 384 tokens")
                if not getattr(self, "_bbox_input_logged", False):
                    import logging
                    logging.getLogger(__name__).info(
                        "BBOX_INPUT mode=red_reference input_ids=%s grid=%s description=%s",
                        tuple(vlm_inputs["input_ids"].shape),
                        vlm_inputs.get("image_grid_thw"), vgen_model.bbox_add_image_description)
                    self._bbox_input_logged = True
            elif bool(self.cfg.model.vlm_text_only):
                vlm_inputs = vgen_model.vlm_interface.build_qwenvl_inputs(
                    images=None, instructions=captions, drop_images=True)
            else:
                vlm_images = rearrange(video_segment[:, :, :, mem_size - 1, :, :], "b c v h w -> b v h w c")
                if bool(getattr(vgen_model, "vlm_head_view_only", False)):
                    head_idx = int(getattr(vgen_model, "vlm_head_view_index", 0))
                    vlm_images = vlm_images[:, head_idx:head_idx + 1]
                vlm_images = (vlm_images.detach() + 1) / 2 * 255.0
                vlm_inputs = vgen_model.vlm_interface.build_qwenvl_inputs(images=vlm_images, instructions=captions)
            profiler.mark("vlm_input_build")
            vlm_forward_kwargs = {}
            if self.cfg.infer.vlm_disable_cache:
                vlm_forward_kwargs["use_cache"] = False
            if self.cfg.infer.vlm_logits_to_keep is not None:
                vlm_forward_kwargs["logits_to_keep"] = (
                    self.cfg.infer.vlm_logits_to_keep
                )
            vlm_outputs = vgen_model.vlm_interface(
                **vlm_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                **vlm_forward_kwargs,
            )
            profiler.mark("vlm_forward")
            encoder_hidden_states = vlm_outputs.hidden_states
            vlm_attention_mask = vlm_inputs.get("attention_mask", None) if hasattr(vlm_inputs, "get") else None
            if bool(getattr(vgen_model, "vlm_text_tokens_only", False)):
                encoder_hidden_states, vlm_attention_mask = slice_vlm_text_part(
                    encoder_hidden_states,
                    vlm_attention_mask,
                    vlm_inputs["input_ids"],
                    vgen_model.vlm_interface.model.config.vision_end_token_id,
                )

            bv = batch_size * n_view
            if bv // encoder_hidden_states[0].shape[0] != 1:
                encoder_hidden_states = [ehs for ehs in encoder_hidden_states]
            else:
                encoder_hidden_states = [ehs.view(bv // n_view, -1, ehs.size(-1)) for ehs in encoder_hidden_states]
                if vlm_attention_mask is not None:
                    vlm_attention_mask = vlm_attention_mask.reshape(bv // n_view, -1)
            vlm_attention_mask = vlm_attention_mask.detach() if vlm_attention_mask is not None else None
            precomputed_vlm_hidden_states = [ehs.detach() for ehs in encoder_hidden_states]
            profiler.mark("vlm_postprocess")
            _timing = getattr(self, "_timing", None)
            if _timing is not None:
                _timing.setdefault("vlm_passes", []).append(
                    profiler.get_ms("vlm_forward")
                )

    timestep_scale = float(getattr(self, "timestep_scale", self.cfg.cotrain.timestep_scale))
    ts = ts * timestep_scale
    ts_r_kwargs = {}
    if bool(getattr(getattr(vgen_model_unwrapped, "config", None), "use_t_and_r_embedding", False)):
        keep = (1 - conditioning_mask) if self.cfg.cotrain.pixel_wise_timestep else (1 - cond_indicator)
        ts = sigmas.unsqueeze(-1) * keep * timestep_scale
        r_sigmas = rearrange(r_sigma_val.unsqueeze(1).repeat(1, n_view), "b v -> (b v)")
        ts_r = r_sigmas.unsqueeze(-1) * keep * timestep_scale
        ts_r_kwargs = {"timestep_r": ts_r}

    profiler.mark("dit_input_preparation")
    out = forward_pass(
        model=vgen_model_unwrapped,
        timesteps=ts,
        noisy_latents=noisy_latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        num_frames=latent_frames,
        height=latent_height,
        width=latent_width,
        n_view=n_view,
        view_ids=batch_segment["view_ids"],
        return_video=True,
        return_action=False,
        video_attention_mask=None,
        history_action_state=None,
        condition_mask=conditioning_mask,
        vlm_inputs=vlm_inputs,
        num_frames_wo_dropout=(self.cfg.model.num_frames_wo_dropout if self.cfg.model.num_frames_wo_dropout is not None else mem_size),
        num_mem_frames=mem_size,
        precomputed_vlm_hidden_states=precomputed_vlm_hidden_states,
        vlm_attention_mask=vlm_attention_mask,
        video_rope_coords=video_rope_coords,
        compute_v_head=compute_v_head,
        **ts_r_kwargs,
    )["latents"]
    u = out["video"]
    profiler.mark("video_dit_forward")

    latents_hat = noisy_latents - (ss - float(return_at_t)) * u
    profiler.mark("euler_update")

    if return_at_t > 0.0:
        profiler.mark("output_postprocess")
        profiler.report(
            metadata={"compute_v_head": compute_v_head},
            order=VIDEO_PROFILE_STAGES,
            extra_ms={"vlm_total": profiler.sum_ms(VLM_PROFILE_STAGES)},
        )
        return latents_hat

    mem_latents_det = mem_latents.detach()
    if mixed_mode_active:
        mem_tok = mem_size * latent_height * latent_width
        dense_tok = n_dense_frames * latent_height * latent_width
        dense_hat = latents_hat[:, mem_tok : mem_tok + dense_tok]
        sparse_hat = latents_hat[:, mem_tok + dense_tok :]
        sparse_h, sparse_w = sparse_video_latents.shape[-2], sparse_video_latents.shape[-1]
        dense_hat_5d = rearrange(
            dense_hat,
            "bv (f h w) c -> bv c f h w",
            f=n_dense_frames,
            h=latent_height,
            w=latent_width,
        )
        sparse_hat_5d = rearrange(
            sparse_hat,
            "bv (f h w) c -> bv c f h w",
            f=n_sparse_frames,
            h=sparse_h,
            w=sparse_w,
        )
        dense_with_mem = torch.cat([mem_latents_det, dense_hat_5d], dim=2)
        profiler.mark("output_postprocess")
        profiler.report(
            metadata={"compute_v_head": compute_v_head},
            order=VIDEO_PROFILE_STAGES,
            extra_ms={"vlm_total": profiler.sum_ms(VLM_PROFILE_STAGES)},
        )
        return (dense_with_mem, sparse_hat_5d)

    latents_hat_5d = rearrange(
        latents_hat,
        "bv (f h w) c -> bv c f h w",
        f=latent_frames,
        h=latent_height,
        w=latent_width,
    )
    latents_hat_5d = torch.cat([mem_latents_det, latents_hat_5d[:, :, mem_size:]], dim=2)
    profiler.mark("output_postprocess")
    profiler.report(
        metadata={"compute_v_head": compute_v_head},
        order=VIDEO_PROFILE_STAGES,
        extra_ms={"vlm_total": profiler.sum_ms(VLM_PROFILE_STAGES)},
    )
    return latents_hat_5d

def build_ae_vision_feature_from_latents(
    self,
    ae_latents_dense,
    ae_latents_sparse,
    n_view,
    *,
    dense_topk_idx=None,
    sparse_topk_idx=None,
):
    n_frame_dense = ae_latents_dense.shape[2]
    sparse_length = ae_latents_sparse.shape[2]

    _, l_c, _, l_h_dense, l_w_dense = ae_latents_dense.shape
    _, _, _, l_h_sparse, l_w_sparse = ae_latents_sparse.shape
    ae_latents_dense = rearrange(ae_latents_dense, 'bv c t h w -> bv (t h w) c')
    ae_latents_sparse = rearrange(ae_latents_sparse, 'bv c t h w -> bv (t h w) c')
    vision_pos_emb_dense = self.action_diffusion_model.get_vision_rope(ae_latents_dense, n_frame_dense, l_h_dense, l_w_dense, is_sparse=False, factor=1)
    video_latents_pos_dense = apply_rotary_emb(ae_latents_dense, vision_pos_emb_dense)
    video_latents_pos_dense = rearrange(video_latents_pos_dense, '(b v) l c -> b v l c', v=n_view)
    dummy_tokens_for_offset = torch.zeros(
        (ae_latents_sparse.shape[0], n_frame_dense * l_h_sparse * l_w_sparse, ae_latents_sparse.shape[-1]),
        device=ae_latents_sparse.device,
        dtype=ae_latents_sparse.dtype
    )
    video_latents_for_rope_sparse = torch.cat([dummy_tokens_for_offset, ae_latents_sparse], dim=1)
    total_frames_for_sparse = n_frame_dense + sparse_length
    vision_pos_emb_sparse_full = self.action_diffusion_model.get_vision_rope(
        video_latents_for_rope_sparse,
        total_frames_for_sparse,
        l_h_sparse,
        l_w_sparse,
        is_sparse=False,
        factor=1
    )
    video_latents_pos_sparse_full = apply_rotary_emb(video_latents_for_rope_sparse, vision_pos_emb_sparse_full)
    offset_token_count = n_frame_dense * l_h_sparse * l_w_sparse
    video_latents_pos_sparse = video_latents_pos_sparse_full[:, offset_token_count:]
    video_latents_pos_sparse = rearrange(video_latents_pos_sparse, '(b v) l c -> b v l c', v=n_view)
    video_latents_pos_dense = rearrange(video_latents_pos_dense, 'b v (t h w) c -> (b v) t (h w) c', t=n_frame_dense, h=l_h_dense, w=l_w_dense)
    video_latents_pos_sparse = rearrange(video_latents_pos_sparse, 'b v (t h w) c -> (b v) t (h w) c', t=sparse_length, h=l_h_sparse, w=l_w_sparse)
    video_latents_clean_dense, sparse_video_latents_dense = random_temporal_token_keep(
        video_latents_pos_dense,
        keep_ratio=1/float(self.cfg.data.mask_ratio),
        mask_cur_obs=False,
        topk_idx=dense_topk_idx,
    )
    video_latents_clean_sparse, sparse_video_latents_sparse = random_temporal_token_keep(
        video_latents_pos_sparse,
        keep_ratio=1,
        mask_cur_obs=True,
        topk_idx=sparse_topk_idx,
    )
    if self.cfg.data.drop_cur_obs:
        vision_cond_final_dense = sparse_video_latents_dense
        vision_cond_final_sparse = torch.cat((video_latents_clean_sparse, sparse_video_latents_sparse), dim=1)
        vision_cond_final = torch.cat((vision_cond_final_dense, vision_cond_final_sparse), dim=1)
    else:
        vision_cond_final_dense = torch.cat((video_latents_clean_dense, sparse_video_latents_dense), dim=1)
        vision_cond_final_sparse = torch.cat((video_latents_clean_sparse, sparse_video_latents_sparse), dim=1)
        vision_cond_final = torch.cat((vision_cond_final_dense, vision_cond_final_sparse), dim=1)
    vision_cond = rearrange(vision_cond_final, '(b v) l c -> b v l c', v=n_view)
    return vision_cond

@torch.no_grad()
def save_vgen_debug_video(self, video_latents_for_ae, n_view, save_path, fps=4):
    try:
        from policy_core.utils.WAM_policy.cpu_video_decoder import CpuVideoDecoder

        decoder = getattr(self, "_cpu_video_decoder", None)
        if decoder is None:
            decoder = CpuVideoDecoder(self.vae, getattr(self, "stats_tracker", None))
            # Keep the CPU decoder outside the GPU task module's child modules.
            object.__setattr__(self, "_cpu_video_decoder", decoder)
        n_save_gen = min(1, video_latents_for_ae[0].shape[0] // n_view if isinstance(video_latents_for_ae, tuple) else video_latents_for_ae.shape[0] // n_view)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(video_latents_for_ae, tuple):
            _dense_lat = video_latents_for_ae[0][:n_save_gen * n_view].detach()
            _decoded_dense = decoder.decode(_dense_lat)
            _sparse_lat = video_latents_for_ae[1][:n_save_gen * n_view].detach()
            _decoded_sparse = decoder.decode(_sparse_lat)
            if _decoded_sparse.shape[-2:] != _decoded_dense.shape[-2:]:
                bv_s, c_s, t_s, h_s, w_s = _decoded_sparse.shape
                _decoded_sparse = rearrange(_decoded_sparse, 'bv c t h w -> (bv t) c h w')
                _decoded_sparse = torch.nn.functional.interpolate(
                    _decoded_sparse, size=_decoded_dense.shape[-2:], mode='bilinear', align_corners=False)
                _decoded_sparse = rearrange(_decoded_sparse, '(bv t) c h w -> bv c t h w', bv=bv_s, t=t_s)
            _decoded_full = torch.cat([_decoded_dense, _decoded_sparse], dim=2)
            _decoded_full = rearrange(_decoded_full, '(b v) c t h w -> b c t h (v w)', v=n_view)
            save_video(_decoded_full[0].cpu(), save_path, fps=fps)
        else:
            _lat = video_latents_for_ae[:n_save_gen * n_view].detach()
            _decoded = decoder.decode(_lat)
            _decoded = rearrange(_decoded, '(b v) c t h w -> b c t h (v w)', v=n_view)
            save_video(_decoded[0].cpu(), save_path, fps=fps)
    except Exception as _e:
        logger.warning("[server] failed to decode/save generated video: %s", _e)
