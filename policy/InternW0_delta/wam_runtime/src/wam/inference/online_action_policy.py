"""Online action diffusion from anchor + recent + current frames."""

from typing import Any, Optional

import torch

from wam.model.modules.codecs import video_latent_codec as video_codec
from wam.model.modules.codecs.utils import normalize_input_image_tensor
from wam.model.modules.conditioning.input_builder import prepare_memory_video_inputs
from wam.model.modules.memory.proprio_encoder import append_proprio_to_context

from .qwen_vl_cache import _get_online_vlm_pack

def _expand_t_mod_to_tokens(t_mod: torch.Tensor, seq_len: int) -> torch.Tensor:
    if t_mod.ndim == 4:
        if int(t_mod.shape[1]) != int(seq_len):
            raise ValueError(
                f"Per-token t_mod length mismatch: {tuple(t_mod.shape)} vs seq_len={seq_len}"
            )
        return t_mod
    if t_mod.ndim != 3:
        raise ValueError(f"Unexpected t_mod shape: {tuple(t_mod.shape)}")
    return t_mod[:, None, :, :].expand(-1, int(seq_len), -1, -1).contiguous()


def _prepare_static_dim_mask(
    mask: Optional[torch.Tensor],
    *,
    feature_dim: int,
    device: torch.device,
    name: str,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.ndim != 1 or int(mask.numel()) != int(feature_dim):
        raise ValueError(f"{name} must be [{feature_dim}], got {tuple(mask.shape)}.")
    return mask


def _prepare_action_hz(
    model,
    action_hz: Optional[torch.Tensor | float],
    *,
    batch_size: int = 1,
) -> Optional[torch.Tensor]:
    if not bool(
        getattr(
            getattr(model, "action_expert", None),
            "physical_time_rope_enabled",
            False,
        )
    ):
        return None
    if action_hz is None:
        raise ValueError(
            "Physical-time Action RoPE requires an explicit `action_hz`."
        )
    value = torch.as_tensor(action_hz, dtype=torch.float32, device="cpu")
    if value.ndim == 0:
        value = value.reshape(1)
    elif value.ndim == 2 and int(value.shape[1]) == 1:
        value = value[:, 0]
    elif value.ndim != 1:
        raise ValueError(
            f"`action_hz` must be scalar, [B], or [B,1], got {tuple(value.shape)}."
        )
    if int(value.numel()) == 1 and int(batch_size) > 1:
        value = value.expand(int(batch_size))
    if int(value.numel()) != int(batch_size):
        raise ValueError(
            f"`action_hz` must contain {batch_size} values, got {int(value.numel())}."
        )
    if not bool(torch.isfinite(value).all().item()) or not bool(
        (value > 0).all().item()
    ):
        raise ValueError("`action_hz` values must be finite and positive.")
    return value.to(device=model.device, dtype=torch.float32)




@torch.no_grad()
def _encode_online_condition_frame(
    model,
    memory_inputs: Optional[dict[str, Any]],
    *,
    name: str,
    height: int,
    width: int,
    tiled: bool,
    video_layout: Any,
    video_view_names: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not memory_inputs or memory_inputs.get(name) is None:
        raise ValueError(
            f"Online frame policy requires memory_inputs[{name!r}]."
        )
    video, pad = prepare_memory_video_inputs(
        model,
        dict(memory_inputs),
        name=name,
        batch_size=1,
        height=height,
        width=width,
    )
    num_frames = int(video.shape[2] if video.ndim == 5 else video.shape[3])
    if num_frames != 1:
        raise ValueError(f"{name} must contain one frame, got {num_frames}.")
    latents = video_codec.encode_video_latents(
        model,
        video,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    if int(latents.shape[2]) != 1:
        raise ValueError(f"{name} must encode to one latent frame.")
    return latents, ~pad[:, :1]


@torch.no_grad()
def _build_mot_attention_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    num_future_delta_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    total_seq_len = int(video_seq_len) + int(action_seq_len)
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
    tokens_per_frame = int(video_tokens_per_frame)
    real_video_seq_len = int(video_seq_len) - num_future_delta_tokens
    num_frames = real_video_seq_len // tokens_per_frame
    for frame_idx in range(num_frames):
        query = slice(
            frame_idx * tokens_per_frame,
            (frame_idx + 1) * tokens_per_frame,
        )
        mask[query, : (frame_idx + 1) * tokens_per_frame] = True
    if num_future_delta_tokens:
        delta_start = real_video_seq_len
        local_video_start = tokens_per_frame
        mask[
            delta_start:video_seq_len,
            local_video_start:real_video_seq_len,
        ] = True
        mask[
            delta_start:video_seq_len,
            delta_start:video_seq_len,
        ] = True
    action_start = video_seq_len
    total_end = action_start + int(action_seq_len)
    mask[action_start:total_end, :real_video_seq_len] = True
    if num_future_delta_tokens:
        mask[
            action_start:total_end,
            real_video_seq_len:video_seq_len,
        ] = True
    mask[action_start:total_end, action_start:total_end] = True
    return mask




@torch.no_grad()
def _predict_online_action_noise_with_cache(
    model,
    *,
    latents_action: torch.Tensor,
    timestep_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_kv_cache: list[dict[str, torch.Tensor]],
    attention_mask: torch.Tensor,
    video_seq_len: int,
    action_hz: Optional[torch.Tensor],
) -> torch.Tensor:
    action_pre = model.action_expert.pre_dit(
        action_tokens=latents_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
        action_hz=action_hz,
    )
    action_context_payload = {
        "context": action_pre["context"],
        "mask": action_pre["context_mask"],
    }
    action_tokens = action_pre["tokens"]
    action_freqs = action_pre["freqs"]
    action_t_mod = _expand_t_mod_to_tokens(
        action_pre["t_mod"], int(action_tokens.shape[1])
    )
    action_tokens = model.mot.forward_action_with_video_cache(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t_mod,
        action_context_payload=action_context_payload,
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=int(video_seq_len),
    )
    return model.action_expert.post_dit(action_tokens, action_pre)


@torch.no_grad()
def infer_online_action_chunk(
    model,
    prompt: Optional[str],
    input_image: torch.Tensor,
    action_horizon: int,
    *,
    action_hz: Optional[torch.Tensor | float] = None,
    proprio: Optional[torch.Tensor] = None,
    action_dim_is_pad: Optional[torch.Tensor] = None,
    memory_inputs: Optional[dict[str, Any]] = None,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    vlm_current_images: Optional[torch.Tensor] = None,
    vlm_current_view_is_pad: Optional[torch.Tensor] = None,
    vlm_view_names: Any = None,
    understanding_prompt: Optional[str] = None,
    num_inference_steps: int = 20,
    sigma_shift: Optional[float] = None,
    seed: Optional[int] = None,
    rand_device: str = "cpu",
    tiled: bool = False,
    memory_chunk_index: int = 0,
    video_layout: Any = None,
    video_view_names: Any = None,
) -> dict[str, Any]:
    model.eval()
    input_image = normalize_input_image_tensor(input_image).to(
        device=model.device, dtype=model.torch_dtype
    )
    height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})"
        )
    action_token_seq_len = model._action_token_seq_len_for_mask(
        action_horizon
    )
    action_hz = _prepare_action_hz(model, action_hz, batch_size=1)
    action_dim_is_pad = _prepare_static_dim_mask(
        action_dim_is_pad,
        feature_dim=int(model.action_expert.action_dim),
        device=model.device,
        name="action_dim_is_pad",
    )
    if proprio is not None:
        if model.proprio_dim is None:
            raise ValueError("`proprio` was provided but `proprio_dim=None`.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or int(proprio.shape[1]) != int(model.proprio_dim):
            raise ValueError(
                f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}"
            )
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
    elif model.understanding_enabled:
        raise ValueError(
            "Action Qwen/state conditioning requires current `proprio`."
        )

    use_prompt = prompt is not None
    use_context = context is not None or context_mask is not None
    if use_prompt and use_context:
        raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
    if not use_prompt and not use_context:
        raise ValueError(
            "Video T5 conditioning requires either `prompt` or both "
            "`context/context_mask`."
        )
    if use_prompt:
        context, context_mask = model.encode_prompt(prompt)
    else:
        if context is None or context_mask is None:
            raise ValueError(
                "`context` and `context_mask` must be both provided together."
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
    if proprio is not None and model.proprio_encoder is not None:
        context, context_mask = append_proprio_to_context(
            model,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
    video_context = context
    video_context_mask = context_mask
    action_context = video_context
    action_context_mask = video_context_mask

    generator = (
        None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    )
    latents_action = torch.randn(
        (1, int(action_horizon), int(model.action_expert.action_dim)),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    mask_invalid_action = action_dim_is_pad is not None
    if mask_invalid_action:
        latents_action = latents_action.masked_fill(
            action_dim_is_pad.view(1, 1, -1), 0.0
        )

    first_frame_latents = video_codec.encode_input_image_latents_tensor(
        model,
        input_image=input_image,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    anchor_latents, anchor_valid = _encode_online_condition_frame(
        model,
        memory_inputs,
        name="memory_video_anchor",
        height=height,
        width=width,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    recent_latents, recent_valid = _encode_online_condition_frame(
        model,
        memory_inputs,
        name="memory_video_recent",
        height=height,
        width=width,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    first_frame_latents = torch.cat(
        [anchor_latents, recent_latents, first_frame_latents], dim=2
    )
    visual_frame_valid = torch.cat(
        [
            anchor_valid,
            recent_valid,
            torch.ones_like(anchor_valid, dtype=torch.bool),
        ],
        dim=1,
    )
    timestep_video = torch.zeros(
        (first_frame_latents.shape[0], int(first_frame_latents.shape[2]) - 1),
        dtype=first_frame_latents.dtype,
        device=model.device,
    )
    if model.understanding_enabled:
        use_understanding_prompt = (
            understanding_prompt if understanding_prompt is not None else (prompt or "")
        )
        vlm_pack = _get_online_vlm_pack(
            model,
            vlm_current_images=vlm_current_images,
            proprio=proprio,
            prompt=use_understanding_prompt,
            vlm_view_names=vlm_view_names,
            vlm_current_view_is_pad=vlm_current_view_is_pad,
        )
        action_context = vlm_pack["vlm_context"].to(
            device=model.device, dtype=model.torch_dtype
        )
        action_context_mask = vlm_pack["vlm_mask"].to(
            device=model.device, dtype=torch.bool
        )
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=video_context,
        context_mask=video_context_mask,
        action=None,
        fuse_vae_embedding_in_latents=bool(
            getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
        ),
        frame_ids=torch.arange(
            int(first_frame_latents.shape[2]),
            device=first_frame_latents.device,
        ),
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    if model.future_delta_enabled:
        model.video_expert.append_future_delta_queries(
            video_pre,
            current_frame_index=int(first_frame_latents.shape[2]) - 1,
        )
    video_seq_len = int(video_pre["tokens"].shape[1])
    video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    video_key_mask = visual_frame_valid.repeat_interleave(
        video_tokens_per_frame, dim=1
    )
    num_future_delta_tokens = int(
        video_pre["meta"].get("num_future_delta_tokens", 0)
    )
    if num_future_delta_tokens:
        video_key_mask = torch.cat(
            [
                video_key_mask,
                torch.ones(
                    (int(video_key_mask.shape[0]), num_future_delta_tokens),
                    dtype=torch.bool,
                    device=video_key_mask.device,
                ),
            ],
            dim=1,
        )
    video_context_payload = {
        "context": video_pre["context"],
        "mask": video_pre["context_mask"],
    }

    joint_attention_mask = _build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_token_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        num_future_delta_tokens=num_future_delta_tokens,
        device=video_pre["tokens"].device,
    )
    video_kv_cache = model.mot.prefill_expert_cache(
        expert_name="video",
        tokens=video_pre["tokens"],
        freqs=video_pre["freqs"],
        t_mod=video_pre["t_mod"],
        context_payload=video_context_payload,
        attention_mask=joint_attention_mask[:video_seq_len, :video_seq_len],
        key_mask=video_key_mask,
    )
    action_attention_mask = joint_attention_mask

    infer_timesteps_action, infer_deltas_action = (
        model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=model.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
    )
    for step_t_action, step_delta_action in zip(
        infer_timesteps_action, infer_deltas_action
    ):
        timestep_action = step_t_action.unsqueeze(0).to(
            dtype=latents_action.dtype, device=model.device
        )
        pred_action = _predict_online_action_noise_with_cache(
            model,
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=action_attention_mask,
            video_seq_len=video_seq_len,
            action_hz=action_hz,
        )
        if mask_invalid_action:
            pred_action = pred_action.masked_fill(
                action_dim_is_pad.view(1, 1, -1), 0.0
            )
        latents_action = model.infer_action_scheduler.step(
            pred_action, step_delta_action, latents_action
        )
        if mask_invalid_action:
            latents_action = latents_action.masked_fill(
                action_dim_is_pad.view(1, 1, -1), 0.0
            )

    return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
