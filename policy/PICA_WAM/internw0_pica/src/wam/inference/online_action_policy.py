"""Online action diffusion from anchor + recent + current frames."""

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch

from wam.model.modules.codecs import video_latent_codec as video_codec
from wam.model.modules.codecs.utils import normalize_input_image_tensor
from wam.model.modules.conditioning.input_builder import prepare_memory_video_inputs, prepare_cached_memory_latents
from wam.model.modules.memory.proprio_encoder import append_proprio_to_context

from .qwen_vl_cache import _get_online_vlm_pack

RTCPrefixSchedule = Literal["linear", "exp", "ones", "zeros"]


@dataclass(frozen=True)
class RTCActionGuidance:
    """Aligned previous-plan target used by real-time chunking guidance.

    ``prev_action_chunk`` must already be aligned to the new chunk's time origin:
    actions consumed before the asynchronous request are removed and the remaining
    suffix is right-padded back to ``[1, H, D]``.  ``inference_delay`` is the
    conservative number of controller ticks that will elapse during inference,
    while ``prefix_attention_horizon`` is the execution horizon ``s`` from RTC.
    """

    prev_action_chunk: torch.Tensor
    inference_delay: int
    prefix_attention_horizon: int
    max_guidance_weight: float = 5.0
    schedule: RTCPrefixSchedule = "exp"


@dataclass(frozen=True)
class RTCActionPrefixCondition:
    """Hard action prefix for a training-time-RTC checkpoint.

    ``prev_action_chunk`` is the previous plan aligned to the new chunk's time
    origin and expressed in normalized model-action space. The first
    ``inference_delay`` actions are already committed and are held fixed while
    the model denoises the remaining postfix.
    """

    prev_action_chunk: torch.Tensor
    inference_delay: int


def get_rtc_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: RTCPrefixSchedule = "exp",
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the prefix weights from the official RTC implementation.

    ``start`` is inclusive: preceding actions are fully frozen. ``end`` is
    exclusive: actions at and after it receive no previous-plan guidance.
    """

    start = int(start)
    end = int(end)
    total = int(total)
    if total <= 0:
        raise ValueError(f"`total` must be positive, got {total}.")
    if start < 0 or end < 0 or start > total or end > total:
        raise ValueError(
            "RTC prefix bounds must lie in [0, total], got "
            f"start={start}, end={end}, total={total}."
        )
    if schedule not in {"linear", "exp", "ones", "zeros"}:
        raise ValueError(f"Unsupported RTC prefix schedule: {schedule!r}.")

    # This matches PI's reference behavior: end takes precedence if end < start.
    start = min(start, end)
    positions = torch.arange(total, device=device, dtype=dtype)
    if schedule == "ones":
        weights = torch.ones(total, device=device, dtype=dtype)
    elif schedule == "zeros":
        weights = (positions < start).to(dtype=dtype)
    else:
        denominator = end - start + 1
        weights = ((start - 1 - positions) / denominator + 1).clamp(0, 1)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / math.expm1(1.0)
    return torch.where(positions >= end, torch.zeros_like(weights), weights)


@dataclass(frozen=True)
class _PreparedRTCActionGuidance:
    prev_action_chunk: torch.Tensor
    prefix_weights: torch.Tensor
    max_guidance_weight: float


@dataclass(frozen=True)
class _PreparedRTCActionPrefixCondition:
    prev_action_chunk: torch.Tensor
    inference_delay: int


def _training_time_rtc_max_delay(model) -> int:
    action_expert = getattr(model, "action_expert", None)
    if not bool(getattr(action_expert, "training_time_rtc_enabled", False)):
        raise ValueError(
            "Hard RTC prefix conditioning requires a checkpoint configured with "
            "training_time_rtc.enabled=true."
        )
    max_delay = getattr(action_expert, "training_time_rtc_max_delay_steps", None)
    if isinstance(max_delay, bool) or not isinstance(max_delay, int):
        raise ValueError(
            "The training-time RTC checkpoint must expose an integer "
            "training_time_rtc_max_delay_steps."
        )
    if max_delay < 0:
        raise ValueError(
            "training_time_rtc_max_delay_steps must be non-negative, got "
            f"{max_delay}."
        )
    return max_delay


def _prepare_rtc_action_prefix_condition(
    condition: RTCActionPrefixCondition,
    *,
    action_horizon: int,
    action_dim: int,
    trained_max_delay: int,
    action_dim_is_pad: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedRTCActionPrefixCondition:
    if not isinstance(condition, RTCActionPrefixCondition):
        raise TypeError(
            "`rtc_prefix_condition` must be an RTCActionPrefixCondition "
            f"instance, got {type(condition).__name__}."
        )
    if isinstance(condition.inference_delay, bool) or not isinstance(
        condition.inference_delay, int
    ):
        raise TypeError("RTC prefix inference_delay must be an integer.")
    delay = int(condition.inference_delay)
    if delay < 0 or delay >= int(action_horizon):
        raise ValueError(
            "RTC prefix inference_delay must lie in [0, action_horizon), got "
            f"{delay} for horizon {action_horizon}."
        )
    if delay > int(trained_max_delay):
        raise ValueError(
            "RTC prefix inference_delay exceeds the checkpoint's trained maximum: "
            f"delay={delay}, trained_max_delay={trained_max_delay}."
        )

    target = torch.as_tensor(condition.prev_action_chunk)
    expected_shape = (1, int(action_horizon), int(action_dim))
    if tuple(target.shape) != expected_shape:
        raise ValueError(
            "RTC prefix previous action chunk must be aligned and have shape "
            f"{expected_shape}, got {tuple(target.shape)}."
        )
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("RTC prefix previous action chunk must contain finite values.")
    target = target.detach().to(device=device, dtype=dtype)
    if action_dim_is_pad is not None:
        target = target.masked_fill(action_dim_is_pad.view(1, 1, -1), 0.0)
    return _PreparedRTCActionPrefixCondition(
        prev_action_chunk=target,
        inference_delay=delay,
    )


def _clamp_rtc_action_prefix(
    latents_action: torch.Tensor,
    condition: _PreparedRTCActionPrefixCondition,
) -> torch.Tensor:
    delay = int(condition.inference_delay)
    if delay == 0:
        return latents_action
    return torch.cat(
        [
            condition.prev_action_chunk[:, :delay],
            latents_action[:, delay:],
        ],
        dim=1,
    )


def _prepare_rtc_action_guidance(
    guidance: RTCActionGuidance,
    *,
    action_horizon: int,
    action_dim: int,
    action_dim_is_pad: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedRTCActionGuidance:
    if not isinstance(guidance, RTCActionGuidance):
        raise TypeError(
            "`rtc_guidance` must be an RTCActionGuidance instance, got "
            f"{type(guidance).__name__}."
        )
    delay = int(guidance.inference_delay)
    prefix_horizon = int(guidance.prefix_attention_horizon)
    if delay < 0 or delay > int(action_horizon):
        raise ValueError(
            "RTC inference delay must lie in [0, action_horizon], got "
            f"{delay} for horizon {action_horizon}."
        )
    if prefix_horizon < delay or prefix_horizon > int(action_horizon):
        raise ValueError(
            "RTC prefix attention horizon must satisfy "
            "inference_delay <= prefix_attention_horizon <= action_horizon, got "
            f"delay={delay}, prefix={prefix_horizon}, horizon={action_horizon}."
        )
    max_guidance_weight = float(guidance.max_guidance_weight)
    if not math.isfinite(max_guidance_weight) or max_guidance_weight < 0:
        raise ValueError(
            "RTC max guidance weight must be finite and non-negative, got "
            f"{guidance.max_guidance_weight!r}."
        )

    target = torch.as_tensor(guidance.prev_action_chunk)
    expected_shape = (1, int(action_horizon), int(action_dim))
    if tuple(target.shape) != expected_shape:
        raise ValueError(
            "RTC previous action chunk must be the already aligned and right-padded "
            f"tensor with shape {expected_shape}, got {tuple(target.shape)}."
        )
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("RTC previous action chunk must contain only finite values.")
    target = target.detach().to(device=device, dtype=dtype)
    if action_dim_is_pad is not None:
        target = target.masked_fill(action_dim_is_pad.view(1, 1, -1), 0.0)
    prefix_weights = get_rtc_prefix_weights(
        delay,
        prefix_horizon,
        int(action_horizon),
        guidance.schedule,
        device=device,
        dtype=dtype,
    ).view(1, int(action_horizon), 1)
    return _PreparedRTCActionGuidance(
        prev_action_chunk=target,
        prefix_weights=prefix_weights,
        max_guidance_weight=max_guidance_weight,
    )


def _rtc_guidance_weight(
    sigma: torch.Tensor,
    *,
    max_guidance_weight: float,
) -> torch.Tensor:
    """Map InternW0-delta's noise-time sigma to RTC's clipped pseudoinverse weight."""

    sigma = sigma.to(dtype=torch.float32)
    clean_time = 1.0 - sigma
    # RTC uses t=0 for noise and t=1 for clean, whereas InternW0-delta uses sigma=1
    # for noise and sigma=0 for clean. This is the reference expression after
    # substituting t = 1 - sigma.
    raw_weight = (
        clean_time.square() + sigma.square()
    ) / (sigma * clean_time)
    raw_weight = torch.nan_to_num(
        raw_weight,
        nan=float(max_guidance_weight),
        posinf=float(max_guidance_weight),
        neginf=0.0,
    )
    return raw_weight.clamp(min=0.0, max=float(max_guidance_weight))


def _apply_rtc_guidance_to_action_velocity(
    *,
    latents_action: torch.Tensor,
    timestep_action: torch.Tensor,
    num_train_timesteps: int,
    predict_velocity: Callable[[torch.Tensor], torch.Tensor],
    guidance: _PreparedRTCActionGuidance,
    action_dim_is_pad: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply RTC's VJP correction to InternW0-delta's ``noise - clean`` velocity."""

    if int(num_train_timesteps) <= 0:
        raise ValueError("`num_train_timesteps` must be positive.")
    with torch.enable_grad():
        action_input = latents_action.detach().requires_grad_(True)
        pred_action = predict_velocity(action_input)
        if action_dim_is_pad is not None:
            pred_action = pred_action.masked_fill(
                action_dim_is_pad.view(1, 1, -1), 0.0
            )
        sigma = (
            timestep_action.to(device=action_input.device, dtype=action_input.dtype)
            / float(num_train_timesteps)
        )
        if sigma.numel() != 1:
            raise ValueError(
                "Online RTC currently requires one scalar action timestep, got "
                f"{tuple(timestep_action.shape)}."
            )
        sigma_sample = sigma.reshape(1, 1, 1)
        # InternW0-delta predicts noise-clean, so its clean estimate has a minus sign.
        clean_action = action_input - sigma_sample * pred_action
        weighted_error = (
            guidance.prev_action_chunk - clean_action
        ) * guidance.prefix_weights
        if action_dim_is_pad is not None:
            weighted_error = weighted_error.masked_fill(
                action_dim_is_pad.view(1, 1, -1), 0.0
            )
        correction = torch.autograd.grad(
            outputs=clean_action,
            inputs=action_input,
            grad_outputs=weighted_error.detach(),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        guidance_weight = _rtc_guidance_weight(
            sigma,
            max_guidance_weight=guidance.max_guidance_weight,
        ).to(device=pred_action.device, dtype=pred_action.dtype)
        # PI adds the correction to clean-noise velocity. InternW0-delta uses the opposite
        # velocity convention, so the correction is subtracted here.
        corrected_velocity = pred_action - guidance_weight * correction
        if action_dim_is_pad is not None:
            corrected_velocity = corrected_velocity.masked_fill(
                action_dim_is_pad.view(1, 1, -1), 0.0
            )
    return corrected_velocity.detach()


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
    if memory_inputs and memory_inputs.get(f"{name}_latents") is not None:
        latents, pad = prepare_cached_memory_latents(
            model,
            dict(memory_inputs),
            name=name,
            batch_size=1,
        )
        if latents is None or pad is None:
            raise RuntimeError(f"Failed to resolve cached online frame {name!r}.")
        return latents, ~pad[:, :1]
    if not memory_inputs or memory_inputs.get(name) is None:
        raise ValueError(
            "Online frame policy requires either "
            f"memory_inputs[{name!r}] or "
            f"memory_inputs[{name + '_latents'!r}]."
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
def _resolve_online_input_image_latents(
    model,
    input_image: torch.Tensor,
    memory_inputs: Optional[dict[str, Any]],
    *,
    tiled: bool,
    video_layout: Any,
    video_view_names: Any,
) -> torch.Tensor:
    latents, _ = prepare_cached_memory_latents(
        model,
        dict(memory_inputs or {}),
        name="video",
        batch_size=1,
    )
    if latents is None:
        latents = video_codec.encode_input_image_latents_tensor(
            model,
            input_image=input_image,
            tiled=tiled,
            video_layout=video_layout,
            video_view_names=video_view_names,
        )
    if (
        latents.ndim != 5
        or int(latents.shape[0]) != 1
        or int(latents.shape[2]) != 1
    ):
        raise ValueError(
            "Online current-frame latents must have shape [1,C,1,H,W], got "
            f"{tuple(latents.shape)}."
        )
    return latents


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




def _predict_online_action_noise_with_cache(
    model,
    *,
    latents_action: torch.Tensor,
    timestep_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_context_kv_cache: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    action_context_kv_cache_stacked: Optional[tuple[torch.Tensor, torch.Tensor]],
    video_kv_cache: list[dict[str, torch.Tensor]],
    video_grouped_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
    attention_mask: torch.Tensor,
    video_seq_len: int,
) -> torch.Tensor:
    action_pre = model.action_expert.pre_dit(
        action_tokens=latents_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    action_context_payload = {
        "context": action_pre["context"],
        "mask": action_pre["context_mask"],
        "kv_cache": action_context_kv_cache,
        "kv_cache_stacked": action_context_kv_cache_stacked,
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
        video_grouped_kv_cache=video_grouped_kv_cache,
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
    rtc_guidance: Optional[RTCActionGuidance] = None,
    rtc_prefix_condition: Optional[RTCActionPrefixCondition] = None,
) -> dict[str, Any]:
    if int(num_inference_steps) < 1:
        raise ValueError("num_inference_steps must be positive")
    model.eval()
    # CUDA Graph Trees retain outputs by invocation. Mark each action chunk as
    # a fresh graph-tree step so consecutive requests cannot recycle a live
    # output tensor.
    mot = getattr(model, "mot", None)
    if getattr(mot, "compile_mode", "off") == "reduce-overhead":
        mark_step_begin = getattr(
            getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
        )
        if mark_step_begin is not None:
            mark_step_begin()
    if rtc_guidance is not None and rtc_prefix_condition is not None:
        raise ValueError(
            "`rtc_guidance` and `rtc_prefix_condition` are mutually exclusive."
        )
    trained_rtc_max_delay = None
    if rtc_prefix_condition is not None:
        trained_rtc_max_delay = _training_time_rtc_max_delay(model)
    if rtc_guidance is not None:
        attention_backend = str(
            getattr(getattr(model, "mot", None), "attention_backend", "")
        ).strip().lower()
        if attention_backend != "sdpa":
            raise ValueError(
                "RTC online action guidance requires MoT attention_backend='sdpa', "
                f"got {attention_backend or '<unknown>'!r}."
            )
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

    prepared_rtc_guidance = None
    if rtc_guidance is not None:
        prepared_rtc_guidance = _prepare_rtc_action_guidance(
            rtc_guidance,
            action_horizon=int(action_horizon),
            action_dim=int(model.action_expert.action_dim),
            action_dim_is_pad=action_dim_is_pad,
            device=model.device,
            dtype=model.torch_dtype,
        )
    prepared_rtc_prefix = None
    if rtc_prefix_condition is not None:
        prepared_rtc_prefix = _prepare_rtc_action_prefix_condition(
            rtc_prefix_condition,
            action_horizon=int(action_horizon),
            action_dim=int(model.action_expert.action_dim),
            trained_max_delay=int(trained_rtc_max_delay),
            action_dim_is_pad=action_dim_is_pad,
            device=model.device,
            dtype=model.torch_dtype,
        )

    first_frame_latents = _resolve_online_input_image_latents(
        model,
        input_image,
        memory_inputs,
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
    mot_compile_mode = str(getattr(model.mot, "compile_mode", "off") or "off")
    prefill_action_context_kv_cache = getattr(
        model.mot, "prefill_action_context_kv_cache", None
    )
    action_context_kv_cache_enabled = str(
        os.environ.get("WAM_ACTION_CONTEXT_KV_CACHE", "1")
    ).lower() not in {"0", "false", "no", "off"}
    action_context_kv_cache = (
        prefill_action_context_kv_cache(action_context)
        if (
            mot_compile_mode != "off"
            and action_context_kv_cache_enabled
            and prefill_action_context_kv_cache is not None
        )
        else None
    )
    action_context_kv_cache_stacked = None
    if action_context_kv_cache is not None:
        action_context_kv_cache_stacked = (
            torch.stack(tuple(layer_kv[0] for layer_kv in action_context_kv_cache)),
            torch.stack(tuple(layer_kv[1] for layer_kv in action_context_kv_cache)),
        )
    video_grouped_kv_cache = (
        (
            torch.stack(tuple(layer_cache["k"] for layer_cache in video_kv_cache)),
            torch.stack(tuple(layer_cache["v"] for layer_cache in video_kv_cache)),
        )
        if mot_compile_mode != "off" and video_kv_cache
        else None
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
        if prepared_rtc_prefix is not None and prepared_rtc_prefix.inference_delay > 0:
            latents_action = _clamp_rtc_action_prefix(
                latents_action, prepared_rtc_prefix
            )
            timestep_action = (
                timestep_action[:, None].expand(-1, int(action_horizon)).clone()
            )
            timestep_action[:, : prepared_rtc_prefix.inference_delay] = 0

        def predict_action_velocity(action_input: torch.Tensor) -> torch.Tensor:
            return _predict_online_action_noise_with_cache(
                model,
                latents_action=action_input,
                timestep_action=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
                action_context_kv_cache=action_context_kv_cache,
                action_context_kv_cache_stacked=action_context_kv_cache_stacked,
                video_kv_cache=video_kv_cache,
                video_grouped_kv_cache=video_grouped_kv_cache,
                attention_mask=action_attention_mask,
                video_seq_len=video_seq_len,
            )

        if (
            prepared_rtc_guidance is not None
            and prepared_rtc_guidance.max_guidance_weight > 0
        ):
            pred_action = _apply_rtc_guidance_to_action_velocity(
                latents_action=latents_action,
                timestep_action=timestep_action,
                num_train_timesteps=int(
                    model.infer_action_scheduler.num_train_timesteps
                ),
                predict_velocity=predict_action_velocity,
                guidance=prepared_rtc_guidance,
                action_dim_is_pad=action_dim_is_pad,
            )
        else:
            pred_action = predict_action_velocity(latents_action)
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
        if prepared_rtc_prefix is not None and prepared_rtc_prefix.inference_delay > 0:
            latents_action = _clamp_rtc_action_prefix(
                latents_action, prepared_rtc_prefix
            )
        # Never retain a denoising graph across scheduler steps. RTC rebuilds
        # the VJP only for the current noisy action latent.
        latents_action = latents_action.detach()

    result = {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
    return result
