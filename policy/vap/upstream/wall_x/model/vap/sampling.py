"""Correctness-first autoregressive samplers for native VAP.

The conditional samplers deliberately fill an IVAVA window one stage at a
time.  Future ground-truth chunks are not inserted before their causal turn:

* action -> video: ``I,A0 -> V0`` then ``I,A0,V0,A1 -> V1``;
* video -> action: ``I -> A0`` then ``I,A0,V0 -> A1``.

That detail keeps validation aligned with the clean-prefix/noisy-suffix
distribution used by training and prevents future-GT leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from .loss import ARStage, ar_chunk_sigmas
from .layout import VideoOnlyParallelTeacherForcingLayout
from .memory import FirstFrameMemoryTokens
from .rope import action_position_ids
from .wan_bridge import (
    EncodedFirstFrameMemory,
    EncodedVideoSequence,
    PatchedVideoSequence,
    patchify_video_sequence,
    patchify_video_sequence_with_memory,
    unpatchify_video_sequence,
)

if TYPE_CHECKING:
    from .training import VAPTrainingModel


@dataclass(frozen=True)
class VAPSequentialSample:
    """Generated normalized actions and continuous ``I,V0,V1`` latents."""

    video_latents: torch.Tensor
    actions: torch.Tensor
    stage_order: tuple[str, ...] = ("A0", "V0", "A1", "V1")


@dataclass(frozen=True)
class VAPVideoFromActionSample:
    """Generated ``I,V0,V1`` latents conditioned on chunk-local actions."""

    video_latents: torch.Tensor
    condition_actions: torch.Tensor
    stage_order: tuple[str, ...] = ("V0", "V1")


@dataclass(frozen=True)
class VAPVideoRolloutChunkSample:
    """One generated video chunk inside the fixed two-slot VAP window."""

    video_latents: torch.Tensor
    generated_chunk_latents: torch.Tensor
    active_slot: int


@dataclass(frozen=True)
class VAPVideoOnlySample:
    """Generated continuous ``I,V0,...,V{K-1}`` video-only latents."""

    video_latents: torch.Tensor
    stage_order: tuple[str, ...] = ("V0", "V1")


@dataclass(frozen=True)
class VAPVideoOnlyRolloutChunkSample:
    """One generated video chunk inside the native video-only VAP window."""

    video_latents: torch.Tensor
    generated_chunk_latents: torch.Tensor
    active_slot: int


@dataclass(frozen=True)
class VAPActionFromVideoSample:
    """Generated normalized ``A0,A1`` conditioned on causal video prefixes."""

    actions: torch.Tensor
    condition_video_latents: torch.Tensor
    stage_order: tuple[str, ...] = ("A0", "A1")


@dataclass(frozen=True)
class VAPActionRolloutChunkSample:
    """One generated action chunk inside the fixed two-slot VAP window."""

    actions: torch.Tensor
    generated_chunk: torch.Tensor
    active_slot: int


@dataclass(frozen=True)
class _SamplingState:
    video: torch.Tensor
    action: torch.Tensor
    view_ids: torch.Tensor
    text_context: torch.Tensor
    action_positions: torch.Tensor
    raw_frames: int
    latent_frames_per_chunk: int
    action_steps_per_chunk: int
    temporal_stride: int
    state_condition: torch.Tensor | None
    first_frame_memory: EncodedFirstFrameMemory | None


def shifted_sigma_schedule(
    num_steps: int,
    *,
    shift: float,
    device: torch.device,
) -> torch.Tensor:
    """Descending straight-flow sigma schedule including both 1 and 0."""

    if num_steps <= 0 or shift <= 0:
        raise ValueError("num_steps and shift must be positive")
    base = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return shift * base / (1 + (shift - 1) * base)


def _stage_chunk_sigmas(
    stage: ARStage,
    sigma: torch.Tensor,
    *,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage_ids = torch.full(
        (sigma.shape[0],), int(stage), device=sigma.device, dtype=torch.long
    )
    video_ranks = torch.arange(2, 2 * num_chunks + 1, 2, device=sigma.device)
    action_ranks = torch.arange(1, 2 * num_chunks, 2, device=sigma.device)
    return (
        ar_chunk_sigmas(stage_ids, chunk_ranks=video_ranks, active_sigma=sigma),
        ar_chunk_sigmas(stage_ids, chunk_ranks=action_ranks, active_sigma=sigma),
    )


def _replace_video_chunk(
    video: torch.Tensor,
    velocity: torch.Tensor,
    *,
    chunk_index: int,
    frames_per_chunk: int,
    delta_sigma: torch.Tensor,
) -> torch.Tensor:
    start = 1 + chunk_index * frames_per_chunk
    end = start + frames_per_chunk
    updated = video.clone()
    scale = delta_sigma.reshape(-1, 1, 1, 1, 1, 1).to(video)
    updated[:, :, :, start:end] = (
        video[:, :, :, start:end] + scale * velocity[:, :, :, start:end]
    )
    return updated


def _replace_action_chunk(
    action: torch.Tensor,
    velocity: torch.Tensor,
    *,
    chunk_index: int,
    steps_per_chunk: int,
    delta_sigma: torch.Tensor,
) -> torch.Tensor:
    start = chunk_index * steps_per_chunk
    end = start + steps_per_chunk
    updated = action.clone()
    scale = delta_sigma.reshape(-1, 1, 1).to(action)
    updated[:, start:end] = action[:, start:end] + scale * velocity[:, start:end]
    return updated


def _copy_video_chunk(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    chunk_index: int,
    frames_per_chunk: int,
) -> torch.Tensor:
    start = 1 + chunk_index * frames_per_chunk
    end = start + frames_per_chunk
    output = destination.clone()
    output[:, :, :, start:end] = source[:, :, :, start:end]
    return output


def _copy_action_chunk(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    chunk_index: int,
    steps_per_chunk: int,
) -> torch.Tensor:
    start = chunk_index * steps_per_chunk
    end = start + steps_per_chunk
    output = destination.clone()
    output[:, start:end] = source[:, start:end]
    return output


def _prepare_sampling_state(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None,
    video_noise: torch.Tensor | None,
    action_noise: torch.Tensor | None,
    state_condition: torch.Tensor | None = None,
    first_frame_memory: EncodedFirstFrameMemory | None = None,
) -> _SamplingState:
    config = model.config
    core = model.core
    if condition_video_latents.ndim != 6 or condition_video_latents.shape[3] != 1:
        raise ValueError("condition_video_latents must be [B,V,C,1,H,W]")
    batch_size, num_views, channels, _, height, width = condition_video_latents.shape
    if tuple(view_ids.shape) != (batch_size, num_views):
        raise ValueError("view_ids must be [B,V]")
    if text_context.ndim != 3 or text_context.shape[0] != batch_size:
        raise ValueError("text_context must be [B,L,C]")

    parameter = next(core.parameters())
    device, dtype = parameter.device, model.torch_dtype
    condition = condition_video_latents.to(device=device, dtype=dtype)
    view_ids = view_ids.to(device=device, dtype=torch.long)
    text_context = text_context.to(device=device, dtype=dtype)
    memory_enabled = bool(getattr(config, "first_frame_memory_enabled", False))
    if memory_enabled != (first_frame_memory is not None):
        raise ValueError("first-frame memory config and encoded Anchor Bank must agree")
    if first_frame_memory is not None:
        available_view_keys = tuple(f"view{index + 1}" for index in range(num_views))
        if not first_frame_memory.view_keys or len(
            set(first_frame_memory.view_keys)
        ) != len(first_frame_memory.view_keys):
            raise ValueError("first-frame memory views must be non-empty and unique")
        missing_view_keys = tuple(
            key
            for key in first_frame_memory.view_keys
            if key not in available_view_keys
        )
        if missing_view_keys:
            raise ValueError(
                "first-frame memory views are absent from serving input: "
                f"{missing_view_keys}; available={available_view_keys}"
            )
        history_offsets = tuple(getattr(config, "history_offsets", ()))
        memory_slots = 1 + len(history_offsets) if history_offsets else 1
        expected_memory_shape = (
            batch_size,
            len(first_frame_memory.view_keys),
            channels,
            memory_slots,
            height,
            width,
        )
        if tuple(first_frame_memory.latents.shape) != expected_memory_shape:
            raise ValueError(
                "first-frame memory latent shape "
                f"{tuple(first_frame_memory.latents.shape)} != "
                f"{expected_memory_shape}"
            )
        expected_slot_shape = (batch_size, memory_slots)
        if history_offsets:
            if first_frame_memory.valid is None or first_frame_memory.age_steps is None:
                raise ValueError(
                    "history memory requires explicit per-slot valid and age_steps"
                )
            if tuple(first_frame_memory.valid.shape) != expected_slot_shape:
                raise ValueError(
                    "history memory valid shape "
                    f"{tuple(first_frame_memory.valid.shape)} != {expected_slot_shape}"
                )
            if tuple(first_frame_memory.age_steps.shape) != expected_slot_shape:
                raise ValueError(
                    "history memory age_steps shape "
                    f"{tuple(first_frame_memory.age_steps.shape)} != "
                    f"{expected_slot_shape}"
                )
        elif first_frame_memory.valid is not None:
            raise ValueError(
                "per-slot history validity was supplied with empty history_offsets"
            )
        first_frame_memory = replace(
            first_frame_memory,
            latents=first_frame_memory.latents.to(device=device, dtype=dtype),
            age_steps=(
                first_frame_memory.age_steps.to(device=device, dtype=torch.long)
                if first_frame_memory.age_steps is not None
                else None
            ),
            valid=(
                first_frame_memory.valid.to(device=device, dtype=torch.bool)
                if first_frame_memory.valid is not None
                else None
            ),
        )
    if getattr(core, "state_film", None) is None:
        if state_condition is not None:
            raise ValueError("state_condition was provided but State FiLM is disabled")
    else:
        if state_condition is None:
            raise ValueError("State FiLM is enabled; provide the current window S0")
        if state_condition.ndim not in (2, 3) or state_condition.shape[0] != batch_size:
            raise ValueError("state_condition must be [B,D], [B,1,D], or [B,E,D]")
        state_condition = state_condition.to(device=device, dtype=dtype)
    num_chunks = int(config.num_interleave_chunks)
    video_frames_per_chunk = int(config.video_frames_per_chunk)
    vae_stride = int(config.vae_temporal_stride)
    latent_frames_per_chunk = video_frames_per_chunk // vae_stride
    action_steps_per_chunk = int(config.action_steps_per_chunk)
    future_latent_shape = (
        batch_size,
        num_views,
        channels,
        num_chunks * latent_frames_per_chunk,
        height,
        width,
    )
    action_shape = (
        batch_size,
        num_chunks * action_steps_per_chunk,
        int(core.action_input_dim),
    )
    if video_noise is None:
        video_noise = torch.randn(
            future_latent_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        video_noise = video_noise.to(device=device, dtype=dtype)
    action_noise_dtype = (
        torch.float32
        if getattr(core, "_v2_action_noise_dtype", "legacy") == "fp32"
        else dtype
    )
    if action_noise is None:
        action_noise = torch.randn(
            action_shape,
            device=device,
            dtype=action_noise_dtype,
            generator=generator,
        )
    else:
        action_noise = action_noise.to(device=device, dtype=action_noise_dtype)
    if tuple(video_noise.shape) != future_latent_shape:
        raise ValueError(
            f"video_noise shape {tuple(video_noise.shape)} != {future_latent_shape}"
        )
    if tuple(action_noise.shape) != action_shape:
        raise ValueError(
            f"action_noise shape {tuple(action_noise.shape)} != {action_shape}"
        )

    return _SamplingState(
        video=torch.cat((condition, video_noise), dim=3),
        action=action_noise,
        view_ids=view_ids,
        text_context=text_context,
        action_positions=action_position_ids(
            batch_size,
            num_chunks,
            action_steps_per_chunk,
            chunk_stride=action_steps_per_chunk,
            device=device,
        ),
        raw_frames=1 + num_chunks * video_frames_per_chunk,
        latent_frames_per_chunk=latent_frames_per_chunk,
        action_steps_per_chunk=action_steps_per_chunk,
        temporal_stride=(vae_stride * action_steps_per_chunk // video_frames_per_chunk),
        state_condition=state_condition,
        first_frame_memory=first_frame_memory,
    )


def _patch_sampling_video(
    model: "VAPTrainingModel",
    state: _SamplingState,
    video: torch.Tensor,
):
    """Patch one sampling window using the checkpoint's Memory injection mode.

    Returns the joint-attention sequence, the original main-video metadata used
    for unpatchification, and an optional external gated Anchor Bank.
    """

    config = model.config
    num_views = video.shape[1]
    encoded = EncodedVideoSequence(
        latents=video,
        view_keys=tuple(f"view{index + 1}" for index in range(num_views)),
        raw_frames=state.raw_frames,
        num_chunks=int(config.num_interleave_chunks),
        future_latent_frames_per_chunk=state.latent_frames_per_chunk,
    )
    if state.first_frame_memory is None:
        patched = patchify_video_sequence(
                model.core,
                encoded,
                view_ids=state.view_ids,
                chunk_stride=state.action_steps_per_chunk,
                temporal_stride=state.temporal_stride,
            )
        return patched, patched, None
    patched, memory = patchify_video_sequence_with_memory(
        model.core,
        encoded,
        state.first_frame_memory,
        view_ids=state.view_ids,
        memory_grid=tuple(config.first_frame_memory_grid),
        chunk_stride=state.action_steps_per_chunk,
        temporal_stride=state.temporal_stride,
        position_mode=getattr(
            config, "first_frame_memory_position_mode", "window_origin"
        ),
    )
    injection_mode = getattr(
        config,
        "first_frame_memory_injection_mode",
        "gated_cross_attention",
    )
    if injection_mode == "prefix_self_attention":
        return _prepend_first_frame_memory_prefix(patched, memory), patched, None
    if injection_mode != "gated_cross_attention":
        raise ValueError(f"unsupported first-frame memory mode: {injection_mode!r}")
    return patched, patched, memory


def _prepend_first_frame_memory_prefix(
    patched: PatchedVideoSequence,
    memory: FirstFrameMemoryTokens,
) -> PatchedVideoSequence:
    """Prepend Memory as read-only condition-video tokens for joint attention."""

    if memory.tokens.shape[0] != patched.tokens.shape[0]:
        raise ValueError("Memory and Video prefix batch sizes must match")
    if memory.tokens.shape[2] != patched.tokens.shape[2]:
        raise ValueError("Memory and Video prefix embedding dimensions must match")
    if memory.position_ids.device != patched.position_ids.device:
        raise ValueError("Memory and Video prefix position IDs must share a device")
    if memory.camera_ids.device != patched.camera_ids.device:
        raise ValueError("Memory and Video prefix camera IDs must share a device")
    memory_token_count = memory.tokens.shape[1]
    memory_condition_mask = torch.ones(
        patched.tokens.shape[0],
        memory_token_count,
        device=patched.condition_token_mask.device,
        dtype=torch.bool,
    )
    return replace(
        patched,
        tokens=torch.cat((memory.tokens, patched.tokens), dim=1),
        position_ids=torch.cat((memory.position_ids, patched.position_ids), dim=1),
        camera_ids=torch.cat((memory.camera_ids, patched.camera_ids), dim=1),
        condition_token_mask=torch.cat(
            (memory_condition_mask, patched.condition_token_mask), dim=1
        ),
        condition_token_count=patched.condition_token_count + memory_token_count,
        memory_valid=memory.valid,
    )


def _denoise_stage(
    model: "VAPTrainingModel",
    state: _SamplingState,
    stage: ARStage,
) -> _SamplingState:
    config = model.config
    core = model.core
    batch_size, num_views = state.video.shape[:2]
    num_chunks = int(config.num_interleave_chunks)
    flow_config = config.video_flow if stage.is_video else config.action_flow
    schedule = shifted_sigma_schedule(
        int(flow_config["inference_steps"]),
        shift=float(flow_config["shift"]),
        device=state.video.device,
    )
    video, action = state.video, state.action
    fp32_action = bool(getattr(core, "_v2_action_fp32_accum", False))
    if fp32_action and not stage.is_video:
        # Keep the original random draw and clean-prefix contents. Promote only
        # the ODE state; network calls still see the checkpoint's model dtype.
        action = action.float()
    for current_sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
        sigma = current_sigma.expand(batch_size)
        video_chunk_sigma, action_chunk_sigma = _stage_chunk_sigmas(
            stage, sigma, num_chunks=num_chunks
        )
        patched, main_patched, first_frame_memory = _patch_sampling_video(
            model, state, video
        )
        video_timestep = patched.per_token_timestep(
            video_chunk_sigma * float(config.video_flow["train_timesteps"])
        )
        action_timestep = action_chunk_sigma.repeat_interleave(
            state.action_steps_per_chunk, dim=1
        ) * float(config.action_flow["train_timesteps"])
        output = core(
            patched.tokens,
            action.to(model.torch_dtype) if fp32_action else action,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            video_position_ids=patched.position_ids,
            condition_video_tokens=patched.condition_token_count,
            action_position_ids=state.action_positions,
            video_camera_ids=patched.camera_ids,
            text_context=state.text_context,
            state_condition=state.state_condition,
            first_frame_memory=first_frame_memory,
            memory_prefix_valid=patched.memory_valid,
        )
        delta = (next_sigma - current_sigma).expand(batch_size)
        if stage.is_video:
            memory_prefix_token_count = (
                patched.condition_token_count - main_patched.condition_token_count
            )
            main_video_velocity = output.video_velocity[:, memory_prefix_token_count:]
            velocity = unpatchify_video_sequence(
                core, main_video_velocity, main_patched
            )
            video = _replace_video_chunk(
                video,
                velocity,
                chunk_index=stage.chunk_index,
                frames_per_chunk=state.latent_frames_per_chunk,
                delta_sigma=delta,
            )
        else:
            action = _replace_action_chunk(
                action,
                output.action_velocity,
                chunk_index=stage.chunk_index,
                steps_per_chunk=state.action_steps_per_chunk,
                delta_sigma=delta,
            )
    return replace(state, video=video, action=action)


def _require_video_only_model(model: "VAPTrainingModel") -> None:
    mode = str(getattr(model.config, "training_mode", "paired_ivava"))
    if mode != "video_only":
        raise ValueError(
            "Stage-1 TI2V sampling requires model.training_mode=video_only, "
            f"got {mode!r}"
        )
    if int(model.config.num_interleave_chunks) < 2:
        raise ValueError("video-only TI2V sampling requires at least two video slots")


def _video_only_velocity(
    model: "VAPTrainingModel",
    state: _SamplingState,
    *,
    active_chunk: int,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one noisy Video branch with the exact training PTF layout.

    The physical standard state is ``[I,V0,...,V{K-1}]``. For ``Vk``, every
    earlier chunk has already been denoised and is supplied as the clean prefix
    branch. For V0, the clean V0 branch is deliberately populated with the
    current value: the
    :class:`VideoOnlyParallelTeacherForcingLayout` makes it causally invisible
    to the noisy V0 query, which is the same invariant used during training.
    Dummy Action branches remain shape-compatible with the checkpoint but are
    isolated from Video by modality and therefore carry no conditioning.
    """

    config = model.config
    core = model.core
    batch_size, num_views = state.video.shape[:2]
    num_chunks = int(config.num_interleave_chunks)
    if not 0 <= active_chunk < num_chunks:
        raise ValueError(
            f"active_chunk must be in [0, {num_chunks}), got {active_chunk}"
        )

    clean_video = state.video
    combined = EncodedVideoSequence(
        latents=torch.cat((clean_video, state.video), dim=0),
        view_keys=tuple(f"view{index}" for index in range(num_views)),
        raw_frames=state.raw_frames,
        num_chunks=num_chunks,
        future_latent_frames_per_chunk=state.latent_frames_per_chunk,
    )
    combined_patched = patchify_video_sequence(
        core,
        combined,
        view_ids=torch.cat((state.view_ids, state.view_ids), dim=0),
        chunk_stride=state.action_steps_per_chunk,
        temporal_stride=state.temporal_stride,
    )
    clean_patched = replace(
        combined_patched,
        tokens=combined_patched.tokens[:batch_size],
        position_ids=combined_patched.position_ids[:batch_size],
        camera_ids=combined_patched.camera_ids[:batch_size],
        condition_token_mask=combined_patched.condition_token_mask[:batch_size],
    )
    noisy_patched = replace(
        combined_patched,
        tokens=combined_patched.tokens[batch_size:],
        position_ids=combined_patched.position_ids[batch_size:],
        camera_ids=combined_patched.camera_ids[batch_size:],
        condition_token_mask=combined_patched.condition_token_mask[batch_size:],
    )
    layout = VideoOnlyParallelTeacherForcingLayout(
        num_chunks=num_chunks,
        condition_video_tokens=clean_patched.condition_token_count,
        video_tokens_per_chunk=clean_patched.video_tokens_per_chunk,
        action_tokens_per_chunk=state.action_steps_per_chunk,
    )
    packed_video = layout.expand_video(clean_patched.tokens, noisy_patched.tokens)
    packed_positions = layout.expand_video(
        clean_patched.position_ids, noisy_patched.position_ids
    )
    packed_camera_ids = layout.expand_video(
        clean_patched.camera_ids[..., None],
        noisy_patched.camera_ids[..., None],
    ).squeeze(-1)

    chunk_sigmas = sigma.new_zeros((batch_size, num_chunks))
    chunk_sigmas[:, active_chunk] = sigma
    clean_video_timestep = sigma.new_zeros(clean_patched.tokens.shape[:2])
    noisy_video_timestep = noisy_patched.per_token_timestep(
        chunk_sigmas * float(config.video_flow["train_timesteps"])
    )
    packed_video_timestep = layout.expand_video(
        clean_video_timestep[..., None], noisy_video_timestep[..., None]
    ).squeeze(-1)

    standard_action = state.action.new_zeros(state.action.shape)
    packed_action = layout.expand_action(standard_action, standard_action)
    base_action_positions = state.action_positions
    packed_action_positions = layout.expand_action(
        base_action_positions, base_action_positions
    )
    packed_action_timestep = sigma.new_zeros(packed_action.shape[:2])

    output = core(
        packed_video,
        packed_action,
        video_timestep=packed_video_timestep,
        action_timestep=packed_action_timestep,
        video_position_ids=packed_positions,
        condition_video_tokens=clean_patched.condition_token_count,
        action_position_ids=packed_action_positions,
        video_camera_ids=packed_camera_ids,
        text_context=state.text_context,
        layout=layout,
    )
    noisy_velocity = layout.noisy_video(output.video_velocity)
    standard_velocity = torch.cat(
        (
            output.video_velocity[:, : clean_patched.condition_token_count],
            noisy_velocity,
        ),
        dim=1,
    )
    return unpatchify_video_sequence(core, standard_velocity, clean_patched)


def _denoise_video_only_chunk(
    model: "VAPTrainingModel",
    state: _SamplingState,
    *,
    active_chunk: int,
) -> _SamplingState:
    config = model.config
    schedule = shifted_sigma_schedule(
        int(config.video_flow["inference_steps"]),
        shift=float(config.video_flow["shift"]),
        device=state.video.device,
    )
    video = state.video
    for current_sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
        sigma = current_sigma.expand(video.shape[0])
        step_state = replace(state, video=video)
        velocity = _video_only_velocity(
            model, step_state, active_chunk=active_chunk, sigma=sigma
        )
        video = _replace_video_chunk(
            video,
            velocity,
            chunk_index=active_chunk,
            frames_per_chunk=state.latent_frames_per_chunk,
            delta_sigma=(next_sigma - current_sigma).expand(video.shape[0]),
        )
    return replace(state, video=video)


def _denoise_action_stage_with_prefix_kv_cache(
    model: "VAPTrainingModel",
    state: _SamplingState,
    stage: ARStage,
) -> _SamplingState:
    """Denoise A0/A1 while exactly reusing immutable clean-prefix BF16 K/V.

    Legacy VAP's first sigma evaluation uses a prefix-only causal forward:
    ``I,A0`` for A0 or ``I,A0,V0,A1`` for A1. It captures every layer's
    earlier-stage K/V from that same pass. Joint-v2 retains the full reference
    shape on this first pass to reduce BF16 shape-dependent rounding; it still
    caches only clean earlier tokens; decode retains reference-shaped scratch
    rows to avoid compact-kernel BF16 drift. Later evaluations update only the
    active Action chunk. No approximation, cache quantization, causally
    invisible future token, or noisy-active K/V reuse is involved.
    """

    if stage.is_video:
        raise ValueError("Action prefix KV cache cannot denoise a Video stage")
    supports_memory_cache = getattr(model.core, "supports_memory_prefix_kv_cache", False)
    if getattr(model.config, "history_offsets", ()) and not supports_memory_cache:
        # Explicit safe fallback until per-token validity is carried by the
        # active-only cache format. Never silently drop invalid-key masking.
        return _denoise_stage(model, state, stage)
    config = model.config
    core = model.core
    batch_size, num_views = state.video.shape[:2]
    num_chunks = int(config.num_interleave_chunks)
    schedule = shifted_sigma_schedule(
        int(config.action_flow["inference_steps"]),
        shift=float(config.action_flow["shift"]),
        device=state.video.device,
    )
    video, action = state.video, state.action
    fp32_action = bool(getattr(core, "_v2_action_fp32_accum", False))
    if fp32_action:
        action = action.float()
    start = stage.chunk_index * state.action_steps_per_chunk
    end = start + state.action_steps_per_chunk
    prefix_cache = None

    for step_index, (current_sigma, next_sigma) in enumerate(
        zip(schedule[:-1], schedule[1:])
    ):
        sigma = current_sigma.expand(batch_size)
        if step_index == 0:
            video_chunk_sigma, action_chunk_sigma = _stage_chunk_sigmas(
                stage, sigma, num_chunks=num_chunks
            )
            patched, _main_patched, first_frame_memory = _patch_sampling_video(
                model, state, video
            )
            video_timestep = patched.per_token_timestep(
                video_chunk_sigma * float(config.video_flow["train_timesteps"])
            )
            action_timestep = action_chunk_sigma.repeat_interleave(
                state.action_steps_per_chunk, dim=1
            ) * float(config.action_flow["train_timesteps"])
            memory_kwargs = (
                {"memory_prefix_valid": patched.memory_valid}
                if supports_memory_cache else {}
            )
            active_velocity, prefix_cache = core.forward_action_stage_with_prefix_cache(
                patched.tokens,
                action.to(model.torch_dtype) if fp32_action else action,
                active_action_chunk=stage.chunk_index,
                video_timestep=video_timestep,
                action_timestep=action_timestep,
                video_position_ids=patched.position_ids,
                condition_video_tokens=patched.condition_token_count,
                action_position_ids=state.action_positions,
                video_camera_ids=patched.camera_ids,
                text_context=state.text_context,
                state_condition=state.state_condition,
                first_frame_memory=first_frame_memory,
                **memory_kwargs,
            )
        else:
            if prefix_cache is None:
                raise AssertionError("first Action denoise step produced no KV cache")
            active_timestep = sigma[:, None].expand(
                -1, state.action_steps_per_chunk
            ) * float(config.action_flow["train_timesteps"])
            active_kwargs = {}
            if state.state_condition is not None:
                active_kwargs["state_condition"] = state.state_condition
            active_velocity = core.forward_active_action_with_prefix_cache(
                (action[:, start:end].to(model.torch_dtype)
                 if fp32_action else action[:, start:end]),
                action_timestep=active_timestep,
                prefix_cache=prefix_cache,
                **active_kwargs,
            )
        if fp32_action:
            active_velocity = active_velocity.float()
        delta = (next_sigma - current_sigma).reshape(-1, 1, 1).to(action)
        updated = action.clone()
        updated[:, start:end] = action[:, start:end] + delta * active_velocity
        action = updated
        if step_index == 0:
            # The per-layer cache owns detached prefix K/V. Do not keep the
            # full patchification/timestep temporaries alive for the remaining
            # active-only denoise steps.
            del patched, video_timestep, action_timestep

    return replace(state, video=video, action=action)


@torch.no_grad()
def sample_latents_sequential(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    state_condition: torch.Tensor | None = None,
) -> VAPSequentialSample:
    """Denoise exactly ``A0 -> V0 -> A1 -> V1`` using clean frozen prefix."""

    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
    )
    was_training = model.training
    model.eval()
    try:
        for stage in (ARStage.A0, ARStage.V0, ARStage.A1, ARStage.V1):
            state = _denoise_stage(model, state, stage)
    finally:
        model.train(was_training)
    return VAPSequentialSample(video_latents=state.video, actions=state.action)


@torch.no_grad()
def sample_video_only_window(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
) -> VAPVideoOnlySample:
    """Generate the complete native ``I -> V0 -> ... -> V{K-1}`` window."""

    _require_video_only_model(model)
    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=None,
    )
    was_training = model.training
    model.eval()
    try:
        for active_chunk in range(int(model.config.num_interleave_chunks)):
            state = _denoise_video_only_chunk(model, state, active_chunk=active_chunk)
    finally:
        model.train(was_training)
    return VAPVideoOnlySample(
        video_latents=state.video,
        stage_order=tuple(
            f"V{index}" for index in range(int(model.config.num_interleave_chunks))
        ),
    )


@torch.no_grad()
def sample_first_video_only_chunk(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
) -> VAPVideoOnlyRolloutChunkSample:
    """Generate only local V0 for a one-chunk Stage-1 rollout."""

    _require_video_only_model(model)
    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=None,
    )
    was_training = model.training
    model.eval()
    try:
        state = _denoise_video_only_chunk(model, state, active_chunk=0)
    finally:
        model.train(was_training)
    start = 1
    end = start + state.latent_frames_per_chunk
    return VAPVideoOnlyRolloutChunkSample(
        video_latents=state.video,
        generated_chunk_latents=state.video[:, :, :, start:end],
        active_slot=0,
    )


@torch.no_grad()
def sample_next_video_only_chunk_given_prefix(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    clean_prefix_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    active_chunk: int = 1,
) -> VAPVideoOnlyRolloutChunkSample:
    """Generate ``V[active_chunk]`` from every clean earlier native chunk."""

    _require_video_only_model(model)
    num_chunks = int(model.config.num_interleave_chunks)
    if not 1 <= active_chunk < num_chunks:
        raise ValueError(
            f"active_chunk must be in [1, {num_chunks}), got {active_chunk}"
        )
    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=None,
    )
    clean_prefix_video_latents = clean_prefix_video_latents.to(
        device=state.video.device, dtype=state.video.dtype
    )
    expected = (
        state.video.shape[0],
        state.video.shape[1],
        state.video.shape[2],
        active_chunk * state.latent_frames_per_chunk,
        state.video.shape[4],
        state.video.shape[5],
    )
    if tuple(clean_prefix_video_latents.shape) != expected:
        raise ValueError(
            "clean_prefix_video_latents shape "
            f"{tuple(clean_prefix_video_latents.shape)} != {expected}"
        )
    video = state.video.clone()
    prefix_end = 1 + active_chunk * state.latent_frames_per_chunk
    video[:, :, :, 1:prefix_end] = clean_prefix_video_latents
    state = replace(state, video=video)
    was_training = model.training
    model.eval()
    try:
        state = _denoise_video_only_chunk(model, state, active_chunk=active_chunk)
    finally:
        model.train(was_training)
    start = 1 + active_chunk * state.latent_frames_per_chunk
    end = start + state.latent_frames_per_chunk
    return VAPVideoOnlyRolloutChunkSample(
        video_latents=state.video,
        generated_chunk_latents=state.video[:, :, :, start:end],
        active_slot=active_chunk,
    )


@torch.no_grad()
def sample_video_given_actions(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    clean_actions: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    state_condition: torch.Tensor | None = None,
) -> VAPVideoFromActionSample:
    """Generate V0/V1 while causally revealing the matching GT action chunk.

    V0 is generated with A0 clean and A1 still at pure noise.  Only after V0
    is complete is A1 copied into the prefix for V1 generation.
    """

    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
    )
    clean_actions = clean_actions.to(
        device=state.action.device, dtype=state.action.dtype
    )
    if tuple(clean_actions.shape) != tuple(state.action.shape):
        raise ValueError(
            f"clean_actions shape {tuple(clean_actions.shape)} != {tuple(state.action.shape)}"
        )

    was_training = model.training
    model.eval()
    try:
        state = replace(
            state,
            action=_copy_action_chunk(
                state.action,
                clean_actions,
                chunk_index=0,
                steps_per_chunk=state.action_steps_per_chunk,
            ),
        )
        state = _denoise_stage(model, state, ARStage.V0)
        state = replace(
            state,
            action=_copy_action_chunk(
                state.action,
                clean_actions,
                chunk_index=1,
                steps_per_chunk=state.action_steps_per_chunk,
            ),
        )
        state = _denoise_stage(model, state, ARStage.V1)
    finally:
        model.train(was_training)
    return VAPVideoFromActionSample(
        video_latents=state.video,
        condition_actions=clean_actions,
    )


@torch.no_grad()
def sample_first_video_chunk_given_action(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    clean_action: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    state_condition: torch.Tensor | None = None,
) -> VAPVideoRolloutChunkSample:
    """Generate the first V0 chunk while keeping the future A1/V1 slots noisy."""

    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
    )
    clean_action = clean_action.to(device=state.action.device, dtype=state.action.dtype)
    expected = (
        state.action.shape[0],
        state.action_steps_per_chunk,
        state.action.shape[-1],
    )
    if tuple(clean_action.shape) != expected:
        raise ValueError(
            f"clean_action shape {tuple(clean_action.shape)} != {expected}"
        )

    was_training = model.training
    model.eval()
    try:
        state = replace(
            state,
            action=_copy_action_chunk(
                state.action,
                clean_action,
                chunk_index=0,
                steps_per_chunk=state.action_steps_per_chunk,
            ),
        )
        state = _denoise_stage(model, state, ARStage.V0)
    finally:
        model.train(was_training)

    start = 1
    end = start + state.latent_frames_per_chunk
    return VAPVideoRolloutChunkSample(
        video_latents=state.video,
        generated_chunk_latents=state.video[:, :, :, start:end],
        active_slot=0,
    )


@torch.no_grad()
def sample_next_video_chunk_given_prefix(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    clean_prefix_video_latents: torch.Tensor,
    clean_actions: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    state_condition: torch.Tensor | None = None,
) -> VAPVideoRolloutChunkSample:
    """Generate slot V1 from a clean overlapping V0 prefix.

    The caller is responsible for shifting trajectory chunks into the local
    two-slot window.  ``condition_video_latents`` is the frame immediately
    before the clean prefix, ``clean_prefix_video_latents`` is that prefix in
    slot V0, and ``clean_actions`` contains the matching local A0/A1 pair.
    """

    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
    )
    clean_prefix_video_latents = clean_prefix_video_latents.to(
        device=state.video.device, dtype=state.video.dtype
    )
    expected_prefix = (
        state.video.shape[0],
        state.video.shape[1],
        state.video.shape[2],
        state.latent_frames_per_chunk,
        state.video.shape[4],
        state.video.shape[5],
    )
    if tuple(clean_prefix_video_latents.shape) != expected_prefix:
        raise ValueError(
            "clean_prefix_video_latents shape "
            f"{tuple(clean_prefix_video_latents.shape)} != {expected_prefix}"
        )
    clean_actions = clean_actions.to(
        device=state.action.device, dtype=state.action.dtype
    )
    if tuple(clean_actions.shape) != tuple(state.action.shape):
        raise ValueError(
            f"clean_actions shape {tuple(clean_actions.shape)} != "
            f"{tuple(state.action.shape)}"
        )

    was_training = model.training
    model.eval()
    try:
        video = state.video.clone()
        prefix_end = 1 + state.latent_frames_per_chunk
        video[:, :, :, 1:prefix_end] = clean_prefix_video_latents
        action = _copy_action_chunk(
            state.action,
            clean_actions,
            chunk_index=0,
            steps_per_chunk=state.action_steps_per_chunk,
        )
        action = _copy_action_chunk(
            action,
            clean_actions,
            chunk_index=1,
            steps_per_chunk=state.action_steps_per_chunk,
        )
        state = replace(state, video=video, action=action)
        state = _denoise_stage(model, state, ARStage.V1)
    finally:
        model.train(was_training)

    start = 1 + state.latent_frames_per_chunk
    end = start + state.latent_frames_per_chunk
    return VAPVideoRolloutChunkSample(
        video_latents=state.video,
        generated_chunk_latents=state.video[:, :, :, start:end],
        active_slot=1,
    )


@torch.no_grad()
def sample_actions_given_video(
    model: "VAPTrainingModel",
    *,
    clean_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    use_kv_cache: bool = False,
    state_condition: torch.Tensor | None = None,
) -> VAPActionFromVideoSample:
    """Predict A0/A1 while causally revealing only the preceding GT video.

    A0 sees only I.  V0 remains pure noise until A0 has completed; then clean
    V0 is inserted and A1 is generated.  Clean V1 is never visible to either
    action chunk because it is after A1 in IVAVA order.
    """

    if clean_video_latents.ndim != 6 or clean_video_latents.shape[3] < 2:
        raise ValueError("clean_video_latents must be [B,V,C,1+future,H,W]")
    state = _prepare_sampling_state(
        model,
        condition_video_latents=clean_video_latents[:, :, :, :1],
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
    )
    clean_video_latents = clean_video_latents.to(
        device=state.video.device, dtype=state.video.dtype
    )
    if tuple(clean_video_latents.shape) != tuple(state.video.shape):
        raise ValueError(
            "clean_video_latents shape "
            f"{tuple(clean_video_latents.shape)} != {tuple(state.video.shape)}"
        )

    was_training = model.training
    model.eval()
    try:
        denoise_action = (
            _denoise_action_stage_with_prefix_kv_cache
            if use_kv_cache
            else _denoise_stage
        )
        state = denoise_action(model, state, ARStage.A0)
        state = replace(
            state,
            video=_copy_video_chunk(
                state.video,
                clean_video_latents,
                chunk_index=0,
                frames_per_chunk=state.latent_frames_per_chunk,
            ),
        )
        state = denoise_action(model, state, ARStage.A1)
    finally:
        model.train(was_training)
    return VAPActionFromVideoSample(
        actions=state.action,
        condition_video_latents=clean_video_latents,
    )


@torch.no_grad()
def sample_first_action_chunk_given_video(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    use_kv_cache: bool = False,
    state_condition: torch.Tensor | None = None,
    first_frame_memory: EncodedFirstFrameMemory | None = None,
) -> VAPActionRolloutChunkSample:
    """Generate trajectory A0 from I before any future video is revealed."""

    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
        first_frame_memory=first_frame_memory,
    )
    was_training = model.training
    model.eval()
    try:
        state = (
            _denoise_action_stage_with_prefix_kv_cache(model, state, ARStage.A0)
            if use_kv_cache
            else _denoise_stage(model, state, ARStage.A0)
        )
    finally:
        model.train(was_training)

    end = state.action_steps_per_chunk
    return VAPActionRolloutChunkSample(
        actions=state.action,
        generated_chunk=state.action[:, :end],
        active_slot=0,
    )


@torch.no_grad()
def sample_next_action_chunk_given_prefix(
    model: "VAPTrainingModel",
    *,
    condition_video_latents: torch.Tensor,
    clean_prefix_video_latents: torch.Tensor,
    clean_prefix_action: torch.Tensor,
    view_ids: torch.Tensor,
    text_context: torch.Tensor,
    generator: torch.Generator | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    use_kv_cache: bool = False,
    state_condition: torch.Tensor | None = None,
    first_frame_memory: EncodedFirstFrameMemory | None = None,
) -> VAPActionRolloutChunkSample:
    """Prefill local A0/V0 and generate only the next local A1 chunk.

    For the first window this implements ``I + A0 + V0 -> A1``.  For every
    later window the caller shifts the preceding prediction/video into local
    A0/V0, so the same operation produces global A2, A3, ... without exposing
    the video chunk that follows the active action.
    """

    state = _prepare_sampling_state(
        model,
        condition_video_latents=condition_video_latents,
        view_ids=view_ids,
        text_context=text_context,
        generator=generator,
        video_noise=video_noise,
        action_noise=action_noise,
        state_condition=state_condition,
        first_frame_memory=first_frame_memory,
    )
    clean_prefix_video_latents = clean_prefix_video_latents.to(
        device=state.video.device, dtype=state.video.dtype
    )
    expected_video = (
        state.video.shape[0],
        state.video.shape[1],
        state.video.shape[2],
        state.latent_frames_per_chunk,
        state.video.shape[4],
        state.video.shape[5],
    )
    if tuple(clean_prefix_video_latents.shape) != expected_video:
        raise ValueError(
            "clean_prefix_video_latents shape "
            f"{tuple(clean_prefix_video_latents.shape)} != {expected_video}"
        )
    clean_prefix_action = clean_prefix_action.to(
        device=state.action.device, dtype=model.torch_dtype
    ).to(dtype=state.action.dtype)
    expected_action = (
        state.action.shape[0],
        state.action_steps_per_chunk,
        state.action.shape[2],
    )
    if tuple(clean_prefix_action.shape) != expected_action:
        raise ValueError(
            "clean_prefix_action shape "
            f"{tuple(clean_prefix_action.shape)} != {expected_action}"
        )

    was_training = model.training
    model.eval()
    try:
        video = state.video.clone()
        video_end = 1 + state.latent_frames_per_chunk
        video[:, :, :, 1:video_end] = clean_prefix_video_latents
        action = state.action.clone()
        action[:, : state.action_steps_per_chunk] = clean_prefix_action
        state = replace(state, video=video, action=action)
        state = (
            _denoise_action_stage_with_prefix_kv_cache(model, state, ARStage.A1)
            if use_kv_cache
            else _denoise_stage(model, state, ARStage.A1)
        )
    finally:
        model.train(was_training)

    start = state.action_steps_per_chunk
    end = 2 * state.action_steps_per_chunk
    return VAPActionRolloutChunkSample(
        actions=state.action,
        generated_chunk=state.action[:, start:end],
        active_slot=1,
    )


__all__ = [
    "VAPActionFromVideoSample",
    "VAPActionRolloutChunkSample",
    "VAPSequentialSample",
    "VAPVideoFromActionSample",
    "VAPVideoOnlyRolloutChunkSample",
    "VAPVideoOnlySample",
    "VAPVideoRolloutChunkSample",
    "sample_actions_given_video",
    "sample_first_action_chunk_given_video",
    "sample_first_video_chunk_given_action",
    "sample_first_video_only_chunk",
    "sample_latents_sequential",
    "sample_next_video_chunk_given_prefix",
    "sample_next_video_only_chunk_given_prefix",
    "sample_next_action_chunk_given_prefix",
    "sample_video_given_actions",
    "sample_video_only_window",
    "shifted_sigma_schedule",
]
