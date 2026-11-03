import torch
import torch.distributed as dist
import torch.nn.functional as F

from wam.model.modules.codecs import video_latent_codec as video_codec
from wam.model.modules.conditioning.frame_attention import (
    build_frame_memory_attention_mask,
)
from wam.model.modules.conditioning.input_builder import build_wam_inputs
from wam.model.modules.understanding.sequence_conditioning import build_frame_vlm_pack


def _distributed_sum_detached(
    local_values: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Sum non-differentiable loss statistics across data-parallel ranks."""
    global_values = local_values.detach().clone()
    if not dist.is_available() or not dist.is_initialized():
        return global_values, 1
    dist.all_reduce(global_values, op=dist.ReduceOp.SUM)
    return global_values, dist.get_world_size()


def _distributed_masked_mean(
    element: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    valid_float = valid.to(device=element.device, dtype=element.dtype)
    local_numerator = (element * valid_float).sum()
    local_denominator = valid_float.sum()
    global_stats, world_size = _distributed_sum_detached(
        torch.stack((local_numerator, local_denominator))
    )
    global_numerator, global_denominator = global_stats.unbind()
    global_denominator = global_denominator.clamp(min=1.0)
    # DDP averages rank gradients, so each local differentiable contribution
    # needs a world-size factor. Keep the returned scalar globally identical.
    backward_loss = world_size * local_numerator / global_denominator
    global_loss = global_numerator / global_denominator
    return backward_loss + (global_loss - backward_loss.detach())


def _frame_future_delta_loss(
    *,
    pred_future_delta: torch.Tensor,
    target_future_delta: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    # (B, S, P) -> (B, S), where P is the predicted latent patch width.
    element = F.mse_loss(
        pred_future_delta.float(),
        target_future_delta.float(),
        reduction="none",
    ).mean(dim=2)
    return _distributed_masked_mean(element, valid)


def _frame_semantic_future_loss(
    *,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    # (B, S, C) -> (B, S).
    cosine = F.cosine_similarity(
        student_hidden.float(),
        teacher_hidden.float(),
        dim=-1,
        eps=1.0e-6,
    )
    return _distributed_masked_mean(1.0 - cosine, valid)


def _encode_single_memory_frame(
    model,
    inputs: dict,
    *,
    name: str,
    tiled: bool,
    deduplicate_identical_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached_latents = inputs.get(f"{name}_latents")
    if cached_latents is not None:
        if not isinstance(cached_latents, torch.Tensor) or cached_latents.ndim != 5:
            raise ValueError(
                f"{name}_latents must be [B,C,1,H,W], got "
                f"{type(cached_latents)} "
                f"{getattr(cached_latents, 'shape', None)}."
            )
        if int(cached_latents.shape[2]) != 1:
            raise ValueError(
                f"{name}_latents must contain one frame, got "
                f"{tuple(cached_latents.shape)}."
            )
        pad = inputs.get(f"{name}_is_pad")
        if pad is None:
            pad = torch.zeros(
                (int(cached_latents.shape[0]), 1),
                dtype=torch.bool,
                device=model.device,
            )
        return cached_latents, pad[:, :1].to(
            device=model.device, dtype=torch.bool
        )

    video = inputs.get(name)
    if video is None:
        raise ValueError(f"Frame-memory training requires {name}.")
    num_frames = int(video.shape[2] if video.ndim == 5 else video.shape[3])
    if num_frames != 1:
        raise ValueError(
            f"{name} must contain exactly one RGB frame, got {num_frames}."
        )
    batch_size = int(video.shape[0])
    encode_video = video
    video_layout = inputs.get("video_layout")
    video_view_names = inputs.get("video_view_names")
    expand_batch = False
    layouts = video_codec.normalize_video_layouts(video_layout, batch_size)
    view_names = video_codec.normalize_video_metadata(
        video_view_names, batch_size, default=""
    )
    metadata_is_identical = len(set(layouts)) == 1 and len(set(view_names)) == 1
    if (
        deduplicate_identical_batch
        and batch_size > 1
        and metadata_is_identical
        and torch.equal(video, video[:1].expand_as(video))
    ):
        encode_video = video[:1]
        video_layout = layouts[0]
        video_view_names = view_names[0]
        expand_batch = True

    latents = video_codec.encode_video_latents(
        model,
        encode_video,
        tiled=tiled,
        video_layout=video_layout,
        video_view_names=video_view_names,
    )
    if expand_batch:
        # Restore the baseline contiguous [B,C,T,H,W] layout so every
        # downstream kernel sees exactly the same shape and strides.
        latents = latents.expand(batch_size, -1, -1, -1, -1).contiguous()
    if int(latents.shape[2]) != 1:
        raise ValueError(
            f"{name} must encode to one latent frame, got {tuple(latents.shape)}."
        )
    pad = inputs.get(f"{name}_is_pad")
    if pad is None:
        pad = torch.zeros(
            (int(latents.shape[0]), 1), dtype=torch.bool, device=model.device
        )
    return latents, pad[:, :1].to(device=model.device, dtype=torch.bool)


def _latent_and_token_spatial_valid_masks(
    pixel_valid_mask: torch.Tensor | None,
    *,
    latents: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Project a canvas-pixel validity mask onto VAE and DiT grids."""
    if pixel_valid_mask is None:
        return None, None
    if pixel_valid_mask.ndim != 3:
        raise ValueError(
            "video_spatial_valid_mask must be [B,H,W], got "
            f"{tuple(pixel_valid_mask.shape)}"
        )
    batch_size = int(latents.shape[0])
    if int(pixel_valid_mask.shape[0]) != batch_size:
        raise ValueError(
            "video_spatial_valid_mask batch mismatch: "
            f"{tuple(pixel_valid_mask.shape)} vs latents {tuple(latents.shape)}"
        )

    latent_h, latent_w = map(int, latents.shape[-2:])
    latent_fraction = F.adaptive_avg_pool2d(
        pixel_valid_mask.to(device=latents.device, dtype=torch.float32).unsqueeze(1),
        output_size=(latent_h, latent_w),
    ).squeeze(1)
    # A latent cell is valid only when its complete source region is valid.
    latent_valid = latent_fraction >= (1.0 - 1.0e-6)

    patch_h = int(patch_size[1])
    patch_w = int(patch_size[2])
    if latent_h % patch_h != 0 or latent_w % patch_w != 0:
        raise ValueError(
            "Latent spatial mask cannot align to DiT patches: "
            f"latent=({latent_h},{latent_w}) patch=({patch_h},{patch_w})"
        )
    token_fraction = F.avg_pool2d(
        latent_valid.to(dtype=torch.float32).unsqueeze(1),
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    ).squeeze(1)
    token_valid = (token_fraction >= (1.0 - 1.0e-6)).reshape(batch_size, -1)
    return latent_valid, token_valid


def _frame_video_loss(
    model,
    *,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    image_is_pad: torch.Tensor | None,
    latent_spatial_valid_mask: torch.Tensor | None,
    timestep: torch.Tensor,
) -> torch.Tensor:
    element = F.mse_loss(
        pred_video.float(), target_video.float(), reduction="none"
    )
    valid = torch.ones_like(element, dtype=torch.bool)
    if latent_spatial_valid_mask is not None:
        expected = (int(pred_video.shape[0]), *map(int, pred_video.shape[-2:]))
        if tuple(latent_spatial_valid_mask.shape) != expected:
            raise ValueError(
                "Video latent spatial loss-mask shape mismatch: "
                f"{tuple(latent_spatial_valid_mask.shape)} vs {expected}"
            )
        # (B, H, W) -> (B, 1, 1, H, W).
        valid &= latent_spatial_valid_mask[:, None, None]
    if image_is_pad is not None:
        temporal_factor = int(model.vae.temporal_downsample_factor)
        tail_pad = image_is_pad[:, 1:]
        if int(tail_pad.shape[1]) % temporal_factor != 0:
            raise ValueError(
                "Future video padding cannot be aligned to VAE latent frames: "
                f"future_frames={int(tail_pad.shape[1])}, factor={temporal_factor}."
            )
        latent_pad = tail_pad.reshape(
            int(tail_pad.shape[0]), -1, temporal_factor
        ).all(dim=2)
        expected = (int(element.shape[0]), int(element.shape[2]))
        if tuple(latent_pad.shape) != expected:
            raise ValueError(
                "Future video loss-mask shape mismatch: "
                f"{tuple(latent_pad.shape)} vs {expected}."
            )
        # (B, T) -> (B, 1, T, 1, 1).
        valid &= ~latent_pad[:, None, :, None, None]
    weight = model.train_video_scheduler.training_weight(timestep).to(
        device=element.device, dtype=element.dtype
    ).reshape(-1)
    weighted_element = element * weight[:, None, None, None, None]
    return _distributed_masked_mean(weighted_element, valid)


def _frame_action_loss(
    model,
    *,
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    action_dim_is_pad: torch.Tensor | None,
    timestep: torch.Tensor,
    action_prefix_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    action_dim_loss_weight: torch.Tensor | None = None,
    action_loss_weight: torch.Tensor | None = None,
    action_loss_weighted_valid_cells: torch.Tensor | None = None,
    action_loss_weighted_valid_cells_pool: torch.Tensor | None = None,
) -> torch.Tensor:
    raw_element = F.mse_loss(
        pred_action.float(), target_action.float(), reduction="none"
    )
    element = raw_element
    if action_dim_loss_weight is not None:
        # Per-sample canonical dimension weights: (B, D) -> (B, 1, D).
        element = element * action_dim_loss_weight.unsqueeze(1)
    valid = torch.ones_like(element, dtype=torch.bool)
    if action_prefix_mask is not None:
        expected_shape = tuple(element.shape[:2])
        if tuple(action_prefix_mask.shape) != expected_shape:
            raise ValueError(
                "`action_prefix_mask` must be [B,T] matching the action, "
                f"got {tuple(action_prefix_mask.shape)} vs {expected_shape}."
            )
        valid &= ~action_prefix_mask.to(
            device=element.device, dtype=torch.bool
        ).unsqueeze(-1)
    if action_mask is not None:
        valid &= action_mask
    if action_is_pad is not None:
        valid &= ~action_is_pad.unsqueeze(-1)
    if action_dim_is_pad is not None:
        valid &= ~action_dim_is_pad.unsqueeze(1)
    valid_float = valid.to(device=element.device, dtype=element.dtype)
    per_sample = (element * valid_float).sum(dim=(1, 2)) / valid_float.sum(
        dim=(1, 2)
    ).clamp(min=1.0)
    weight = model.train_action_scheduler.training_weight(timestep).to(
        device=per_sample.device, dtype=per_sample.dtype
    ).reshape(-1)
    if action_loss_weighted_valid_cells is None:
        policy = torch.zeros_like(per_sample, dtype=torch.bool)
    else:
        policy = action_loss_weighted_valid_cells.to(
            device=per_sample.device, dtype=torch.bool
        ).reshape(-1)
        if int(policy.numel()) != int(per_sample.shape[0]):
            raise ValueError(
                "action_loss_weighted_valid_cells batch mismatch: "
                f"{tuple(policy.shape)} vs {tuple(per_sample.shape)}"
            )
    if action_loss_weighted_valid_cells_pool is None:
        # Backward-compatible single weighted pool for callers that only pass
        # the historical boolean policy.
        weighted_pool = policy[:, None]
    else:
        weighted_pool = action_loss_weighted_valid_cells_pool.to(
            device=per_sample.device, dtype=torch.bool
        )
        if weighted_pool.ndim == 1:
            weighted_pool = weighted_pool.unsqueeze(-1)
        if weighted_pool.ndim != 2 or int(weighted_pool.shape[0]) != int(
            per_sample.shape[0]
        ):
            raise ValueError(
                "action_loss_weighted_valid_cells_pool must have shape [B,K], "
                f"got {tuple(weighted_pool.shape)} for batch "
                f"{tuple(per_sample.shape)}"
            )
        if int(weighted_pool.shape[1]) <= 0:
            raise ValueError(
                "action_loss_weighted_valid_cells_pool must contain at least "
                "one family pool"
            )
        memberships = weighted_pool.sum(dim=1)
        if bool((memberships > 1).any()):
            raise ValueError(
                "Each action sample may belong to at most one weighted-valid-"
                "cell family pool"
            )
        pool_policy = memberships == 1
        if action_loss_weighted_valid_cells is not None and not bool(
            torch.equal(pool_policy, policy)
        ):
            raise ValueError(
                "action_loss_weighted_valid_cells disagrees with its family "
                "pool membership"
            )
        policy = pool_policy
    legacy = ~policy

    cell_weight = valid_float
    if action_dim_loss_weight is not None:
        cell_weight = cell_weight * action_dim_loss_weight.unsqueeze(1)
    if action_loss_weight is None:
        sample_loss_weight = torch.ones_like(per_sample)
    else:
        sample_loss_weight = action_loss_weight.to(
            device=per_sample.device, dtype=per_sample.dtype
        ).reshape(-1)
        if int(sample_loss_weight.numel()) != int(per_sample.shape[0]):
            raise ValueError(
                "action_loss_weight batch mismatch: "
                f"{tuple(sample_loss_weight.shape)} vs {tuple(per_sample.shape)}"
            )
        if not bool(torch.isfinite(sample_loss_weight).all()) or bool(
            (sample_loss_weight <= 0).any()
        ):
            raise ValueError("action_loss_weight must be finite and positive")
    # Non-unit legacy source weights apply uniformly, including the gripper.
    # Other legacy rows retain their canonical dimension weights.
    nonunit_legacy = legacy & ((sample_loss_weight - 1.0).abs() > 1.0e-7)
    uniform_per_sample = (raw_element * valid_float).sum(dim=(1, 2)) / valid_float.sum(
        dim=(1, 2)
    ).clamp(min=1.0)
    per_sample = torch.where(
        nonunit_legacy, uniform_per_sample * sample_loss_weight, per_sample
    )
    legacy_sum = (per_sample[legacy] * weight[legacy]).sum()
    legacy_count = legacy.sum().to(per_sample.dtype)
    weighted_numerators: list[torch.Tensor] = []
    weighted_denominators: list[torch.Tensor] = []
    weighted_valid_samples: list[torch.Tensor] = []
    for pool_index in range(int(weighted_pool.shape[1])):
        in_pool = weighted_pool[:, pool_index]
        absolute_weight = cell_weight[in_pool]
        relative_weight = absolute_weight / sample_loss_weight[
            in_pool, None, None
        ]
        weighted_denominators.append(relative_weight.sum())
        weighted_valid_samples.append(
            (relative_weight.sum(dim=(1, 2)) > 0)
            .sum()
            .to(per_sample.dtype)
        )
        weighted_numerators.append(
            (
                raw_element[in_pool]
                * absolute_weight
                * weight[in_pool, None, None]
            ).sum()
        )

    global_stats, world_size = _distributed_sum_detached(
        torch.stack(
            (
                legacy_sum,
                legacy_count,
                *weighted_numerators,
                *weighted_denominators,
                *weighted_valid_samples,
            )
        )
    )
    pool_count = int(weighted_pool.shape[1])
    global_legacy_sum = global_stats[0]
    global_legacy_count = global_stats[1]
    global_weighted_numerators = global_stats[2 : 2 + pool_count]
    global_weighted_denominators = global_stats[
        2 + pool_count : 2 + 2 * pool_count
    ].clamp(min=1.0e-12)
    global_weighted_valid_samples = global_stats[
        2 + 2 * pool_count : 2 + 3 * pool_count
    ]
    global_total_count = (
        global_legacy_count + global_weighted_valid_samples.sum()
    ).clamp(min=1.0)
    local_weighted_objective = legacy_sum.new_zeros(())
    global_weighted_objective = global_legacy_sum.new_zeros(())
    for pool_index in range(pool_count):
        scale = (
            global_weighted_valid_samples[pool_index]
            / global_weighted_denominators[pool_index]
        )
        local_weighted_objective = (
            local_weighted_objective
            + weighted_numerators[pool_index] * scale
        )
        global_weighted_objective = (
            global_weighted_objective
            + global_weighted_numerators[pool_index] * scale
        )
    local_objective_sum = legacy_sum + local_weighted_objective
    # Preserve legacy-per-sample versus family-separated weighted-cell mixing
    # while making the gradient equal to one globally materialized batch.
    backward_loss = world_size * local_objective_sum / global_total_count
    global_loss = (
        global_legacy_sum
        + global_weighted_objective
    ) / global_total_count
    return backward_loss + (global_loss - backward_loss.detach())


frame_action_loss = _frame_action_loss


def _sample_training_time_rtc_delays(
    *,
    batch_size: int,
    action_horizon: int,
    max_delay_steps: int,
    action_is_pad: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Sample an inclusive uniform RTC delay, capped by valid action length."""
    batch_size = int(batch_size)
    action_horizon = int(action_horizon)
    max_delay_steps = int(max_delay_steps)
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
    if action_horizon <= 0:
        raise ValueError(
            f"`action_horizon` must be positive, got {action_horizon}."
        )
    if max_delay_steps < 0 or max_delay_steps >= action_horizon:
        raise ValueError(
            "`max_delay_steps` must satisfy 0 <= max_delay_steps < "
            f"action_horizon, got {max_delay_steps} and {action_horizon}."
        )

    per_sample_max = torch.full(
        (batch_size,),
        max_delay_steps,
        device=device,
        dtype=torch.long,
    )
    if action_is_pad is not None:
        if tuple(action_is_pad.shape) != (batch_size, action_horizon):
            raise ValueError(
                "`action_is_pad` must be [B,T] matching the action, "
                f"got {tuple(action_is_pad.shape)} vs "
                f"{(batch_size, action_horizon)}."
            )
        valid_action_steps = (~action_is_pad.to(
            device=device, dtype=torch.bool
        )).sum(dim=1)
        # Dataset chunks contain at least one valid action. Capping at
        # valid_steps - 1 leaves one or more valid postfix targets for every
        # such sample, including chunks padded at the end of an episode.
        per_sample_max = torch.minimum(
            per_sample_max, (valid_action_steps - 1).clamp(min=0)
        )

    uniform = torch.rand((batch_size,), device=device, dtype=torch.float32)
    return torch.floor(
        uniform * (per_sample_max + 1).to(dtype=torch.float32)
    ).to(dtype=torch.long)


def _frame_pica_align_loss(model, sample, *, student_hidden, input_latents, noise_video,
                        timestep_video, noisy_action, timestep_action, video_context,
                        video_context_mask, action_context, action_context_mask,
                        attention_mask, video_key_mask, action_key_mask, action_valid,
                        video_query_mask, tiled):
    pica = model.pica_alignment
    clean_sample = dict(sample)
    for name in ("video", "memory_video_anchor", "memory_video_recent"):
        key = name + "_canonical"
        if key not in sample:
            raise ValueError(f"PICA alignment requires {key} in the training batch.")
        clean_sample[name] = sample[key]
    # Encoder cache artifacts describe the student view and cannot feed the teacher.
    clean_inputs = build_wam_inputs(model, clean_sample, tiled=tiled)
    clean_latents = clean_inputs["input_latents"]
    anchor, _ = _encode_single_memory_frame(model, clean_inputs, name="memory_video_anchor", tiled=tiled)
    recent, _ = _encode_single_memory_frame(model, clean_inputs, name="memory_video_recent", tiled=tiled)
    gap = torch.rand_like(timestep_video) * pica.sigma_gap_max * model.train_video_scheduler.num_train_timesteps
    teacher_t = (timestep_video - gap).clamp(min=0)
    teacher_window = model.train_video_scheduler.add_noise(clean_latents, noise_video, teacher_t)
    teacher_window[:, :, :1] = clean_latents[:, :, :1]
    teacher_latents = torch.cat([anchor, recent, teacher_window], dim=2)
    teacher_frame_t = torch.cat([teacher_t.new_zeros((teacher_t.shape[0], 2)),
                                teacher_t[:, None].expand(-1, clean_latents.shape[2]-1)], dim=1)
    teacher = pica.teacher(model.mot)
    with torch.no_grad():
        video_pre = teacher.mixtures["video"].pre_dit(
            x=teacher_latents, timestep=teacher_frame_t, context=video_context,
            context_mask=video_context_mask,
            fuse_vae_embedding_in_latents=clean_inputs["fuse_vae_embedding_in_latents"],
        )
        if model.future_delta_enabled:
            teacher.mixtures["video"].append_future_delta_queries(video_pre, current_frame_index=2)
        action_pre = teacher.mixtures["action"].pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=action_context, context_mask=action_context_mask,
        )
        _, captured = teacher._forward_eager(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                         "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            key_masks_all={"video": video_key_mask, "action": action_key_mask},
            query_masks_all={"video": video_query_mask} if video_query_mask is not None else None,
            capture_layers={"action": {pica.teacher_layer}}, max_layer=pica.teacher_layer,
        )
        teacher_hidden = captured["action"][pica.teacher_layer]
    projected = pica.head(student_hidden)
    if projected.shape != teacher_hidden.shape:
        raise ValueError("PICA alignment student and teacher action tokens must have matching shapes.")
    valid = action_valid.any(dim=-1)
    cosine = F.cosine_similarity(projected.float(), teacher_hidden.float(), dim=-1)
    loss = -_distributed_masked_mean(cosine, valid)
    identity = F.cosine_similarity(student_hidden.detach().float(), teacher_hidden.float(), dim=-1)
    metrics = {
        "align_cos": -loss.detach(),
        "align_identity_cos": _distributed_masked_mean(identity, valid).detach(),
        "align_sigma_gap": ((timestep_video-teacher_t) / model.train_video_scheduler.num_train_timesteps).mean().detach(),
        "align_teacher_view_gap": (clean_latents-input_latents).float().square().mean().sqrt().detach(),
        "align_ema_steps": loss.new_tensor(pica.ema_steps),
    }
    return loss, metrics


def frame_training_loss(model, sample, tiled: bool = False, global_step: int | None = None):
    """FastWAM-style one-window joint video/action objective.

    Each sample contains a clean anchor frame, one previous-decision recent
    frame, and one current/future window. Only the future latent frames and
    action horizon are denoised.
    """
    inputs = build_wam_inputs(model, sample, tiled=tiled)
    input_latents = inputs["input_latents"]
    if input_latents is None:
        raise ValueError("Frame training requires encoded window latents.")
    anchor_latents, anchor_pad = _encode_single_memory_frame(
        model,
        inputs,
        name="memory_video_anchor",
        tiled=tiled,
        deduplicate_identical_batch=bool(
            getattr(model, "deduplicate_identical_anchor_batch", True)
        ),
    )
    recent_latents, recent_pad = _encode_single_memory_frame(
        model, inputs, name="memory_video_recent", tiled=tiled
    )
    latent_spatial_valid_mask, spatial_token_valid_mask = (
        _latent_and_token_spatial_valid_masks(
            inputs.get("video_spatial_valid_mask"),
            latents=input_latents,
            patch_size=tuple(int(v) for v in model.video_expert.patch_size),
        )
    )
    if latent_spatial_valid_mask is not None:
        latent_valid = latent_spatial_valid_mask[:, None, None]
        input_latents = input_latents.masked_fill(~latent_valid, 0.0)
        anchor_latents = anchor_latents.masked_fill(~latent_valid, 0.0)
        recent_latents = recent_latents.masked_fill(~latent_valid, 0.0)
    vlm_pack = build_frame_vlm_pack(model, inputs=inputs)

    batch_size = int(input_latents.shape[0])
    num_condition_frames = 3
    action = inputs["action"]
    action_mask = inputs.get("action_mask")
    action_dim_loss_weight = inputs.get("action_dim_loss_weight")
    action_loss_weight = inputs.get("action_loss_weight")
    action_loss_weighted_valid_cells = inputs.get(
        "action_loss_weighted_valid_cells"
    )
    action_loss_weighted_valid_cells_pool = inputs.get(
        "action_loss_weighted_valid_cells_pool"
    )
    action_is_pad = inputs.get("action_is_pad")
    action_dim_is_pad = inputs.get("action_dim_is_pad")
    video_context = inputs["context"]
    video_context_mask = inputs["context_mask"]
    action_context = video_context
    action_context_mask = video_context_mask
    if model.understanding_enabled:
        action_context = vlm_pack["vlm_context"].to(
            device=model.device, dtype=model.torch_dtype
        )
        action_context_mask = vlm_pack["vlm_mask"].to(
            device=model.device, dtype=torch.bool
        )

    noise_video = torch.randn_like(input_latents)
    if latent_spatial_valid_mask is not None:
        noise_video = noise_video.masked_fill(
            ~latent_spatial_valid_mask[:, None, None], 0.0
        )
    timestep_video = model.train_video_scheduler.sample_training_t(
        batch_size=batch_size,
        device=model.device,
        dtype=input_latents.dtype,
    )
    semantic_active = torch.zeros(
        (batch_size,), dtype=torch.bool, device=model.device
    )
    if model.semantic_future_alignment_enabled:
        semantic_active = timestep_video.float() <= (
            model.semantic_future_max_sigma
            * float(model.train_video_scheduler.num_train_timesteps)
        )
    noisy_window = model.train_video_scheduler.add_noise(
        input_latents, noise_video, timestep_video
    )
    target_window = model.train_video_scheduler.training_target(
        input_latents, noise_video, timestep_video
    )
    if latent_spatial_valid_mask is not None:
        latent_valid = latent_spatial_valid_mask[:, None, None]
        noisy_window = noisy_window.masked_fill(~latent_valid, 0.0)
        target_window = target_window.masked_fill(~latent_valid, 0.0)
    noisy_window[:, :, :1] = input_latents[:, :, :1]
    video_latents = torch.cat(
        [anchor_latents, recent_latents, noisy_window], dim=2
    )
    future_latent_frames = int(input_latents.shape[2]) - 1
    if model.future_delta_enabled:
        patch_size = tuple(int(v) for v in model.video_expert.patch_size)
        patch_h = int(patch_size[1])
        patch_w = int(patch_size[2])
        future_delta = input_latents[:, :, 1:] - input_latents[:, :, :-1]
        delta_batch, delta_channels, delta_frames, delta_h, delta_w = (
            future_delta.shape
        )
        future_delta_target = F.unfold(
            future_delta.permute(0, 2, 1, 3, 4).reshape(
                delta_batch * delta_frames,
                delta_channels,
                delta_h,
                delta_w,
            ),
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        ).transpose(1, 2).reshape(
            delta_batch,
            -1,
            delta_channels * patch_h * patch_w,
        ).contiguous()
    video_frame_timesteps = torch.cat(
        [
            torch.zeros(
                (batch_size, num_condition_frames - 1),
                device=model.device,
                dtype=timestep_video.dtype,
            ),
            timestep_video[:, None].expand(batch_size, future_latent_frames),
        ],
        dim=1,
    )

    action_valid = torch.ones_like(action, dtype=torch.bool)
    if action_mask is not None:
        action_valid &= action_mask
    if action_is_pad is not None:
        action_valid &= ~action_is_pad.unsqueeze(-1)
    if action_dim_is_pad is not None:
        action_valid &= ~action_dim_is_pad.unsqueeze(1)
    action = action.masked_fill(~action_valid, 0.0)
    noise_action = torch.randn_like(action).masked_fill(~action_valid, 0.0)
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=model.device,
        dtype=action.dtype,
    )
    rtc_prefix_mask = None
    rtc_delay = None
    action_model_timestep = timestep_action
    if getattr(model.action_expert, "training_time_rtc_enabled", False):
        action_horizon = int(action.shape[1])
        rtc_delay = _sample_training_time_rtc_delays(
            batch_size=batch_size,
            action_horizon=action_horizon,
            max_delay_steps=(
                model.action_expert.training_time_rtc_max_delay_steps
            ),
            action_is_pad=action_is_pad,
            device=action.device,
        )
        rtc_prefix_mask = (
            torch.arange(action_horizon, device=action.device).unsqueeze(0)
            < rtc_delay.unsqueeze(1)
        )
        action_model_timestep = timestep_action.unsqueeze(1).expand(
            batch_size, action_horizon
        ).clone()
        action_model_timestep.masked_fill_(rtc_prefix_mask, 0)
        noisy_action = model.train_action_scheduler.add_noise(
            action,
            noise_action,
            action_model_timestep,
            time_dim=1,
        )
    else:
        noisy_action = model.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
    noisy_action = noisy_action.masked_fill(~action_valid, 0.0)
    target_action = model.train_action_scheduler.training_target(
        action, noise_action, timestep_action
    ).masked_fill(~action_valid, 0.0)

    video_pre = model.video_expert.pre_dit(
        x=video_latents,
        timestep=video_frame_timesteps,
        context=video_context,
        context_mask=video_context_mask,
        action=action if model.video_expert.action_conditioned else None,
        fuse_vae_embedding_in_latents=inputs[
            "fuse_vae_embedding_in_latents"
        ],
    )
    if model.future_delta_enabled:
        model.video_expert.append_future_delta_queries(
            video_pre,
            current_frame_index=num_condition_frames - 1,
        )
    action_pre = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=action_model_timestep,
        context=action_context,
        context_mask=action_context_mask,
    )

    video_tokens = video_pre["tokens"]
    action_tokens = action_pre["tokens"]
    tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    num_future_delta_tokens = int(
        video_pre["meta"].get("num_future_delta_tokens", 0)
    )
    video_spatial_query_mask = None
    if spatial_token_valid_mask is not None:
        if int(spatial_token_valid_mask.shape[1]) != tokens_per_frame:
            raise ValueError(
                "Spatial token mask does not match tokens_per_frame: "
                f"{tuple(spatial_token_valid_mask.shape)} vs {tokens_per_frame}"
            )
        num_real_video_frames = int(video_pre["meta"]["grid_size"][0])
        video_spatial_query_mask = spatial_token_valid_mask.repeat(
            1, num_real_video_frames
        )
        if num_future_delta_tokens:
            if num_future_delta_tokens != tokens_per_frame:
                raise ValueError(
                    "Future-delta spatial mask expects one token map, got "
                    f"{num_future_delta_tokens} vs {tokens_per_frame}"
                )
            video_spatial_query_mask = torch.cat(
                [video_spatial_query_mask, spatial_token_valid_mask], dim=1
            )
    attention_mask = build_frame_memory_attention_mask(
        video_seq_len=int(video_tokens.shape[1]),
        action_seq_len=int(action_tokens.shape[1]),
        video_tokens_per_frame=tokens_per_frame,
        num_condition_frames=num_condition_frames,
        device=video_tokens.device,
        num_future_delta_tokens=num_future_delta_tokens,
        num_anchor_condition_frames=1,
    )

    image_is_pad = inputs.get("image_is_pad")
    if image_is_pad is None:
        current_pad = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=model.device
        )
        future_latent_pad = torch.zeros(
            (batch_size, future_latent_frames),
            dtype=torch.bool,
            device=model.device,
        )
    else:
        current_pad = image_is_pad[:, :1]
        temporal_factor = int(model.vae.temporal_downsample_factor)
        future_latent_pad = image_is_pad[:, 1:].reshape(
            batch_size, -1, temporal_factor
        ).all(dim=2)
    video_frame_valid = ~torch.cat(
        [anchor_pad, recent_pad, current_pad, future_latent_pad], dim=1
    )
    video_key_mask = video_frame_valid.repeat_interleave(
        tokens_per_frame, dim=1
    )
    if num_future_delta_tokens:
        video_key_mask = torch.cat(
            [
                video_key_mask,
                torch.ones(
                    (batch_size, num_future_delta_tokens),
                    dtype=torch.bool,
                    device=video_key_mask.device,
                ),
            ],
            dim=1,
        )
    if video_spatial_query_mask is not None:
        if tuple(video_spatial_query_mask.shape) != tuple(video_key_mask.shape):
            raise ValueError(
                "Video spatial attention-mask shape mismatch: "
                f"{tuple(video_spatial_query_mask.shape)} vs "
                f"{tuple(video_key_mask.shape)}"
            )
        video_key_mask &= video_spatial_query_mask
    if action_mask is not None:
        action_step_key_mask = action_valid.any(dim=-1)
    else:
        action_step_key_mask = (
            torch.ones(
                action.shape[:2], dtype=torch.bool, device=action_tokens.device
            )
            if action_is_pad is None
            else ~action_is_pad
        )
    action_key_mask = action_step_key_mask
    action_context_payload = {
        "context": action_pre["context"],
        "mask": action_pre["context_mask"],
    }

    teacher_feature = sample.get("track_teacher_feature") if model.track_bridge is not None else None
    teacher_valid = sample.get("track_teacher_valid")
    track_mask = None
    if model.track_bridge is not None:
        if model.track_bridge.training and teacher_feature is None:
            raise ValueError("4D distillation training requires a teacher cache.")
        if teacher_feature is not None:
            expected = (batch_size, 1, model.track_bridge.teacher_feature_dim)
            if tuple(teacher_feature.shape) != expected:
                raise ValueError(f"Teacher features must have shape {expected}.")
            if teacher_valid is None or tuple(teacher_valid.shape) not in {(batch_size,), (batch_size, 1)}:
                raise ValueError("Teacher validity must have shape [B] or [B,1].")
            track_mask = teacher_valid.to(device=model.device, dtype=torch.bool).reshape(batch_size)
            if not track_mask.any():
                raise ValueError("4D distillation batch has no valid teacher sample.")

    semantic_capture_layers = {}
    if model.semantic_future_alignment_enabled:
        semantic_capture_layers = {
            "video": {
                model.semantic_future_student_layer,
                model.semantic_future_teacher_layer,
            },
        }

    pica_active = getattr(model, "pica_alignment", None) is not None and model.mot.training
    if pica_active:
        semantic_capture_layers.setdefault("action", set()).add(model.pica_alignment.student_layer)

    capture_batch_masks = {}
    track_sparse_capture = False
    if track_mask is not None:
        layer = model.track_bridge.source_layer
        track_sparse_capture = layer not in semantic_capture_layers.get("video", set())
        semantic_capture_layers.setdefault("video", set()).add(layer)
        if track_sparse_capture:
            capture_batch_masks = {"video": {layer: track_mask}}

    video_context_attention_mask = video_pre["context_mask"]
    if video_spatial_query_mask is not None:
        if video_context_attention_mask.ndim == 3:
            video_context_attention_mask = (
                video_context_attention_mask
                & video_spatial_query_mask[:, :, None]
            )
        elif video_context_attention_mask.ndim == 4:
            video_context_attention_mask = (
                video_context_attention_mask
                & video_spatial_query_mask[:, None, :, None]
            )
        else:
            raise ValueError(
                "Video context mask must be [B,S,L] or [B,1,S,L], got "
                f"{tuple(video_context_attention_mask.shape)}"
            )

    mot_output = model.mot(
        embeds_all={"video": video_tokens, "action": action_tokens},
        attention_mask=attention_mask,
        freqs_all={
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        },
        context_all={
            "video": {
                "context": video_pre["context"],
                "mask": video_context_attention_mask,
            },
            "action": action_context_payload,
        },
        t_mod_all={
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        },
        key_masks_all={
            "video": video_key_mask,
            "action": action_key_mask,
        },
        query_masks_all=(
            {"video": video_spatial_query_mask}
            if video_spatial_query_mask is not None
            else None
        ),
        capture_layers=semantic_capture_layers or None,
        capture_batch_masks=capture_batch_masks or None,
    )
    if semantic_capture_layers:
        tokens_out, semantic_states = mot_output
    else:
        tokens_out = mot_output

    loss_track_raw = tokens_out["action"].new_zeros(())
    track_weight = 0.0
    if track_mask is not None:
        hidden = semantic_states["video"][model.track_bridge.source_layer]
        if not track_sparse_capture:
            hidden = hidden[track_mask]
        valid_memory = video_frame_valid[:, :num_condition_frames].repeat_interleave(tokens_per_frame, dim=1)
        student = model.track_bridge.encode_video_tokens(
            hidden, tokens_per_frame=tokens_per_frame,
            num_condition_frames=num_condition_frames, memory_valid=valid_memory[track_mask],
        )
        teacher = teacher_feature[track_mask.to(teacher_feature.device)]
        loss_track_raw = model.track_bridge.alignment_loss(student, teacher, None)
        track_weight = model.track_bridge.alignment_weight(global_step)

    pred_video_all = model.video_expert.post_dit(
        tokens_out["video"], video_pre
    )
    pred_video = pred_video_all[:, :, num_condition_frames:]
    target_video = target_window[:, :, 1:]
    pred_action = model.action_expert.post_dit(
        tokens_out["action"], action_pre
    )
    loss_video = _frame_video_loss(
        model,
        pred_video=pred_video,
        target_video=target_video,
        image_is_pad=image_is_pad,
        latent_spatial_valid_mask=latent_spatial_valid_mask,
        timestep=timestep_video,
    )
    loss_action = _frame_action_loss(
        model,
        action_prefix_mask=rtc_prefix_mask,
        pred_action=pred_action,
        target_action=target_action,
        action_is_pad=action_is_pad,
        action_dim_is_pad=action_dim_is_pad,
        timestep=timestep_action,
        action_mask=action_mask,
        action_dim_loss_weight=action_dim_loss_weight,
        action_loss_weight=action_loss_weight,
        action_loss_weighted_valid_cells=action_loss_weighted_valid_cells,
        action_loss_weighted_valid_cells_pool=(
            action_loss_weighted_valid_cells_pool
        ),
    )
    loss_future_delta = loss_action.new_zeros(())
    if model.future_delta_enabled:
        pred_future_delta = model.video_expert.predict_future_delta(
            tokens_out["video"],
            video_pre,
        )
        future_delta_valid = (~future_latent_pad)[:, :, None].expand(
            -1, -1, tokens_per_frame
        )
        if spatial_token_valid_mask is not None:
            future_delta_valid = (
                future_delta_valid & spatial_token_valid_mask[:, None, :]
            )
        future_delta_valid = future_delta_valid.reshape(
            batch_size, -1
        )
        loss_future_delta = _frame_future_delta_loss(
            pred_future_delta=pred_future_delta,
            target_future_delta=future_delta_target,
            valid=future_delta_valid,
        )

    loss_semantic_future = loss_action.new_zeros(())
    if model.semantic_future_alignment_enabled:
        student_hidden_all = semantic_states["video"][
            model.semantic_future_student_layer
        ]
        teacher_hidden_all = semantic_states["video"][
            model.semantic_future_teacher_layer
        ]
        future_start = (
            num_condition_frames + future_latent_frames - 1
        ) * tokens_per_frame
        future_end = future_start + tokens_per_frame
        teacher_hidden = teacher_hidden_all[
            :, future_start:future_end
        ].detach()
        delta_start = int(video_pre["meta"]["real_video_seq_len"])
        delta_end = delta_start + num_future_delta_tokens
        student_hidden = student_hidden_all[:, delta_start:delta_end]
        semantic_valid = (
            semantic_active & ~future_latent_pad[:, -1]
        )[:, None]
        if spatial_token_valid_mask is not None:
            semantic_valid = semantic_valid & spatial_token_valid_mask
        else:
            semantic_valid = semantic_valid.expand(
                -1, int(student_hidden.shape[1])
            )
        loss_semantic_future = _frame_semantic_future_loss(
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            valid=semantic_valid,
        )

    loss_pica_align = loss_action.new_zeros(())
    pica_metrics = {}
    if pica_active:
        loss_pica_align, pica_metrics = _frame_pica_align_loss(
            model, sample, student_hidden=semantic_states["action"][model.pica_alignment.student_layer],
            input_latents=input_latents, noise_video=noise_video,
            timestep_video=timestep_video, noisy_action=noisy_action,
            timestep_action=action_model_timestep, video_context=video_context,
            video_context_mask=video_context_mask, action_context=action_context,
            action_context_mask=action_context_mask, attention_mask=attention_mask,
            video_key_mask=video_key_mask, action_key_mask=action_key_mask,
            action_valid=action_valid, video_query_mask=video_spatial_query_mask, tiled=tiled,
        )

    loss_total = (
        model.loss_lambda_video * loss_video
        + model.loss_lambda_action * loss_action
        + model.loss_lambda_future_delta * loss_future_delta
        + model.loss_lambda_semantic_future * loss_semantic_future
        + track_weight * loss_track_raw
        + (model.pica_alignment.loss_weight * loss_pica_align if pica_active else 0)
    )
    loss_dict = {
        "loss_video": model.loss_lambda_video * loss_video.detach(),
        "loss_action": model.loss_lambda_action * loss_action.detach(),
        "loss_future_delta": model.loss_lambda_future_delta
        * loss_future_delta.detach(),
        "loss_semantic_future": model.loss_lambda_semantic_future
        * loss_semantic_future.detach(),
    }
    if model.track_bridge is not None:
        loss_dict.update(
            loss_track=track_weight * loss_track_raw.detach(),
            loss_track_raw=loss_track_raw.detach(),
            track_loss_weight=loss_track_raw.new_tensor(track_weight),
            track_valid_fraction=(track_mask.float().mean() if track_mask is not None else loss_track_raw.new_zeros(())),
        )
    if pica_active:
        loss_dict.update(loss_pica_align=model.pica_alignment.loss_weight * loss_pica_align.detach(), **pica_metrics)
    if rtc_delay is not None:
        loss_dict["rtc_delay_mean"] = rtc_delay.float().mean().detach()
    return loss_total, loss_dict
