"""Independent Video/Action flow-matching corruption and loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch


class ARStage(IntEnum):
    """Logical IVAVA rank of the one chunk being denoised."""

    A0 = 1
    V0 = 2
    A1 = 3
    V1 = 4

    @property
    def chunk_index(self) -> int:
        return (int(self) - 1) // 2

    @property
    def is_video(self) -> bool:
        return int(self) % 2 == 0


@dataclass(frozen=True)
class VAPARFlowBatch:
    """Teacher-forced AR corruption with clean prefix and pure-noise suffix."""

    noisy_video: torch.Tensor
    noisy_action: torch.Tensor
    video_target: torch.Tensor
    action_target: torch.Tensor
    video_sigma: torch.Tensor
    action_sigma: torch.Tensor
    video_chunk_sigma: torch.Tensor
    action_chunk_sigma: torch.Tensor
    video_loss_mask: torch.Tensor
    action_loss_mask: torch.Tensor
    stage_ids: torch.Tensor


@dataclass(frozen=True)
class VAPParallelFlowBatch:
    """All-stage clean/noisy flow inputs for packed teacher forcing."""

    noisy_video: torch.Tensor
    noisy_action: torch.Tensor
    action_noise: torch.Tensor
    video_target: torch.Tensor
    action_target: torch.Tensor
    video_chunk_sigma: torch.Tensor
    action_sigma: torch.Tensor
    video_loss_mask: torch.Tensor
    action_loss_mask: torch.Tensor


@dataclass(frozen=True)
class VAPFlowBatch:
    noisy_video: torch.Tensor
    noisy_action: torch.Tensor
    video_target: torch.Tensor
    action_target: torch.Tensor
    video_sigma: torch.Tensor
    action_sigma: torch.Tensor
    video_loss_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ActionPrefixCorruption:
    """Corrupted clean Action context plus observable training statistics."""

    actions: torch.Tensor
    sample_fraction: torch.Tensor
    rms: torch.Tensor


@dataclass(frozen=True)
class ActionSOARBatch:
    """One-step on-policy Action correction states for SOAR training."""

    noisy_action: torch.Tensor
    correction_target: torch.Tensor
    aux_sigma: torch.Tensor
    next_sigma: torch.Tensor
    rollout_rms: torch.Tensor


def corrupt_clean_action_prefix(
    clean_action: torch.Tensor,
    dof_mask: torch.Tensor,
    *,
    action_steps_per_chunk: int,
    probability: float,
    std: float,
    exclude_dims: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
) -> ActionPrefixCorruption:
    """Perturb only clean local A0 context in normalized Action space.

    The returned tensor retains clean A1 and every ineligible/padded element.
    The caller must continue constructing diffusion inputs and flow targets
    from the original ``clean_action`` tensor.
    """

    if clean_action.ndim != 3:
        raise ValueError("clean_action must be [B,T,D]")
    if tuple(dof_mask.shape) != tuple(clean_action.shape):
        raise ValueError("dof_mask must match clean_action")
    if action_steps_per_chunk <= 0 or action_steps_per_chunk >= clean_action.shape[1]:
        raise ValueError(
            "action_steps_per_chunk must select A0 inside a multi-chunk action"
        )
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be in [0,1]")
    if float(std) < 0:
        raise ValueError("std must be non-negative")
    if len(set(exclude_dims)) != len(exclude_dims):
        raise ValueError("exclude_dims must be unique")
    if any(index < 0 or index >= clean_action.shape[-1] for index in exclude_dims):
        raise ValueError("exclude_dims contains an out-of-range Action dimension")

    zero = clean_action.new_zeros((), dtype=torch.float32)
    if probability == 0 or std == 0:
        return ActionPrefixCorruption(clean_action, zero, zero)

    batch_size, _, action_dim = clean_action.shape
    selected = torch.rand(
        batch_size,
        device=clean_action.device,
        generator=generator,
    ) < float(probability)
    dim_mask = torch.ones(action_dim, device=clean_action.device, dtype=torch.bool)
    if exclude_dims:
        dim_mask[list(exclude_dims)] = False
    eligible = (
        selected[:, None, None]
        & dof_mask[:, :action_steps_per_chunk].to(dtype=torch.bool)
        & dim_mask[None, None, :]
    )
    noise = torch.randn(
        (batch_size, action_steps_per_chunk, action_dim),
        device=clean_action.device,
        dtype=clean_action.dtype,
        generator=generator,
    ) * float(std)
    original_a0 = clean_action[:, :action_steps_per_chunk]
    candidate = (original_a0 + noise).clamp(-1.0, 1.0)
    corrupted_a0 = torch.where(eligible, candidate, original_a0)
    actions = torch.cat((corrupted_a0, clean_action[:, action_steps_per_chunk:]), dim=1)
    delta = (corrupted_a0.float() - original_a0.float()).square()
    eligible_count = eligible.sum().to(dtype=torch.float32)
    rms = torch.where(
        eligible_count > 0,
        (delta.sum() / eligible_count.clamp_min(1.0)).sqrt(),
        zero,
    )
    return ActionPrefixCorruption(
        actions=actions,
        sample_fraction=selected.float().mean(),
        rms=rms,
    )


def _expand_sigma(sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if sigma.ndim != 1 or sigma.shape[0] != target.shape[0]:
        raise ValueError(
            f"sigma must be [B] matching target batch, got {tuple(sigma.shape)}"
        )
    return sigma.reshape(sigma.shape[0], *((1,) * (target.ndim - 1))).to(
        device=target.device, dtype=target.dtype
    )


def sample_shifted_sigma(
    batch_size: int,
    *,
    shift: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    sample_mode: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Sample a base sigma then apply Wan's rational shift.

    ``uniform`` preserves the historical VAP objective exactly. In
    ``logit_normal`` mode, the base coordinate is ``sigmoid(N(mean, std))``;
    the same rational shift is then applied so ``shift`` keeps one consistent
    meaning across both sampling distributions.
    """

    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if sample_mode == "uniform":
        sigma = torch.rand(batch_size, device=device, generator=generator)
    elif sample_mode == "logit_normal":
        if not math.isfinite(logit_mean):
            raise ValueError(f"logit_mean must be finite, got {logit_mean}")
        if not math.isfinite(logit_std) or logit_std <= 0:
            raise ValueError(f"logit_std must be finite and positive, got {logit_std}")
        logits = torch.randn(batch_size, device=device, generator=generator)
        sigma = torch.sigmoid(logits * float(logit_std) + float(logit_mean))
    else:
        raise ValueError(
            "sample_mode must be 'uniform' or 'logit_normal', "
            f"got {sample_mode!r}"
        )
    return shift * sigma / (1 + (shift - 1) * sigma)


def step_shifted_sigma_toward_clean(
    sigma: torch.Tensor,
    *,
    shift: float,
    inference_steps: int,
) -> torch.Tensor:
    """Move a shifted-flow sigma by one deployed ODE step toward clean.

    Training samples a continuous base coordinate and applies Wan's rational
    sigma shift.  SOAR must step in that unshifted coordinate; subtracting
    ``1 / inference_steps`` directly from a shifted sigma would be wrong when
    ``shift != 1``.
    """

    if shift <= 0:
        raise ValueError("shift must be positive")
    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    sigma_fp32 = sigma.float()
    denominator = float(shift) - (float(shift) - 1.0) * sigma_fp32
    base = sigma_fp32 / denominator.clamp_min(torch.finfo(torch.float32).eps)
    next_base = (base - 1.0 / float(inference_steps)).clamp_min(0.0)
    next_sigma = float(shift) * next_base / (1.0 + (float(shift) - 1.0) * next_base)
    return next_sigma.to(device=sigma.device, dtype=sigma.dtype)


def build_action_soar_batch(
    clean_action: torch.Tensor,
    noisy_action: torch.Tensor,
    action_noise: torch.Tensor,
    base_velocity: torch.Tensor,
    *,
    action_sigma: torch.Tensor,
    action_steps_per_chunk: int,
    shift: float,
    inference_steps: int,
    min_aux_sigma: float,
    generator: torch.Generator | None = None,
) -> ActionSOARBatch:
    """Construct SOAR off-trajectory Action states with the base noise ray.

    ``base_velocity`` is detached here even if the caller already detached it:
    the rollout state is a target-construction path and must never backpropagate
    through the model prediction that created it.  The correction target stays
    fp32 while the model input retains the Action activation dtype.
    """

    if clean_action.ndim != 3:
        raise ValueError("clean_action must be [B,T,D]")
    for name, value in (
        ("noisy_action", noisy_action),
        ("action_noise", action_noise),
        ("base_velocity", base_velocity),
    ):
        if tuple(value.shape) != tuple(clean_action.shape):
            raise ValueError(f"{name} must match clean_action")
    if action_steps_per_chunk <= 0:
        raise ValueError("action_steps_per_chunk must be positive")
    batch_size, total_steps, _ = clean_action.shape
    if total_steps % action_steps_per_chunk:
        raise ValueError("Action length must be divisible by steps per chunk")
    num_chunks = total_steps // action_steps_per_chunk
    if tuple(action_sigma.shape) != (batch_size, num_chunks):
        raise ValueError("action_sigma must be [B,num_chunks]")
    if not 0.0 < float(min_aux_sigma) < 1.0:
        raise ValueError("min_aux_sigma must be inside (0,1)")

    next_sigma = step_shifted_sigma_toward_clean(
        action_sigma,
        shift=float(shift),
        inference_steps=int(inference_steps),
    )
    lower = next_sigma.float().clamp_min(float(min_aux_sigma))
    uniform = torch.rand(
        action_sigma.shape,
        device=action_sigma.device,
        generator=generator,
        dtype=torch.float32,
    )
    aux_sigma = lower + uniform * (1.0 - lower)

    current_scale = action_sigma.repeat_interleave(action_steps_per_chunk, dim=1).to(
        device=clean_action.device, dtype=clean_action.dtype
    )[..., None]
    next_scale_fp32 = (
        next_sigma.float()
        .repeat_interleave(action_steps_per_chunk, dim=1)
        .to(device=clean_action.device)[..., None]
    )
    aux_scale_fp32 = aux_sigma.repeat_interleave(action_steps_per_chunk, dim=1).to(
        device=clean_action.device, dtype=torch.float32
    )[..., None]
    next_scale = next_scale_fp32.to(dtype=clean_action.dtype)

    rollout = noisy_action + (next_scale - current_scale) * base_velocity.detach()
    ideal_next = (1.0 - next_scale) * clean_action + next_scale * action_noise
    rollout_rms = rollout.float().sub(ideal_next.float()).square().mean().sqrt()

    alpha = (
        (aux_scale_fp32 - next_scale_fp32)
        / (1.0 - next_scale_fp32).clamp_min(torch.finfo(torch.float32).eps)
    ).clamp(0.0, 1.0)
    auxiliary = (1.0 - alpha) * rollout.float() + alpha * action_noise.float()
    correction_target = (auxiliary - clean_action.float()) / aux_scale_fp32.clamp_min(
        float(min_aux_sigma)
    )
    return ActionSOARBatch(
        noisy_action=auxiliary.to(dtype=clean_action.dtype),
        correction_target=correction_target,
        aux_sigma=aux_sigma.to(device=action_sigma.device, dtype=action_sigma.dtype),
        next_sigma=next_sigma,
        rollout_rms=rollout_rms,
    )


def sample_ar_stages(
    batch_size: int,
    *,
    num_chunks: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniformly choose one of ``A0,V0,...`` independently per sample."""

    if batch_size <= 0 or num_chunks <= 0:
        raise ValueError("batch_size and num_chunks must be positive")
    return torch.randint(
        1,
        2 * num_chunks + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )


def ar_chunk_sigmas(
    stage_ids: torch.Tensor,
    *,
    chunk_ranks: torch.Tensor,
    active_sigma: torch.Tensor,
) -> torch.Tensor:
    """Return 0 for clean prefix, sampled sigma for active, 1 for suffix."""

    ranks = chunk_ranks.to(device=stage_ids.device)[None, :]
    stages = stage_ids[:, None]
    active = active_sigma[:, None]
    return torch.where(
        ranks < stages,
        torch.zeros_like(active).expand(-1, ranks.shape[1]),
        torch.where(
            ranks == stages,
            active.expand(-1, ranks.shape[1]),
            torch.ones_like(active).expand(-1, ranks.shape[1]),
        ),
    )


def build_ar_flow_batch(
    clean_video: torch.Tensor,
    clean_action: torch.Tensor,
    *,
    num_chunks: int,
    future_video_frames_per_chunk: int,
    action_steps_per_chunk: int,
    stage_ids: torch.Tensor,
    video_sigma: torch.Tensor,
    action_sigma: torch.Tensor,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> VAPARFlowBatch:
    """Build strict block-AR flow inputs for ``I,A0,V0,A1,V1``.

    Tokens before each sample's selected stage are clean teacher-forced prefix;
    the selected chunk follows its sampled straight-flow path; later chunks are
    pure base noise. Only the selected chunk contributes a loss.
    """

    if clean_video.ndim != 6:
        raise ValueError("clean_video must be [B,V,C,F,H,W]")
    if clean_action.ndim != 3:
        raise ValueError("clean_action must be [B,T,D]")
    batch_size = clean_video.shape[0]
    if clean_action.shape[0] != batch_size:
        raise ValueError("video/action batch sizes must match")
    expected_frames = 1 + num_chunks * future_video_frames_per_chunk
    expected_actions = num_chunks * action_steps_per_chunk
    if clean_video.shape[3] != expected_frames:
        raise ValueError(
            f"video latent frames {clean_video.shape[3]} != {expected_frames}"
        )
    if clean_action.shape[1] != expected_actions:
        raise ValueError(f"action steps {clean_action.shape[1]} != {expected_actions}")
    for name, value in (
        ("stage_ids", stage_ids),
        ("video_sigma", video_sigma),
        ("action_sigma", action_sigma),
    ):
        if tuple(value.shape) != (batch_size,):
            raise ValueError(f"{name} must be [B]")
    if stage_ids.dtype == torch.bool or stage_ids.is_floating_point():
        raise ValueError("stage_ids must use an integer dtype")
    if stage_ids.min().item() < 1 or stage_ids.max().item() > 2 * num_chunks:
        raise ValueError("stage_ids must select one logical Action/Video chunk")

    if video_noise is None:
        video_noise = torch.randn(
            clean_video.shape,
            device=clean_video.device,
            dtype=clean_video.dtype,
            generator=generator,
        )
    if action_noise is None:
        action_noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
    if video_noise.shape != clean_video.shape:
        raise ValueError("video_noise shape must equal clean_video")
    if action_noise.shape != clean_action.shape:
        raise ValueError("action_noise shape must equal clean_action")

    video_ranks = torch.arange(2, 2 * num_chunks + 1, 2, device=stage_ids.device)
    action_ranks = torch.arange(1, 2 * num_chunks, 2, device=stage_ids.device)
    video_chunk_sigma = ar_chunk_sigmas(
        stage_ids, chunk_ranks=video_ranks, active_sigma=video_sigma
    )
    action_chunk_sigma = ar_chunk_sigmas(
        stage_ids, chunk_ranks=action_ranks, active_sigma=action_sigma
    )

    video_frame_sigma = video_chunk_sigma.repeat_interleave(
        future_video_frames_per_chunk, dim=1
    )
    video_frame_sigma = torch.cat(
        (torch.zeros(batch_size, 1, device=stage_ids.device), video_frame_sigma),
        dim=1,
    ).to(device=clean_video.device, dtype=clean_video.dtype)
    video_scale = video_frame_sigma[:, None, None, :, None, None]
    action_token_sigma = action_chunk_sigma.repeat_interleave(
        action_steps_per_chunk, dim=1
    ).to(device=clean_action.device, dtype=clean_action.dtype)
    action_scale = action_token_sigma[:, :, None]

    video_frame_ranks = video_ranks.repeat_interleave(future_video_frames_per_chunk)
    video_frame_ranks = torch.cat(
        (
            torch.zeros(1, device=stage_ids.device, dtype=video_ranks.dtype),
            video_frame_ranks,
        )
    )
    action_token_ranks = action_ranks.repeat_interleave(action_steps_per_chunk)
    video_loss_mask = (video_frame_ranks[None, :] == stage_ids[:, None])[
        :, None, None, :, None, None
    ].to(device=clean_video.device)
    action_loss_mask = (action_token_ranks[None, :] == stage_ids[:, None])[
        :, :, None
    ].to(device=clean_action.device)

    return VAPARFlowBatch(
        noisy_video=(1 - video_scale) * clean_video + video_scale * video_noise,
        noisy_action=(1 - action_scale) * clean_action + action_scale * action_noise,
        video_target=video_noise - clean_video,
        action_target=action_noise - clean_action,
        video_sigma=video_sigma,
        action_sigma=action_sigma,
        video_chunk_sigma=video_chunk_sigma,
        action_chunk_sigma=action_chunk_sigma,
        video_loss_mask=video_loss_mask,
        action_loss_mask=action_loss_mask,
        stage_ids=stage_ids,
    )


def build_parallel_teacher_forcing_flow_batch(
    clean_video: torch.Tensor,
    clean_action: torch.Tensor,
    *,
    num_chunks: int,
    future_video_frames_per_chunk: int,
    action_steps_per_chunk: int,
    video_chunk_sigma: torch.Tensor,
    action_sigma: torch.Tensor,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> VAPParallelFlowBatch:
    """Corrupt every IVAVA target while retaining separate clean copies.

    Video and action chunks receive independent timesteps.  Action chunks also
    retain independent Gaussian noise.  The returned tensors have the standard
    physical shapes ``[I,V0,V1]`` and ``[A0,A1]``; duplication into clean/noisy
    transformer streams happens after patchification.
    """

    if clean_video.ndim != 6:
        raise ValueError("clean_video must be [B,V,C,F,H,W]")
    if clean_action.ndim != 3:
        raise ValueError("clean_action must be [B,T,D]")
    batch_size = clean_video.shape[0]
    expected_frames = 1 + num_chunks * future_video_frames_per_chunk
    expected_actions = num_chunks * action_steps_per_chunk
    if clean_video.shape[3] != expected_frames:
        raise ValueError(
            f"video latent frames {clean_video.shape[3]} != {expected_frames}"
        )
    if clean_action.shape[:2] != (batch_size, expected_actions):
        raise ValueError(
            f"action shape {tuple(clean_action.shape[:2])} != "
            f"{(batch_size, expected_actions)}"
        )
    if tuple(video_chunk_sigma.shape) != (batch_size, num_chunks):
        raise ValueError("video_chunk_sigma must be [B,num_chunks]")
    if tuple(action_sigma.shape) != (batch_size, num_chunks):
        raise ValueError("action_sigma must be [B,num_chunks]")

    if video_noise is None:
        video_noise = torch.randn(
            clean_video.shape,
            device=clean_video.device,
            dtype=clean_video.dtype,
            generator=generator,
        )
    if action_noise is None:
        action_noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
    if video_noise.shape != clean_video.shape:
        raise ValueError("video_noise shape must equal clean_video")
    if action_noise.shape != clean_action.shape:
        raise ValueError("action_noise shape must equal clean_action")

    frame_sigma = video_chunk_sigma.repeat_interleave(
        future_video_frames_per_chunk, dim=1
    )
    frame_sigma = torch.cat(
        (
            torch.zeros(
                batch_size,
                1,
                device=video_chunk_sigma.device,
                dtype=video_chunk_sigma.dtype,
            ),
            frame_sigma,
        ),
        dim=1,
    ).to(device=clean_video.device, dtype=clean_video.dtype)
    video_scale = frame_sigma[:, None, None, :, None, None]
    action_scale = action_sigma.repeat_interleave(action_steps_per_chunk, dim=1).to(
        device=clean_action.device, dtype=clean_action.dtype
    )[..., None]

    video_loss_mask = torch.ones(
        batch_size,
        1,
        1,
        expected_frames,
        1,
        1,
        device=clean_video.device,
        dtype=torch.bool,
    )
    video_loss_mask[:, :, :, 0] = False
    action_loss_mask = torch.ones(
        batch_size,
        expected_actions,
        1,
        device=clean_action.device,
        dtype=torch.bool,
    )
    return VAPParallelFlowBatch(
        noisy_video=(1 - video_scale) * clean_video + video_scale * video_noise,
        noisy_action=(1 - action_scale) * clean_action + action_scale * action_noise,
        action_noise=action_noise,
        video_target=video_noise - clean_video,
        action_target=action_noise - clean_action,
        video_chunk_sigma=video_chunk_sigma,
        action_sigma=action_sigma,
        video_loss_mask=video_loss_mask,
        action_loss_mask=action_loss_mask,
    )


def build_independent_flow_batch(
    clean_video: torch.Tensor,
    clean_action: torch.Tensor,
    *,
    video_sigma: torch.Tensor | None = None,
    action_sigma: torch.Tensor | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    video_condition_mask: torch.Tensor | None = None,
) -> VAPFlowBatch:
    """Corrupt each expert with a separately sampled straight flow path."""

    if clean_video.shape[0] != clean_action.shape[0]:
        raise ValueError("video/action batch sizes must match")
    batch_size = clean_video.shape[0]
    if video_sigma is None:
        video_sigma = torch.rand(
            batch_size, device=clean_video.device, generator=generator
        )
    if action_sigma is None:
        action_sigma = torch.rand(
            batch_size, device=clean_action.device, generator=generator
        )
    if video_noise is None:
        video_noise = torch.randn(
            clean_video.shape,
            device=clean_video.device,
            dtype=clean_video.dtype,
            generator=generator,
        )
    if action_noise is None:
        action_noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
    if video_noise.shape != clean_video.shape:
        raise ValueError("video_noise shape must equal clean_video")
    if action_noise.shape != clean_action.shape:
        raise ValueError("action_noise shape must equal clean_action")

    video_scale = _expand_sigma(video_sigma, clean_video)
    action_scale = _expand_sigma(action_sigma, clean_action)
    noisy_video = (1 - video_scale) * clean_video + video_scale * video_noise
    video_loss_mask = None
    if video_condition_mask is not None:
        try:
            condition = video_condition_mask.to(
                device=clean_video.device, dtype=torch.bool
            ).expand_as(clean_video)
        except RuntimeError as exc:
            raise ValueError(
                "video_condition_mask is not broadcastable to clean_video"
            ) from exc
        noisy_video = torch.where(condition, clean_video, noisy_video)
        video_loss_mask = ~condition
    return VAPFlowBatch(
        noisy_video=noisy_video,
        noisy_action=(1 - action_scale) * clean_action + action_scale * action_noise,
        video_target=video_noise - clean_video,
        action_target=action_noise - clean_action,
        video_sigma=video_sigma,
        action_sigma=action_sigma,
        video_loss_mask=video_loss_mask,
    )


def flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    squared = (prediction.float() - target.float()).square()
    if mask is None:
        return squared.mean()
    try:
        weighted = squared * mask.to(device=squared.device, dtype=squared.dtype)
    except RuntimeError as exc:
        raise ValueError("mask is not broadcastable to prediction") from exc
    denominator = mask.expand_as(squared).sum().clamp_min(1).to(squared.dtype)
    return weighted.sum() / denominator


def flow_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor,
    element_weights: torch.Tensor | None = None,
    element_weight_normalization: str = "weighted_mean",
) -> torch.Tensor:
    """Masked weighted reduction; samples with no active token return zero.

    ``element_weights`` is normalized by its valid masked sum.  A relative
    With ``weighted_mean``, weights are normalized by their valid masked sum.
    With ``valid_mean``, the denominator is the raw valid-element count, so
    weights also change the aggregate loss scale.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    squared = (prediction.float() - target.float()).square()
    try:
        expanded_mask = mask.to(device=squared.device, dtype=squared.dtype).expand_as(
            squared
        )
    except RuntimeError as exc:
        raise ValueError("mask is not broadcastable to prediction") from exc
    denominator_mask = expanded_mask
    if element_weights is not None:
        try:
            expanded_weights = element_weights.to(
                device=squared.device, dtype=squared.dtype
            ).expand_as(squared)
        except RuntimeError as exc:
            raise ValueError(
                "element_weights is not broadcastable to prediction"
            ) from exc
        weighted_mask = expanded_mask * expanded_weights
        if element_weight_normalization == "weighted_mean":
            denominator_mask = weighted_mask
        elif element_weight_normalization == "valid_mean":
            denominator_mask = expanded_mask
        else:
            raise ValueError(
                "element_weight_normalization must be weighted_mean or valid_mean"
            )
        expanded_mask = weighted_mask
    reduce_dims = tuple(range(1, squared.ndim))
    numerator = (squared * expanded_mask).sum(dim=reduce_dims)
    denominator = denominator_mask.sum(dim=reduce_dims).clamp_min(1)
    return numerator / denominator


__all__ = [
    "ActionPrefixCorruption",
    "ActionSOARBatch",
    "ARStage",
    "VAPARFlowBatch",
    "VAPFlowBatch",
    "VAPParallelFlowBatch",
    "ar_chunk_sigmas",
    "build_ar_flow_batch",
    "build_action_soar_batch",
    "build_parallel_teacher_forcing_flow_batch",
    "corrupt_clean_action_prefix",
    "build_independent_flow_batch",
    "flow_mse",
    "flow_mse_per_sample",
    "sample_ar_stages",
    "sample_shifted_sigma",
    "step_shifted_sigma_toward_clean",
]
