"""Strict adapters from refactor_2604 VGA batches to VAP chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VGABatchReport:
    batch_size: int
    view_keys: tuple[str, ...]
    history_frames: int
    future_frames: int
    action_steps: int
    action_dim: int
    equal_view_lengths: bool
    action_steps_per_video_frame: float


@dataclass(frozen=True)
class VAPBatchChunks:
    """Chunked tensors retaining the clean observation prefix.

    video_prefix/view and video_chunks/view tensors keep the source HWC layout.
    `video_chunks[view]` is [B,E,Tv,H,W,C], actions is [B,E,Ta,D].
    """

    video_prefix: Mapping[str, torch.Tensor]
    video_chunks: Mapping[str, torch.Tensor]
    actions: torch.Tensor
    dof_mask: torch.Tensor
    proprio: torch.Tensor
    metadata: Mapping[str, Any]
    # A true episode/segment-start observation, kept separate from the local
    # rolling-window I.  It is populated only when the explicit Memory switch
    # is enabled; otherwise this field remains None and is never inspected.
    episode_first_video: Mapping[str, torch.Tensor] | None = None

    def chunk_start_states(self, *, num_chunks: int) -> torch.Tensor:
        """Return one normalized conditioning state per Action chunk.

        New chunk-local datasets provide ``[B,E,D]`` directly (for example
        ``state[t]`` and ``state[t+32]``).  A legacy ``[B,1,D]`` S0 remains
        valid and is broadcast by :class:`ChunkStateFiLM` for backward
        compatibility. Any other temporal width is ambiguous and rejected.
        """

        if self.proprio.ndim != 3:
            raise ValueError(
                f"proprio must be [B,T,D], got {tuple(self.proprio.shape)}"
            )
        if self.proprio.shape[1] not in (1, int(num_chunks)):
            raise ValueError(
                "proprio state axis must be 1 or num_chunks="
                f"{num_chunks}, got {self.proprio.shape[1]}"
            )
        return self.proprio

    def continuous_video_sequence(
        self, *, vae_temporal_stride: int = 4
    ) -> Mapping[str, torch.Tensor]:
        """Return continuous ``I + V0 + ... + V{E-1}`` Wan inputs.

        Each view is ``[B,1+E*Tv,H,W,C]``. The single initial observation is
        the causal condition; future chunks are concatenated without
        duplicating or re-encoding the V0/V1 boundary. The complete sequence
        must satisfy Wan's ``4N+1`` temporal contract.
        """

        if vae_temporal_stride <= 0:
            raise ValueError("vae_temporal_stride must be positive")
        sequences = {}
        for key, chunks in self.video_chunks.items():
            if chunks.ndim != 6:
                raise ValueError(
                    f"video_chunks[{key!r}] must be [B,E,T,H,W,C], "
                    f"got {tuple(chunks.shape)}"
                )
            total_future = chunks.shape[1] * chunks.shape[2]
            if total_future % vae_temporal_stride:
                raise ValueError(
                    f"{key} has {total_future} future frames; expected a multiple "
                    f"of Wan VAE stride {vae_temporal_stride} so I+future is 4N+1"
                )
            prefix = self.video_prefix[key]
            if prefix.ndim != 5 or prefix.shape[1] < 1:
                raise ValueError(
                    f"video_prefix[{key!r}] must be [B,T,H,W,C] with T>=1, "
                    f"got {tuple(prefix.shape)}"
                )
            future = chunks.flatten(1, 2)
            sequence = torch.cat((prefix[:, -1:], future), dim=1)
            if (sequence.shape[1] - 1) % vae_temporal_stride:
                raise AssertionError("continuous Wan sequence is not 4N+1")
            sequences[key] = sequence
        return sequences


def _rotation_6d_rows_to_matrix(values: torch.Tensor) -> torch.Tensor:
    """Convert the repo's row-major 6D rotation representation to SO(3)."""

    if values.shape[-1] != 6:
        raise ValueError("rotation_6D values must have width 6")
    row0 = torch.nn.functional.normalize(values[..., :3], dim=-1)
    row1_raw = values[..., 3:6]
    row1 = row1_raw - (row1_raw * row0).sum(dim=-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1)
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rebase_relative_action_chunks(
    actions: torch.Tensor,
    dof_config: Mapping[str, int],
    *,
    action_steps_per_chunk: int,
    num_chunks: int = 2,
) -> torch.Tensor:
    """Rebase each later action chunk onto the preceding chunk boundary.

    x2robot_v1 produces every relative target against the sample's initial
    pose. IVAVA instead treats A1 as the policy rollout from V0's final state.
    Translation deltas are subtracted; row-major 6D rotations are composed as
    ``R(t->k) @ R(t->boundary).T``. Absolute gripper targets are unchanged.
    """

    if actions.ndim < 2:
        raise ValueError("actions must end in [T,D]")
    expected_steps = num_chunks * action_steps_per_chunk
    if actions.shape[-2] != expected_steps:
        raise ValueError(
            f"action length {actions.shape[-2]} != {num_chunks} * "
            f"{action_steps_per_chunk} = {expected_steps}"
        )
    expected_dim = sum(int(width) for width in dof_config.values())
    if actions.shape[-1] != expected_dim:
        raise ValueError(
            f"action dim {actions.shape[-1]} != dof_config sum {expected_dim}"
        )

    output = actions.clone()
    offset = 0
    slices: list[tuple[str, slice]] = []
    for name, width in dof_config.items():
        width = int(width)
        slices.append((name, slice(offset, offset + width)))
        offset += width

    for chunk_index in range(1, num_chunks):
        start = chunk_index * action_steps_per_chunk
        end = start + action_steps_per_chunk
        boundary = actions[..., start - 1, :]
        for name, dof_slice in slices:
            width = dof_slice.stop - dof_slice.start
            if "rotation_6D_relative" in name:
                if width != 6:
                    raise ValueError(f"{name} must have width 6, got {width}")
                target_matrix = _rotation_6d_rows_to_matrix(
                    actions[..., start:end, dof_slice]
                )
                boundary_matrix = _rotation_6d_rows_to_matrix(boundary[..., dof_slice])
                local_matrix = target_matrix @ boundary_matrix.unsqueeze(-3).transpose(
                    -1, -2
                )
                output[..., start:end, dof_slice] = local_matrix[..., :2, :].flatten(-2)
            elif "pos_relative" in name:
                if width != 3:
                    raise ValueError(f"{name} must have width 3, got {width}")
                output[..., start:end, dof_slice] = (
                    actions[..., start:end, dof_slice] - boundary[..., None, dof_slice]
                )
    return output


def rebase_normalized_relative_action_chunks(
    actions: torch.Tensor,
    normalizer: nn.Module,
    dataset_names,
    dof_config: Mapping[str, int],
    *,
    action_steps_per_chunk: int,
    num_chunks: int = 2,
    dof_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rebase an already-normalized x2robot_v2 action window.

    The v2 data pipeline applies its normalizer before returning a batch, while
    :func:`rebase_relative_action_chunks` must operate on physical translation
    and rotation values.  Decode with the model-side normalizer (constructed
    from the same v2 stats), rebase later chunks, then encode again.  Keeping
    this bridge inside VAP avoids changing the shared dataset package.
    """

    # VAP's canonical action layout is always full-width. ``dof_mask`` marks
    # invalid/loss-excluded coordinates; it must never compact the normalizer
    # vectors because that would turn a 20-D action into an 18-D transform for
    # embodiments with two inactive internal slots (for example dex hands).
    raw_actions = normalizer.unnormalize_data(
        actions,
        dataset_names,
        None,
    )
    local_actions = rebase_relative_action_chunks(
        raw_actions,
        dof_config,
        action_steps_per_chunk=action_steps_per_chunk,
        num_chunks=num_chunks,
    )
    return normalizer.normalize_data(local_actions, dataset_names)


class IVAVAChunkLocalActionNormalizer(nn.Module):
    """Rebase raw x2robot actions immediately before baseline normalization."""

    def __init__(
        self,
        normalizer: nn.Module,
        dof_config: Mapping[str, int],
        *,
        action_steps_per_chunk: int,
        num_chunks: int = 2,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.dof_config = dict(dof_config)
        self.action_steps_per_chunk = int(action_steps_per_chunk)
        self.num_chunks = int(num_chunks)

    @property
    def min(self):
        """Expose dataset-type statistics required by x2robot_dataset."""

        return self.normalizer.min

    def normalize_data(self, values, dataset_names):
        should_rebase = (
            torch.is_tensor(values)
            and values.ndim >= 3
            and values.shape[-2] == self.num_chunks * self.action_steps_per_chunk
        )
        if should_rebase:
            values = rebase_relative_action_chunks(
                values,
                self.dof_config,
                action_steps_per_chunk=self.action_steps_per_chunk,
                num_chunks=self.num_chunks,
            )
        return self.normalizer.normalize_data(values, dataset_names)

    def unnormalize_data(self, values, dataset_names, dof_mask=None):
        return self.normalizer.unnormalize_data(values, dataset_names, dof_mask)


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must be a torch.Tensor, got {type(value)}")
    return value


def inspect_vga_batch(batch: Mapping[str, Any]) -> VGABatchReport:
    """Inspect the fixed-horizon VGA contract and fail on ambiguous shapes."""

    obs = batch.get("obs")
    next_obs = batch.get("next_obs")
    if not isinstance(obs, Mapping) or not isinstance(next_obs, Mapping):
        raise TypeError("VGA batch requires mapping-valued obs and next_obs")

    view_keys = tuple(
        sorted(
            (key for key in next_obs if key.startswith("view") and key[4:].isdigit()),
            key=lambda key: int(key[4:]),
        )
    )
    if not view_keys:
        raise ValueError("VGA batch has no next_obs viewN tensors")
    if any(k not in obs for k in view_keys):
        raise ValueError("obs and next_obs must expose the same view slots")

    history_lengths = set()
    future_lengths = set()
    batch_sizes = set()
    for key in view_keys:
        obs_view = _require_tensor(obs, key)
        future_view = _require_tensor(next_obs, key)
        if obs_view.ndim != 5 or future_view.ndim != 5:
            raise ValueError(
                f"{key} must be [B,T,H,W,C]; got obs={tuple(obs_view.shape)}, "
                f"next_obs={tuple(future_view.shape)}"
            )
        batch_sizes.update((obs_view.shape[0], future_view.shape[0]))
        history_lengths.add(obs_view.shape[1])
        future_lengths.add(future_view.shape[1])

    action = _require_tensor(batch, "action")
    dof_mask = _require_tensor(batch, "dof_mask")
    if action.ndim != 3:
        raise ValueError(f"action must be [B,T,D], got {tuple(action.shape)}")
    if dof_mask.shape != action.shape:
        raise ValueError(
            f"dof_mask shape {tuple(dof_mask.shape)} != action {tuple(action.shape)}"
        )
    batch_sizes.add(action.shape[0])
    if len(batch_sizes) != 1:
        raise ValueError(f"inconsistent batch sizes: {sorted(batch_sizes)}")
    if len(history_lengths) != 1 or len(future_lengths) != 1:
        raise ValueError(
            "all camera views must have equal history/future lengths; "
            f"history={sorted(history_lengths)}, future={sorted(future_lengths)}"
        )
    history_frames = next(iter(history_lengths))
    future_frames = next(iter(future_lengths))
    if history_frames <= 0 or future_frames <= 0 or action.shape[1] <= 0:
        raise ValueError(
            "history, future-video and action horizons must all be positive"
        )

    agent_pos = _require_tensor(obs, "agent_pos")
    if agent_pos.ndim != 3 or agent_pos.shape[0] != action.shape[0]:
        raise ValueError(f"obs.agent_pos must be [B,T,D], got {tuple(agent_pos.shape)}")

    return VGABatchReport(
        batch_size=action.shape[0],
        view_keys=view_keys,
        history_frames=history_frames,
        future_frames=future_frames,
        action_steps=action.shape[1],
        action_dim=action.shape[2],
        equal_view_lengths=True,
        action_steps_per_video_frame=action.shape[1] / future_frames,
    )


def _split_time(x: torch.Tensor, num_chunks: int, name: str) -> torch.Tensor:
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    length = x.shape[1]
    if length % num_chunks:
        raise ValueError(
            f"{name} length {length} is not divisible by num_chunks={num_chunks}; "
            "VAP refuses silent padding/truncation"
        )
    chunk_len = length // num_chunks
    return x.reshape(x.shape[0], num_chunks, chunk_len, *x.shape[2:])


def _effective_dof_mask(batch: Mapping[str, Any]) -> torch.Tensor:
    """Combine per-DoF and per-timestep validity without implicit reshaping.

    x2robot_dataset_v2 emits ``action_valid_mask`` as ``[B,T]`` for windows
    that can contain padded or otherwise invalid action timesteps.  The VAP
    loss consumes a ``[B,T,D]`` mask, so validity must be applied before the
    IVAVA split.  Older datasets that do not expose this optional field retain
    their existing ``dof_mask`` behavior.
    """

    action = _require_tensor(batch, "action")
    dof_mask = _require_tensor(batch, "dof_mask")
    validity = batch.get("action_valid_mask")
    if validity is None:
        return dof_mask
    if not isinstance(validity, torch.Tensor):
        raise TypeError(
            "batch['action_valid_mask'] must be a torch.Tensor, "
            f"got {type(validity)}"
        )
    if validity.ndim == 2:
        if validity.shape != action.shape[:2]:
            raise ValueError(
                "action_valid_mask [B,T] shape "
                f"{tuple(validity.shape)} != action prefix {tuple(action.shape[:2])}"
            )
        validity = validity.unsqueeze(-1)
    elif validity.ndim == 3:
        if validity.shape != action.shape:
            raise ValueError(
                "action_valid_mask [B,T,D] shape "
                f"{tuple(validity.shape)} != action {tuple(action.shape)}"
            )
    else:
        raise ValueError(
            "action_valid_mask must be [B,T] or [B,T,D], "
            f"got {tuple(validity.shape)}"
        )
    if not torch.isfinite(validity).all():
        raise ValueError("action_valid_mask contains non-finite values")
    if torch.any((validity < 0) | (validity > 1)):
        raise ValueError("action_valid_mask values must lie in [0, 1]")
    return dof_mask * validity.to(device=dof_mask.device, dtype=dof_mask.dtype)


def split_vga_batch(
    batch: Mapping[str, Any],
    *,
    num_chunks: int = 2,
    video_frames_per_chunk: int = 20,
    action_steps_per_chunk: int = 40,
    require_episode_first_obs: bool = False,
) -> VAPBatchChunks:
    """Strictly split a fixed-horizon VGA clip into equal VAP chunks.

    The default is the first VAP milestone: two windows with logical order
    ``I -> A0 -> V0 -> A1 -> V1``. Each chunk keeps the previous single-window
    horizon; therefore the input batch must contain ``num_chunks`` windows.
    """

    report = inspect_vga_batch(batch)
    expected_video = num_chunks * video_frames_per_chunk
    expected_action = num_chunks * action_steps_per_chunk
    if report.future_frames != expected_video:
        raise ValueError(
            f"future video horizon {report.future_frames} != "
            f"{num_chunks} * {video_frames_per_chunk} = {expected_video}"
        )
    if report.action_steps != expected_action:
        raise ValueError(
            f"action horizon {report.action_steps} != "
            f"{num_chunks} * {action_steps_per_chunk} = {expected_action}"
        )
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    video_prefix = {key: obs[key] for key in report.view_keys}
    video_chunks = {
        key: _split_time(next_obs[key], num_chunks, f"next_obs.{key}")
        for key in report.view_keys
    }
    actions = _split_time(batch["action"], num_chunks, "action")
    dof_mask = _split_time(_effective_dof_mask(batch), num_chunks, "effective_dof_mask")
    metadata = {
        key: batch[key]
        for key in ("instruction", "caption", "dataset_name", "camera_names")
        if key in batch
    }
    episode_first_video = None
    if require_episode_first_obs:
        raw_anchor = batch.get("episode_first_obs")
        if not isinstance(raw_anchor, Mapping):
            raise ValueError(
                "first-frame memory is enabled but batch['episode_first_obs'] "
                "is missing; using the current window I as a fallback is forbidden"
            )
        anchor_keys = tuple(
            sorted(
                (
                    key
                    for key in raw_anchor
                    if key.startswith("view") and key[4:].isdigit()
                ),
                key=lambda key: int(key[4:]),
            )
        )
        if anchor_keys != report.view_keys:
            raise ValueError(
                "episode_first_obs views must exactly match the query views: "
                f"{anchor_keys} != {report.view_keys}"
            )
        episode_first_video = {}
        memory_slots = 1
        if "memory_valid" in batch:
            valid = batch["memory_valid"]
            ages = batch.get("memory_age_steps")
            if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.ndim != 2:
                raise ValueError("memory_valid must be bool [B,M]")
            if not isinstance(ages, torch.Tensor) or ages.shape != valid.shape:
                raise ValueError("memory_age_steps must match memory_valid [B,M]")
            if ages.is_floating_point() or bool((ages < 0).any()) or not bool(valid[:, 0].all()):
                raise ValueError("Memory requires integer nonnegative ages and valid first anchor")
            memory_slots = valid.shape[1]
            metadata["memory_valid"] = valid
            metadata["memory_age_steps"] = ages
        for key in report.view_keys:
            anchor = raw_anchor[key]
            prefix = video_prefix[key]
            if not isinstance(anchor, torch.Tensor):
                raise TypeError(
                    f"episode_first_obs.{key} must be a torch.Tensor, "
                    f"got {type(anchor)}"
                )
            expected = (prefix.shape[0], memory_slots, *prefix.shape[2:])
            if tuple(anchor.shape) != expected:
                raise ValueError(
                    f"episode_first_obs.{key} must be {expected}, "
                    f"got {tuple(anchor.shape)}"
                )
            if anchor.dtype != prefix.dtype:
                raise ValueError(
                    f"episode_first_obs.{key} dtype {anchor.dtype} does not match "
                    f"obs.{key} dtype {prefix.dtype}"
                )
            episode_first_video[key] = anchor
        if "episode_first_local_idx" in batch:
            local_idx = batch["episode_first_local_idx"]
            if not isinstance(local_idx, torch.Tensor):
                local_idx = torch.as_tensor(local_idx)
            if local_idx.numel() != report.batch_size or bool(
                (local_idx.reshape(-1) != 0).any().item()
            ):
                raise ValueError(
                    "episode_first_local_idx must contain one zero per sample"
                )
            metadata["episode_first_local_idx"] = local_idx
        if "episode_first_frame_idx" in batch:
            metadata["episode_first_frame_idx"] = batch["episode_first_frame_idx"]
        if "episode_first_memory_age_steps" in batch:
            age_steps = batch["episode_first_memory_age_steps"]
            if not isinstance(age_steps, torch.Tensor):
                age_steps = torch.as_tensor(age_steps)
            age_steps = age_steps.reshape(-1)
            if age_steps.numel() != report.batch_size:
                raise ValueError(
                    "episode_first_memory_age_steps must contain one value per "
                    "sample"
                )
            if age_steps.is_floating_point() and not torch.equal(
                age_steps, age_steps.round()
            ):
                raise ValueError(
                    "episode_first_memory_age_steps must contain integer values"
                )
            if bool((age_steps < 0).any().item()):
                raise ValueError("episode_first_memory_age_steps must be non-negative")
            metadata["episode_first_memory_age_steps"] = age_steps.to(dtype=torch.long)
    return VAPBatchChunks(
        video_prefix=video_prefix,
        video_chunks=video_chunks,
        actions=actions,
        dof_mask=dof_mask,
        proprio=obs["agent_pos"],
        metadata=metadata,
        episode_first_video=episode_first_video,
    )


__all__ = [
    "VGABatchReport",
    "VAPBatchChunks",
    "IVAVAChunkLocalActionNormalizer",
    "inspect_vga_batch",
    "rebase_normalized_relative_action_chunks",
    "rebase_relative_action_chunks",
    "split_vga_batch",
]
