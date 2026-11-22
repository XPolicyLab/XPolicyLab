"""Shape-safe bridge from continuous IVAVA video to Wan2.2 latent tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from .data import VAPBatchChunks
from .memory import FirstFrameMemoryTokens
from .rope import video_position_ids_from_grid


@dataclass(frozen=True)
class EncodedVideoSequence:
    """Continuous Wan latents in ``[B,V,C,F,H,W]`` order.

    Temporal frame 0 is I. The remaining frames are equally split into the
    configured future chunks without duplicating chunk boundaries.
    """

    latents: torch.Tensor
    view_keys: tuple[str, ...]
    raw_frames: int
    num_chunks: int
    future_latent_frames_per_chunk: int

    @property
    def condition_mask(self) -> torch.Tensor:
        """Broadcastable mask selecting the single I latent for every view."""

        mask = torch.zeros(
            self.latents.shape[0],
            self.latents.shape[1],
            1,
            self.latents.shape[3],
            1,
            1,
            dtype=torch.bool,
            device=self.latents.device,
        )
        mask[:, :, :, 0] = True
        return mask


@dataclass(frozen=True)
class EncodedFirstFrameMemory:
    """Episode-start visual anchor plus its distance from the current window."""

    latents: torch.Tensor
    view_keys: tuple[str, ...]
    age_steps: torch.Tensor | None = None
    valid: torch.Tensor | None = None  # optional bool [B,M]; None is legacy one-frame


@dataclass(frozen=True)
class PatchedVideoSequence:
    """Video tokens physically stored as ``[I,V0,V1]``."""

    tokens: torch.Tensor
    position_ids: torch.Tensor
    camera_ids: torch.Tensor
    condition_token_mask: torch.Tensor
    grid: tuple[int, int, int]
    num_chunks: int
    num_views: int
    condition_token_count: int
    video_tokens_per_chunk: int
    memory_valid: torch.Tensor | None = None  # bool [B,memory_tokens]

    def per_token_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        batch_size = self.tokens.shape[0]
        if tuple(timestep.shape) == (batch_size,):
            expanded = timestep[:, None].expand(-1, self.tokens.shape[1]).clone()
            return expanded.masked_fill(self.condition_token_mask, 0)
        if tuple(timestep.shape) != (batch_size, self.num_chunks):
            raise ValueError("video timestep must be [B] or per-chunk [B,E]")
        condition = torch.zeros(
            batch_size,
            self.condition_token_count,
            device=timestep.device,
            dtype=timestep.dtype,
        )
        future = timestep.repeat_interleave(self.video_tokens_per_chunk, dim=1)
        return torch.cat((condition, future), dim=1)


def _normalize_pixels(frames: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    frames = frames.to(dtype=torch.float32)
    if frames.max() <= 1.0:
        frames = frames * 2.0 - 1.0
    else:
        frames = frames * (2.0 / 255.0) - 1.0
    return frames.to(dtype=dtype)


def encode_video_sequence(
    vae,
    chunks: VAPBatchChunks,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    vae_temporal_stride: int = 4,
    tiled: bool = False,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
) -> EncodedVideoSequence:
    """Encode ``I+V0+...`` once per view with Wan's ``4N+1`` contract."""

    sequences = chunks.continuous_video_sequence(
        vae_temporal_stride=vae_temporal_stride
    )
    view_keys = tuple(sorted(sequences, key=lambda key: int(key[4:])))
    shapes = {tuple(sequences[key].shape) for key in view_keys}
    if len(shapes) != 1:
        raise ValueError(f"all video sequences must share shape, got {shapes}")

    # [B,V,T,H,W,C] and one continuous VAE call over B*V streams.
    pixels = torch.stack([sequences[key] for key in view_keys], dim=1)
    batch, views, raw_frames, height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError(f"Wan VAE expects RGB channels=3, got {channels}")
    sample_chunks = next(iter(chunks.video_chunks.values()))
    num_chunks = sample_chunks.shape[1]
    expected_raw_frames = 1 + num_chunks * sample_chunks.shape[2]
    if raw_frames != expected_raw_frames:
        raise AssertionError(
            f"raw video length {raw_frames} != expected {expected_raw_frames}"
        )

    flat = pixels.reshape(batch * views, raw_frames, height, width, channels).to(
        device=device
    )
    flat = _normalize_pixels(flat, dtype=dtype).permute(0, 4, 1, 2, 3)
    latents = vae.encode(
        flat,
        device=device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    ).to(device=device, dtype=dtype)
    expected_latent_frames = (raw_frames - 1) // vae_temporal_stride + 1
    if latents.shape[2] != expected_latent_frames:
        raise RuntimeError(
            f"Wan VAE returned F={latents.shape[2]}, expected "
            f"{expected_latent_frames} from T={raw_frames}"
        )
    future_latent_frames = expected_latent_frames - 1
    if future_latent_frames % num_chunks:
        raise RuntimeError(
            f"future latent frames {future_latent_frames} cannot split into "
            f"{num_chunks} IVAVA chunks"
        )
    latents = latents.reshape(batch, views, *latents.shape[1:])
    return EncodedVideoSequence(
        latents=latents,
        view_keys=view_keys,
        raw_frames=raw_frames,
        num_chunks=num_chunks,
        future_latent_frames_per_chunk=future_latent_frames // num_chunks,
    )


def encode_first_frame_memory(
    vae,
    chunks: VAPBatchChunks,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    tiled: bool = False,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
    view_keys: Sequence[str] | None = None,
) -> EncodedFirstFrameMemory:
    """Encode the strict ``episode_first_obs`` bank once per source sample."""

    anchors = chunks.episode_first_video
    if anchors is None:
        raise ValueError("first-frame memory is enabled but chunks contain no anchor")
    available_view_keys = tuple(sorted(anchors, key=lambda key: int(key[4:])))
    if available_view_keys != tuple(chunks.video_prefix):
        raise ValueError("first-frame memory views do not match video prefix views")
    selected_view_keys = (
        available_view_keys if not view_keys else tuple(str(key) for key in view_keys)
    )
    if not selected_view_keys:
        raise ValueError("first-frame memory must select at least one view")
    if len(set(selected_view_keys)) != len(selected_view_keys):
        raise ValueError("first-frame memory selected views must be unique")
    missing_view_keys = tuple(
        key for key in selected_view_keys if key not in available_view_keys
    )
    if missing_view_keys:
        raise ValueError(
            "first-frame memory selected views are absent from the batch: "
            f"{missing_view_keys}; available={available_view_keys}"
        )
    view_keys = selected_view_keys
    shapes = {tuple(anchors[key].shape) for key in view_keys}
    if len(shapes) != 1:
        raise ValueError(f"all first-frame views must share shape, got {shapes}")
    pixels = torch.stack([anchors[key] for key in view_keys], dim=1)
    batch, views, frames, height, width, channels = pixels.shape
    valid = chunks.metadata.get("memory_valid")
    if (frames != 1 and valid is None) or channels != 3:
        raise ValueError(
            "first-frame memory expects one RGB frame per view, got "
            f"T={frames}, C={channels}"
        )
    flat = pixels.reshape(batch * views * frames, 1, height, width, channels).to(device=device)
    if valid is not None:
        valid = valid.to(device=device, dtype=torch.bool)
        if tuple(valid.shape) != (batch, frames):
            raise ValueError("Memory frame validity must match [B,M]")
        flat_valid = valid[:, None, :].expand(batch, views, frames).reshape(-1)
        flat = flat[flat_valid]
    flat = _normalize_pixels(flat, dtype=dtype).permute(0, 4, 1, 2, 3)
    latents = vae.encode(
        flat,
        device=device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    ).to(device=device, dtype=dtype)
    if latents.shape[2] != 1:
        raise RuntimeError(
            "Wan VAE must preserve a one-frame first-frame memory latent"
        )
    if valid is not None:
        all_latents = latents.new_zeros((batch * views * frames, *latents.shape[1:]))
        all_latents[flat_valid] = latents
        latents = all_latents.reshape(batch, views, frames, *latents.shape[1:])
        latents = latents.squeeze(4).permute(0, 1, 3, 2, 4, 5).contiguous()
        return EncodedFirstFrameMemory(
            latents=latents, view_keys=view_keys, valid=valid,
            age_steps=chunks.metadata["memory_age_steps"].to(device=device),
        )
    return EncodedFirstFrameMemory(
        latents=latents.reshape(batch, views, *latents.shape[1:]),
        view_keys=view_keys,
        age_steps=(
            torch.as_tensor(
                chunks.metadata["episode_first_memory_age_steps"],
                device=device,
                dtype=torch.long,
            ).reshape(-1)
            if "episode_first_memory_age_steps" in chunks.metadata
            else None
        ),
    )


def _patched_video_from_grid(
    patched: torch.Tensor,
    encoded: EncodedVideoSequence,
    *,
    view_ids: torch.Tensor,
    chunk_stride: int,
    temporal_stride: int,
) -> PatchedVideoSequence:
    """Assemble the standard physical Video storage from patch-grid tensors."""

    batch, views, dim, latent_frames, height_tokens, width_tokens = patched.shape
    if latent_frames != encoded.latents.shape[3]:
        raise RuntimeError("Wan patchify unexpectedly changed temporal length")

    condition = patched[:, :, :, :1]
    future = patched[:, :, :, 1:]
    future = future.unflatten(
        3, (encoded.num_chunks, encoded.future_latent_frames_per_chunk)
    ).permute(0, 3, 1, 2, 4, 5, 6)
    condition_tokens = rearrange(condition, "b v d f h w -> b (v f h w) d").contiguous()
    future_tokens = rearrange(future, "b e v d f h w -> b (e v f h w) d").contiguous()
    tokens = torch.cat((condition_tokens, future_tokens), dim=1)

    condition_camera_ids = (
        view_ids[:, :, None, None, None]
        .expand(batch, views, 1, height_tokens, width_tokens)
        .flatten(1)
    )
    future_camera_ids = (
        view_ids[:, None, :, None, None, None]
        .expand(
            batch,
            encoded.num_chunks,
            views,
            encoded.future_latent_frames_per_chunk,
            height_tokens,
            width_tokens,
        )
        .flatten(1)
    )
    camera_ids = torch.cat((condition_camera_ids, future_camera_ids), dim=1).to(
        device=tokens.device, dtype=torch.long
    )

    condition_positions = video_position_ids_from_grid(
        batch_size=batch,
        num_chunks=1,
        num_views=views,
        latent_frames=1,
        height_tokens=height_tokens,
        width_tokens=width_tokens,
        chunk_stride=chunk_stride,
        temporal_stride=temporal_stride,
        time_offset=0,
        device=tokens.device,
    )
    future_positions = video_position_ids_from_grid(
        batch_size=batch,
        num_chunks=encoded.num_chunks,
        num_views=views,
        latent_frames=encoded.future_latent_frames_per_chunk,
        height_tokens=height_tokens,
        width_tokens=width_tokens,
        chunk_stride=chunk_stride,
        temporal_stride=temporal_stride,
        time_offset=1,
        device=tokens.device,
    )
    position_ids = torch.cat((condition_positions, future_positions), dim=1)
    condition_token_count = condition_tokens.shape[1]
    condition_mask = torch.zeros(
        batch, tokens.shape[1], dtype=torch.bool, device=tokens.device
    )
    condition_mask[:, :condition_token_count] = True
    return PatchedVideoSequence(
        tokens=tokens,
        position_ids=position_ids,
        camera_ids=camera_ids,
        condition_token_mask=condition_mask,
        grid=(latent_frames, height_tokens, width_tokens),
        num_chunks=encoded.num_chunks,
        num_views=views,
        condition_token_count=condition_token_count,
        video_tokens_per_chunk=future_tokens.shape[1] // encoded.num_chunks,
    )


def patchify_video_sequence(
    dit,
    encoded: EncodedVideoSequence,
    *,
    view_ids: torch.Tensor,
    chunk_stride: int,
    temporal_stride: int = 1,
) -> PatchedVideoSequence:
    """Patchify continuous latents as physical ``I,V0,...,V{E-1}``."""

    latents = encoded.latents
    batch, views, channels, frames, height, width = latents.shape
    if tuple(view_ids.shape) != (batch, views):
        raise ValueError(
            f"view_ids must be {(batch, views)}, got {tuple(view_ids.shape)}"
        )
    flat = latents.reshape(batch * views, channels, frames, height, width)
    patched = dit.patchify(flat)
    _, dim, latent_frames, height_tokens, width_tokens = patched.shape
    patched = patched.reshape(
        batch, views, dim, latent_frames, height_tokens, width_tokens
    )
    return _patched_video_from_grid(
        patched,
        encoded,
        view_ids=view_ids,
        chunk_stride=chunk_stride,
        temporal_stride=temporal_stride,
    )


def patchify_video_sequence_with_memory(
    dit,
    encoded: EncodedVideoSequence,
    memory: EncodedFirstFrameMemory,
    *,
    view_ids: torch.Tensor,
    memory_grid: tuple[int, int],
    chunk_stride: int,
    temporal_stride: int = 1,
    position_mode: str = "window_origin",
) -> tuple[PatchedVideoSequence, FirstFrameMemoryTokens]:
    """Patch main Video and Anchor Bank in one DMuon boundary call."""

    latents = encoded.latents
    anchor = memory.latents
    slots = anchor.shape[3]
    history_mode = memory.valid is not None
    if not memory.view_keys or len(set(memory.view_keys)) != len(memory.view_keys):
        raise ValueError("memory view keys must be non-empty and unique")
    missing_view_keys = tuple(
        key for key in memory.view_keys if key not in encoded.view_keys
    )
    if missing_view_keys:
        raise ValueError(
            "memory views are absent from the main Video stream: "
            f"{missing_view_keys}; available={encoded.view_keys}"
        )
    memory_view_indices = tuple(
        encoded.view_keys.index(key) for key in memory.view_keys
    )
    if (
        anchor.shape[0] != latents.shape[0]
        or anchor.shape[1] != len(memory_view_indices)
        or anchor.shape[2] != latents.shape[2]
        or (slots != 1 and not history_mode)
    ):
        raise ValueError(
            "memory/main latent batch, selected-view, or channel shape differs"
        )
    if anchor.shape[-2:] != latents.shape[-2:]:
        raise ValueError("memory/main latent spatial geometry differs")
    batch, views, channels, frames, height, width = latents.shape
    if tuple(view_ids.shape) != (batch, views):
        raise ValueError(
            f"view_ids must be {(batch, views)}, got {tuple(view_ids.shape)}"
        )
    if position_mode not in {"window_origin", "relative_age"}:
        raise ValueError(
            "first-frame memory position_mode must be window_origin or " "relative_age"
        )
    memory_age_steps = None
    if position_mode == "relative_age":
        if memory.age_steps is None:
            raise ValueError(
                "relative-age first-frame Memory PE requires one age_steps "
                "value per sample"
            )
        memory_age_steps = memory.age_steps.to(
            device=latents.device, dtype=torch.float32
        )
        expected_age_shape = (batch, slots) if history_mode else (batch,)
        if tuple(memory_age_steps.shape) != expected_age_shape:
            raise ValueError(
                f"first-frame memory age_steps must be {(batch,)}, got "
                f"{tuple(memory_age_steps.shape)}"
            )
        if not torch.isfinite(memory_age_steps).all() or bool(
            (memory_age_steps < 0).any().item()
        ):
            raise ValueError(
                "first-frame memory age_steps must be finite and non-negative"
            )
    # DMuon requires a single patchify boundary.  Populate the selected main
    # view slots with real Anchor latents and use throwaway current-I latents
    # for unselected slots; those placeholder patches are discarded below.
    if history_mode and (memory.valid.dtype != torch.bool or tuple(memory.valid.shape) != (batch, slots)):
        raise ValueError("History validity must be bool [B,M]")
    anchor_by_main_view = latents[:, :, :, :1].expand(-1, -1, -1, slots, -1, -1).clone()
    for memory_index, main_index in enumerate(memory_view_indices):
        anchor_by_main_view[:, main_index] = anchor[:, memory_index]
    combined = torch.cat((anchor_by_main_view, latents), dim=3).reshape(
        batch * views, channels, frames + slots, height, width
    )
    patched = dit.patchify(combined)
    _, dim, patched_frames, height_tokens, width_tokens = patched.shape
    if patched_frames != frames + slots:
        raise RuntimeError("Wan patchify unexpectedly changed temporal length")
    patched = patched.reshape(
        batch, views, dim, patched_frames, height_tokens, width_tokens
    )
    anchor_patches = patched[:, memory_view_indices, :, :slots]
    main_patches = patched[:, :, :, slots:]
    main = _patched_video_from_grid(
        main_patches,
        encoded,
        view_ids=view_ids,
        chunk_stride=chunk_stride,
        temporal_stride=temporal_stride,
    )

    pool_h, pool_w = (int(memory_grid[0]), int(memory_grid[1]))
    if pool_h > height_tokens or pool_w > width_tokens:
        raise ValueError(
            "first_frame_memory_grid cannot exceed the Video patch grid: "
            f"{memory_grid} vs {(height_tokens, width_tokens)}"
        )
    memory_views = len(memory_view_indices)
    pooled = F.adaptive_avg_pool2d(
        anchor_patches.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * memory_views * slots, dim, height_tokens, width_tokens
        ),
        (pool_h, pool_w),
    ).reshape(batch, memory_views, slots, dim, pool_h, pool_w)
    memory_tokens = rearrange(pooled, "b v m d h w -> b (v m h w) d").contiguous()
    memory_view_ids = view_ids[:, memory_view_indices]
    memory_camera_ids = (
        memory_view_ids[:, :, None, None, None]
        .expand(batch, memory_views, slots, pool_h, pool_w)
        .flatten(1)
        .to(device=memory_tokens.device, dtype=torch.long)
    )
    # Preserve approximate spatial alignment after adaptive pooling by using
    # the source patch-bin centers rather than pretending the bank is a new
    # contiguous 4x4 image coordinate system.
    h_centers = (
        torch.floor(
            (torch.arange(pool_h, device=memory_tokens.device) + 0.5)
            * height_tokens
            / pool_h
        )
        .long()
        .clamp_max(height_tokens - 1)
    )
    w_centers = (
        torch.floor(
            (torch.arange(pool_w, device=memory_tokens.device) + 0.5)
            * width_tokens
            / pool_w
        )
        .long()
        .clamp_max(width_tokens - 1)
    )
    hh, ww = torch.meshgrid(h_centers, w_centers, indexing="ij")
    per_view_positions = torch.stack(
        (
            torch.zeros_like(hh),
            torch.zeros_like(hh),
            hh,
            ww,
        ),
        dim=-1,
    ).reshape(1, pool_h * pool_w, 4)
    memory_positions = per_view_positions.repeat(batch, memory_views * slots, 1).reshape(
        batch, memory_views * slots * pool_h * pool_w, 4
    )
    if memory_age_steps is not None:
        # Main IVAVA tokens remain window-local.  Placing the immutable
        # episode anchor at -age is RoPE-equivalent to shifting the complete
        # current window forward by query_local_idx, but does not perturb the
        # pretrained/current-window position contract.
        memory_positions = memory_positions.to(dtype=torch.float32)
        memory_positions[..., 0] = (
            -memory_age_steps[:, None, :, None].expand(batch, memory_views, slots, pool_h * pool_w).reshape(batch, -1)
            if history_mode else -memory_age_steps[:, None]
        )
    token_valid = (memory.valid[:, None, :, None].expand(batch, memory_views, slots, pool_h * pool_w).reshape(batch, -1)
                   if history_mode else None)
    if token_valid is not None:
        memory_tokens = memory_tokens.masked_fill(~token_valid[..., None], 0)
    return main, FirstFrameMemoryTokens(
        tokens=memory_tokens,
        position_ids=memory_positions,
        camera_ids=memory_camera_ids,
        grid=(pool_h, pool_w),
        num_views=memory_views,
        valid=token_valid,
    )


def unpatchify_video_sequence(
    dit, patch_velocity: torch.Tensor, patched: PatchedVideoSequence
) -> torch.Tensor:
    """Return continuous latent velocity in ``[B,V,C,F,H,W]`` order."""

    batch = patch_velocity.shape[0]
    frames, height, width = patched.grid
    condition = patch_velocity[:, : patched.condition_token_count]
    future = patch_velocity[:, patched.condition_token_count :]
    future_frames_per_chunk = (frames - 1) // patched.num_chunks
    expected_future = (
        patched.num_chunks
        * patched.num_views
        * future_frames_per_chunk
        * height
        * width
    )
    if future.shape[1] != expected_future:
        raise ValueError(
            f"future patch token length {future.shape[1]} != {expected_future}"
        )
    condition = rearrange(
        condition,
        "b (v f h w) d -> (b v) (f h w) d",
        v=patched.num_views,
        f=1,
        h=height,
        w=width,
    )
    future = rearrange(
        future,
        "b (e v f h w) d -> (b v) (e f h w) d",
        e=patched.num_chunks,
        v=patched.num_views,
        f=future_frames_per_chunk,
        h=height,
        w=width,
    )
    flat = torch.cat((condition, future), dim=1)
    latent_velocity = dit.unpatchify(flat, patched.grid)
    return latent_velocity.reshape(batch, patched.num_views, *latent_velocity.shape[1:])


__all__ = [
    "EncodedFirstFrameMemory",
    "EncodedVideoSequence",
    "PatchedVideoSequence",
    "encode_video_sequence",
    "encode_first_frame_memory",
    "patchify_video_sequence",
    "patchify_video_sequence_with_memory",
    "unpatchify_video_sequence",
]
