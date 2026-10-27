import logging
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from policy_core.utils.WAM_policy.data_utils import decode_latents


logger = logging.getLogger(__name__)


@dataclass
class MixedVideoLayout:
    frame_token_counts: torch.LongTensor
    frame_offsets: torch.LongTensor
    frame_latent_hw: torch.LongTensor
    frame_is_sparse: torch.BoolTensor
    video_rope_coords: torch.FloatTensor

    @property
    def total_frames(self) -> int:
        return int(self.frame_token_counts.numel())

    @property
    def total_tokens(self) -> int:
        return int(self.frame_offsets[-1].item())


def normalize_sparse_downsample_factor(sparse_downsample_factor: int) -> int:
    factor = int(sparse_downsample_factor)
    if factor < 1:
        raise ValueError(f"video_sparse_downsample_factor must be >= 1, got {sparse_downsample_factor}.")
    return factor


def normalize_sparse_spatial_rope_mode(sparse_spatial_rope_mode: str) -> str:
    mode = str(sparse_spatial_rope_mode).strip().lower()
    if mode in {"continuous", "center", "centered"}:
        return "continuous"
    if mode in {"discrete", "grid"}:
        return "discrete"
    raise ValueError(
        "video_sparse_spatial_rope_mode must be 'continuous' or 'discrete'. "
        f"Got {sparse_spatial_rope_mode!r}."
    )


def centered_spatial_rope_axis(
    size: int,
    canonical_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    size = int(size)
    canonical_size = int(canonical_size)
    if size <= 0 or canonical_size <= 0:
        raise ValueError(
            f"Spatial RoPE sizes must be positive, got size={size}, canonical_size={canonical_size}."
        )

    axis = torch.arange(size, device=device, dtype=torch.float32)
    if size == canonical_size:
        return axis

    scale = canonical_size / size
    return axis * scale + (scale - 1.0) / 2.0


def validate_mixed_video_sparse_mode(
    *,
    sparse_downsample_factor: int,
    sparse_spatial_rope_mode: str = "continuous",
    force_mixed_path: bool = False,
    vae_encoding_mode: str,
    patch_size: int,
    patch_size_t: int,
    use_joint_ae_handoff: bool = False,
    frame_indices: Optional[torch.Tensor] = None,
    dense_frame_count: Optional[int] = None,
    require_infer_schedule: bool = False,
) -> int:
    factor = normalize_sparse_downsample_factor(sparse_downsample_factor)
    normalize_sparse_spatial_rope_mode(sparse_spatial_rope_mode)
    if factor == 1 and not force_mixed_path:
        return factor

    if str(vae_encoding_mode).lower().strip() != "2dvae":
        raise ValueError(
            "Mixed sparse video mode only supports vae_encoding_mode == '2dvae'. "
            f"Got {vae_encoding_mode!r}."
        )
    if int(patch_size) != 1 or int(patch_size_t) != 1:
        raise ValueError(
            "Mixed sparse video mode requires transformer patch sizes of 1. "
            f"Got patch_size={patch_size}, patch_size_t={patch_size_t}."
        )
    if use_joint_ae_handoff:
        raise ValueError(
            "Mixed sparse video mode does not support passing video-gen latents into the action branch."
        )
    if require_infer_schedule and frame_indices is None:
        raise ValueError(
            "Mixed sparse video mode requires explicit frame_indices at inference/validation time."
        )
    if require_infer_schedule and dense_frame_count is None:
        raise ValueError(
            "Mixed sparse video mode requires dense_frame_count at inference/validation time."
        )
    return factor


def expand_frame_indices_for_views(
    frame_indices: Optional[torch.Tensor],
    *,
    batch_size: int,
    n_view: int,
    device: torch.device,
) -> Optional[torch.LongTensor]:
    if frame_indices is None:
        return None

    frame_indices = torch.as_tensor(frame_indices, device=device)
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0)

    if frame_indices.shape[0] == batch_size:
        return frame_indices.repeat_interleave(n_view, dim=0).long()
    if frame_indices.shape[0] == batch_size * n_view:
        return frame_indices.long()
    if frame_indices.shape[0] == 1:
        return frame_indices.repeat(batch_size * n_view, 1).long()

    raise ValueError(
        "frame_indices must have leading dimension 1, batch_size, or batch_size * n_view. "
        f"Got shape {tuple(frame_indices.shape)} for batch_size={batch_size}, n_view={n_view}."
    )


def bilinear_downsample_video(video: torch.Tensor, factor: int) -> torch.Tensor:
    factor = normalize_sparse_downsample_factor(factor)
    if factor == 1:
        return video
    if video.shape[-2] % factor != 0 or video.shape[-1] % factor != 0:
        raise ValueError(
            f"Expected video size divisible by sparse factor {factor}, got {tuple(video.shape[-2:])}."
        )

    height = video.shape[-2] // factor
    width = video.shape[-1] // factor
    if video.ndim == 5:
        batch_size, channels, num_frames, _, _ = video.shape
        flat = rearrange(video, "b c f h w -> (b f) c h w")
        flat = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
        return rearrange(flat, "(b f) c h w -> b c f h w", b=batch_size, f=num_frames)
    if video.ndim == 4:
        return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    raise ValueError(f"Expected a 4D or 5D tensor for video downsample, got shape {tuple(video.shape)}.")


def _encode_2dvae_frames(vae, frames: torch.Tensor, batch_size_2dvae: int, stats_tracker=None) -> torch.Tensor:
    encoded = []
    for start in range(0, frames.shape[0], batch_size_2dvae):
        end = min(start + batch_size_2dvae, frames.shape[0])
        z = vae.encode(frames[start:end]).latent
        z = stats_tracker.normalize(z) if stats_tracker is not None else z * vae.config.scaling_factor
        encoded.append(z)
    return torch.cat(encoded, dim=0)


def encode_mixed_video_2dvae(
    vae,
    mem: torch.Tensor,
    future_video: torch.Tensor,
    *,
    dense_frame_count: int,
    sparse_downsample_factor: int,
    batch_size_2dvae: int,
    stats_tracker=None,
) -> dict:
    factor = normalize_sparse_downsample_factor(sparse_downsample_factor)
    dense_frame_count = min(int(dense_frame_count), int(future_video.shape[2]))
    sparse_frame_count = int(future_video.shape[2]) - dense_frame_count

    mem_size = mem.shape[2]
    batch_size = mem.shape[0]

    mem_frames = rearrange(mem, "b c f h w -> (b f) c h w")
    mem_latents = _encode_2dvae_frames(vae, mem_frames, batch_size_2dvae, stats_tracker=stats_tracker)
    mem_latents = rearrange(mem_latents, "(b f) c h w -> b c f h w", b=batch_size, f=mem_size)

    dense_latents = None
    if dense_frame_count > 0:
        dense_frames = rearrange(future_video[:, :, :dense_frame_count], "b c f h w -> (b f) c h w")
        dense_latents = _encode_2dvae_frames(vae, dense_frames, batch_size_2dvae, stats_tracker=stats_tracker)
        dense_latents = rearrange(
            dense_latents, "(b f) c h w -> b c f h w", b=batch_size, f=dense_frame_count
        )

    sparse_latents = None
    sparse_init_latents = None
    if sparse_frame_count > 0:
        sparse_video = bilinear_downsample_video(future_video[:, :, dense_frame_count:], factor)
        sparse_frames = rearrange(sparse_video, "b c f h w -> (b f) c h w")
        sparse_latents = _encode_2dvae_frames(vae, sparse_frames, batch_size_2dvae, stats_tracker=stats_tracker)
        sparse_latents = rearrange(
            sparse_latents, "(b f) c h w -> b c f h w", b=batch_size, f=sparse_frame_count
        )

        sparse_mem = bilinear_downsample_video(mem[:, :, :1], factor)
        sparse_mem = rearrange(sparse_mem, "b c f h w -> (b f) c h w")
        sparse_init_latents = _encode_2dvae_frames(vae, sparse_mem, batch_size_2dvae, stats_tracker=stats_tracker)
        sparse_init_latents = rearrange(sparse_init_latents, "(b f) c h w -> b c f h w", b=batch_size, f=1)

    return {
        "mem_latents": mem_latents,
        "dense_latents": dense_latents,
        "sparse_latents": sparse_latents,
        "sparse_init_latents": sparse_init_latents,
        "dense_frame_count": dense_frame_count,
        "sparse_frame_count": sparse_frame_count,
    }


def pack_frame_latent_groups(
    mem_latents: torch.Tensor,
    dense_latents: Optional[torch.Tensor] = None,
    sparse_latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    groups = [rearrange(mem_latents, "b c f h w -> b (f h w) c")]
    if dense_latents is not None and dense_latents.shape[2] > 0:
        groups.append(rearrange(dense_latents, "b c f h w -> b (f h w) c"))
    if sparse_latents is not None and sparse_latents.shape[2] > 0:
        groups.append(rearrange(sparse_latents, "b c f h w -> b (f h w) c"))
    return torch.cat(groups, dim=1)


def build_frame_latent_spec(
    *,
    mem_frame_count: int,
    dense_frame_count: int,
    sparse_frame_count: int,
    full_latent_hw: Sequence[int],
    sparse_latent_hw: Optional[Sequence[int]],
    device: torch.device,
    sparse_downsample_factor: int,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    factor = normalize_sparse_downsample_factor(sparse_downsample_factor)
    full_h, full_w = int(full_latent_hw[0]), int(full_latent_hw[1])

    frame_latent_hw = [[full_h, full_w] for _ in range(mem_frame_count + dense_frame_count)]
    frame_is_sparse = [False for _ in range(mem_frame_count + dense_frame_count)]

    if sparse_frame_count > 0:
        if sparse_latent_hw is None:
            if full_h % factor != 0 or full_w % factor != 0:
                raise ValueError(
                    f"Expected latent size divisible by sparse factor {factor}, got {(full_h, full_w)}."
                )
            sparse_h, sparse_w = full_h // factor, full_w // factor
        else:
            sparse_h, sparse_w = int(sparse_latent_hw[0]), int(sparse_latent_hw[1])
        frame_latent_hw.extend([[sparse_h, sparse_w] for _ in range(sparse_frame_count)])
        frame_is_sparse.extend([factor > 1 for _ in range(sparse_frame_count)])

    frame_latent_hw = torch.tensor(frame_latent_hw, device=device, dtype=torch.long)
    frame_is_sparse = torch.tensor(frame_is_sparse, device=device, dtype=torch.bool)
    return frame_latent_hw, frame_is_sparse


def build_mixed_video_layout(
    *,
    frame_indices: torch.Tensor,
    frame_latent_hw: torch.LongTensor,
    frame_is_sparse: torch.BoolTensor,
    sparse_downsample_factor: int,
    sparse_spatial_rope_mode: str = "continuous",
) -> MixedVideoLayout:
    factor = normalize_sparse_downsample_factor(sparse_downsample_factor)
    sparse_spatial_rope_mode = normalize_sparse_spatial_rope_mode(sparse_spatial_rope_mode)
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0)
    if frame_indices.shape[1] != frame_latent_hw.shape[0]:
        raise ValueError(
            f"frame_indices length {frame_indices.shape[1]} does not match frame spec length {frame_latent_hw.shape[0]}."
        )

    device = frame_indices.device
    batch_size = frame_indices.shape[0]
    frame_token_counts = frame_latent_hw[:, 0] * frame_latent_hw[:, 1]
    frame_offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.long), frame_token_counts.cumsum(dim=0)],
        dim=0,
    )

    rebased_frame_indices = torch.arange(
        frame_indices.shape[1], device=device, dtype=torch.float32
    ).unsqueeze(0).expand(batch_size, -1)
    rope_chunks = []
    for frame_idx in range(frame_latent_hw.shape[0]):
        latent_h = int(frame_latent_hw[frame_idx, 0].item())
        latent_w = int(frame_latent_hw[frame_idx, 1].item())

        if bool(frame_is_sparse[frame_idx].item()):
            if sparse_spatial_rope_mode == "continuous":
                ys = centered_spatial_rope_axis(latent_h, latent_h * factor, device=device)
                xs = centered_spatial_rope_axis(latent_w, latent_w * factor, device=device)
            else:
                ys = torch.arange(latent_h, device=device, dtype=torch.float32) * factor
                xs = torch.arange(latent_w, device=device, dtype=torch.float32) * factor
        else:
            ys = torch.arange(latent_h, device=device, dtype=torch.float32)
            xs = torch.arange(latent_w, device=device, dtype=torch.float32)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        time_coord = rebased_frame_indices[:, frame_idx][:, None].expand(batch_size, latent_h * latent_w)
        y_coord = grid_y.reshape(1, -1).expand(batch_size, -1)
        x_coord = grid_x.reshape(1, -1).expand(batch_size, -1)
        rope_chunks.append(torch.stack((time_coord, y_coord, x_coord), dim=-1))

    video_rope_coords = torch.cat(rope_chunks, dim=1)
    return MixedVideoLayout(
        frame_token_counts=frame_token_counts,
        frame_offsets=frame_offsets,
        frame_latent_hw=frame_latent_hw,
        frame_is_sparse=frame_is_sparse,
        video_rope_coords=video_rope_coords,
    )


def build_mixed_conditioning_tokens(
    *,
    mem_latents: torch.Tensor,
    dense_frame_count: int,
    sparse_frame_count: int,
    sparse_init_latents: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    noise_to_condition_frames: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_frame_count = int(dense_frame_count)
    sparse_frame_count = int(sparse_frame_count)
    batch_size = mem_latents.shape[0]

    mem_tokens = rearrange(mem_latents, "b c f h w -> b (f h w) c")
    groups = [mem_tokens]

    if dense_frame_count > 0:
        dense_template = mem_latents[:, :, :1].repeat(1, 1, dense_frame_count, 1, 1)
        groups.append(rearrange(dense_template, "b c f h w -> b (f h w) c"))

    if sparse_frame_count > 0:
        if sparse_init_latents is None:
            raise ValueError("sparse_init_latents is required when sparse_frame_count > 0.")
        sparse_template = sparse_init_latents.repeat(1, 1, sparse_frame_count, 1, 1)
        groups.append(rearrange(sparse_template, "b c f h w -> b (f h w) c"))

    init_latents = torch.cat(groups, dim=1)
    conditioning_mask = init_latents.new_zeros((batch_size, init_latents.shape[1]))
    conditioning_mask[:, : mem_tokens.shape[1]] = 1.0

    rand_noise_ff = random.random() * noise_to_condition_frames
    memory_mask = conditioning_mask * (1.0 - rand_noise_ff)

    noise = randn_tensor(
        init_latents.shape,
        generator=generator,
        device=init_latents.device,
        dtype=init_latents.dtype,
    )
    latents = init_latents * memory_mask.unsqueeze(-1) + noise * (1.0 - memory_mask.unsqueeze(-1))
    cond_indicator = conditioning_mask.clone()
    return latents, conditioning_mask, cond_indicator


def unpack_frame_token_group(
    tokens: torch.Tensor,
    *,
    frame_offsets: torch.LongTensor,
    frame_latent_hw: torch.LongTensor,
    frame_start: int,
    frame_count: int,
) -> Optional[torch.Tensor]:
    frame_count = int(frame_count)
    if frame_count <= 0:
        return None

    group_hw = frame_latent_hw[frame_start : frame_start + frame_count]
    if not torch.equal(group_hw, group_hw[:1].expand_as(group_hw)):
        raise ValueError("All frames inside a decoded group must share the same latent resolution.")

    start_offset = int(frame_offsets[frame_start].item())
    end_offset = int(frame_offsets[frame_start + frame_count].item())
    latent_h, latent_w = group_hw[0].tolist()
    return rearrange(
        tokens[:, start_offset:end_offset],
        "b (f h w) c -> b c f h w",
        f=frame_count,
        h=int(latent_h),
        w=int(latent_w),
    )


def upsample_video_bilinear(video: Optional[torch.Tensor], *, target_height: int, target_width: int) -> Optional[torch.Tensor]:
    if video is None:
        return None
    batch_size, channels, num_frames = video.shape[:3]
    flat = rearrange(video, "b c f h w -> (b f) c h w")
    flat = F.interpolate(flat, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return rearrange(flat, "(b f) c h w -> b c f h w", b=batch_size, f=num_frames)


@torch.no_grad()
def roundtrip_video_2dvae(
    vae,
    video: torch.Tensor,
    *,
    n_prev: int,
    dense_frame_count: int,
    sparse_downsample_factor: int,
    mixed_mode_active: bool,
    batch_size_2dvae: int,
    stats_tracker=None,
) -> torch.Tensor:
    if str(getattr(vae, "vae_encoding_mode", "3dvae")).lower().strip() != "2dvae":
        logger.warning(
            "Saving raw validation GT video: VAE round-trip reference is currently "
            "implemented only for vae_encoding_mode='2dvae'."
        )
        return video

    batch_size, _, n_view, _, height, width = video.shape
    video_bv = rearrange(video, "b c v t h w -> (b v) c t h w")
    mem = video_bv[:, :, :n_prev]
    future_video = video_bv[:, :, n_prev:]
    future_frame_count = future_video.shape[2]
    encoded = encode_mixed_video_2dvae(
        vae,
        mem,
        future_video,
        dense_frame_count=(
            min(int(dense_frame_count), int(future_frame_count))
            if mixed_mode_active else int(future_frame_count)
        ),
        sparse_downsample_factor=(
            sparse_downsample_factor if mixed_mode_active else 1
        ),
        batch_size_2dvae=batch_size_2dvae,
        stats_tracker=stats_tracker,
    )

    decoded_groups = [decode_latents(vae, encoded["mem_latents"], stats_tracker=stats_tracker)]
    if encoded["dense_latents"] is not None:
        decoded_groups.append(decode_latents(vae, encoded["dense_latents"], stats_tracker=stats_tracker))
    if encoded["sparse_latents"] is not None:
        decoded_sparse = decode_latents(vae, encoded["sparse_latents"], stats_tracker=stats_tracker)
        decoded_groups.append(
            upsample_video_bilinear(
                decoded_sparse, target_height=height, target_width=width,
            )
        )

    decoded = torch.cat(decoded_groups, dim=2)
    return rearrange(
        decoded, "(b v) c t h w -> b c v t h w", b=batch_size, v=n_view,
    )
