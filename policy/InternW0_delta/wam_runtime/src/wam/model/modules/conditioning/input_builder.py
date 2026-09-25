from contextlib import nullcontext
from typing import Any, Optional, Sequence

import torch

from wam.cache.codec import EncodedTensor, materialize_cache_tensor

from ..codecs import video_latent_codec as video_codec
from ..codecs.utils import video_batch_size, video_num_frames, video_spatial_shape
from ..memory.proprio_encoder import append_proprio_to_context


def _profile_section(profiler, name: str):
    """Return a low-overhead nested profiler section when available."""
    return profiler.section(name) if profiler is not None else nullcontext()


def _move_vlm_images_to_device(
    images: Any,
    *,
    device: torch.device | str,
) -> Any:
    """Move raw VLM images to the target device while preserving uint8 storage."""
    if images is None:
        return None
    if isinstance(images, torch.Tensor):
        return images.to(device=device, non_blocking=True)
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        moved = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Each `vlm_current_images` item must be a torch.Tensor, "
                    f"got {type(image)}."
                )
            moved.append(image.to(device=device, non_blocking=True))
        return moved
    raise TypeError(
        "`vlm_current_images` must be a tensor or sequence of tensors, "
        f"got {type(images)}."
    )


def _zero_padded_feature_dimensions(
    value: torch.Tensor,
    dim_is_pad: Optional[torch.Tensor],
    *,
    name: str,
) -> torch.Tensor:
    """Zero sample-specific missing slots in a shared canonical feature space."""
    if dim_is_pad is None:
        return value
    pad = dim_is_pad.to(device=value.device, dtype=torch.bool, non_blocking=True)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    elif pad.ndim == 2 and int(pad.shape[0]) == 1 and int(value.shape[0]) > 1:
        pad = pad.expand(int(value.shape[0]), -1)
    if pad.ndim != 2 or tuple(pad.shape) != (
        int(value.shape[0]),
        int(value.shape[-1]),
    ):
        raise ValueError(
            f"`{name}` must be [D] or [B,D] and match {tuple(value.shape)}; "
            f"got {tuple(pad.shape)}."
        )
    return value.masked_fill(pad.unsqueeze(1), 0.0)


def prepare_memory_proprio_inputs(
    model,
    sample: dict[str, Any],
    *,
    name: str,
    batch_size: int,
    dim_is_pad: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    proprio = sample.get(name, None)
    if proprio is None:
        return None, None
    if model.proprio_dim is None or model.proprio_encoder is None:
        raise ValueError(f"`{name}` was provided but proprio encoder is disabled.")
    if proprio.ndim == 2:
        proprio = proprio.unsqueeze(0)
    elif proprio.ndim == 3 and int(proprio.shape[0]) == 1 and batch_size > 1:
        proprio = proprio.expand(batch_size, -1, -1)
    if (
        proprio.ndim != 3
        or int(proprio.shape[0]) != int(batch_size)
        or int(proprio.shape[2]) != int(model.proprio_dim)
    ):
        raise ValueError(
            f"`{name}` must be [T,D] or [B,T,D], got {tuple(proprio.shape)} "
            f"for batch={batch_size}, proprio_dim={model.proprio_dim}."
        )
    proprio = proprio.to(
        device=model.device, dtype=model.torch_dtype, non_blocking=True
    )
    proprio = _zero_padded_feature_dimensions(
        proprio,
        dim_is_pad,
        name="sample['proprio_dim_is_pad']",
    )
    pad = sample.get(f"{name}_is_pad", None)
    if pad is None:
        pad = torch.zeros(
            (batch_size, int(proprio.shape[1])), dtype=torch.bool, device=model.device
        )
    else:
        if pad.ndim == 1:
            pad = pad.unsqueeze(0).expand(batch_size, -1)
        elif pad.ndim == 2 and int(pad.shape[0]) == 1 and batch_size > 1:
            pad = pad.expand(batch_size, -1)
        if pad.ndim != 2 or tuple(pad.shape) != tuple(proprio.shape[:2]):
            raise ValueError(
                f"`{name}_is_pad` must be [T] or [B,T], got {tuple(pad.shape)} "
                f"for proprio shape {tuple(proprio.shape)}."
            )
        pad = pad.to(device=model.device, dtype=torch.bool, non_blocking=True)
    return proprio, pad


def prepare_memory_video_inputs(
    model,
    sample: dict[str, Any],
    *,
    name: str,
    batch_size: int,
    height: int,
    width: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    video = sample.get(name, None)
    pad = sample.get(f"{name}_is_pad", None)
    if video is None:
        return None, None
    if video.ndim == 4:
        video = video.unsqueeze(0)
    elif (
        video.ndim == 5
        and int(video.shape[1]) == 3
        and int(video.shape[0]) != int(batch_size)
    ):
        video = video.unsqueeze(0)
    if video.ndim == 5:
        if int(video.shape[0]) != int(batch_size) or int(video.shape[1]) != 3:
            raise ValueError(
                f"`sample['{name}']` shape mismatch: got {tuple(video.shape)} vs batch={batch_size}, channels=3"
            )
    elif video.ndim == 6:
        if int(video.shape[0]) != int(batch_size) or int(video.shape[2]) != 3:
            raise ValueError(
                f"`sample['{name}']` shape mismatch: got {tuple(video.shape)} vs batch={batch_size}, channels=3"
            )
    else:
        raise ValueError(
            f"`sample['{name}']` must be [B,3,K,H,W] or [B,N,3,K,H,W], got {tuple(video.shape)}"
        )
    if int(video.shape[-2]) != int(height) or int(video.shape[-1]) != int(width):
        raise ValueError(
            f"`sample['{name}']` spatial shape must match current video HxW=({height},{width}), "
            f"got {tuple(video.shape[-2:])}"
        )
    video = video.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
    num_frames = video_num_frames(video)
    if pad is None:
        pad = torch.zeros(
            (batch_size, num_frames), dtype=torch.bool, device=model.device
        )
    else:
        if pad.ndim == 1:
            pad = pad.unsqueeze(0).expand(batch_size, -1)
        elif pad.ndim == 2 and pad.shape[0] == 1 and batch_size > 1:
            pad = pad.expand(batch_size, -1)
        if pad.ndim != 2 or tuple(pad.shape) != (batch_size, num_frames):
            raise ValueError(
                f"`sample['{name}_is_pad']` must be [K] or [B,K], got {tuple(pad.shape)}"
            )
        pad = pad.to(device=model.device, dtype=torch.bool, non_blocking=True)
    return video, pad


def prepare_cached_memory_latents(
    model,
    sample: dict[str, Any],
    *,
    name: str,
    batch_size: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    latents = sample.get(f"{name}_latents", None)
    if latents is None:
        return None, None
    if not isinstance(latents, (torch.Tensor, EncodedTensor)) or latents.ndim != 5:
        raise ValueError(
            f"`sample['{name}_latents']` must be [B,C,1,H,W], got "
            f"{type(latents)} {getattr(latents, 'shape', None)}."
        )
    if int(latents.shape[0]) != int(batch_size) or int(latents.shape[2]) != 1:
        raise ValueError(
            f"`sample['{name}_latents']` must have batch={batch_size} and one "
            f"latent frame, got {tuple(latents.shape)}."
        )
    latents = materialize_cache_tensor(
        latents,
        device=model.device,
        dtype=model.torch_dtype,
        non_blocking=True,
    )
    pad = sample.get(f"{name}_is_pad", None)
    if pad is None:
        pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=model.device)
    else:
        if not isinstance(pad, torch.Tensor):
            pad = torch.as_tensor(pad)
        if pad.ndim == 1:
            pad = pad.unsqueeze(0).expand(batch_size, -1)
        elif pad.ndim == 2 and int(pad.shape[0]) == 1 and batch_size > 1:
            pad = pad.expand(batch_size, -1)
        if pad.ndim != 2 or tuple(pad.shape) != (batch_size, 1):
            raise ValueError(
                f"`sample['{name}_is_pad']` must be [1] or [B,1], got "
                f"{tuple(pad.shape)}."
            )
        pad = pad.to(device=model.device, dtype=torch.bool, non_blocking=True)
    return latents, pad


def _resolve_single_memory_latent(
    model,
    inputs: dict[str, Any],
    *,
    name: str,
    tiled: bool,
    profiler=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one clean memory latent from an injected value or online RGB."""

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
    with _profile_section(profiler, "vae_forward"):
        latents = video_codec.encode_video_latents(
            model,
            video,
            tiled=tiled,
            video_layout=inputs.get("video_layout"),
            video_view_names=inputs.get("video_view_names"),
        )
    if int(latents.shape[2]) != 1:
        raise ValueError(
            f"{name} must encode to one latent frame, got {tuple(latents.shape)}."
        )
    pad = inputs.get(f"{name}_is_pad")
    if pad is None:
        pad = torch.zeros(
            (int(latents.shape[0]), 1),
            dtype=torch.bool,
            device=model.device,
        )
    return latents, pad[:, :1].to(device=model.device, dtype=torch.bool)


def resolve_frame_memory_latents(
    model,
    inputs: dict[str, Any],
    *,
    tiled: bool,
    profiler=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clean anchor/recent latents without exposing source policy."""

    resolved: list[torch.Tensor] = []
    for short_name, input_name in (
        ("memory_anchor", "memory_video_anchor"),
        ("memory_recent", "memory_video_recent"),
    ):
        source = (
            "cache_use"
            if inputs.get(f"{input_name}_latents") is not None
            else "vae_encode"
        )
        with _profile_section(profiler, f"{short_name}_{source}"):
            latents, pad = _resolve_single_memory_latent(
                model,
                inputs,
                name=input_name,
                tiled=tiled,
                profiler=profiler,
            )
        resolved.extend((latents, pad))
    return resolved[0], resolved[1], resolved[2], resolved[3]


def prepare_memory_frame_id_inputs(
    model, sample: dict[str, Any], *, name: str, batch_size: int
) -> Optional[torch.Tensor]:
    frame_ids = sample.get(f"{name}_frame_ids", None)
    if frame_ids is None:
        return None
    if not torch.is_tensor(frame_ids):
        frame_ids = torch.as_tensor(frame_ids)
    if frame_ids.ndim == 1:
        frame_ids = frame_ids.unsqueeze(0).expand(batch_size, -1)
    elif frame_ids.ndim == 2 and int(frame_ids.shape[0]) == 1 and batch_size > 1:
        frame_ids = frame_ids.expand(batch_size, -1)
    if frame_ids.ndim != 2 or int(frame_ids.shape[0]) != int(batch_size):
        raise ValueError(
            f"`sample['{name}_frame_ids']` must be [K] or [B,K], got {tuple(frame_ids.shape)} "
            f"for batch={batch_size}."
        )
    return frame_ids.to(device=model.device, dtype=torch.long, non_blocking=True)


def build_wam_inputs(model, sample, tiled: bool = False, profiler=None):
    video = sample.get("video")
    cached_latents = sample.get("video_latents", None)
    understanding_enabled = bool(getattr(model, "understanding_enabled", False))
    if "context" not in sample or "context_mask" not in sample:
        raise ValueError(
            "Video T5 conditioning requires `sample['context']` and "
            "`sample['context_mask']`."
        )
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    proprio = sample.get("proprio", None)
    proprio_dim_is_pad = sample.get("proprio_dim_is_pad", None)
    if video is None:
        if not isinstance(cached_latents, (torch.Tensor, EncodedTensor)):
            raise ValueError(
                "A sample without `video` requires cached `video_latents`."
            )
        if cached_latents.ndim != 5:
            raise ValueError(
                "Cached `video_latents` must be [B,C,T,H,W], got "
                f"{tuple(cached_latents.shape)}"
            )
        video_shape = sample.get("video_shape")
        if not isinstance(video_shape, torch.Tensor):
            raise TypeError(
                "A latent-only sample requires tensor `video_shape` metadata."
            )
        if video_shape.ndim == 1:
            video_shape = video_shape.unsqueeze(0)
        if video_shape.ndim != 2 or int(video_shape.shape[1]) != 4:
            raise ValueError(
                "`sample['video_shape']` must be [B,4] with [C,T,H,W], got "
                f"{tuple(video_shape.shape)}"
            )
        batch_size = int(cached_latents.shape[0])
        if int(video_shape.shape[0]) != batch_size:
            raise ValueError(
                "Latent-only `video_shape` batch mismatch: "
                f"shape_rows={int(video_shape.shape[0])} batch={batch_size}."
            )
        first_video_shape = video_shape[0].to(device="cpu", dtype=torch.long)
        if not bool(
            (
                video_shape.to(device="cpu", dtype=torch.long)
                == first_video_shape
            )
            .all()
            .item()
        ):
            raise ValueError("All latent-only samples in a batch must share video_shape.")
        channels, num_frames, height, width = (
            int(value) for value in first_video_shape.tolist()
        )
        if channels != 3:
            raise ValueError(
                f"Latent-only video channel dimension must be 3, got {channels}."
            )
    else:
        if video.ndim == 5:
            if int(video.shape[1]) != 3:
                raise ValueError(
                    f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
                )
        elif video.ndim == 6:
            if int(video.shape[2]) != 3:
                raise ValueError(
                    f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
                )
        else:
            raise ValueError(
                "`sample['video']` must be [B,3,T,H,W] or [B,N,3,T,H,W], "
                f"got shape {tuple(video.shape)}"
            )

        batch_size = video_batch_size(video)
        num_frames = video_num_frames(video)
        height, width = video_spatial_shape(video)
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
        )
    if num_frames % 4 != 1:
        raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
    if num_frames <= 1:
        raise ValueError(
            f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
        )

    if "action" not in sample:
        raise ValueError("`sample['action']` is required for WAM training.")

    action = sample["action"]
    if action.ndim != 3:
        raise ValueError(
            f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
        )
    action_horizon = int(action.shape[1])
    model._action_token_seq_len_for_mask(action_horizon)
    if action_horizon % (num_frames - 1) != 0:
        raise ValueError(
            f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
        )

    action_hz = None
    if bool(
        getattr(
            getattr(model, "action_expert", None),
            "physical_time_rope_enabled",
            False,
        )
    ):
        raw_action_hz = sample.get("action_hz")
        if raw_action_hz is None:
            raise ValueError(
                "Physical-time Action RoPE requires `sample['action_hz']`."
            )
        action_hz = torch.as_tensor(
            raw_action_hz, dtype=torch.float32, device="cpu"
        )
        if action_hz.ndim == 0:
            action_hz = action_hz.reshape(1)
        elif action_hz.ndim == 2 and int(action_hz.shape[1]) == 1:
            action_hz = action_hz[:, 0]
        elif action_hz.ndim != 1:
            raise ValueError(
                "`sample['action_hz']` must be scalar, [B], or [B,1], "
                f"got {tuple(action_hz.shape)}."
            )
        if int(action_hz.numel()) == 1 and batch_size > 1:
            action_hz = action_hz.expand(batch_size)
        if int(action_hz.numel()) != batch_size:
            raise ValueError(
                "`sample['action_hz']` must contain one value per sample: "
                f"batch={batch_size}, got={int(action_hz.numel())}."
            )
        if not bool(torch.isfinite(action_hz).all().item()) or not bool(
            (action_hz > 0).all().item()
        ):
            raise ValueError(
                "`sample['action_hz']` values must be finite and positive."
            )
        action_hz = action_hz.to(
            device=model.device, dtype=torch.float32, non_blocking=True
        )

    action_is_pad = sample.get("action_is_pad", None)
    if action_is_pad is not None:
        if action_is_pad.ndim != 2:
            raise ValueError(
                f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
            )
        if (
            action_is_pad.shape[0] != batch_size
            or action_is_pad.shape[1] != action_horizon
        ):
            raise ValueError(
                "`sample['action_is_pad']` shape mismatch: "
                f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
            )

    action_dim_is_pad = sample.get("action_dim_is_pad", None)
    if action_dim_is_pad is not None:
        if action_dim_is_pad.ndim == 1:
            action_dim_is_pad = action_dim_is_pad.unsqueeze(0).expand(batch_size, -1)
        elif (
            action_dim_is_pad.ndim == 2
            and action_dim_is_pad.shape[0] == 1
            and batch_size > 1
        ):
            action_dim_is_pad = action_dim_is_pad.expand(batch_size, -1)
        if action_dim_is_pad.ndim != 2:
            raise ValueError(
                f"`sample['action_dim_is_pad']` must be [D] or [B, D], got shape {tuple(action_dim_is_pad.shape)}"
            )
        if (
            action_dim_is_pad.shape[0] != batch_size
            or action_dim_is_pad.shape[1] != action.shape[2]
        ):
            raise ValueError(
                "`sample['action_dim_is_pad']` shape mismatch: "
                f"got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action.shape[2]})"
            )

    image_is_pad = sample.get("image_is_pad", None)
    if image_is_pad is not None:
        if image_is_pad.ndim != 2:
            raise ValueError(
                f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
            )
        if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
            raise ValueError(
                "`sample['image_is_pad']` shape mismatch: "
                f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
            )

    # Qwen uses this validity mask in Python control flow. Keep it on CPU
    # before the general padding mask is moved to CUDA below.
    vlm_current_valid = None
    if understanding_enabled:
        if image_is_pad is None:
            vlm_current_valid = torch.ones((batch_size,), dtype=torch.bool)
        else:
            vlm_current_valid = ~image_is_pad[:, 0].detach().to(
                device="cpu", dtype=torch.bool
            )

    input_video = None
    fuse_flag = bool(
        getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
    )
    with _profile_section(profiler, "current_vae_cache_lookup"):
        if cached_latents is not None:
            if (video is not None and video.ndim not in (5, 6)) or tiled:
                raise ValueError(
                    "Cached current latents require a supported single- or "
                    "multi-view, non-tiled frame-window batch."
                )
            if not isinstance(cached_latents, (torch.Tensor, EncodedTensor)):
                raise TypeError(
                    "`sample['video_latents']` must be a tensor or encoded tensor "
                    "when provided, "
                    f"got {type(cached_latents)}"
                )
            if cached_latents.ndim != 5:
                raise ValueError(
                    "`sample['video_latents']` must be [B,C,T,H,W], got "
                    f"{tuple(cached_latents.shape)}"
                )
            if int(cached_latents.shape[0]) != int(batch_size):
                raise ValueError(
                    "`sample['video_latents']` batch dimension must match video: "
                    f"got {int(cached_latents.shape[0])} vs {batch_size}"
                )

    if cached_latents is not None:
        # DataLoader pin_memory handles the CPU tensor returned by the worker;
        # non_blocking preserves the raw-video path's transfer contract.
        with _profile_section(profiler, "current_vae_cache_h2d"):
            input_latents = materialize_cache_tensor(
                cached_latents,
                device=model.device,
                dtype=model.torch_dtype,
                non_blocking=True,
            )
    else:
        with _profile_section(profiler, "video_h2d"):
            input_video = video.to(
                device=model.device, dtype=model.torch_dtype, non_blocking=True
            )
        with _profile_section(profiler, "current_vae_encode"):
            input_latents = video_codec.encode_video_latents(
                model,
                input_video,
                tiled=tiled,
                video_layout=sample.get("video_layout", None),
                video_view_names=sample.get("video_view_names", None),
            )
    first_frame_latents = input_latents[:, :, 0:1] if fuse_flag else None

    vlm_context_cache = sample.get("vlm_context_cache", None)
    vlm_mask_cache = sample.get("vlm_mask_cache", None)
    vlm_cache_hit_mask = sample.get("vlm_cache_hit_mask", None)
    if vlm_cache_hit_mask is not None:
        if not isinstance(vlm_cache_hit_mask, torch.Tensor):
            raise TypeError("`vlm_cache_hit_mask` must be a tensor when provided.")
        vlm_cache_hit_mask = vlm_cache_hit_mask.detach().to(
            device="cpu", dtype=torch.bool
        ).reshape(-1)
        if int(vlm_cache_hit_mask.numel()) != int(batch_size):
            raise ValueError(
                "`vlm_cache_hit_mask` must have one value per sample: "
                f"got={tuple(vlm_cache_hit_mask.shape)} batch={batch_size}."
            )
    if (vlm_context_cache is None) != (vlm_mask_cache is None):
        raise ValueError("VLM cache context and mask must be supplied together.")
    if vlm_cache_hit_mask is not None:
        expected_hits = int(vlm_cache_hit_mask.sum().item())
        actual_hits = 0 if vlm_context_cache is None else int(vlm_context_cache.shape[0])
        if actual_hits != expected_hits:
            raise ValueError(
                "Compact VLM cache rows must match `vlm_cache_hit_mask`: "
                f"rows={actual_hits} hits={expected_hits}."
            )
    if vlm_context_cache is not None:
        with _profile_section(profiler, "vlm_cache_h2d_dequant"):
            vlm_context_cache = materialize_cache_tensor(
                vlm_context_cache,
                device=model.device,
                dtype=model.torch_dtype,
                non_blocking=True,
            )
            vlm_mask_cache = materialize_cache_tensor(
                vlm_mask_cache,
                device=model.device,
                dtype=torch.bool,
                non_blocking=True,
            )
    vlm_current_images = sample.get("vlm_current_images", None)
    vlm_needs_online = vlm_context_cache is None or bool(
        vlm_cache_hit_mask is not None
        and not bool(vlm_cache_hit_mask.all().item())
    )
    if understanding_enabled and vlm_needs_online and vlm_current_images is not None:
        with _profile_section(profiler, "vlm_image_h2d"):
            vlm_current_images = _move_vlm_images_to_device(
                vlm_current_images,
                device=model.device,
            )

    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError(
            f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
        )
    with _profile_section(profiler, "context_h2d"):
        context = context.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
    current_proprio = None
    if model.proprio_encoder is not None:
        if proprio is None:
            raise ValueError(
                "`sample['proprio']` is required when `proprio_dim` is enabled."
            )
        if proprio.ndim != 3:
            raise ValueError(
                f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[2] != model.proprio_dim:
            raise ValueError(
                f"`sample['proprio']` last dim must be {model.proprio_dim}, got {proprio.shape[2]}"
            )
        with _profile_section(profiler, "context_proprio"):
            proprio = proprio.to(
                device=model.device,
                dtype=model.torch_dtype,
                non_blocking=True,
            )
            proprio = _zero_padded_feature_dimensions(
                proprio,
                proprio_dim_is_pad,
                name="sample['proprio_dim_is_pad']",
            )
            proprio = proprio[:, 0, :]  # [B, D]
            current_proprio = proprio
            context, context_mask = append_proprio_to_context(
                model,
                context=context,
                context_mask=context_mask,
                proprio=current_proprio,
            )
    with _profile_section(profiler, "action_h2d"):
        action = action.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        action = _zero_padded_feature_dimensions(
            action,
            action_dim_is_pad,
            name="sample['action_dim_is_pad']",
        )

    with _profile_section(profiler, "memory_anchor_cache_h2d"):
        memory_video_anchor_latents, memory_video_anchor_is_pad = (
            prepare_cached_memory_latents(
                model,
                sample,
                name="memory_video_anchor",
                batch_size=batch_size,
            )
        )
    with _profile_section(profiler, "memory_recent_cache_h2d"):
        memory_video_recent_latents, memory_video_recent_is_pad = (
            prepare_cached_memory_latents(
                model,
                sample,
                name="memory_video_recent",
                batch_size=batch_size,
            )
        )
    with _profile_section(profiler, "memory_video_prepare"):
        if memory_video_anchor_latents is None:
            memory_video_anchor, memory_video_anchor_is_pad = (
                prepare_memory_video_inputs(
                    model,
                    sample,
                    name="memory_video_anchor",
                    batch_size=batch_size,
                    height=height,
                    width=width,
                )
            )
        else:
            memory_video_anchor = None
        if memory_video_recent_latents is None:
            memory_video_recent, memory_video_recent_is_pad = (
                prepare_memory_video_inputs(
                    model,
                    sample,
                    name="memory_video_recent",
                    batch_size=batch_size,
                    height=height,
                    width=width,
                )
            )
        else:
            memory_video_recent = None
    with _profile_section(profiler, "memory_proprio_prepare"):
        memory_video_anchor_proprio, memory_video_anchor_proprio_is_pad = (
            prepare_memory_proprio_inputs(
                model,
                sample,
                name="memory_video_anchor_proprio",
                batch_size=batch_size,
                dim_is_pad=proprio_dim_is_pad,
            )
        )
        memory_video_recent_proprio, memory_video_recent_proprio_is_pad = (
            prepare_memory_proprio_inputs(
                model,
                sample,
                name="memory_video_recent_proprio",
                batch_size=batch_size,
                dim_is_pad=proprio_dim_is_pad,
            )
        )
    with _profile_section(profiler, "mask_prepare"):
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=model.device, dtype=torch.bool, non_blocking=True
            )
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(
                device=model.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=model.device, dtype=torch.bool, non_blocking=True
            )

    return {
        "context": context,
        "context_mask": context_mask,
        "current_proprio": current_proprio,
        "input_video": input_video,
        "input_latents": input_latents,
        "first_frame_latents": first_frame_latents,
        "fuse_vae_embedding_in_latents": fuse_flag,
        "action": action,
        "action_hz": action_hz,
        "memory_video_anchor": memory_video_anchor,
        "memory_video_anchor_latents": memory_video_anchor_latents,
        "memory_video_anchor_is_pad": memory_video_anchor_is_pad,
        "memory_video_anchor_proprio": memory_video_anchor_proprio,
        "memory_video_anchor_proprio_is_pad": memory_video_anchor_proprio_is_pad,
        "memory_video_recent": memory_video_recent,
        "memory_video_recent_latents": memory_video_recent_latents,
        "memory_video_recent_is_pad": memory_video_recent_is_pad,
        "memory_video_recent_proprio": memory_video_recent_proprio,
        "memory_video_recent_proprio_is_pad": memory_video_recent_proprio_is_pad,
        "action_is_pad": action_is_pad,
        "action_dim_is_pad": action_dim_is_pad,
        "image_is_pad": image_is_pad,
        "vlm_current_valid": vlm_current_valid,
        "vlm_context_cache": vlm_context_cache,
        "vlm_mask_cache": vlm_mask_cache,
        "vlm_cache_hit_mask": vlm_cache_hit_mask,
        "prompt": sample.get("prompt", None),
        "vlm_current_images": vlm_current_images,
        "vlm_current_view_is_pad": sample.get(
            "vlm_current_view_is_pad", None
        ),
        "video_layout": sample.get("video_layout", None),
        "video_view_names": sample.get("video_view_names", None),
        "video_canvas_layout": sample.get("video_canvas_layout", None),
        "video_canvas_view_names": sample.get("video_canvas_view_names", None),
    }
