from typing import Any, Optional, Sequence

import torch

from ..codecs import video_latent_codec as video_codec
from ..codecs.utils import video_batch_size, video_num_frames, video_spatial_shape
from ..memory.proprio_encoder import append_proprio_to_context


def _move_vlm_images_to_device(
    images: Any,
    *,
    device: torch.device | str,
) -> Any:
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


def _cached_video_geometry(
    sample: dict[str, Any],
    cached_latents: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Recover source-video geometry without materializing cached RGB video."""

    if not isinstance(cached_latents, torch.Tensor) or cached_latents.ndim != 5:
        raise ValueError(
            "Image-free cache projection requires `video_latents` with shape "
            f"[B,C,T,H,W], got {type(cached_latents)} "
            f"{getattr(cached_latents, 'shape', None)}."
        )
    raw_shape = sample.get("video_shape")
    if raw_shape is None:
        raise ValueError(
            "Image-free cache projection requires `sample['video_shape']`."
        )
    shape = torch.as_tensor(raw_shape, dtype=torch.long)
    batch_size = int(cached_latents.shape[0])
    if shape.ndim == 2:
        if int(shape.shape[0]) not in (1, batch_size):
            raise ValueError(
                "`sample['video_shape']` batch mismatch: "
                f"got {tuple(shape.shape)} for batch={batch_size}."
            )
        if int(shape.shape[0]) > 1 and not bool(
            torch.all(shape == shape[0:1]).item()
        ):
            raise ValueError("All cached samples in a batch must share video_shape.")
        shape = shape[0]
    if shape.ndim != 1 or int(shape.numel()) not in (4, 5):
        raise ValueError(
            "`sample['video_shape']` must describe [C,T,H,W] or [N,C,T,H,W], "
            f"got {tuple(shape.shape)} with values={shape.tolist()}."
        )
    values = [int(value) for value in shape.tolist()]
    channel_index = 0 if len(values) == 4 else 1
    time_index = 1 if len(values) == 4 else 2
    if values[channel_index] != 3:
        raise ValueError(
            "Cached source video must have three channels, got "
            f"video_shape={values}."
        )
    num_frames = values[time_index]
    height, width = values[-2:]
    return batch_size, num_frames, height, width


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
    latents = sample.get(f"{name}_latents")
    if latents is None:
        return None, None
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError(
            f"`sample['{name}_latents']` must be [B,C,1,H,W], got "
            f"{type(latents)} {getattr(latents, 'shape', None)}."
        )
    if int(latents.shape[0]) != batch_size or int(latents.shape[2]) != 1:
        raise ValueError(
            f"`sample['{name}_latents']` must have batch={batch_size} and one "
            f"latent frame, got {tuple(latents.shape)}."
        )
    latents = latents.to(
        device=model.device, dtype=model.torch_dtype, non_blocking=True
    )
    pad = sample.get(f"{name}_is_pad")
    if pad is None:
        pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=model.device)
    else:
        pad = torch.as_tensor(pad)
        if pad.ndim == 1:
            pad = pad.unsqueeze(0).expand(batch_size, -1)
        elif pad.ndim == 2 and int(pad.shape[0]) == 1 and batch_size > 1:
            pad = pad.expand(batch_size, -1)
        if tuple(pad.shape) != (batch_size, 1):
            raise ValueError(
                f"`sample['{name}_is_pad']` must be [1] or [B,1], got "
                f"{tuple(pad.shape)}."
            )
        pad = pad.to(device=model.device, dtype=torch.bool, non_blocking=True)
    return latents, pad


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


def build_wam_inputs(model, sample, tiled: bool = False):
    video = sample.get("video")
    cached_latents = sample.get("video_latents")
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
        batch_size, num_frames, height, width = _cached_video_geometry(
            sample, cached_latents
        )
    elif video.ndim == 5:
        if int(video.shape[1]) != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )
        batch_size = video_batch_size(video)
        num_frames = video_num_frames(video)
        height, width = video_spatial_shape(video)
    elif video.ndim == 6:
        if int(video.shape[2]) != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )
        batch_size = video_batch_size(video)
        num_frames = video_num_frames(video)
        height, width = video_spatial_shape(video)
    else:
        raise ValueError(
            "`sample['video']` must be [B,3,T,H,W] or [B,N,3,T,H,W], "
            f"got {type(video)} shape={getattr(video, 'shape', None)}"
        )
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
        raise ValueError("`sample['action']` is required for InternW0-delta training.")

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

    action_mask = sample.get("action_mask", None)
    if action_mask is not None:
        if action_mask.ndim != 3 or tuple(action_mask.shape) != tuple(action.shape):
            raise ValueError(
                "`sample['action_mask']` must be [B,T,D] matching action, "
                f"got {tuple(action_mask.shape)} vs {tuple(action.shape)}"
            )

    action_dim_loss_weight = sample.get("action_dim_loss_weight", None)
    if action_dim_loss_weight is not None:
        if action_dim_loss_weight.ndim == 1:
            action_dim_loss_weight = action_dim_loss_weight.unsqueeze(0).expand(batch_size, -1)
        if tuple(action_dim_loss_weight.shape) != (batch_size, int(action.shape[2])):
            raise ValueError(
                "`sample['action_dim_loss_weight']` must be [D] or [B,D], "
                f"got {tuple(action_dim_loss_weight.shape)}."
            )

    action_loss_weight = sample.get("action_loss_weight", None)
    action_loss_weighted_valid_cells = sample.get(
        "action_loss_weighted_valid_cells", None
    )
    action_loss_weighted_valid_cells_pool = sample.get(
        "action_loss_weighted_valid_cells_pool", None
    )
    if action_loss_weighted_valid_cells is not None:
        if action_loss_weighted_valid_cells.ndim != 1 or int(
            action_loss_weighted_valid_cells.shape[0]
        ) != batch_size:
            raise ValueError(
                "`sample['action_loss_weighted_valid_cells']` must be [B], "
                f"got {tuple(action_loss_weighted_valid_cells.shape)}"
            )
    if action_loss_weighted_valid_cells_pool is not None:
        if (
            action_loss_weighted_valid_cells_pool.ndim != 2
            or int(action_loss_weighted_valid_cells_pool.shape[0])
            != batch_size
            or int(action_loss_weighted_valid_cells_pool.shape[1]) <= 0
        ):
            raise ValueError(
                "`sample['action_loss_weighted_valid_cells_pool']` must be "
                "[B,K] with K > 0, got "
                f"{tuple(action_loss_weighted_valid_cells_pool.shape)}"
            )
    if action_loss_weight is not None:
        if action_loss_weight.ndim != 1 or int(action_loss_weight.shape[0]) != batch_size:
            raise ValueError(
                "`sample['action_loss_weight']` must be [B], "
                f"got {tuple(action_loss_weight.shape)}"
            )
        if not bool(torch.isfinite(action_loss_weight).all()) or bool(
            (action_loss_weight <= 0).any()
        ):
            raise ValueError("`sample['action_loss_weight']` must be finite and positive")

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

    video_spatial_valid_mask = sample.get("video_spatial_valid_mask", None)
    if video_spatial_valid_mask is not None:
        if video_spatial_valid_mask.ndim == 2:
            video_spatial_valid_mask = video_spatial_valid_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        elif (
            video_spatial_valid_mask.ndim == 3
            and int(video_spatial_valid_mask.shape[0]) == 1
            and batch_size > 1
        ):
            video_spatial_valid_mask = video_spatial_valid_mask.expand(
                batch_size, -1, -1
            )
        expected_spatial_mask_shape = (batch_size, height, width)
        if tuple(video_spatial_valid_mask.shape) != expected_spatial_mask_shape:
            raise ValueError(
                "`sample['video_spatial_valid_mask']` must be [H,W] or [B,H,W], "
                f"got {tuple(video_spatial_valid_mask.shape)} vs expected "
                f"{expected_spatial_mask_shape}"
            )
        # Keep the legacy path byte-for-byte for batches whose whole canvas is
        # valid. The explicit all-ones masks exist only so mixed-source batches
        # can be collated with Ego samples that carry a real spatial mask.
        if bool(video_spatial_valid_mask.all().item()):
            video_spatial_valid_mask = None

    fuse_flag = bool(
        getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
    )
    input_video = None
    if cached_latents is not None:
        if tiled:
            raise ValueError("Cached current latents require tiled=false.")
        if not isinstance(cached_latents, torch.Tensor) or cached_latents.ndim != 5:
            raise ValueError(
                "`sample['video_latents']` must be [B,C,T,H,W], got "
                f"{type(cached_latents)} {getattr(cached_latents, 'shape', None)}."
            )
        if int(cached_latents.shape[0]) != batch_size:
            raise ValueError("Cached current latent batch does not match video batch.")
        input_latents = cached_latents.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
    else:
        input_video = video.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        input_latents = video_codec.encode_video_latents(
            model,
            input_video,
            tiled=tiled,
            video_layout=sample.get("video_layout", None),
            video_view_names=sample.get("video_view_names", None),
        )
    first_frame_latents = input_latents[:, :, 0:1] if fuse_flag else None

    vlm_current_images = sample.get("vlm_current_images")
    if sample.get("vlm_context_cache") is not None and sample.get("vlm_cache_hit_mask") is None:
        vlm_current_images = None
    defer_vlm_transfer = bool(getattr(model, "understanding_cfg", {}).get("defer_image_transfer", False)) or sample.get("vlm_cache_hit_mask") is not None
    if bool(getattr(model, "understanding_enabled", False)) and not defer_vlm_transfer:
        vlm_current_images = _move_vlm_images_to_device(
            vlm_current_images, device=model.device
        )

    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError(
            f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
        )
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
        proprio = proprio.to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        proprio = _zero_padded_feature_dimensions(
            proprio,
            proprio_dim_is_pad,
            name="sample['proprio_dim_is_pad']",
        )
        proprio = proprio[:, 0, :]  # [B, D]
        current_proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = append_proprio_to_context(
            model,
            context=context,
            context_mask=context_mask,
            proprio=current_proprio,
        )
    action = action.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)

    memory_video_anchor_latents, memory_video_anchor_is_pad = (
        prepare_cached_memory_latents(
            model, sample, name="memory_video_anchor", batch_size=batch_size
        )
    )
    memory_video_recent_latents, memory_video_recent_is_pad = (
        prepare_cached_memory_latents(
            model, sample, name="memory_video_recent", batch_size=batch_size
        )
    )
    memory_video_anchor = None
    if memory_video_anchor_latents is None:
        memory_video_anchor, memory_video_anchor_is_pad = prepare_memory_video_inputs(
            model,
            sample,
            name="memory_video_anchor",
            batch_size=batch_size,
            height=height,
            width=width,
        )
    memory_video_recent = None
    if memory_video_recent_latents is None:
        memory_video_recent, memory_video_recent_is_pad = prepare_memory_video_inputs(
            model,
            sample,
            name="memory_video_recent",
            batch_size=batch_size,
            height=height,
            width=width,
        )
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
    if action_is_pad is not None:
        action_is_pad = action_is_pad.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
    if action_dim_is_pad is not None:
        action_dim_is_pad = action_dim_is_pad.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
        action = _zero_padded_feature_dimensions(
            action,
            action_dim_is_pad,
            name="sample['action_dim_is_pad']",
        )
    if action_mask is not None:
        action_mask = action_mask.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
        action = action.masked_fill(~action_mask, 0.0)
    if action_dim_loss_weight is not None:
        action_dim_loss_weight = action_dim_loss_weight.to(
            device=model.device, dtype=torch.float32, non_blocking=True
        )
    if image_is_pad is not None:
        image_is_pad = image_is_pad.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
    if video_spatial_valid_mask is not None:
        video_spatial_valid_mask = video_spatial_valid_mask.to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )

    if action_loss_weight is not None:
        action_loss_weight = action_loss_weight.to(device=model.device, dtype=torch.float32)
    return {
        "context": context,
        "context_mask": context_mask,
        "current_proprio": current_proprio,
        "input_video": input_video,
        "input_latents": input_latents,
        "first_frame_latents": first_frame_latents,
        "fuse_vae_embedding_in_latents": fuse_flag,
        "action": action,
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
        "action_mask": action_mask,
        "action_dim_loss_weight": action_dim_loss_weight,
        "action_loss_weight": action_loss_weight,
        "action_loss_weighted_valid_cells": action_loss_weighted_valid_cells,
        "action_loss_weighted_valid_cells_pool": action_loss_weighted_valid_cells_pool,
        "image_is_pad": image_is_pad,
        "video_spatial_valid_mask": video_spatial_valid_mask,
        "prompt": sample.get("prompt", None),
        "vlm_current_images": vlm_current_images,
        "vlm_context_cache": (sample["vlm_context_cache"].to(device=model.device, dtype=model.torch_dtype, non_blocking=True) if sample.get("vlm_context_cache") is not None else None),
        "vlm_mask_cache": (sample["vlm_mask_cache"].to(device=model.device, dtype=torch.bool, non_blocking=True) if sample.get("vlm_mask_cache") is not None else None),
        "vlm_cache_hit_mask": sample.get("vlm_cache_hit_mask"),
        "vlm_current_view_is_pad": sample.get("vlm_current_view_is_pad", None),
        "video_layout": sample.get("video_layout", None),
        "video_view_names": sample.get("video_view_names", None),
        "video_canvas_layout": sample.get("video_canvas_layout", None),
        "video_canvas_view_names": sample.get("video_canvas_view_names", None),
    }
