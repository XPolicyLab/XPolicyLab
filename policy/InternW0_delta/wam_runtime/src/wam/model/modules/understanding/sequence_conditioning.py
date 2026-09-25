"""Build Action-side Qwen-VL and robot-state conditions for WAM."""

import inspect
from contextlib import nullcontext
from typing import Any, Optional, Sequence

import torch

from ..memory.proprio_encoder import append_proprio_to_context


def _profile_section(profiler, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


def build_vlm_condition(
    model,
    *,
    frames: torch.Tensor,
    prompts: Sequence[str],
    view_names: Sequence[str] | None,
    frame_labels: Sequence[Sequence[str]] | Sequence[str] | None = None,
    valid_mask: Optional[torch.Tensor] = None,
    view_valid_mask: Optional[torch.Tensor] = None,
    profiler=None,
) -> Optional[dict[str, torch.Tensor]]:
    if not model.understanding_enabled:
        return None
    kwargs = dict(
        frames=frames,
        prompts=prompts,
        view_names=view_names,
        frame_labels=frame_labels,
        valid_mask=valid_mask,
    )
    # The optimized extractor accepts profiling sections, while lightweight
    # adapters and older evaluation wrappers may not. Preserve their public
    # call contract when profiling is disabled.
    call_target = (
        model.understanding.forward
        if isinstance(model.understanding, torch.nn.Module)
        else model.understanding.__call__
    )
    accepts_profiler = "profiler" in inspect.signature(call_target).parameters
    accepts_view_mask = (
        "view_valid_mask" in inspect.signature(call_target).parameters
    )
    if view_valid_mask is not None:
        if not accepts_view_mask:
            raise TypeError(
                "The configured understanding encoder does not accept "
                "`view_valid_mask`."
            )
        kwargs["view_valid_mask"] = view_valid_mask
    if profiler is not None or accepts_profiler:
        kwargs["profiler"] = profiler
    return model.understanding(**kwargs)

def normalize_batch_text(
    value: Any, *, batch_size: int, default: str = ""
) -> list[str]:
    if value is None:
        return [default] * int(batch_size)
    if isinstance(value, str):
        return [value] * int(batch_size)
    if isinstance(value, Sequence):
        values = [default if item is None else str(item) for item in value]
        if len(values) == int(batch_size):
            return values
        if len(values) == 1:
            return values * int(batch_size)
    return [default if value is None else str(value)] * int(batch_size)



def build_frame_vlm_pack(
    model,
    *,
    inputs: dict[str, Any],
    profiler=None,
) -> dict[str, Any]:
    """Encode current multi-camera observations for Action conditioning.

    Qwen receives the original-resolution current camera images as independent
    images. Episode anchor/recent frames remain video-expert conditions only.
    Encoded proprioception is appended after the VLM features for Action.
    """
    if not model.understanding_enabled:
        return {"vlm_context": None, "vlm_mask": None}

    cached_context = inputs.get("vlm_context_cache")
    cached_mask = inputs.get("vlm_mask_cache")
    cache_hit_mask = inputs.get("vlm_cache_hit_mask")
    if cached_context is not None or cached_mask is not None:
        if not isinstance(cached_context, torch.Tensor) or not isinstance(
            cached_mask, torch.Tensor
        ):
            raise TypeError("Cached VLM context and mask must both be tensors.")
        if cached_context.ndim != 3 or cached_mask.ndim != 2:
            raise ValueError(
                "Cached VLM context/mask must be [B,S,H]/[B,S], got "
                f"{tuple(cached_context.shape)} and {tuple(cached_mask.shape)}."
            )
        if tuple(cached_context.shape[:2]) != tuple(cached_mask.shape):
            raise ValueError("Cached VLM context and mask shapes do not align.")
    if cache_hit_mask is not None:
        if not isinstance(cache_hit_mask, torch.Tensor):
            raise TypeError("VLM cache hit mask must be a tensor.")
        cache_hit_mask = cache_hit_mask.detach().to(
            device="cpu", dtype=torch.bool
        ).reshape(-1)
        cached_rows = 0 if cached_context is None else int(cached_context.shape[0])
        if cached_rows != int(cache_hit_mask.sum().item()):
            raise ValueError(
                "Compact cached VLM rows do not match the cache hit mask: "
                f"rows={cached_rows} hits={int(cache_hit_mask.sum().item())}."
            )
    if cached_context is not None and (
        cache_hit_mask is None or bool(cache_hit_mask.all().item())
    ):
        vlm_context, vlm_mask = append_proprio_to_context(
            model,
            context=cached_context,
            context_mask=cached_mask.to(
                device=cached_context.device, dtype=torch.bool
            ),
            proprio=inputs.get("current_proprio"),
            proprio_encoder=model.action_proprio_encoder,
        )
        return {"vlm_context": vlm_context, "vlm_mask": vlm_mask}

    current_images = inputs["vlm_current_images"]
    if isinstance(current_images, torch.Tensor):
        frames: torch.Tensor | list[torch.Tensor] = current_images.unsqueeze(1)
    elif isinstance(current_images, Sequence):
        frames = []
        for images in current_images:
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError(
                    "Each vlm_current_images sample must be [V,C,H,W], "
                    f"got {type(images)} with shape={getattr(images, 'shape', None)}"
                )
            frames.append(images.unsqueeze(0))
    else:
        raise TypeError(
            "vlm_current_images must be a tensor or a sequence of tensors, "
            f"got {type(current_images)}"
        )
    frame_labels = ["current"]
    batch_size = int(frames.shape[0]) if isinstance(frames, torch.Tensor) else len(frames)
    if cache_hit_mask is not None and int(cache_hit_mask.numel()) != batch_size:
        raise ValueError(
            "VLM cache hit mask must have one value per input: "
            f"got={tuple(cache_hit_mask.shape)} batch={batch_size}."
        )

    with _profile_section(profiler, "prepare_metadata"):
        current_valid = inputs.get("vlm_current_valid")
        if current_valid is None:
            image_is_pad = inputs.get("image_is_pad")
            if image_is_pad is None:
                current_valid = torch.ones((batch_size,), dtype=torch.bool)
            else:
                current_valid = ~image_is_pad[:, 0].detach().to(
                    device="cpu", dtype=torch.bool
                )
        else:
            current_valid = current_valid.detach().to(
                device="cpu", dtype=torch.bool
            ).reshape(-1)
        if int(current_valid.numel()) != batch_size:
            raise ValueError(
                "`vlm_current_valid` must have one value per VLM input: "
                f"got {tuple(current_valid.shape)} for batch={batch_size}."
            )
        online_valid = current_valid.clone()
        if cache_hit_mask is not None:
            online_valid &= ~cache_hit_mask
        has_online = bool(online_valid.any().item())

        view_valid_mask = None
        if bool(
            model.understanding_cfg.get("use_view_valid_mask", True)
        ):
            view_is_pad = inputs.get("vlm_current_view_is_pad")
            if view_is_pad is not None:
                if not isinstance(current_images, torch.Tensor):
                    raise ValueError(
                        "`vlm_current_view_is_pad` requires batched tensor "
                        "`vlm_current_images`."
                    )
                view_is_pad = torch.as_tensor(
                    view_is_pad, device="cpu", dtype=torch.bool
                )
                if view_is_pad.ndim != 2 or tuple(
                    view_is_pad.shape
                ) != tuple(current_images.shape[:2]):
                    raise ValueError(
                        "`vlm_current_view_is_pad` must be [B,V] and match "
                        "current images, got "
                        f"{tuple(view_is_pad.shape)} vs "
                        f"{tuple(current_images.shape)}."
                    )
                view_valid_mask = (~view_is_pad).unsqueeze(1)
                if not bool(view_valid_mask.any(dim=(1, 2)).all().item()):
                    raise ValueError(
                        "Every VLM sample must contain at least one valid view."
                    )

        prompts = normalize_batch_text(
            inputs.get("prompt"), batch_size=batch_size, default=""
        )
        view_names = normalize_batch_text(
            inputs.get("video_canvas_view_names"),
            batch_size=batch_size,
            default=str(model.understanding_cfg.get("default_view_names", "")),
        )
    if has_online:
        vlm_pack = build_vlm_condition(
            model,
            frames=frames,
            prompts=prompts,
            view_names=view_names,
            frame_labels=frame_labels,
            valid_mask=(
                None if bool(online_valid.all().item()) else online_valid
            ),
            view_valid_mask=view_valid_mask,
            profiler=profiler,
        )
        if vlm_pack is None:
            return {"vlm_context": None, "vlm_mask": None}
        vlm_context = vlm_pack["vlm_context"]
        vlm_mask = vlm_pack["vlm_mask"].to(
            device=vlm_context.device, dtype=torch.bool
        )
        vlm_mask &= online_valid.to(device=vlm_mask.device).unsqueeze(-1)
    elif cached_context is not None:
        vlm_context = cached_context.new_zeros(
            (batch_size, int(cached_context.shape[1]), int(cached_context.shape[2]))
        )
        vlm_mask = torch.zeros(
            (batch_size, int(cached_mask.shape[1])),
            device=cached_context.device,
            dtype=torch.bool,
        )
    else:
        raise ValueError("VLM understanding received no valid current images.")

    if cached_context is not None:
        assert cached_mask is not None and cache_hit_mask is not None
        target_length = max(
            int(vlm_context.shape[1]), int(cached_context.shape[1])
        )
        if int(vlm_context.shape[1]) < target_length:
            padding = target_length - int(vlm_context.shape[1])
            vlm_context = torch.cat(
                [
                    vlm_context,
                    vlm_context.new_zeros(
                        (batch_size, padding, int(vlm_context.shape[2]))
                    ),
                ],
                dim=1,
            )
            vlm_mask = torch.cat(
                [
                    vlm_mask,
                    torch.zeros(
                        (batch_size, padding),
                        device=vlm_mask.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        if int(cached_context.shape[1]) < target_length:
            padding = target_length - int(cached_context.shape[1])
            cached_context = torch.cat(
                [
                    cached_context,
                    cached_context.new_zeros(
                        (int(cached_context.shape[0]), padding, int(cached_context.shape[2]))
                    ),
                ],
                dim=1,
            )
            cached_mask = torch.cat(
                [
                    cached_mask,
                    torch.zeros(
                        (int(cached_mask.shape[0]), padding),
                        device=cached_mask.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        hit_indices = cache_hit_mask.nonzero(as_tuple=False).flatten().to(
            device=vlm_context.device
        )
        vlm_context.index_copy_(0, hit_indices, cached_context)
        vlm_mask.index_copy_(
            0,
            hit_indices.to(device=vlm_mask.device),
            cached_mask.to(device=vlm_mask.device, dtype=torch.bool),
        )
    vlm_context, vlm_mask = append_proprio_to_context(
        model,
        context=vlm_context,
        context_mask=vlm_mask,
        proprio=inputs.get("current_proprio"),
        proprio_encoder=model.action_proprio_encoder,
    )
    return {
        "vlm_context": vlm_context,
        "vlm_mask": vlm_mask,
    }
