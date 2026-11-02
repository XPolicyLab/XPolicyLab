"""Build Action-side Qwen-VL and robot-state conditions for InternW0-delta."""

from typing import Any, Optional, Sequence

import torch

from ..memory.proprio_encoder import append_proprio_to_context


def build_vlm_condition(
    model,
    *,
    frames: Any,
    prompts: Sequence[str],
    view_names: Sequence[str] | None,
    frame_labels: Sequence[Sequence[str]] | Sequence[str] | None = None,
    valid_mask: Optional[torch.Tensor] = None,
    view_valid_mask: Optional[torch.Tensor] = None,
) -> Optional[dict[str, torch.Tensor]]:
    if not model.understanding_enabled:
        return None
    return model.understanding(
        frames=frames,
        prompts=prompts,
        view_names=view_names,
        frame_labels=frame_labels,
        valid_mask=valid_mask,
        view_valid_mask=view_valid_mask,
    )


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


def _build_online_frame_vlm_pack(
    model,
    *,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Encode current multi-camera observations for Action conditioning.

    Qwen receives the fixed-size current camera images as independent images.
    Episode anchor/recent frames remain video-expert conditions only.
    Encoded proprioception is appended after the VLM features for Action.
    """
    if not model.understanding_enabled:
        return {"vlm_context": None, "vlm_mask": None}

    current_images = inputs["vlm_current_images"]
    frames = current_images.unsqueeze(1)
    batch_size = int(frames.shape[0])
    frame_labels = ["current"]

    view_valid_mask = None
    view_is_pad = inputs.get("vlm_current_view_is_pad")
    if view_is_pad is not None:
        if view_is_pad.ndim != 2 or tuple(view_is_pad.shape) != tuple(
            current_images.shape[:2]
        ):
            raise ValueError(
                "`vlm_current_view_is_pad` must be [B,V] and match current images, "
                f"got {tuple(view_is_pad.shape)} vs {tuple(current_images.shape)}."
            )
        view_valid_mask = ~view_is_pad.to(
            device=current_images.device, dtype=torch.bool
        )
        view_valid_mask = view_valid_mask.unsqueeze(1)

    current_valid = torch.ones((batch_size,), dtype=torch.bool)
    image_is_pad = inputs.get("image_is_pad")
    if image_is_pad is not None:
        current_valid &= ~image_is_pad[:, 0].to(device="cpu", dtype=torch.bool)
    if not bool(current_valid.any().item()):
        raise ValueError("VLM understanding received no valid current images.")
    current_all_valid = bool(current_valid.all().item())
    prompts = normalize_batch_text(
        inputs.get("prompt"), batch_size=batch_size, default=""
    )
    view_names = normalize_batch_text(
        inputs.get("video_canvas_view_names"),
        batch_size=batch_size,
        default=str(model.understanding_cfg.get("default_view_names", "")),
    )
    vlm_pack = build_vlm_condition(
        model,
        frames=frames,
        prompts=prompts,
        view_names=view_names,
        frame_labels=frame_labels,
        valid_mask=None if current_all_valid else current_valid,
        view_valid_mask=view_valid_mask,
    )
    if vlm_pack is None:
        return {"vlm_context": None, "vlm_mask": None}
    vlm_context = vlm_pack["vlm_context"]
    vlm_mask = vlm_pack["vlm_mask"].to(device=vlm_context.device, dtype=torch.bool)
    if not current_all_valid:
        vlm_mask &= current_valid.to(device=vlm_mask.device).unsqueeze(-1)
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


def build_frame_vlm_pack(model, *, inputs: dict[str, Any]) -> dict[str, Any]:
    """Combine frozen cached observations with online misses, then encode state."""
    context = inputs.get("vlm_context_cache")
    mask = inputs.get("vlm_mask_cache")
    hits = inputs.get("vlm_cache_hit_mask")
    if context is None and mask is None and hits is None:
        return _build_online_frame_vlm_pack(model, inputs=inputs)
    if not model.understanding_enabled:
        raise ValueError("VLM artifacts require understanding.enabled=true.")
    if bool(getattr(model.understanding, "train_vlm", False)):
        raise ValueError("Frozen VLM artifacts cannot be used with train_vlm=true.")
    if context is not None or mask is not None:
        if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("VLM cached context and mask must both be tensors.")
        if context.ndim != 3 or mask.ndim != 2 or context.shape[:2] != mask.shape:
            raise ValueError(
                "VLM cached context/mask must have matching [B,S,H]/[B,S] shapes."
            )
    if hits is None:
        if context is None:
            raise ValueError("VLM cache has no context.")
        hits = torch.ones(context.shape[0], dtype=torch.bool)
    else:
        hits = hits.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    batch_size = len(hits)
    hit_count = int(hits.sum())
    if (0 if context is None else context.shape[0]) != hit_count:
        raise ValueError("Compact VLM cached rows do not match the hit mask.")
    if hit_count == batch_size:
        combined, combined_mask = context, mask
    else:
        images = inputs.get("vlm_current_images")
        if not isinstance(images, torch.Tensor) or images.shape[0] != batch_size:
            raise ValueError(
                "VLM cache misses require current images for every batch row."
            )
        valid = torch.ones(batch_size, dtype=torch.bool)
        image_is_pad = inputs.get("image_is_pad")
        if image_is_pad is not None:
            valid &= ~image_is_pad[:, 0].detach().to(device="cpu", dtype=torch.bool)
        online_valid = valid & ~hits
        view_pad = inputs.get("vlm_current_view_is_pad")
        view_valid = (
            None
            if view_pad is None
            else ~view_pad.detach().to(device="cpu", dtype=torch.bool).unsqueeze(1)
        )
        if bool(online_valid.any()):
            pack = build_vlm_condition(
                model,
                frames=images.unsqueeze(1),
                prompts=normalize_batch_text(
                    inputs.get("prompt"), batch_size=batch_size
                ),
                view_names=normalize_batch_text(
                    inputs.get("video_canvas_view_names"),
                    batch_size=batch_size,
                    default=str(model.understanding_cfg.get("default_view_names", "")),
                ),
                frame_labels=["current"],
                valid_mask=online_valid,
                view_valid_mask=view_valid,
            )
            combined, combined_mask = pack["vlm_context"], pack["vlm_mask"]
            combined_mask = (
                combined_mask & online_valid.to(combined_mask.device)[:, None]
            )
        elif context is not None:
            combined = context.new_zeros(
                (batch_size, context.shape[1], context.shape[2])
            )
            combined_mask = mask.new_zeros((batch_size, mask.shape[1]))
        else:
            raise ValueError("VLM understanding received no valid current images.")
        if context is not None:
            import torch.nn.functional as F

            length = max(combined.shape[1], context.shape[1])
            combined = F.pad(combined, (0, 0, 0, length - combined.shape[1]))
            combined_mask = F.pad(
                combined_mask, (0, length - combined_mask.shape[1]), value=False
            )
            context = F.pad(context, (0, 0, 0, length - context.shape[1]))
            mask = F.pad(mask, (0, length - mask.shape[1]), value=False)
            indices = hits.nonzero(as_tuple=False).flatten().to(combined.device)
            combined = combined.index_copy(0, indices, context.to(combined))
            combined_mask = combined_mask.index_copy(
                0, indices.to(combined_mask.device), mask.to(combined_mask)
            )
    combined, combined_mask = append_proprio_to_context(
        model,
        context=combined,
        context_mask=combined_mask.bool(),
        proprio=inputs.get("current_proprio"),
        proprio_encoder=model.action_proprio_encoder,
    )
    return {"vlm_context": combined, "vlm_mask": combined_mask}
