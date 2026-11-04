"""VGGT-style alternating attention shared by both backbones.

Most layers keep view-local self-attention. Selected layers reuse the same
block with a mask that lets each view also attend to every other view.
"""
from typing import Any, Dict, Optional, Sequence

import torch

from flexwam.models.architectures.structures import _video_visual_view_slices


def resolve_cross_view_layers(
    num_layers: int,
    interval: int = 0,
    layers: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Return zero-based cross-view layers; explicit `layers` override `interval`."""
    num_layers = int(num_layers)
    if layers is not None:
        resolved = sorted({int(idx) for idx in layers})
        for idx in resolved:
            if not 0 <= idx < num_layers:
                raise ValueError(
                    f"`cross_view_attention_layers` entry {idx} is outside depth={num_layers}."
                )
        return tuple(resolved)
    interval = int(interval)
    if interval < 0:
        raise ValueError(f"`cross_view_attention_interval` must be >= 0, got {interval}.")
    if interval == 0:
        return ()
    return tuple(range(interval - 1, num_layers, interval))


def build_cross_view_attention_mask(
    view_mask: torch.Tensor,
    view_slices: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Give each view's queries the same key pattern in every other view.

    Prefix rows and columns are unchanged, and temporal rules carry over, so a
    first-frame query still reads only first-frame tokens of other views.
    """
    mask = view_mask.clone()
    for q_start, q_end in view_slices:
        local = view_mask[q_start:q_end, q_start:q_end]
        for k_start, k_end in view_slices:
            if k_end - k_start != q_end - q_start:
                raise ValueError("Cross-view attention requires equal token counts per view.")
            mask[q_start:q_end, k_start:k_end] = local
    return mask


class CrossViewAttentionMixin:
    """Backbones must define `blocks` before calling `configure_cross_view_attention`."""

    cross_view_layer_indices: tuple[int, ...] = ()

    def configure_cross_view_attention(
        self,
        interval: int = 0,
        layers: Optional[Sequence[int]] = None,
    ) -> None:
        self.cross_view_attention_interval = int(interval)
        self.cross_view_attention_layers = None if layers is None else [int(idx) for idx in layers]
        self.cross_view_layer_indices = resolve_cross_view_layers(len(self.blocks), interval, layers)

    def build_cross_view_mask(
        self,
        video_mask: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        if video_mask is None or not self.cross_view_layer_indices:
            return None
        if int(meta.get("num_views", 1)) <= 1:
            return None
        view_slices = _video_visual_view_slices(int(video_mask.shape[0]), meta)
        return build_cross_view_attention_mask(video_mask, view_slices)
