"""Shared camera condition configuration and feature packing."""

import dataclasses
from collections.abc import Mapping

import torch


@dataclasses.dataclass(frozen=True)
class ConditionViewConfig:
    """A physical camera crop of a decoded or prepared RGB view.

    Bounds are normalized (left, top, right, bottom), after view resizing.
    The caller selects the observed frame and preserves its tensor/array layout.
    """

    source: str
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    crops_by_size: Mapping[str, tuple[float, float, float, float]] | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("condition view source must not be empty")
        crop = tuple(float(value) for value in self.crop)
        if len(crop) != 4 or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError("condition crop must be normalized (left, top, right, bottom)")
        object.__setattr__(self, "crop", crop)
        if self.crops_by_size is not None:
            normalized = {}
            for size, bounds in self.crops_by_size.items():
                axes = size.split("x")
                if len(axes) != 2 or any(not axis.isdigit() or int(axis) <= 0 for axis in axes):
                    raise ValueError("condition crop size keys must use positive HxW dimensions")
                normalized[size] = ConditionViewConfig(self.source, bounds).crop
            if not normalized:
                raise ValueError("crops_by_size must not be empty")
            object.__setattr__(self, "crops_by_size", normalized)

    def crop_bounds(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Return integer (left, top, right, bottom) bounds for this view size."""
        crop = self.crop
        size = f"{height}x{width}"
        if self.crops_by_size is not None:
            if size not in self.crops_by_size:
                raise ValueError(f"condition source {self.source!r} has no crop configured for {size}")
            crop = self.crops_by_size[size]
        left, top, right, bottom = crop
        x0, x1 = round(left * width), round(right * width)
        y0, y1 = round(top * height), round(bottom * height)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"condition source {self.source!r} has an empty crop at {size}")
        return x0, y0, x1, y1


def pack_action_condition(
    features: dict[int, list[torch.Tensor]],
    cp_size: int,
) -> dict:
    """Pack target-layer features, padding each sample for context parallelism."""
    first = next(iter(features.values()))
    lengths = [value.shape[0] for value in first]
    padded = [length + (-length) % cp_size for length in lengths]
    packed = {}
    for layer, samples in features.items():
        if [value.shape[0] for value in samples] != lengths:
            raise ValueError("condition layers must share sample sequence lengths")
        packed[layer] = torch.cat([
            torch.cat([value, value.new_zeros(padding - length, value.shape[-1])])
            for value, length, padding in zip(samples, lengths, padded, strict=True)
        ])
    return {
        "features": packed,
        "lengths": torch.tensor(lengths, dtype=torch.int32),
        "padded_lengths": torch.tensor(padded, dtype=torch.int32),
    }
