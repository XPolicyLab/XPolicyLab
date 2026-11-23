# SPDX-License-Identifier: Apache-2.0

"""Composable prompt and image transforms for the orchestrator.

Transforms deliberately operate on copies and return metadata.  This keeps
the raw simulator observation available while allowing the same transform to
be applied independently to the VLA, VLM planner, or VLM monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class PromptTransform(Protocol):
    """Modify a prompt for a particular orchestration phase."""

    def transform_prompt(
        self, prompt: str, obs: dict, state, phase: str
    ) -> str: ...


class ImageTransform(Protocol):
    """Modify an image for a particular consumer."""

    def transform_image(
        self, image: np.ndarray, obs: dict, state, consumer: str
    ) -> tuple[np.ndarray, dict]: ...


@dataclass(frozen=True)
class IdentityPromptTransform:
    """Explicit no-op prompt transform useful in configuration lists."""

    def transform_prompt(self, prompt: str, obs: dict, state, phase: str) -> str:
        return prompt


@dataclass(frozen=True)
class PrefixPromptTransform:
    prefix: str

    def transform_prompt(self, prompt: str, obs: dict, state, phase: str) -> str:
        return f"{self.prefix}{prompt}"


@dataclass(frozen=True)
class SuffixPromptTransform:
    suffix: str

    def transform_prompt(self, prompt: str, obs: dict, state, phase: str) -> str:
        return f"{prompt}{self.suffix}"


@dataclass
class PromptTransformPipeline:
    transforms: list[PromptTransform] = field(default_factory=list)

    def apply(self, prompt: str, obs: dict, state, phase: str) -> tuple[str, list[dict]]:
        value = prompt
        metadata: list[dict] = []
        for transform in self.transforms:
            before = value
            value = transform.transform_prompt(value, obs, state, phase)
            metadata.append({
                "transform": type(transform).__name__,
                "phase": phase,
                "changed": value != before,
            })
        return value, metadata


@dataclass
class ImageTransformPipeline:
    transforms: list[ImageTransform] = field(default_factory=list)
    consumers: set[str] = field(default_factory=set)

    def apply(
        self,
        image: np.ndarray,
        obs: dict,
        state,
        consumer: str,
    ) -> tuple[np.ndarray, list[dict]]:
        value = np.array(image, copy=True)
        metadata: list[dict] = []
        if consumer not in self.consumers:
            return value, metadata
        for transform in self.transforms:
            value, item = transform.transform_image(value, obs, state, consumer)
            item = dict(item or {})
            item.setdefault("transform", type(transform).__name__)
            item.setdefault("consumer", consumer)
            metadata.append(item)
        return value, metadata


class BBox:
    """Axis-aligned pixel bounding box used by simple image transforms."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1, self.y1 = int(x1), int(y1)
        self.x2, self.y2 = int(x2), int(y2)

    def clamp(self, height: int, width: int) -> "BBox":
        return BBox(
            max(0, min(self.x1, width)),
            max(0, min(self.y1, height)),
            max(0, min(self.x2, width)),
            max(0, min(self.y2, height)),
        )

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def edit_highlight(
    image: np.ndarray,
    bbox: BBox,
    border_px: int = 3,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Return a copy with a colored border around ``bbox``."""
    result = np.array(image, copy=True)
    h, w = result.shape[:2]
    box = bbox.clamp(h, w)
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    result[max(0, y1 - border_px):y1, x1:x2] = color
    result[y2:min(h, y2 + border_px), x1:x2] = color
    result[max(0, y1 - border_px):min(h, y2 + border_px), max(0, x1 - border_px):x1] = color
    result[max(0, y1 - border_px):min(h, y2 + border_px), x2:min(w, x2 + border_px)] = color
    return result


def edit_dim(image: np.ndarray, bbox: BBox, dim_factor: float = 0.3) -> np.ndarray:
    """Return a copy with pixels outside ``bbox`` dimmed."""
    h, w = image.shape[:2]
    box = bbox.clamp(h, w)
    result = (image.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)
    result[box.y1:box.y2, box.x1:box.x2] = image[box.y1:box.y2, box.x1:box.x2]
    return result


def edit_both(image: np.ndarray, bbox: BBox) -> np.ndarray:
    return edit_highlight(edit_dim(image, bbox), bbox)


def apply_edit(image: np.ndarray, bbox: BBox | None, mode: str) -> np.ndarray:
    """Apply ``none``, ``highlight``, ``dim``, or ``both`` without mutation."""
    if bbox is None or mode == "none":
        return np.array(image, copy=True)
    if mode == "highlight":
        return edit_highlight(image, bbox)
    if mode == "dim":
        return edit_dim(image, bbox)
    if mode == "both":
        return edit_both(image, bbox)
    raise ValueError(f"Unknown image edit mode: {mode!r}")


@dataclass(frozen=True)
class BBoxImageTransform:
    """Small, detector-free image transform for programmatic experiments."""

    bbox: BBox
    mode: str = "highlight"

    def transform_image(
        self, image: np.ndarray, obs: dict, state, consumer: str
    ) -> tuple[np.ndarray, dict]:
        return apply_edit(image, self.bbox, self.mode), {
            "mode": self.mode,
            "bbox": self.bbox.to_dict(),
        }


__all__ = [
    "BBox",
    "BBoxImageTransform",
    "IdentityPromptTransform",
    "ImageTransform",
    "ImageTransformPipeline",
    "PrefixPromptTransform",
    "PromptTransform",
    "PromptTransformPipeline",
    "SuffixPromptTransform",
    "apply_edit",
    "edit_both",
    "edit_dim",
    "edit_highlight",
]
