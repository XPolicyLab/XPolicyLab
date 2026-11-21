"""Perception data contracts. Every estimate carries its provenance and quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from knowin_agent.runtime.backend import CameraFrame


class PerceptionError(RuntimeError):
    """A measurement could not be made; the message says why (never a guessed value)."""


@dataclass
class Mask:
    score: float
    mask: np.ndarray  # HxW bool
    box_xyxy: tuple = ()

    @property
    def pixels(self) -> int:
        return int(self.mask.sum())


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray, text: str, *, max_masks: int = 8, min_score: float = 0.5) -> list[Mask]: ...


@runtime_checkable
class DepthSource(Protocol):
    def depth(self, frame: CameraFrame) -> Optional[np.ndarray]:
        """HxW metres along the optical axis, NaN where unknown; None if unavailable."""


def _round(value, digits=4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _round(value.tolist(), digits)
    if isinstance(value, np.generic):
        return _round(value.item(), digits)
    return value


@dataclass
class ObjectEstimate:
    """A grounded object. center_xyz is the top-face centre's XY at mid height."""

    label: str
    camera: str
    obs_id: str
    center_xyz: list
    top_z: float
    height_m: float
    length_m: float
    width_m: float
    long_axis_deg: Optional[float]
    grasp_yaw_deg: Optional[float]
    score: float
    pixel_centroid: list
    mask_pixels: int
    method: str
    quality: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _round(asdict(self))


@dataclass
class AxisTipEstimate:
    label: str
    camera: str
    obs_id: str
    axis_dir: list  # unit vector from base toward tip
    tip_xyz: list
    base_xyz: list
    length_m: float
    method: str
    quality: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _round(asdict(self))


@dataclass
class SupportPlaneEstimate:
    camera: str
    obs_id: str
    z: float
    normal: list
    inlier_ratio: float
    center_xyz: list
    method: str
    quality: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _round(asdict(self))


@dataclass
class OpeningEstimate:
    label: str
    camera: str
    obs_id: str
    center_xyz: list
    radius_m: float
    rim_z: float
    normal: list
    method: str
    quality: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _round(asdict(self))
