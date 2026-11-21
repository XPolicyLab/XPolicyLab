"""Backend-neutral semantic action field declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import torch

Array: TypeAlias = np.ndarray | torch.Tensor
ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)
ModelRepr: TypeAlias = Literal["absolute", "relative"]
RotRepr: TypeAlias = Literal["rpy", "rvec", "rot6d", "xyzw", "wxyz"]
NormMethod: TypeAlias = Literal["zscore", "minmax", "quantile"]
NumericValue: TypeAlias = float | tuple[float, ...]
Matrix4: TypeAlias = Array | tuple[tuple[float, float, float, float], ...]

ROT_DIM: dict[RotRepr, int] = {
    "rpy": 3,
    "rvec": 3,
    "rot6d": 6,
    "xyzw": 4,
    "wxyz": 4,
}


@dataclass(frozen=True)
class ActionResult:
    """Action conversion output and its optional full-layout mask."""

    action: Array
    mask: Array | None


@dataclass(frozen=True)
class ActionField(ABC):
    """One semantic action representation.

    ``name`` is required when the field is placed in an ``ActionProfile``. It
    may be omitted when a caller already owns naming, such as a dataset source
    declaration.
    """

    name: str | None = None
    model_repr: ModelRepr = "absolute"
    input_used: bool = True
    model_used: bool = True
    output_used: bool = True

    def __post_init__(self) -> None:
        if self.name is not None and not self.name:
            raise ValueError("action field name must not be empty")
        if self.model_repr not in ("absolute", "relative"):
            raise ValueError(f"unknown model representation: {self.model_repr!r}")
        for label, value in (
            ("input_used", self.input_used),
            ("model_used", self.model_used),
            ("output_used", self.output_used),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{label} must be bool, got {type(value).__name__}")

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of values in the full input layout."""

    @property
    @abstractmethod
    def model_dim(self) -> int:
        """Number of values in the full model layout."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of values in the full output layout."""

    def encode(
        self,
        action: Array,
        *,
        ref_state: Array | None = None,
        ref_frame: Array | None = None,
    ) -> Array:
        """Convert this field from input to model representation."""
        from morph.convs import encode_field

        return encode_field(
            self,
            action,
            ref_state=ref_state,
            ref_frame=ref_frame,
        )

    def decode(
        self,
        action: Array,
        *,
        ref_state: Array | None = None,
        ref_frame: Array | None = None,
    ) -> Array:
        """Convert this field from model to output representation."""
        from morph.convs import decode_field

        return decode_field(
            self,
            action,
            ref_state=ref_state,
            ref_frame=ref_frame,
        )

    @property
    def mask_is_atomic(self) -> bool:
        """Whether invalid rows must be skipped as one whole component."""
        return False


@dataclass(frozen=True)
class Qpos(ActionField):
    """Joint-space or scalar field with affine calibration."""

    dim: int | None = None
    trans: NumericValue = 0.0
    scale: NumericValue = 1.0
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim is not None and self.dim < 1:
            raise ValueError(f"Qpos dim must be positive, got {self.dim}")
        for label, value in (("trans", self.trans), ("scale", self.scale)):
            if (
                self.dim is not None
                and isinstance(value, tuple)
                and len(value) != self.dim
            ):
                raise ValueError(
                    f"Qpos {label} length {len(value)} does not match dim "
                    f"{self.dim}"
                )
        scales = self.scale if isinstance(self.scale, tuple) else (self.scale,)
        if any(value == 0 for value in scales):
            raise ValueError("Qpos scale must be non-zero on every dimension")
        if self.clip is not None and (
            len(self.clip) != 2 or self.clip[0] > self.clip[1]
        ):
            raise ValueError(
                f"Qpos clip must be an ordered (low, high) pair, got {self.clip}"
            )

    @property
    def input_dim(self) -> int:
        return self._resolved_dim

    @property
    def model_dim(self) -> int:
        return self._resolved_dim

    @property
    def output_dim(self) -> int:
        return self._resolved_dim

    @property
    def _resolved_dim(self) -> int:
        if self.dim is None:
            raise ValueError("Qpos dim must be set before layout conversion")
        return self.dim


@dataclass(frozen=True)
class Gripper(Qpos):
    """One-dimensional continuous gripper field."""

    dim: int | None = field(default=1, init=False)


@dataclass(frozen=True)
class BinaryQpos(ActionField):
    """Binary actuator represented as -1 or +1 in model space."""

    model_repr: ModelRepr = field(default="absolute", init=False)
    min_value: float = 0.0
    max_value: float = 1.0
    threshold: float = 0.5
    invert: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_value >= self.max_value:
            raise ValueError("BinaryQpos min_value must be less than max_value")
        if not self.min_value <= self.threshold <= self.max_value:
            raise ValueError("BinaryQpos threshold must lie within min/max values")

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def model_dim(self) -> int:
        return 1

    @property
    def output_dim(self) -> int:
        return 1


@dataclass(frozen=True)
class EEF(ActionField):
    """End-effector pose with explicit rotations and an atomic step mask.

    Within each action step, every EEF mask value must be uniformly valid or
    uniformly invalid. Invalid EEF steps skip geometric conversion.
    """

    input_rot: RotRepr = "rot6d"
    model_rot: RotRepr = "rot6d"
    output_rot: RotRepr = "rot6d"
    eef_tf: Matrix4 | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for label, value in (
            ("input_rot", self.input_rot),
            ("model_rot", self.model_rot),
            ("output_rot", self.output_rot),
        ):
            if value not in ROT_DIM:
                raise ValueError(f"unknown {label}: {value!r}")
        _validate_transform(self.eef_tf, "EEF eef_tf")

    @property
    def input_dim(self) -> int:
        return 3 + ROT_DIM[self.input_rot]

    @property
    def model_dim(self) -> int:
        return 3 + ROT_DIM[self.model_rot]

    @property
    def output_dim(self) -> int:
        return 3 + ROT_DIM[self.output_rot]

    @property
    def mask_is_atomic(self) -> bool:
        """Whether invalid rows must be skipped as one whole component."""
        return True


@dataclass(frozen=True)
class Chassis(ActionField):
    """Planar chassis pose represented as reference-local dx, dy, dyaw."""

    model_repr: ModelRepr = field(default="relative", init=False)

    @property
    def input_dim(self) -> int:
        return 3

    @property
    def model_dim(self) -> int:
        return 3

    @property
    def output_dim(self) -> int:
        return 3


@dataclass(frozen=True)
class Padding(ActionField):
    """Zero-filled field reserving dimensions in every full layout."""

    model_repr: ModelRepr = field(default="absolute", init=False)
    input_used: bool = field(default=False, init=False)
    model_used: bool = field(default=False, init=False)
    output_used: bool = field(default=False, init=False)
    dim: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim < 1:
            raise ValueError(f"Padding dim must be positive, got {self.dim}")

    @property
    def input_dim(self) -> int:
        return self.dim

    @property
    def model_dim(self) -> int:
        return self.dim

    @property
    def output_dim(self) -> int:
        return self.dim


@dataclass(frozen=True)
class LandmarkEEF(EEF):
    """EEF derived from per-frame landmark positions and orientations."""

    wrist_index: int = 0
    center_indices: tuple[int, ...] = (5, 9, 13, 17)
    center_alpha: float = 0.5
    source_rotation: Literal["xyzw", "wxyz"] = "wxyz"
    position_filter_window: int | None = None
    position_filter_polyorder: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.wrist_index < 0:
            raise ValueError("LandmarkEEF wrist_index must be non-negative")
        if not self.center_indices or any(index < 0 for index in self.center_indices):
            raise ValueError(
                "LandmarkEEF center_indices must contain non-negative indices"
            )
        if not 0.0 <= self.center_alpha <= 1.0:
            raise ValueError("LandmarkEEF center_alpha must be in [0, 1]")
        if self.source_rotation not in ("xyzw", "wxyz"):
            raise ValueError("LandmarkEEF source_rotation must be 'xyzw' or 'wxyz'")
        if self.position_filter_window is not None:
            if self.position_filter_window < 1:
                raise ValueError("LandmarkEEF position_filter_window must be positive")
            if self.position_filter_window % 2 == 0:
                raise ValueError("LandmarkEEF position_filter_window must be odd")
            if self.position_filter_polyorder >= self.position_filter_window:
                raise ValueError(
                    "LandmarkEEF polyorder must be smaller than filter window"
                )


@dataclass(frozen=True)
class UrdfEEF(EEF):
    """EEF computed from qpos by URDF forward kinematics."""

    urdf_path: str = ""
    frame: str = ""
    joint_names: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.urdf_path:
            raise ValueError("UrdfEEF requires urdf_path")
        if not self.frame:
            raise ValueError("UrdfEEF requires frame")
        if isinstance(self.joint_names, list):
            object.__setattr__(self, "joint_names", tuple(self.joint_names))
        if isinstance(self.source_names, list):
            object.__setattr__(self, "source_names", tuple(self.source_names))


def _validate_transform(value: Matrix4 | None, label: str) -> None:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {array.shape}")
    if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"{label} last row must be [0, 0, 0, 1]")
