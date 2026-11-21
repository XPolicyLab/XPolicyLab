"""Backend-preserving image resize and spatial shape planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as torch_functional

from morph.types.fields import Array
from morph.types.fields import ArrayT

Interpolation = Literal["nearest", "bilinear", "bicubic", "lanczos"]
FixedResizeMode = Literal["direct", "padding"]
ResizeMode = Literal["direct", "padding", "area"]

_INTERPOLATIONS = frozenset(Interpolation.__args__)
_RESIZE_MODES = frozenset(ResizeMode.__args__)
_FIXED_RESIZE_MODES = frozenset(FixedResizeMode.__args__)


@dataclass(frozen=True)
class ResizeConfig:
    """Complete fixed-shape or target-area image resize configuration."""

    width: int | None = None
    height: int | None = None
    mode: ResizeMode = "direct"
    interpolation: Interpolation = "bilinear"
    target_area: int | None = None
    multiple: int | None = None

    def __post_init__(self) -> None:
        _validate_interpolation(self.interpolation)
        if self.mode not in _RESIZE_MODES:
            raise ValueError(
                f"resize mode must be one of {sorted(_RESIZE_MODES)}"
            )
        if self.mode in _FIXED_RESIZE_MODES:
            if self.width is None or self.height is None:
                raise ValueError(
                    f"resize mode={self.mode!r} requires width and height"
                )
            _validate_size(self.width, self.height)
            if self.target_area is not None or self.multiple is not None:
                raise ValueError(
                    f"resize mode={self.mode!r} does not accept "
                    "target_area or multiple"
                )
            return
        if self.width is not None or self.height is not None:
            raise ValueError(
                "resize mode='area' does not accept width or height"
            )
        if self.target_area is None or self.target_area <= 0:
            raise ValueError(
                "resize mode='area' requires positive target_area"
            )
        if self.multiple is None or self.multiple <= 0:
            raise ValueError(
                "resize mode='area' requires positive multiple"
            )

    def apply(
        self,
        value: ArrayT,
        *,
        spatial_axes: tuple[int, int] = (-2, -1),
    ) -> ArrayT:
        """Apply this configuration while preserving the input backend."""

        if self.mode == "area":
            assert self.target_area is not None
            assert self.multiple is not None
            return resize_to_area(
                value,
                target_area=self.target_area,
                multiple=self.multiple,
                interpolation=self.interpolation,
                spatial_axes=spatial_axes,
            )
        assert self.width is not None
        assert self.height is not None
        return resize(
            value,
            width=self.width,
            height=self.height,
            mode=self.mode,
            interpolation=self.interpolation,
            spatial_axes=spatial_axes,
        )


@dataclass(frozen=True)
class ResizePlan:
    """Integer resize and padding geometry for a fixed output shape."""

    resize_width: int
    resize_height: int
    output_width: int
    output_height: int
    pad_left: int
    pad_top: int

    @property
    def pad_right(self) -> int:
        """Return the number of pixels padded on the right."""

        return self.output_width - self.resize_width - self.pad_left

    @property
    def pad_bottom(self) -> int:
        """Return the number of pixels padded on the bottom."""

        return self.output_height - self.resize_height - self.pad_top


@dataclass(frozen=True)
class AreaResizePlan:
    """Integer resize and crop geometry for target-area resize."""

    resize_width: int
    resize_height: int
    output_width: int
    output_height: int
    crop_left: int
    crop_top: int

    @property
    def crop_right(self) -> int:
        """Return the exclusive horizontal crop boundary."""

        return self.crop_left + self.output_width

    @property
    def crop_bottom(self) -> int:
        """Return the exclusive vertical crop boundary."""

        return self.crop_top + self.output_height


def resize(
    value: ArrayT,
    *,
    width: int,
    height: int,
    mode: FixedResizeMode = "direct",
    interpolation: Interpolation = "bicubic",
    spatial_axes: tuple[int, int] = (-2, -1),
    padding_value: float = 0.0,
) -> ArrayT:
    """Resize two spatial axes directly or with aspect-preserving padding.

    Args:
        value: NumPy array or PyTorch tensor.
        width: Exact output width.
        height: Exact output height.
        mode: ``direct`` stretches to the requested shape. ``padding`` keeps
            aspect ratio, fits inside the requested shape, and pads equally
            on opposite sides, with any odd pixel placed on bottom or right.
        interpolation: Spatial interpolation kernel.
        spatial_axes: Height and width axes in ``value``.
        padding_value: Constant used only by ``padding`` mode.

    Returns:
        A resized value with the same backend, dtype, device, and axis order.
    """

    _validate_array(value)
    _validate_size(width, height)
    _validate_interpolation(interpolation)
    if mode not in _FIXED_RESIZE_MODES:
        raise ValueError(
            "fixed resize mode must be one of "
            f"{sorted(_FIXED_RESIZE_MODES)}"
        )
    axes = _normalize_spatial_axes(value.ndim, spatial_axes)
    plan = resize_plan(
        source_width=int(value.shape[axes[1]]),
        source_height=int(value.shape[axes[0]]),
        width=width,
        height=height,
        mode=mode,
    )
    resized = _resize_direct(
        value,
        width=plan.resize_width,
        height=plan.resize_height,
        interpolation=interpolation,
        spatial_axes=axes,
    )
    if mode == "direct":
        return resized
    return _pad_spatial(
        resized,
        axes,
        left=plan.pad_left,
        right=plan.pad_right,
        top=plan.pad_top,
        bottom=plan.pad_bottom,
        padding_value=padding_value,
    )


def resize_plan(
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    mode: FixedResizeMode = "direct",
) -> ResizePlan:
    """Plan direct resize or aspect-preserving resize with centered padding."""

    _validate_size(source_width, source_height, label="source")
    _validate_size(width, height)
    if mode not in _FIXED_RESIZE_MODES:
        raise ValueError(
            "fixed resize mode must be one of "
            f"{sorted(_FIXED_RESIZE_MODES)}"
        )
    if mode == "direct":
        return ResizePlan(
            resize_width=width,
            resize_height=height,
            output_width=width,
            output_height=height,
            pad_left=0,
            pad_top=0,
        )
    scale = min(width / source_width, height / source_height)
    resize_width = min(
        width,
        max(1, _round_half_up(source_width * scale)),
    )
    resize_height = min(
        height,
        max(1, _round_half_up(source_height * scale)),
    )
    return ResizePlan(
        resize_width=resize_width,
        resize_height=resize_height,
        output_width=width,
        output_height=height,
        pad_left=(width - resize_width) // 2,
        pad_top=(height - resize_height) // 2,
    )


def resize_to_area(
    value: ArrayT,
    *,
    target_area: int,
    multiple: int,
    interpolation: Interpolation = "bicubic",
    spatial_axes: tuple[int, int] = (-2, -1),
) -> ArrayT:
    """Resize with preserved aspect ratio, then crop to multiples.

    Width-aligned and height-aligned scales around ``target_area`` are tested.
    The resized area closest to the target wins, with crop size as the first
    tie-breaker. Vertical remainder is removed from the top; horizontal
    remainder is center-cropped with the odd pixel removed from the right.
    """

    _validate_array(value)
    _validate_interpolation(interpolation)
    axes = _normalize_spatial_axes(value.ndim, spatial_axes)
    plan = area_resize_plan(
        source_width=int(value.shape[axes[1]]),
        source_height=int(value.shape[axes[0]]),
        target_area=target_area,
        multiple=multiple,
    )
    resized = _resize_direct(
        value,
        width=plan.resize_width,
        height=plan.resize_height,
        interpolation=interpolation,
        spatial_axes=axes,
    )
    moved = _move_spatial_last(resized, axes)
    cropped = moved[
        ...,
        plan.crop_top : plan.crop_bottom,
        plan.crop_left : plan.crop_right,
    ]
    return _restore_spatial_axes(cropped, axes)


def area_resize_plan(
    *,
    source_width: int,
    source_height: int,
    target_area: int,
    multiple: int,
) -> AreaResizePlan:
    """Plan aspect-preserving resize and minimum-loss multiple crop."""

    _validate_size(source_width, source_height, label="source")
    if target_area <= 0:
        raise ValueError("target_area must be positive")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    ideal_scale = math.sqrt(
        target_area / (source_width * source_height)
    )
    ideal_width = source_width * ideal_scale
    ideal_height = source_height * ideal_scale
    candidates: list[tuple[tuple[int, ...], AreaResizePlan]] = []
    for resize_width in _multiple_neighbors(ideal_width, multiple):
        resize_height = max(
            1,
            _round_half_up(
                resize_width * source_height / source_width
            ),
        )
        output_height = resize_height // multiple * multiple
        if output_height <= 0:
            continue
        candidates.append(
            _area_candidate(
                resize_width=resize_width,
                resize_height=resize_height,
                output_width=resize_width,
                output_height=output_height,
                crop_left=0,
                crop_top=resize_height - output_height,
                target_area=target_area,
            )
        )
    for resize_height in _multiple_neighbors(ideal_height, multiple):
        resize_width = max(
            1,
            _round_half_up(
                resize_height * source_width / source_height
            ),
        )
        output_width = resize_width // multiple * multiple
        if output_width <= 0:
            continue
        crop_width = resize_width - output_width
        candidates.append(
            _area_candidate(
                resize_width=resize_width,
                resize_height=resize_height,
                output_width=output_width,
                output_height=resize_height,
                crop_left=crop_width // 2,
                crop_top=0,
                target_area=target_area,
            )
        )
    if not candidates:
        raise ValueError(
            "target_area and multiple cannot produce a positive output shape"
        )
    return min(candidates, key=lambda item: item[0])[1]


def _area_candidate(
    *,
    resize_width: int,
    resize_height: int,
    output_width: int,
    output_height: int,
    crop_left: int,
    crop_top: int,
    target_area: int,
) -> tuple[tuple[int, ...], AreaResizePlan]:
    plan = AreaResizePlan(
        resize_width=resize_width,
        resize_height=resize_height,
        output_width=output_width,
        output_height=output_height,
        crop_left=crop_left,
        crop_top=crop_top,
    )
    resized_area = resize_width * resize_height
    output_area = output_width * output_height
    score = (
        abs(resized_area - target_area),
        resized_area - output_area,
        abs(output_area - target_area),
        resized_area,
    )
    return score, plan


def _multiple_neighbors(value: float, multiple: int) -> tuple[int, ...]:
    lower = max(multiple, math.floor(value / multiple) * multiple)
    upper = max(multiple, math.ceil(value / multiple) * multiple)
    return tuple(sorted({lower, upper}))


def _resize_direct(
    value: ArrayT,
    *,
    width: int,
    height: int,
    interpolation: Interpolation,
    spatial_axes: tuple[int, int],
) -> ArrayT:
    source_height = int(value.shape[spatial_axes[0]])
    source_width = int(value.shape[spatial_axes[1]])
    if (source_height, source_width) == (height, width):
        return value
    moved = _move_spatial_last(value, spatial_axes)
    prefix = moved.shape[:-2]
    flat = moved.reshape(
        (math.prod(prefix), source_height, source_width)
    )
    if interpolation != "lanczos" and flat.shape[0] > 0:
        output = _torch_resize(
            flat,
            width=width,
            height=height,
            interpolation=interpolation,
        )
        if output is not None:
            output = output.reshape(prefix + (height, width))
            return _restore_spatial_axes(output, spatial_axes)
    work = _working_array(flat)
    height_indices, height_weights = _axis_weights(
        source_height,
        height,
        interpolation,
    )
    width_indices, width_weights = _axis_weights(
        source_width,
        width,
        interpolation,
    )
    if isinstance(work, torch.Tensor):
        h_indices = torch.as_tensor(
            height_indices,
            dtype=torch.long,
            device=work.device,
        )
        h_weights = torch.as_tensor(
            height_weights,
            dtype=work.dtype,
            device=work.device,
        )
        w_indices = torch.as_tensor(
            width_indices,
            dtype=torch.long,
            device=work.device,
        )
        w_weights = torch.as_tensor(
            width_weights,
            dtype=work.dtype,
            device=work.device,
        )
        vertical = (
            work[:, h_indices, :]
            * h_weights[None, :, :, None]
        ).sum(dim=2)
        output = (
            vertical[:, :, w_indices]
            * w_weights[None, None, :, :]
        ).sum(dim=-1)
    else:
        vertical = (
            work[:, height_indices, :]
            * height_weights[None, :, :, None]
        ).sum(axis=2)
        output = (
            vertical[:, :, width_indices]
            * width_weights[None, None, :, :]
        ).sum(axis=-1)
    output = _restore_dtype(output, value)
    output = output.reshape(prefix + (height, width))
    return _restore_spatial_axes(output, spatial_axes)


def _torch_resize(
    value: Array,
    *,
    width: int,
    height: int,
    interpolation: Interpolation,
) -> Array | None:
    """Use optimized Torch kernels while preserving the caller's backend."""
    numpy_input = isinstance(value, np.ndarray)
    if numpy_input:
        try:
            source = torch.from_numpy(np.ascontiguousarray(value))
        except TypeError:
            return None
    else:
        source = value
    work = _working_array(source)
    mode = "nearest-exact" if interpolation == "nearest" else interpolation
    options = {} if interpolation == "nearest" else {"align_corners": False}
    output = torch_functional.interpolate(
        work[:, None],
        size=(height, width),
        mode=mode,
        **options,
    )[:, 0]
    output = _restore_dtype(output, source)
    if numpy_input:
        return output.numpy()
    return output


def _axis_weights(
    source_size: int,
    target_size: int,
    interpolation: Interpolation,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = (
        (np.arange(target_size, dtype=np.float64) + 0.5)
        * source_size
        / target_size
        - 0.5
    )
    if interpolation == "nearest":
        indices = np.floor(coordinates + 0.5).astype(np.int64)[:, None]
        indices = np.clip(indices, 0, source_size - 1)
        return indices, np.ones((target_size, 1), dtype=np.float64)
    if interpolation == "bilinear":
        offsets = np.asarray((0, 1), dtype=np.int64)
        kernel = _linear_kernel
    elif interpolation == "bicubic":
        offsets = np.arange(-1, 3, dtype=np.int64)
        kernel = _cubic_kernel
    else:
        offsets = np.arange(-3, 5, dtype=np.int64)
        kernel = _lanczos4_kernel
    base = np.floor(coordinates).astype(np.int64)
    raw_indices = base[:, None] + offsets[None, :]
    weights = kernel(coordinates[:, None] - raw_indices)
    indices = np.clip(raw_indices, 0, source_size - 1)
    weights /= weights.sum(axis=1, keepdims=True)
    return indices, weights


def _linear_kernel(distance: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.abs(distance), 0.0, None)


def _cubic_kernel(distance: np.ndarray) -> np.ndarray:
    absolute = np.abs(distance)
    parameter = -0.75
    inner = (
        (parameter + 2.0) * absolute**3
        - (parameter + 3.0) * absolute**2
        + 1.0
    )
    outer = (
        parameter * absolute**3
        - 5.0 * parameter * absolute**2
        + 8.0 * parameter * absolute
        - 4.0 * parameter
    )
    return np.where(
        absolute <= 1.0,
        inner,
        np.where(absolute < 2.0, outer, 0.0),
    )


def _lanczos4_kernel(distance: np.ndarray) -> np.ndarray:
    absolute = np.abs(distance)
    return np.where(
        absolute < 4.0,
        np.sinc(distance) * np.sinc(distance / 4.0),
        0.0,
    )


def _working_array(value: Array) -> Array:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype not in {
            torch.float16,
            torch.bfloat16,
        }:
            return value
        return value.to(torch.float32)
    if np.issubdtype(value.dtype, np.floating):
        if value.dtype == np.float16:
            return value.astype(np.float32)
        return value
    return value.astype(np.float32)


def _restore_dtype(value: Array, source: Array) -> Array:
    if isinstance(value, torch.Tensor):
        if source.is_floating_point():
            return value.to(source.dtype)
        if source.dtype == torch.bool:
            return (value >= 0.5).to(source.dtype)
        limits = torch.iinfo(source.dtype)
        return value.round().clamp(limits.min, limits.max).to(source.dtype)
    if np.issubdtype(source.dtype, np.floating):
        return value.astype(source.dtype, copy=False)
    if np.issubdtype(source.dtype, np.bool_):
        return (value >= 0.5).astype(source.dtype)
    limits = np.iinfo(source.dtype)
    return np.clip(np.rint(value), limits.min, limits.max).astype(source.dtype)


def _pad_spatial(
    value: ArrayT,
    spatial_axes: tuple[int, int],
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    padding_value: float,
) -> ArrayT:
    moved = _move_spatial_last(value, spatial_axes)
    if isinstance(moved, torch.Tensor):
        padded = torch_functional.pad(
            moved,
            (left, right, top, bottom),
            mode="constant",
            value=padding_value,
        )
    else:
        widths = [(0, 0)] * moved.ndim
        widths[-2] = (top, bottom)
        widths[-1] = (left, right)
        padded = np.pad(
            moved,
            widths,
            mode="constant",
            constant_values=padding_value,
        )
    return _restore_spatial_axes(padded, spatial_axes)


def _move_spatial_last(
    value: ArrayT,
    spatial_axes: tuple[int, int],
) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.movedim(value, spatial_axes, (-2, -1))
    return np.moveaxis(value, spatial_axes, (-2, -1))


def _restore_spatial_axes(
    value: ArrayT,
    spatial_axes: tuple[int, int],
) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.movedim(value, (-2, -1), spatial_axes)
    return np.moveaxis(value, (-2, -1), spatial_axes)


def _normalize_spatial_axes(
    ndim: int,
    spatial_axes: tuple[int, int],
) -> tuple[int, int]:
    if ndim < 2:
        raise ValueError(f"image value must have at least 2 dims, got {ndim}")
    if len(spatial_axes) != 2:
        raise ValueError("spatial_axes must contain height and width axes")
    axes = tuple(axis % ndim for axis in spatial_axes)
    if axes[0] == axes[1]:
        raise ValueError("spatial_axes must contain two distinct axes")
    return axes


def _validate_array(value: Array) -> None:
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(
            "image resize requires a NumPy array or PyTorch tensor"
        )


def _validate_size(width: int, height: int, *, label: str = "output") -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} width and height must be positive")


def _validate_interpolation(interpolation: str) -> None:
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            "resize interpolation must be one of "
            f"{sorted(_INTERPOLATIONS)}"
        )


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


__all__ = [
    "AreaResizePlan",
    "FixedResizeMode",
    "Interpolation",
    "ResizeConfig",
    "ResizeMode",
    "ResizePlan",
    "area_resize_plan",
    "resize",
    "resize_plan",
    "resize_to_area",
]
