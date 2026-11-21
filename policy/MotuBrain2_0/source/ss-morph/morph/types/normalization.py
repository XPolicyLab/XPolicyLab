"""Normalization declarations independent of array backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

from morph.types.fields import ActionField
from morph.types.fields import NormMethod

StatValues: TypeAlias = (
    np.ndarray | torch.Tensor | Sequence[float] | Sequence[Sequence[float]]
)
EmbodimentInput: TypeAlias = str | np.ndarray | Sequence[object]


@dataclass(frozen=True)
class NormalizationStats:
    """Resolved affine statistics expressed as center and scale."""

    center: StatValues
    scale: StatValues
    mask: StatValues | None = None


StatsSource: TypeAlias = NormalizationStats | Mapping[str, object] | str | Path
ResolvedStats: TypeAlias = NormalizationStats | Mapping[str, NormalizationStats]

_EXPORTED_QUANTILE_NAMES = {
    (0.01, 0.99): "q99",
    (0.001, 0.999): "q999",
    (0.0001, 0.9999): "q9999",
}


@dataclass(frozen=True)
class NormalizationConfig:
    """Action normalization statistics and output bounds.

    ``stats`` may describe one fixed affine transform or contain an
    ActionNormExporter JSON object keyed first by embodiment. Embodiment-keyed
    stats use stepwise entries for relative fields and global entries for
    absolute fields. Fixed affine statistics are already resolved; their
    array shape determines broadcasting.
    ``clip`` optionally bounds active dimensions after normalization; it does
    not affect denormalization.
    """

    method: NormMethod
    stats: StatsSource
    quantiles: tuple[float, float] | None = None
    eps: float = 1e-6
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.method not in ("zscore", "minmax", "quantile"):
            raise ValueError(f"unknown normalization method: {self.method!r}")
        if self.eps <= 0:
            raise ValueError(f"normalization eps must be positive, got {self.eps}")
        if self.method == "quantile":
            if self.quantiles is None:
                raise ValueError("quantile normalization requires quantiles")
            low, high = self.quantiles
            if not 0 <= low < high <= 1:
                raise ValueError(
                    "normalization quantiles must satisfy 0 <= low < high <= 1"
                )
        elif self.quantiles is not None:
            raise ValueError("quantiles are only valid for quantile normalization")
        if self.clip is not None and (
            len(self.clip) != 2 or self.clip[0] > self.clip[1]
        ):
            raise ValueError(
                "normalization clip must be an ordered (low, high) pair, "
                f"got {self.clip}"
            )

    def build(self, *, fields: Sequence[ActionField]):
        """Resolve this declaration into a backend-polymorphic normalizer."""
        from morph.convs import normalization

        return normalization.build_normalizer(
            self,
            fields=fields,
        )


def resolve_stats(
    config: NormalizationConfig,
    *,
    fields: Sequence[ActionField],
) -> ResolvedStats:
    """Load fixed or embodiment-keyed statistics configured by ``config``."""
    source = config.stats
    if isinstance(source, NormalizationStats):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if "://" in str(source):
            raise ValueError("remote normalization paths must be loaded by the caller")
        with path.open(encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, Mapping):
            raise ValueError("normalization stats file must contain an object")
    else:
        values = source
    if values and all(
        isinstance(value, Mapping) and "action_norm" in value
        for value in values.values()
    ):
        return {
            str(embodiment): _stats_from_export(
                config,
                str(embodiment),
                value,
                fields,
            )
            for embodiment, value in values.items()
        }
    return _stats_from_mapping(config.method, values)


def _stats_from_export(
    config: NormalizationConfig,
    embodiment: str,
    values: Mapping[str, object],
    fields: Sequence[ActionField],
) -> NormalizationStats:
    """Select statistics by representation, then combine one full-layout affine."""
    dims = [item.model_dim for item in fields]
    relative = np.repeat([item.model_repr == "relative" for item in fields], dims)
    active = np.repeat([item.model_used for item in fields], dims)
    stats = {}
    for scope, selected in (("global", ~relative), ("stepwise", relative)):
        if not np.any(selected & active):
            continue
        try:
            entry = values["action_norm"][scope][_exported_entry_name(config)]
            stats[scope] = _stats_from_mapping(config.method, entry)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                f"invalid or missing {embodiment!r} action_norm.{scope}."
                f"{_exported_entry_name(config)}"
            ) from error
        item = stats[scope]
        shape = np.shape(item.center)
        if (
            shape != np.shape(item.scale)
            or len(shape) != (2 if scope == "stepwise" else 1)
            or shape[-1] != len(relative)
        ):
            raise ValueError(
                f"{embodiment} {scope} statistics have invalid center/scale shapes"
            )
        if item.mask is not None and np.shape(item.mask) not in (
            (len(relative),),
            shape,
        ):
            raise ValueError(
                f"{embodiment} {scope} normalization mask shape differs from stats"
            )
    if len(stats) == 1:
        return next(iter(stats.values()))
    global_stats, stepwise = stats["global"], stats["stepwise"]
    return NormalizationStats(
        center=np.where(relative, stepwise.center, global_stats.center),
        scale=np.where(relative, stepwise.scale, global_stats.scale),
        mask=np.where(
            relative,
            True if stepwise.mask is None else stepwise.mask,
            True if global_stats.mask is None else global_stats.mask,
        ),
    )


def _exported_entry_name(config: NormalizationConfig) -> str:
    """Map a quantile declaration to ActionNormExporter's stable key."""
    if config.method != "quantile":
        return config.method
    for bounds, name in _EXPORTED_QUANTILE_NAMES.items():
        if all(
            abs(actual - expected) < 1e-12
            for actual, expected in zip(config.quantiles, bounds, strict=True)
        ):
            return name
    raise ValueError(
        "exported quantile normalization requires one of "
        f"{sorted(_EXPORTED_QUANTILE_NAMES)}, got {config.quantiles}"
    )


def _stats_from_mapping(
    method: NormMethod,
    values: Mapping[str, object],
) -> NormalizationStats:
    mask = values.get("mask")
    if method == "zscore":
        mean = values.get("mean", values.get("action_mean"))
        std = values.get("std", values.get("action_std"))
        if mean is not None and std is not None:
            return NormalizationStats(center=mean, scale=std, mask=mask)
        if mean is not None or std is not None:
            raise ValueError("zscore stats require both mean and std")

    low = values.get("low", values.get("min", values.get("action_min")))
    high = values.get("high", values.get("max", values.get("action_max")))
    if low is None or high is None:
        raise ValueError(f"{method} stats require low/high or min/max values")
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    if low_array.shape != high_array.shape:
        raise ValueError(
            "normalization low/high shapes differ: "
            f"{low_array.shape} and {high_array.shape}"
        )
    return NormalizationStats(
        center=(low_array + high_array) / 2,
        scale=(high_array - low_array) / 2,
        mask=mask,
    )
