"""Backend-preserving affine normalization and checkpoint adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from morph.convs import geometry
from morph.types.fields import ActionField
from morph.types.fields import Array
from morph.types.fields import ArrayT
from morph.types.normalization import EmbodimentInput
from morph.types.normalization import NormalizationConfig
from morph.types.normalization import NormalizationStats
from morph.types.normalization import resolve_stats

ActionNorm = tuple[Array, Array, Array]
BUFFER_NAMES = (
    "action_norm_min",
    "action_norm_max",
    "action_norm_mask",
)


@dataclass(frozen=True)
class ActionNormalizer:
    """Resolved affine normalizer for one full model layout."""

    center: object
    scale: object
    mask: object
    clip: tuple[float, float] | None = None

    def _align(self, action: Array) -> tuple[Array, Array, Array]:
        center = geometry.as_like(self.center, action, dtype=_float_dtype(action))
        scale = geometry.as_like(self.scale, action, dtype=_float_dtype(action))
        mask = geometry.bool_array(self.mask, action)
        if action.ndim == 0 or action.shape[-1] != center.shape[-1]:
            raise ValueError(
                f"action shape {action.shape} is incompatible with "
                f"normalization shape {center.shape}"
            )
        if center.ndim == 2:
            if action.ndim < 2:
                raise ValueError(
                    "stepwise normalization requires an action step dimension"
                )
            length = action.shape[-2]
            if length > center.shape[0]:
                raise ValueError(
                    f"action length {length} exceeds normalization length "
                    f"{center.shape[0]}"
                )
            center = center[:length]
            scale = scale[:length]
            mask = mask[:length]
        return center, scale, mask

    def normalize(
        self,
        action: ArrayT,
        embodiment: EmbodimentInput | None = None,
    ) -> ArrayT:
        """Normalize active dimensions while preserving backend and dtype."""
        del embodiment
        center, scale, mask = self._align(action)
        normalized = (_to_float(action) - center) / scale
        if self.clip is not None:
            normalized = _clip(normalized, self.clip)
        return _where(mask, _cast(normalized, action), action)

    def denormalize(
        self,
        action: ArrayT,
        embodiment: EmbodimentInput | None = None,
    ) -> ArrayT:
        """Restore normalized active dimensions."""
        del embodiment
        center, scale, mask = self._align(action)
        restored = _to_float(action) * scale + center
        return _where(mask, _cast(restored, action), action)


@dataclass(frozen=True)
class EmbodimentActionNormalizer:
    """Select affine statistics using labels aligned with action batch dims."""

    normalizers: Mapping[str, ActionNormalizer]

    def _labels(
        self,
        action: Array,
        embodiment: EmbodimentInput | None,
    ) -> np.ndarray:
        if action.ndim < 2:
            raise ValueError(
                "embodiment normalization requires action shape " "batch_shape + (T, D)"
            )
        batch_shape = action.shape[:-2]
        if not batch_shape:
            if not isinstance(embodiment, str) or not embodiment:
                raise ValueError(
                    "embodiment normalization requires embodiment as a "
                    "non-empty string for action shape (T, D)"
                )
            return np.asarray([embodiment], dtype=object)
        if embodiment is None:
            raise ValueError("embodiment normalization requires embodiment labels")
        labels = np.asarray(embodiment, dtype=object)
        if labels.shape != batch_shape:
            raise ValueError(
                f"embodiment shape {labels.shape} must match action batch "
                f"shape {batch_shape} for action shape {action.shape}"
            )
        return labels.reshape(-1)

    def _apply(
        self,
        action: ArrayT,
        embodiment: EmbodimentInput | None,
        method: Literal["normalize", "denormalize"],
    ) -> ArrayT:
        labels = self._labels(action, embodiment)
        flat_action = action.reshape((-1,) + action.shape[-2:])
        outputs = []
        for index, label in enumerate(labels):
            if not isinstance(label, str) or not label:
                raise ValueError(
                    "embodiment labels must be non-empty strings, got " f"{label!r}"
                )
            try:
                normalizer = self.normalizers[label]
            except KeyError as error:
                raise ValueError(
                    f"normalization stats for embodiment {label!r} are "
                    f"unavailable; available embodiments: "
                    f"{sorted(self.normalizers)}"
                ) from error
            if method == "normalize":
                outputs.append(normalizer.normalize(flat_action[index]))
            else:
                outputs.append(normalizer.denormalize(flat_action[index]))
        if isinstance(action, torch.Tensor):
            result = torch.stack(outputs)
        else:
            result = np.stack(outputs)
        return result.reshape(action.shape)

    def normalize(
        self,
        action: ArrayT,
        embodiment: EmbodimentInput | None = None,
    ) -> ArrayT:
        """Normalize action sequences with batch-aligned embodiment stats.

        ``action`` has shape ``batch_shape + (T, D)``. ``embodiment`` is one
        string when ``batch_shape`` is empty, otherwise its shape must exactly
        equal ``batch_shape``; labels are not repeated along ``T``.
        """
        return self._apply(action, embodiment, "normalize")

    def denormalize(
        self,
        action: ArrayT,
        embodiment: EmbodimentInput | None = None,
    ) -> ArrayT:
        """Denormalize action sequences with batch-aligned embodiment stats.

        ``action`` has shape ``batch_shape + (T, D)``. ``embodiment`` is one
        string when ``batch_shape`` is empty, otherwise its shape must exactly
        equal ``batch_shape``; labels are not repeated along ``T``.
        """
        return self._apply(action, embodiment, "denormalize")


def _select_stats_entry(
    stats: dict[str, Any],
    *,
    scope: str,
    mode: str,
) -> dict[str, Any]:
    try:
        return stats["action_norm"][scope][mode]
    except KeyError as error:
        scopes = stats.get("action_norm", {})
        available = {
            scope_name: sorted(scope_stats)
            for scope_name, scope_stats in scopes.items()
        }
        raise KeyError(
            f"action normalization {scope=}, {mode=} was not found; "
            f"available entries: {available}"
        ) from error


def _entry_to_low_high(
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert min/max or mean/std stats to a shared affine interval."""
    if "action_mean" in entry and "action_std" in entry:
        mean = np.asarray(entry["action_mean"], dtype=np.float32)
        std = np.asarray(entry["action_std"], dtype=np.float32)
        if mean.shape != std.shape:
            raise ValueError(
                "action normalization mean/std shapes differ: "
                f"{mean.shape} and {std.shape}"
            )
        return mean - std, mean + std
    return (
        np.asarray(entry["action_min"], dtype=np.float32),
        np.asarray(entry["action_max"], dtype=np.float32),
    )


def load_action_norm(
    path: str | Path,
    *,
    action_dim: int,
    scope: str = "global",
    mode: str = "minmax",
    dims: list[int] | None = None,
    eps: float = 1e-2,
    like: Array | None = None,
) -> ActionNorm:
    """Load action bounds and their active mask from an exported stats file.

    Stats are loaded as NumPy metadata by default. Passing ``like`` materializes
    them on that array's backend and, for tensors, its device.
    """
    with Path(path).open(encoding="utf-8") as file:
        stats = json.load(file)
    entry = _select_stats_entry(stats, scope=scope, mode=mode)
    low, high = _entry_to_low_high(entry)
    if low.shape != high.shape:
        raise ValueError(
            "action normalization min/max shapes differ: "
            f"{low.shape} and {high.shape}"
        )
    expected_rank = 2 if scope == "stepwise" else 1
    if low.ndim != expected_rank:
        raise ValueError(
            f"{scope} action normalization must have rank {expected_rank}, "
            f"got {low.ndim}"
        )
    if low.shape[-1] != action_dim:
        raise ValueError(
            f"action normalization dimension {low.shape[-1]} does not match "
            f"action_dim={action_dim}"
        )

    dim_mask = np.zeros(action_dim, dtype=bool)
    if dims is None:
        dim_mask.fill(True)
    else:
        invalid = [dim for dim in dims if not 0 <= dim < action_dim]
        if invalid:
            raise ValueError(
                "action_norm_dims contains out-of-range dimensions: " f"{invalid}"
            )
        dim_mask[dims] = True
    mask = dim_mask if low.ndim == 1 else np.broadcast_to(dim_mask, low.shape).copy()
    mask &= high - low >= eps
    if like is None:
        return low, high, mask
    return (
        geometry.as_like(low, like, dtype=_float_dtype(like)),
        geometry.as_like(high, like, dtype=_float_dtype(like)),
        geometry.bool_array(mask, like),
    )


def _shape(value: object) -> tuple[int, ...]:
    return tuple(value.shape) if hasattr(value, "shape") else np.asarray(value).shape


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_normalizer(
    config: NormalizationConfig,
    *,
    fields: Sequence[ActionField],
) -> ActionNormalizer | EmbodimentActionNormalizer:
    """Validate stats and build a backend-neutral normalizer."""
    stats = resolve_stats(config, fields=fields)
    action_dim = sum(item.model_dim for item in fields)
    model_mask = [item.model_used for item in fields for _ in range(item.model_dim)]
    if isinstance(stats, Mapping):
        return EmbodimentActionNormalizer(
            normalizers={
                embodiment: _build_normalizer(
                    config,
                    stats=embodiment_stats,
                    action_dim=action_dim,
                    model_mask=model_mask,
                )
                for embodiment, embodiment_stats in stats.items()
            }
        )
    return _build_normalizer(
        config,
        stats=stats,
        action_dim=action_dim,
        model_mask=model_mask,
    )


def _build_normalizer(
    config: NormalizationConfig,
    *,
    stats: NormalizationStats,
    action_dim: int,
    model_mask: object,
) -> ActionNormalizer:
    """Validate one resolved statistics entry and build its normalizer."""
    center = stats.center
    scale = stats.scale
    if _shape(center) != _shape(scale):
        raise ValueError(
            "normalization center and scale shapes differ: "
            f"{_shape(center)} and {_shape(scale)}"
        )
    rank = len(_shape(center))
    if rank not in (1, 2):
        raise ValueError(f"normalization stats must have rank 1 or 2, got {rank}")
    if _shape(center)[-1] != action_dim:
        raise ValueError(
            f"normalization dim {_shape(center)[-1]} does not match "
            f"model dim {action_dim}"
        )
    model_mask_array = _as_numpy(model_mask).astype(bool, copy=False)
    if model_mask_array.shape != (action_dim,):
        raise ValueError(
            f"model mask must have shape ({action_dim},), got "
            f"{model_mask_array.shape}"
        )
    if rank == 2:
        mask = np.broadcast_to(
            model_mask_array,
            _shape(center),
        ).copy()
    else:
        mask = model_mask_array.copy()
    if stats.mask is not None:
        stats_mask = _as_numpy(stats.mask).astype(bool, copy=False)
        if stats_mask.shape == (action_dim,) and rank == 2:
            stats_mask = np.broadcast_to(stats_mask, _shape(center))
        if stats_mask.shape != _shape(center):
            raise ValueError(
                "normalization mask shape is incompatible with stats: "
                f"{stats_mask.shape} and {_shape(center)}"
            )
        mask &= stats_mask
    scale_array = _as_numpy(scale)
    mask &= np.abs(scale_array) >= config.eps
    safe_scale = np.where(mask, scale_array, 1.0)
    return ActionNormalizer(
        center=center,
        scale=safe_scale,
        mask=mask,
        clip=config.clip,
    )


def _float_dtype(action: Array) -> object:
    if isinstance(action, torch.Tensor):
        return torch.float64 if action.dtype == torch.float64 else torch.float32
    return np.float64 if action.dtype == np.float64 else np.float32


def _to_float(action: Array) -> Array:
    if isinstance(action, torch.Tensor):
        return action.to(_float_dtype(action))
    return action.astype(_float_dtype(action), copy=False)


def _cast(value: Array, like: Array) -> Array:
    if isinstance(value, torch.Tensor):
        dtype = like.dtype if like.is_floating_point() else torch.float32
        return value.to(dtype)
    dtype = like.dtype if np.issubdtype(like.dtype, np.floating) else np.float32
    return value.astype(dtype, copy=False)


def _clip(value: Array, bounds: tuple[float, float]) -> Array:
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=bounds[0], max=bounds[1])
    return np.clip(value, bounds[0], bounds[1])


def _where(mask: Array, left: Array, right: Array) -> Array:
    if isinstance(mask, torch.Tensor):
        return torch.where(mask, left, right)
    return np.where(mask, left, right)


def _align_norm(action: Array, norm: ActionNorm) -> ActionNorm:
    norm_min, norm_max, norm_mask = norm
    norm_min = geometry.as_like(
        norm_min,
        action,
        dtype=_float_dtype(action),
    )
    norm_max = geometry.as_like(
        norm_max,
        action,
        dtype=_float_dtype(action),
    )
    norm_mask = geometry.bool_array(norm_mask, action)
    if norm_min.shape != norm_max.shape:
        raise ValueError(
            f"action norm min shape {norm_min.shape} does not match "
            f"max shape {norm_max.shape}"
        )
    if norm_min.ndim not in (1, 2):
        raise ValueError(
            f"action norm must have shape [D] or [T, D], got {norm_min.shape}"
        )
    if action.ndim == 0 or action.shape[-1] != norm_min.shape[-1]:
        raise ValueError(
            f"action shape {action.shape} is incompatible with norm shape "
            f"{norm_min.shape}"
        )
    if norm_mask.shape not in (norm_min.shape, norm_min.shape[-1:]):
        raise ValueError(
            f"action norm mask shape {norm_mask.shape} is incompatible with "
            f"norm shape {norm_min.shape}"
        )
    if norm_min.ndim == 2:
        if action.ndim < 2:
            raise ValueError("stepwise action norm requires an action step dimension")
        length = action.shape[-2]
        if length > norm_min.shape[0]:
            raise ValueError(
                f"action length {length} exceeds norm length {norm_min.shape[0]}"
            )
        norm_min = norm_min[:length]
        norm_max = norm_max[:length]
        if norm_mask.ndim == 2:
            norm_mask = norm_mask[:length]
    return norm_min, norm_max, norm_mask


def normalize(action: ArrayT, norm: ActionNorm) -> ArrayT:
    """Normalize selected checkpoint action dimensions to ``[-1, 1]``."""
    norm_min, norm_max, norm_mask = _align_norm(action, norm)
    denominator = norm_max - norm_min
    if isinstance(action, torch.Tensor):
        safe = torch.where(norm_mask, denominator, torch.ones_like(denominator))
    else:
        safe = np.where(norm_mask, denominator, np.ones_like(denominator))
    normalized = (_to_float(action) - norm_min) / safe * 2 - 1
    return _where(norm_mask, _cast(normalized, action), action)


def denormalize(action: ArrayT, norm: ActionNorm) -> ArrayT:
    """Restore selected checkpoint action dimensions from ``[-1, 1]``."""
    norm_min, norm_max, norm_mask = _align_norm(action, norm)
    restored = (_to_float(action) + 1) / 2 * (norm_max - norm_min) + norm_min
    return _where(norm_mask, _cast(restored, action), action)


def is_checkpoint_key(key: str) -> bool:
    """Return whether a checkpoint key stores an action norm buffer."""
    return any(key == name or key.startswith(f"{name}_") for name in BUFFER_NAMES)


def resolve_checkpoint_keys(
    keys: Iterable[str],
    embodiment: str | None,
) -> dict[str, str]:
    """Map standard buffer names to selected checkpoint keys."""
    key_set = set(keys)
    legacy = {name: name for name in BUFFER_NAMES}
    legacy_complete = all(name in key_set for name in BUFFER_NAMES)
    if embodiment:
        selected = {name: f"{name}_{embodiment}" for name in BUFFER_NAMES}
        if all(key in key_set for key in selected.values()):
            return selected
        if legacy_complete:
            return legacy
    elif legacy_complete:
        return legacy

    prefix = f"{BUFFER_NAMES[0]}_"
    candidates = []
    for key in key_set:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            candidate = {name: f"{name}_{suffix}" for name in BUFFER_NAMES}
            if all(item in key_set for item in candidate.values()):
                candidates.append((suffix, candidate))
    if embodiment and candidates:
        available = sorted(suffix for suffix, _ in candidates)
        raise ValueError(
            f"action norm for embodiment {embodiment!r} is unavailable; "
            f"available embodiments: {available}"
        )
    if not embodiment and len(candidates) == 1:
        return candidates[0][1]
    if not embodiment and len(candidates) > 1:
        available = sorted(suffix for suffix, _ in candidates)
        raise ValueError(
            "checkpoint contains multiple embodiment-specific action norms; "
            f"select one of {available}"
        )
    norm_keys = sorted(key for key in key_set if is_checkpoint_key(key))
    if norm_keys:
        raise ValueError(
            f"checkpoint contains incomplete action norm buffers: {norm_keys}"
        )
    return {}
