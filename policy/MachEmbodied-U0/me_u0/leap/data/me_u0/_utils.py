"""Shared logging, tensor mapping, and statistics helpers for the data layer."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Sequence

import torch


def select_lerobot_dimensions(
    value: torch.Tensor,
    meta: Dict[str, Any],
) -> torch.Tensor:
    """Project a source action/state column onto its canonical coordinates.

    ``raw_shape`` describes the tensor *after* an optional ``select_indices``
    projection. This keeps runtime sharded reads and map-style stats
    construction on the same action/state coordinates.
    """

    if value.ndim == 0:
        raise ValueError(
            "LeRobot action/state columns must have a feature dimension"
        )
    if value.ndim == 1:
        # Arrow scalar columns are materialized as [T], while vector columns
        # are [T, D]. Keep the common time-major contract before projection.
        value = value.unsqueeze(-1)

    raw_indices = meta.get("select_indices")
    selected = value
    if raw_indices is not None:
        indices = tuple(int(index) for index in raw_indices)
        key = meta.get("lerobot_key", meta.get("key", "<unknown>"))
        if not indices:
            raise ValueError(f"select_indices must not be empty for {key!r}")
        if min(indices) < 0 or max(indices) >= value.shape[-1]:
            raise IndexError(
                f"select_indices={list(indices)} exceed source width "
                f"{value.shape[-1]} for {key!r}"
            )
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        selected = value.index_select(-1, index)

    expected_width = meta.get("raw_shape")
    if expected_width is not None and selected.shape[-1] != int(expected_width):
        key = meta.get("lerobot_key", meta.get("key", "<unknown>"))
        raise ValueError(
            f"LeRobot column {key!r} has canonical width {selected.shape[-1]}, "
            f"expected raw_shape={int(expected_width)}"
        )
    return selected




def supervised_anchor_lengths(
    episode_lengths: Sequence[int],
    minimum_future_offset: int,
) -> List[int]:
    """Return per-episode anchors that have a real future video observation.

    An anchor ``a`` is eligible only when ``a + minimum_future_offset < ep_len``.
    This removes terminal samples whose forward-video target is entirely padded
    while preserving the source episode length for window clamping and stats.
    """

    minimum_future_offset = int(minimum_future_offset)
    if minimum_future_offset <= 0:
        raise ValueError("`minimum_future_offset` must be positive")
    result: List[int] = []
    for raw_length in episode_lengths:
        length = int(raw_length)
        if length < 0:
            raise ValueError("Episode lengths must be non-negative")
        result.append(max(0, length - minimum_future_offset))
    return result




def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def dict_apply(
    x: Dict[str, torch.Tensor],
    func: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result: Dict = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


_WORK_DIR: str | None = None
_DEFAULT_WORK_DIR = "./runs/"


def register_work_dir(path: str | os.PathLike | None) -> None:
    global _WORK_DIR
    _WORK_DIR = str(path) if path is not None else None
    if path is not None:
        os.makedirs(path, exist_ok=True)


def get_work_dir() -> str | None:
    return _WORK_DIR if _WORK_DIR is not None else _DEFAULT_WORK_DIR
