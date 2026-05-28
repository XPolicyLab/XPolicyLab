"""Helpers for aligning env proprio dimensions with checkpoint ``norm_stats``."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def proprio_pad_ndim_from_norm_stats(norm_stats: Optional[dict[str, Any]]) -> int:
    """Target last-axis size for :class:`~dexbotic.data.dataset.transform.action.PadState`.

    Must match ``norm_stats['state']['mean']`` length (Pi0 LIBERO checkpoints typically use 32).
    Using ``action_dim`` here is wrong: env ``observation/state`` is often 8-D while actions are 7-D.
    """
    if norm_stats is None or "state" not in norm_stats:
        return 32
    st = norm_stats["state"]
    if isinstance(st, dict) and "mean" in st:
        return int(np.asarray(st["mean"], dtype=np.float64).size)
    return 32
