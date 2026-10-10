"""Per-embodiment normalisation of state and action vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mmabc.canonical.layout import CanonicalLayout

_EPS = 1e-6
STD_FLOOR = 1e-3


class RunningStats:
    """Streaming mean/std/min/max plus coarse quantiles, per dimension.

    Welford keeps the variance stable over tens of millions of frames, and the
    histogram gives q01/q99 for diagnostics without a second pass.
    """

    def __init__(self, dim: int, *, hist_bins: int = 2048, hist_range: float = 50.0) -> None:
        self.dim = dim
        self.count = np.zeros(dim, dtype=np.int64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        self.hist = np.zeros((dim, hist_bins), dtype=np.int64)

    def update(self, values: np.ndarray, mask: np.ndarray | None = None) -> None:
        """values: (N, dim); mask: (N, dim) bool selecting supervised entries."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if mask is None:
            mask = np.ones_like(values, dtype=bool)
        mask = np.asarray(mask, dtype=bool)

        n = mask.sum(axis=0)
        if not n.any():
            return
        safe = np.where(mask, values, 0.0)
        col_sum = safe.sum(axis=0)

        new_count = self.count + n
        active = n > 0
        delta = np.zeros(self.dim, dtype=np.float64)
        delta[active] = col_sum[active] / n[active] - self.mean[active]

        # Chan et al. parallel variance merge: treat this batch as one block.
        batch_mean = np.zeros(self.dim, dtype=np.float64)
        batch_mean[active] = col_sum[active] / n[active]
        centred = np.where(mask, values - batch_mean[None, :], 0.0)
        batch_m2 = (centred**2).sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            correction = np.where(
                new_count > 0, delta**2 * self.count * n / np.maximum(new_count, 1), 0.0
            )
        self.m2 += batch_m2 + correction
        self.mean = np.where(active, self.mean + delta * n / np.maximum(new_count, 1), self.mean)
        self.count = new_count

        masked_min = np.where(mask, values, np.inf).min(axis=0)
        masked_max = np.where(mask, values, -np.inf).max(axis=0)
        self.min = np.minimum(self.min, masked_min)
        self.max = np.maximum(self.max, masked_max)

        clipped = np.clip(values, -self.hist_range, self.hist_range)
        bins = ((clipped + self.hist_range) / (2 * self.hist_range) * (self.hist_bins - 1)).astype(
            np.int32
        )
        for d in np.flatnonzero(active):
            np.add.at(self.hist[d], bins[mask[:, d], d], 1)

    def quantile(self, q: float) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float64)
        edges = np.linspace(-self.hist_range, self.hist_range, self.hist_bins)
        for d in range(self.dim):
            total = self.hist[d].sum()
            if total == 0:
                continue
            cdf = np.cumsum(self.hist[d]) / total
            out[d] = edges[int(np.searchsorted(cdf, q, side="left").clip(0, self.hist_bins - 1))]
        return out

    def result(self) -> dict[str, list[float]]:
        var = np.zeros(self.dim, dtype=np.float64)
        ok = self.count > 1
        var[ok] = self.m2[ok] / (self.count[ok] - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": np.where(np.isfinite(self.min), self.min, 0.0).tolist(),
            "max": np.where(np.isfinite(self.max), self.max, 0.0).tolist(),
            "q01": self.quantile(0.01).tolist(),
            "q99": self.quantile(0.99).tolist(),
        }


@dataclass
class NormStats:
    """Statistics for one embodiment, in model space for actions."""

    embodiment_tag: str
    action: dict[str, np.ndarray] = field(default_factory=dict)
    state: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        blob = json.loads(Path(path).read_text())
        to_arr = lambda d: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
        return cls(
            embodiment_tag=blob["embodiment_tag"],
            action=to_arr(blob["action"]),
            state=to_arr(blob["state"]),
        )

    def save(self, path: str | Path) -> None:
        blob = {
            "embodiment_tag": self.embodiment_tag,
            "action": {k: np.asarray(v).tolist() for k, v in self.action.items()},
            "state": {k: np.asarray(v).tolist() for k, v in self.state.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(blob, indent=1))


class Normalizer:
    """Applies the layout's per-segment normalisation policy.

    Scale/offset are baked into dense 80-dim vectors once, so the hot path is a
    single fused affine op rather than a loop over segments.
    """

    def __init__(self, layout: CanonicalLayout, stats: NormStats) -> None:
        self.layout = layout
        self.stats = stats
        self.action_offset, self.action_scale = self._build(
            stats.action, layout.segments, layout.total_dim
        )
        # State may have its own (decoupled) layout; falls back to the action
        # layout for legacy embodiments where state and action share a space.
        self.state_offset, self.state_scale = self._build(
            stats.state, layout.state_segments, layout.state_dim
        )

    def _build(
        self, table: dict[str, np.ndarray], segments, dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        offset = np.zeros(dim, dtype=np.float32)
        scale = np.ones(dim, dtype=np.float32)
        if not table:
            return offset, scale

        mean = np.asarray(table.get("mean", np.zeros(dim)), dtype=np.float64)
        std = np.asarray(table.get("std", np.ones(dim)), dtype=np.float64)
        lo = np.asarray(table.get("min", np.zeros(dim)), dtype=np.float64)
        hi = np.asarray(table.get("max", np.ones(dim)), dtype=np.float64)

        for seg in segments:
            sl = seg.slice
            if seg.norm == "meanstd":
                # A near-constant dim would otherwise be amplified into noise.
                s = np.maximum(std[sl], STD_FLOOR)
                offset[sl] = mean[sl]
                scale[sl] = s
            elif seg.norm == "minmax":
                span = np.maximum(hi[sl] - lo[sl], _EPS)
                # Map to [-1, 1] so bounded dims share the scale of the rest.
                offset[sl] = lo[sl] + span * 0.5
                scale[sl] = span * 0.5
            elif seg.norm == "identity":
                offset[sl] = 0.0
                scale[sl] = 1.0
            else:
                raise ValueError(f"unknown norm {seg.norm!r} on segment {seg.name}")
        return offset, scale

    def normalize_action(self, target: np.ndarray) -> np.ndarray:
        return (target - self.action_offset) / self.action_scale

    def denormalize_action(self, target: np.ndarray) -> np.ndarray:
        return target * self.action_scale + self.action_offset

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_offset) / self.state_scale

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return state * self.state_scale + self.state_offset


def identity_stats(
    embodiment_tag: str, dim: int = 80, state_dim: int | None = None
) -> NormStats:
    """Placeholder statistics, for smoke tests before a stats pass has run."""

    def base(d: int) -> dict[str, np.ndarray]:
        return {
            "mean": np.zeros(d),
            "std": np.ones(d),
            "min": -np.ones(d),
            "max": np.ones(d),
        }

    return NormStats(
        embodiment_tag=embodiment_tag,
        action=base(dim),
        state=base(dim if state_dim is None else state_dim),
    )
