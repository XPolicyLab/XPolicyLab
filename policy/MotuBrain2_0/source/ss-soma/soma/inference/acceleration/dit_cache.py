"""DiT Caching: DreamZero dual-threshold velocity caching with extrapolation.

Skips redundant DIT forward passes when velocity predictions converge,
using linear extrapolation from previous steps to fill skipped predictions.

Usage:
    cache = DiTCache(threshold=0.95, skip_steps=4)
    for step_i in range(N):
        v_vel, a_vel = cache.step(t_cur, predict_fn, step_i, skip_video)
"""

import logging

import torch
import torch.nn.functional as F

from soma.inference.types import Velocity
from soma.inference.types import VelocityThunk

logger = logging.getLogger(__name__)


class DiTCache:
    """Tracks skip state and previous predictions for DiT step caching.

    Args:
        threshold: primary cosine similarity threshold (None = disabled)
        threshold_2: secondary (lower) threshold for fewer skip steps
        skip_steps: steps to skip when primary threshold is exceeded
        skip_steps_2: steps to skip when secondary threshold is exceeded
        damping: extrapolation damping factor (0 = reuse last, >0 = linear extrap)
    """

    def __init__(
        self,
        threshold: float | None = None,
        threshold_2: float | None = None,
        skip_steps: int = 4,
        skip_steps_2: int = 2,
        damping: float = 0.25,
    ) -> None:
        self.threshold: float | None = threshold
        self.threshold_2: float | None = threshold_2
        self.skip_steps: int = skip_steps
        self.skip_steps_2: int = skip_steps_2
        self.damping: float = damping
        self.skip_countdown: int
        self.prev_preds: list[tuple[float, torch.Tensor | None, torch.Tensor]]
        self.real_forward_count: int
        self.skip_forward_count: int
        self.trigger_count: int
        self.primary_trigger_count: int
        self.secondary_trigger_count: int
        self.similarity_count: int
        self.similarity_sum: float
        self.max_similarity: float | None
        self.reset()

    def reset(self) -> None:
        self.skip_countdown = 0
        self.prev_preds = []
        self.real_forward_count = 0
        self.skip_forward_count = 0
        self.trigger_count = 0
        self.primary_trigger_count = 0
        self.secondary_trigger_count = 0
        self.similarity_count = 0
        self.similarity_sum = 0.0
        self.max_similarity = None

    def log_stats(self) -> None:
        """Log cache hit summary for the current denoising run."""
        if self.threshold is None:
            return

        avg_similarity = self.similarity_sum / self.similarity_count if self.similarity_count > 0 else float("nan")
        max_similarity = self.max_similarity if self.max_similarity is not None else float("nan")
        logger.info(
            "DiT cache stats: triggers=%d (primary=%d, secondary=%d), "
            "skipped_forwards=%d, real_forwards=%d, avg_similarity=%.4f, max_similarity=%.4f",
            self.trigger_count,
            self.primary_trigger_count,
            self.secondary_trigger_count,
            self.skip_forward_count,
            self.real_forward_count,
            avg_similarity,
            max_similarity,
        )

    def step(
        self,
        t_cur: torch.Tensor,
        predict_fn: VelocityThunk,
        step_i: int,
        skip_video: bool,
    ) -> Velocity:
        """Run one denoising step with optional caching.

        Args:
            t_cur: current timestep (GPU scalar tensor)
            predict_fn: callable() -> (v_vel | None, a_vel)
            step_i: step index (0-based)
            skip_video: bool, if True video velocity may be None
        Returns:
            (v_vel | None, a_vel)
        """
        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            self.skip_forward_count += 1
            return self._extrapolate(t_cur)

        if self.skip_countdown == 1:
            self.skip_countdown = 0

        self.real_forward_count += 1
        v_vel, a_vel = predict_fn()
        self._check_and_update(t_cur, v_vel, a_vel, step_i, skip_video)
        return v_vel, a_vel

    def _extrapolate(self, t_cur: torch.Tensor) -> Velocity:
        """Velocity extrapolation from 2 previous predictions."""
        if len(self.prev_preds) >= 2 and self.damping > 0:
            t1, v1, a1 = self.prev_preds[-1]
            t2, v2, a2 = self.prev_preds[-2]
            h_past = t1 - t2
            if abs(h_past) > 1e-8:
                h_curr = t_cur.float() - t1
                a_vel = a1 + (a1 - a2) / h_past * h_curr * self.damping
                v_vel = (v1 + (v1 - v2) / h_past * h_curr * self.damping) if v1 is not None else None
                return v_vel, a_vel
        if self.prev_preds:
            return self.prev_preds[-1][1], self.prev_preds[-1][2]
        raise RuntimeError("DiTCache cannot extrapolate without a prediction")

    def _check_and_update(
        self,
        t_cur: torch.Tensor,
        v_vel: torch.Tensor | None,
        a_vel: torch.Tensor,
        step_i: int,
        skip_video: bool,
    ) -> None:
        """Check convergence and update prev_preds."""
        check_vel = a_vel if (skip_video or v_vel is None) else v_vel

        if self.threshold is not None and len(self.prev_preds) >= 1 and step_i >= 2:
            prev_v, prev_a = self.prev_preds[-1][1], self.prev_preds[-1][2]
            prev_check = prev_a if (skip_video or prev_v is None) else prev_v
            sim = F.cosine_similarity(
                check_vel.detach().flatten().unsqueeze(0),
                prev_check.detach().flatten().unsqueeze(0),
            ).item()
            self.similarity_count += 1
            self.similarity_sum += sim
            self.max_similarity = sim if self.max_similarity is None else max(self.max_similarity, sim)
            if sim > self.threshold:
                self.skip_countdown = self.skip_steps
                self.trigger_count += 1
                self.primary_trigger_count += 1
            elif self.threshold_2 is not None and sim > self.threshold_2:
                self.skip_countdown = self.skip_steps_2
                self.trigger_count += 1
                self.secondary_trigger_count += 1

        t_val = t_cur.float().item()
        self.prev_preds.append(
            (
                t_val,
                v_vel.detach().clone() if v_vel is not None else None,
                a_vel.detach().clone(),
            )
        )
        if len(self.prev_preds) > 2:
            self.prev_preds.pop(0)
