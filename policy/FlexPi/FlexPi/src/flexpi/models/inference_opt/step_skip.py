"""Shared step-skip controller for joint-denoising inference loops.

Implements DreamZero's DiT-cache step skipping as a modality-agnostic helper.
When consecutive velocity predictions are sufficiently cosine-similar, the
controller skips upcoming MoT forward passes and reuses the last real
prediction — amortizing the dominant cost of joint denoising.

Usage::

    ctrl = StepSkipController(
        enabled=dynamic_step_skip,
        thresholds=[(0.95, 4), (0.93, 2)],
        sim_aggregation="mean",   # how to combine per-stream cos-sims
    )
    for step in range(num_inference_steps):
        if ctrl.should_run():
            pred_video, pred_aux, pred_action = model.forward(...)
            ctrl.record(
                sim_refs=[pred_video, pred_aux],
                preds=(pred_video, pred_aux, pred_action),
            )
        else:
            pred_video, pred_aux, pred_action = ctrl.cached()
        # scheduler steps as usual

Multi-stream similarity aggregation (``sim_aggregation``):
    - ``"mean"`` (default): cos-sim per stream, then average. Each modality
      contributes equally regardless of token count — the modality-balanced
      reading. Skip when the average crosses a threshold.
    - ``"min"``: cos-sim per stream, then min. Conservative — skip only when
      *all* listed streams have plateaued.
    - ``"concat"``: flatten+concat all streams into one vector, then a single
      cos-sim. Legacy behavior; weighted by each stream's L2 norm so
      large-token streams (typically video) dominate. Matches DreamZero
      when ``sim_refs`` has length 1.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class StepSkipController:
    """Intra-loop, modality-agnostic step-skip state.

    The controller is purely loop-local — no cross-call state, no cross-chunk
    state. Each ``infer_action`` call constructs a fresh controller.
    """

    VALID_AGGREGATIONS = ("mean", "min", "concat")

    def __init__(
        self,
        enabled: bool,
        thresholds: Sequence[Tuple[float, int]] = ((0.95, 4), (0.93, 2)),
        sim_aggregation: str = "mean",
    ):
        if sim_aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"sim_aggregation must be one of {self.VALID_AGGREGATIONS}; "
                f"got {sim_aggregation!r}."
            )
        self.enabled = bool(enabled)
        self.thresholds: List[Tuple[float, int]] = [
            (float(t), int(n)) for t, n in thresholds
        ]
        self.sim_aggregation = sim_aggregation
        # Each entry is a list of per-stream flattened velocity tensors at one
        # denoising step: _prev_refs[i] = [v_stream1, v_stream2, ...]. Length
        # capped at 2 (the cosine-sim compares the last two recorded steps).
        self._prev_refs: List[List[torch.Tensor]] = []
        self._skip_countdown: int = 0
        self._cached_preds: Any = None
        self.steps_run: int = 0
        self.last_sim: Optional[float] = None       # most recent aggregated cos-sim
        self.sim_history: List[float] = []          # all aggregated cos-sims (one per decision call)
        self.per_stream_history: List[List[float]] = []  # per-stream cos-sims at each decision (same order as sim_refs)
        self.decision_history: List[str] = []       # outcome of each decision: "skip{N}" or "run"

    def should_run(self) -> bool:
        """Decide whether to run a fresh forward pass at this denoising step."""
        if not self.enabled:
            return True
        if len(self._prev_refs) < 2:
            return True                             # need two samples to compare
        if self._skip_countdown > 1:
            self._skip_countdown -= 1
            return False
        if self._skip_countdown == 1:
            self._skip_countdown = 0
            return True                             # refresh after a skip run
        last_streams = self._prev_refs[-1]
        prev_streams = self._prev_refs[-2]
        # Always compute per-stream cos-sims (cheap; needed for logging in any
        # mode). For concat mode we additionally compute the L2-weighted
        # aggregate; for mean/min the per-stream values are also the inputs.
        per_stream = [
            F.cosine_similarity(l, p, dim=1).mean().item()
            for l, p in zip(last_streams, prev_streams)
        ]
        self.per_stream_history.append(per_stream)
        if self.sim_aggregation == "concat":
            # L2-magnitude-weighted across streams (large streams like video dominate).
            last = torch.cat(last_streams, dim=1)
            prev = torch.cat(prev_streams, dim=1)
            sim = F.cosine_similarity(last, prev, dim=1).mean().item()
        elif self.sim_aggregation == "mean":
            sim = sum(per_stream) / len(per_stream)
        else:  # "min" — only skip when ALL streams have plateaued
            sim = min(per_stream)
        self.last_sim = sim
        self.sim_history.append(sim)
        for threshold, count in self.thresholds:
            if sim > threshold:
                self._skip_countdown = count
                self.decision_history.append(f"skip{count}")
                return False
        self.decision_history.append("run")
        return True

    def record(self, sim_refs: Sequence[torch.Tensor], preds: Any) -> None:
        """Store a real forward's outputs.

        Args:
            sim_refs: per-stream tensors used for the similarity check. Each is
                flattened to ``[B, -1]`` and stored independently. The ``should_run``
                decision aggregates per-stream cos-sims according to
                ``sim_aggregation``.
            preds: arbitrary payload (e.g. a tuple of predictions) returned
                from ``cached()`` when a step is skipped.
        """
        self._cached_preds = preds
        self.steps_run += 1
        if not self.enabled or not sim_refs:
            return
        flat_streams = [r.flatten(1).float().detach() for r in sim_refs]
        self._prev_refs.append(flat_streams)
        if len(self._prev_refs) > 2:
            self._prev_refs.pop(0)

    def cached(self) -> Any:
        """Return the last recorded ``preds`` payload (reuse on a skipped step)."""
        if self._cached_preds is None:
            raise RuntimeError(
                "StepSkipController.cached() called before any record() — no "
                "cached prediction to reuse. This indicates a logic error in "
                "the denoising loop (should_run() must return True on the "
                "first iteration)."
            )
        return self._cached_preds


def resolve_sim_sources(
    explicit: Optional[Sequence[str]],
    denoised_flags: dict[str, bool],
) -> List[str]:
    """Resolve the list of sim-source names for the current joint regime.

    Args:
        explicit: user-supplied list (e.g. ``["video", "dino"]``) or ``None``.
        denoised_flags: mapping from valid source name → whether that stream
            is being denoised this run. Auto-pick ignores False entries.

    Returns:
        Resolved list of source names. Empty when nothing is being denoised
        (caller should disable the controller in that case).

        Auto-pick (``explicit=None``) returns ALL denoised streams, in the
        insertion order of ``denoised_flags``. Combined with the default
        ``sim_aggregation="mean"`` in :class:`StepSkipController`, this
        averages per-stream cosine similarities — each modality contributes
        equally regardless of token count. This is more principled than
        single-stream auto-pick under flex inference (where which stream is
        denoised varies per call). Callers wanting the legacy single-stream
        behavior should pass ``explicit=["video"]`` (or whichever stream).

    Raises:
        ValueError: explicit name not in ``denoised_flags`` keys, or named a
            stream that is not currently being denoised.
    """
    valid_names = list(denoised_flags.keys())
    if explicit is None:
        # Auto-pick: every denoised stream. With sim_aggregation="mean" (the
        # default), the controller averages per-stream cos-sims and skips
        # when the average crosses threshold — modality-balanced view.
        return [name for name in valid_names if denoised_flags[name]]
    if len(explicit) == 0:
        raise ValueError(
            "`step_skip_sim_sources` must be None (auto) or a non-empty list; "
            "got an empty list."
        )
    resolved: List[str] = []
    for name in explicit:
        if name not in denoised_flags:
            raise ValueError(
                f"Unknown step_skip_sim_source {name!r}; valid for this model: "
                f"{valid_names}."
            )
        if not denoised_flags[name]:
            raise ValueError(
                f"step_skip_sim_source {name!r} requires the corresponding "
                f"joint flag to be True (stream is not being denoised)."
            )
        resolved.append(name)
    return resolved
