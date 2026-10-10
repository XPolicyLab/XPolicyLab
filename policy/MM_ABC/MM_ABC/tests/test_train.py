"""Trainer pieces that are easy to get subtly wrong."""

from __future__ import annotations

import math
from collections import deque

import pytest
import torch

from mmabc.train.fsdp import clip_grad_norm
from mmabc.train.trainer import Trainer, cosine_with_warmup


def _params(grads):
    out = []
    for g in grads:
        p = torch.nn.Parameter(torch.zeros_like(g))
        p.grad = g.clone()
        out.append(p)
    return out


def test_clip_matches_reference_for_plain_tensors():
    """The DTensor-aware path must stay equivalent on unsharded gradients."""
    torch.manual_seed(0)
    grads = [torch.randn(8, 4), torch.randn(16), torch.randn(3, 3, 2)]

    mine = _params(grads)
    reference = _params(grads)
    norm_mine = clip_grad_norm(mine, 1.0)
    norm_ref = torch.nn.utils.clip_grad_norm_(reference, 1.0)

    assert torch.allclose(norm_mine, norm_ref)
    for a, b in zip(mine, reference):
        assert torch.allclose(a.grad, b.grad)


def test_clip_actually_scales_to_max_norm():
    grads = [torch.full((4,), 10.0)]
    params = _params(grads)
    clip_grad_norm(params, 1.0)
    assert abs(params[0].grad.norm().item() - 1.0) < 1e-5


def test_clip_leaves_small_gradients_untouched():
    grads = [torch.full((4,), 0.01)]
    params = _params(grads)
    before = params[0].grad.clone()
    clip_grad_norm(params, 1.0)
    assert torch.allclose(params[0].grad, before)


def test_clip_reports_nonfinite_rather_than_raising():
    """The trainer relies on getting a non-finite norm back so it can skip."""
    grads = [torch.tensor([float("inf"), 0.0])]
    norm = clip_grad_norm(_params(grads), 1.0)
    assert not torch.isfinite(norm)


def test_clip_handles_no_gradients():
    p = torch.nn.Parameter(torch.zeros(4))
    assert float(clip_grad_norm([p], 1.0)) == 0.0


def test_warmup_then_cosine_decay():
    warmup, total, min_ratio = 100, 1000, 0.02
    assert cosine_with_warmup(0, warmup=warmup, total=total, min_ratio=min_ratio) < 0.02
    assert math.isclose(
        cosine_with_warmup(warmup - 1, warmup=warmup, total=total, min_ratio=min_ratio), 1.0
    )
    mid = cosine_with_warmup(550, warmup=warmup, total=total, min_ratio=min_ratio)
    assert 0.4 < mid < 0.6
    end = cosine_with_warmup(total, warmup=warmup, total=total, min_ratio=min_ratio)
    assert math.isclose(end, min_ratio, abs_tol=1e-6)
    # Must not run away past the end of the schedule.
    assert math.isclose(
        cosine_with_warmup(total * 3, warmup=warmup, total=total, min_ratio=min_ratio),
        min_ratio,
        abs_tol=1e-6,
    )


def test_schedule_is_monotone_after_warmup():
    vals = [
        cosine_with_warmup(s, warmup=10, total=200, min_ratio=0.0) for s in range(10, 201)
    ]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def _monitor(**kw):
    """A Trainer carrying only the outlier-rejection state.

    Built without __init__ because the real one needs a process group, a mesh
    and a 4B backbone, none of which this logic touches.
    """
    t = Trainer.__new__(Trainer)
    t.skip_factor = kw.get("skip_factor", 8.0)
    t.skip_min_history = kw.get("min_history", 3)
    t.grad_norm_ceiling = kw.get("ceiling", 1.0e4)
    t.grad_norm_floor = kw.get("floor", 0.0)
    t.max_skip_frac = kw.get("max_skip_frac", 0.25)
    t.grad_norm_history = deque(maxlen=kw.get("window", 200))
    t.skip_window = deque(maxlen=kw.get("rate_window", 8))
    t.skips = {}
    return t


def test_only_the_ceiling_applies_until_history_is_banked():
    """With no baseline there is nothing to be a *relative* outlier against.

    The absolute ceiling still has to hold, because the steps taken while the
    history fills are the ones that wreck a run: an accepted 1e10 gradient in
    the first ten steps leaves its mark in Adam's second moment and the run
    never comes back.
    """
    t = _monitor(min_history=3, ceiling=1.0e4)
    assert t._grad_norm_threshold() == 1.0e4
    for n in (0.4, 0.5, 0.6):
        t.grad_norm_history.append(n)
    assert t._grad_norm_threshold() == 8.0 * 0.5


def test_ceiling_caps_the_median_threshold_too():
    """A median that has drifted high must not license an arbitrary spike."""
    t = _monitor(min_history=3, ceiling=100.0)
    for n in [1e3] * 10:
        t.grad_norm_history.append(n)
    assert t._grad_norm_threshold() == 100.0


def test_threshold_follows_the_median_not_the_mean():
    """One 1e5 outlier must not raise the bar that would have rejected it."""
    t = _monitor(min_history=3)
    for n in [0.5] * 20 + [1e5]:
        t.grad_norm_history.append(n)
    assert t._grad_norm_threshold() == 8.0 * 0.5


def test_threshold_tracks_a_decaying_norm():
    """A constant limit would stop rejecting anything as the run settles."""
    t = _monitor(min_history=3, window=10, ceiling=1.0e6)
    for n in [10.0] * 10:
        t.grad_norm_history.append(n)
    early = t._grad_norm_threshold()
    for n in [0.1] * 10:
        t.grad_norm_history.append(n)
    assert t._grad_norm_threshold() < early / 50


def test_sustained_skipping_aborts_but_isolated_skips_do_not():
    t = _monitor(max_skip_frac=0.25, rate_window=8)
    # Two skips in eight steps is under the limit and must be tolerated.
    for verdict in (0, 0, 1, 0, 0, 1, 0, 0):
        if verdict:
            t._record_skip("spike")
        else:
            t.skip_window.append(0)
    assert t.skips["spike"] == 2

    # A run that skips most steps is not training, and should say so.
    t = _monitor(max_skip_frac=0.25, rate_window=4)
    with pytest.raises(RuntimeError, match="of the last"):
        for _ in range(4):
            t._record_skip("spike")


def test_skip_reasons_are_counted_separately():
    """Which failure dominates decides where to look, so do not merge them."""
    t = _monitor(rate_window=100)
    t._record_skip("spike")
    t._record_skip("nonfinite_grad")
    t._record_skip("spike")
    assert t.skips == {"spike": 2, "nonfinite_grad": 1}


def test_threshold_never_drops_below_the_clipping_norm():
    t = _monitor(floor=1.0)
    t.grad_norm_history.extend([0.01] * 10)
    # 8 x median would be 0.08, but a norm the clipper accepts is never an outlier.
    assert t._grad_norm_threshold() == 1.0
