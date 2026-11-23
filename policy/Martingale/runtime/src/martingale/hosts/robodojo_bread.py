# Copyright © 2026 Martingale. All Rights Reserved.
import math
import time

import numpy as np



def rotation(yaw, tilt, lean):
    """Slab axes: width, thickness, height; yaw about world Z, then local Y/X."""
    y, t, l = np.deg2rad([yaw, tilt, lean])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    rx = np.array([[1, 0, 0], [0, np.cos(l), -np.sin(l)], [0, np.sin(l), np.cos(l)]])
    return rz @ ry @ rx


def slab_solid(sight, grid, parameters):
    """Intersect each vertical grid line with a rotated rectangular slab."""
    p = np.asarray(parameters, float)
    center, half = p[:3], p[6:9] / 2
    axes = rotation(*p[3:6])
    xy = grid.xy - center[:2]
    lower = np.full(len(xy), -np.inf)
    upper = np.full(len(xy), np.inf)
    valid = np.ones(len(xy), bool)
    for axis, radius in zip(axes.T, half):
        horizontal = xy @ axis[:2]
        if abs(axis[2]) < 1e-10:
            valid &= np.abs(horizontal) <= radius
        else:
            a = (-radius - horizontal) / axis[2]
            b = (radius - horizontal) / axis[2]
            lower = np.maximum(lower, np.minimum(a, b))
            upper = np.minimum(upper, np.maximum(a, b))
    valid &= upper >= lower
    low = np.where(valid, lower + center[2] - grid.base_z, np.nan)
    high = np.where(valid, upper + center[2] - grid.base_z, np.nan)
    return sight._Solid(grid, low, high)


def score_views(sight, solid, observations):
    marks = [sight._match(view, solid, pixels) for view, pixels in observations]
    values = [0.0 if mark is None else float(mark) for mark in marks]
    return float(np.mean(values)), values


def fit_bread(sight, captured, estimate, progress=None):
    """Add a slab candidate, freezing baseline camera/instance selection.

    Returns (chosen_capture, slab_capture, diagnostics). The second may be None
    if fewer than two baseline-selected views constrain this object.
    """
    head, trusted, old_solid = captured.get("head"), captured.get("trusted") or [], captured.get("solid")
    if head is None or old_solid is None or not trusted:
        return captured, None, {"selected": "baseline", "skipped": "fewer than two selected views"}
    rows, cols = np.nonzero(head.mask)
    stride = max(1, math.ceil(len(rows) / 1500))
    observations = [(head, (rows[::stride], cols[::stride])), *trusted]
    grid = old_solid.grid
    base_score, base_marks = score_views(sight, old_solid, observations)
    baseline_yaw = float(estimate.get("long_axis_deg") or 0)
    initial_z = grid.base_z
    ray = head.view.rays(np.array([rows.mean()]), np.array([cols.mean()]))[0]
    center = head.view.pos + ray * ((initial_z - head.view.pos[2]) / ray[2])
    span = grid.cell * max(grid.nx, grid.ny)
    limits = np.array([
        [grid.lo[0], grid.lo[0] + (grid.nx - 1) * grid.cell],
        [grid.lo[1], grid.lo[1] + (grid.ny - 1) * grid.cell],
        [grid.base_z, grid.base_z + span],
        [-360., 360.], [-90., 90.], [-90., 90.],
        [grid.cell, span], [grid.cell, span], [grid.cell, span],
    ])
    cache, evaluations = {}, 0
    started = time.perf_counter()

    def objective(p):
        nonlocal evaluations
        key = tuple(np.round(p, 8))
        if key in cache:
            return cache[key]
        axes = rotation(*p[3:6])
        corners_extent = np.abs(axes) @ (p[6:9] / 2)
        if p[2] - corners_extent[2] < grid.base_z - 0.002:
            return -1.0
        if np.any(p[:2] - corners_extent[:2] < grid.lo) or np.any(
                p[:2] + corners_extent[:2] > grid.lo + np.array([grid.nx - 1, grid.ny - 1]) * grid.cell):
            return -1.0
        solid = slab_solid(sight, grid, p)
        score, _ = score_views(sight, solid, observations)
        evaluations += 1
        cache[key] = score
        if progress and evaluations % 100 == 0:
            progress({"evaluations": evaluations, "elapsed_s": round(time.perf_counter() - started, 1),
                      "best_score_seen": max(cache.values())})
        return score

    dims = np.array([span / 4.0, grid.cell, span / 4.0])
    p = np.clip(np.r_[center, baseline_yaw, 0., 0., dims], limits[:, 0], limits[:, 1])
    value = objective(p)
    step = grid.cell
    for dimension in range(len(p)):
        candidates = []
        for sign in (-1, 1):
            trial = p.copy()
            trial[dimension] = np.clip(trial[dimension] + sign * step, *limits[dimension])
            candidates.append((objective(trial), trial))
        score, trial = max(candidates, key=lambda pair: pair[0])
        if score > value:
            p, value = trial, score
    starts = [{"score": value, "parameters": p.tolist()}]
    best_score, best_p = value, p.copy()
    if best_p is None or best_score < 0:
        return captured, None, {"selected": "baseline", "skipped": "no valid slab in search bounds", "starts": starts}
    solid = slab_solid(sight, grid, best_p)
    score, marks = score_views(sight, solid, observations)
    depths = sight._march(head.view, rows, cols, solid)
    hit = np.isfinite(depths)
    result = {"points": head.view.pos + head.view.rays(rows[hit], cols[hit]) * depths[hit, None],
              "rows": rows[hit], "cols": cols[hit], "kind": "bread_slab", "views": len(observations),
              "match": score, "height_m": float(np.nanmax(solid.high)), "remembered": False,
              "outgrown": False, "cell_m": grid.cell}
    info = {"parameters": best_p.tolist(), "baseline_mean_score": base_score,
            "slab_mean_score": score, "baseline_per_view_scores": base_marks, "slab_per_view_scores": marks,
            "evaluations": evaluations, "elapsed_s": time.perf_counter() - started, "starts": starts,
            "objective": "Existing _match F1 averaged across primary and fixed supporting views"}
    accepted = score > base_score and min(marks) >= sight.SECOND_VIEW_MIN
    info["selected"] = "bread_slab" if accepted else "baseline"
    candidate = {"head": head, "trusted": trusted, "solid": solid, "result": result,
                 "trace": [{"bread_slab": info}]}
    return candidate if accepted else captured, candidate, info
