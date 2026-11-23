# Copyright © 2026 Martingale. All Rights Reserved.

from __future__ import annotations
from martingale import self_play_constants as sp

import itertools
import re

import numpy as np

from . import robodojo_kinematics as kin
from . import robodojo_sight as sight

KINDS = ("slab", "box", "pyramid")
MIN_IOU = 0.639

ANGLES = np.arange(48) * (2 * np.pi / 48)
DIRECTIONS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)
SCALE = np.array([0.1, 0.1, 0.1, 1., 1., 1., 0.1, 0.1, 0.1])


def shape_for_prompt(prompt: str) -> str | None:
    """Recognize piece descriptions, not arbitrary containers or whole towers."""
    target = re.search(r"\b(pyramid|plank|board|slab|block|cube|cuboid)\b", prompt.lower())
    if target:
        noun = target.group(1)
        if noun == "pyramid":
            return "pyramid"
        if noun in ("plank", "board", "slab"):
            return "slab"
        return "box"
    return None


def rotation(angles: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles), np.sin(angles)
    rz = np.array([[c[0], -s[0], 0], [s[0], c[0], 0], [0, 0, 1]])
    ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    rx = np.array([[1, 0, 0], [0, c[2], -s[2]], [0, s[2], c[2]]])
    return rz @ ry @ rx


def vertices(kind: str, p: np.ndarray) -> np.ndarray:
    if kind == "pyramid":
        local = np.array([[-.5, -.5, -.5], [-.5, .5, -.5],
                          [.5, .5, -.5], [.5, -.5, -.5], [0, 0, .5]])
    elif kind == "triangular_prism":
        local = np.array([[-.5, -.5, -.5], [.5, -.5, -.5],
                          [.5, .5, -.5], [-.5, .5, -.5], [0, -.5, .5], [0, .5, .5]])
    else:
        local = np.array(list(itertools.product((-.5, .5), repeat=3)))
    return (local * p[6:9]) @ rotation(p[3:6]).T + p[:3]


def ray_depth(view: sight.View, rows: np.ndarray, cols: np.ndarray,
              kind: str, p: np.ndarray) -> np.ndarray:
    """Analytic intersection with convex halfspaces, in optical-axis metres."""
    r = rotation(p[3:6])
    origin = (view.pos - p[:3]) @ r
    rays = view.rays(rows, cols) @ r
    if kind in ("pyramid", "triangular_prism"):
        w, d, h = p[6:9]
        normals = np.array([[2/w, 0, 1/h], [-2/w, 0, 1/h],
                            [0, 2/d, 1/h], [0, -2/d, 1/h], [0, 0, -1.]])
        bounds = np.array([.5, .5, .5, .5, h/2])
        if kind == "triangular_prism":
            normals[2:4] = [[0, 1, 0], [0, -1, 0]]
            bounds[2:4] = d/2
    else:
        normals = np.vstack([np.eye(3), -np.eye(3)])
        bounds = np.tile(p[6:9] / 2, 2)
    low = np.zeros(np.shape(rows))
    high = np.full(np.shape(rows), np.inf)
    valid = np.ones(np.shape(rows), dtype=bool)
    for normal, bound in zip(normals, bounds):
        denom = rays @ normal
        numer = bound - origin @ normal
        parallel = np.abs(denom) < 1e-10
        valid &= ~(parallel & (numer < 0))
        t = numer / np.where(parallel, 1., denom)
        low = np.maximum(low, np.where((~parallel) & (denom < 0), t, -np.inf))
        high = np.minimum(high, np.where((~parallel) & (denom > 0), t, np.inf))
    return np.where(valid & (high >= low) & (low > 0), low, np.nan)


def silhouette(observation: sight.Sight, kind: str, p: np.ndarray) -> np.ndarray:
    rows, cols, depth = observation.view.project(vertices(kind, p))
    result = np.zeros(observation.mask.shape, dtype=bool)
    if np.min(depth) > .01:
        sight._fill_convex(result, np.stack([rows, cols], axis=1))
    return result


def iou(observation: sight.Sight, predicted: np.ndarray) -> float:
    valid = ~observation.unknown if observation.unknown is not None else True
    a, b = observation.mask & valid, predicted & valid
    return float((a & b).sum() / max(1, (a | b).sum()))


def _support(observation: sight.Sight):
    rows, cols = np.nonzero(observation.mask)
    projected = np.stack([cols, rows], axis=1) @ DIRECTIONS.T
    target = projected.max(axis=0)
    unknown = observation.unknown
    touches_unknown = np.zeros(len(rows), dtype=bool)
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc = rows+dr,cols+dc
        inside = (rr>=0)&(rr<unknown.shape[0])&(cc>=0)&(cc<unknown.shape[1])
        touches_unknown |= ~inside
        touches_unknown[inside] |= unknown[rr[inside],cc[inside]]
    clipped = np.any((projected >= target-.5)&touches_unknown[:,None],axis=0)
    return target, clipped


def _initial_center(observations: list[sight.Sight]) -> np.ndarray:
    a, b = np.zeros((3, 3)), np.zeros(3)
    for ob in observations:
        rows, cols = np.nonzero(ob.mask)
        ray = ob.view.rays(np.array([(rows.min()+rows.max())/2]),
                           np.array([(cols.min()+cols.max())/2]))[0]
        ray /= np.linalg.norm(ray)
        projection = np.eye(3) - np.outer(ray, ray)
        a += projection
        b += projection @ ob.view.pos
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _solve(residual, initial, lower, upper):
    """Bounded damped Gauss-Newton; nine parameters, finite differences."""
    x = np.clip(initial / SCALE, lower / SCALE, upper / SCALE)
    lo, hi = lower / SCALE, upper / SCALE
    damping = .01
    values = residual(x * SCALE)
    loss = values @ values
    for _ in range(65):
        jac = np.empty((len(values), len(x)))
        for j in range(len(x)):
            trial = x.copy()
            trial[j] += 1e-4
            jac[:, j] = (residual(trial * SCALE) - values) / 1e-4
        normal = jac.T @ jac
        delta = np.linalg.solve(normal + damping * np.diag(np.maximum(np.diag(normal), 1)),
                                -jac.T @ values)
        trial = np.clip(x + delta, lo, hi)
        new_values = residual(trial * SCALE)
        new_loss = new_values @ new_values
        if new_loss < loss:
            improvement = loss - new_loss
            x, values, loss = trial, new_values, new_loss
            damping = max(1e-6, damping / 3)
            if improvement < 1e-7 or np.linalg.norm(delta) < 1e-6:
                break
        else:
            damping *= 5
            if damping > 1e8:
                break
    return x * SCALE, float(loss)


def _fit(observations: list[sight.Sight], kind: str):
    supports = [_support(ob) for ob in observations]
    initial_center = _initial_center(observations)
    lower = np.array([kin.TABLE_X[0], kin.TABLE_Y[0], kin.TABLE_Z + .002,
                      -2*np.pi, -np.pi, -np.pi, .008, .008, .004])
    upper = np.array([kin.TABLE_X[1], kin.TABLE_Y[1], kin.TABLE_Z + .55,
                      2*np.pi, np.pi, np.pi, .45, .30, .25])
    if kind == "slab":
        upper[8] = .065
    if kind in ("pyramid", "triangular_prism"):
        lower[4:6], upper[4:6] = -.65, .65
    center = np.clip(initial_center, lower[:3], upper[:3])
    spans = []
    for ob in observations:
        rows, cols = np.nonzero(ob.mask)
        _, _, depth = ob.view.project(center)
        spans.append(max(np.ptp(rows) / ob.view.fy, np.ptp(cols) / ob.view.fx) * depth)
    size = float(np.clip(np.median(spans), sp.SHAPE_SIZE_FLOOR_M, .25))

    def residual(p):
        v = vertices(kind, p)
        chunks = []
        for ob, (target, clipped) in zip(observations, supports):
            rows, cols, depth = ob.view.project(v)
            predicted = (np.stack([cols, rows], axis=1) @ DIRECTIONS.T).max(axis=0)
            error = predicted-target
            error = np.where(clipped, np.minimum(error,0.),error)
            chunks.append(np.clip(error, -300, 300))
        chunks.append(np.array([min(0., v[:, 2].min() - kin.TABLE_Z + .002) * 1000]))
        return np.concatenate(chunks)

    pitches = (0., np.pi/2) if kind == "slab" else (0.,)
    best = None
    for yaw, pitch in itertools.product(np.arange(4)*np.pi/4, pitches):
        dims = [size, size*sp.SLAB_INITIAL_WIDTH_RATIO, sp.SLAB_INITIAL_THICKNESS_M] if kind == "slab" else [size, *(size*x for x in sp.BOX_INITIAL_RATIOS)]
        if kind in ("pyramid", "triangular_prism"):
            dims = [size, size, size]
        p = np.r_[center, yaw, pitch, 0., dims]
        candidate, loss = _solve(residual, p, lower, upper)
        if best is None or loss < best[1]:
            best = (candidate, loss)
    return best


def measure(head: sight.Sight, others, kind: str):
    """Return a validated shape measurement or a reason to request more views.

    Each supporting view may have several SAM candidate masks. Compare complete
    joint fits, not per-camera guesses of the instance. No previous pose,
    depth image or fixed object dimensions enter the fit.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown shape: {kind}")
    if head.mask.sum() < 30:
        return None, {"reason": "too few primary-mask pixels"}
    groups = []
    for entry in others:
        group = entry if isinstance(entry, (list, tuple)) else [entry]
        group = [max(group, key=lambda ob: int(ob.mask.sum()))]
        if int(group[0].mask.sum()) <= 0:
            continue
        groups.append(group)
    if not groups:
        return None, {"reason": "shape and elevation need at least two current camera views"}
    if max(np.linalg.norm(ob.view.pos-head.view.pos) for g in groups for ob in g) < .025:
        return None, {"reason": "camera viewpoints are too close to resolve elevation"}
    best = None
    models = ("pyramid", "triangular_prism") if kind == "pyramid" else (kind,)
    for combination, model in itertools.product(itertools.product(*groups), models):
        observations = [head, *combination]
        p, loss = _fit(observations, model)
        marks = [iou(ob, silhouette(ob, model, p)) for ob in observations]
        if best is None or np.mean(marks) > np.mean(best[2]):
            best = (p, observations, marks, loss, model)
    if best is None:
        return None, {"reason": "object is cropped; obtain complete silhouettes"}
    p, observations, marks, loss, model = best
    info = {"kind": model, "shape_family": kind, "parameters": p.tolist(), "per_view_iou": marks,
            "support_rmse_px": float(np.sqrt(loss/(48*len(observations))))}
    reliable_marks = [marks[0]] + [mark for ob, mark in zip(observations[1:], marks[1:])
                                  if int(ob.mask.sum()) > 0]
    if min(reliable_marks) < MIN_IOU or info["support_rmse_px"] > 2.5:
        return None, {**info, "reason": "shape does not agree across current views; check masks or occlusion"}
    rows, cols = np.nonzero(head.mask)
    depth = ray_depth(head.view, rows, cols, model, p)
    hit = np.isfinite(depth)
    v = vertices(model, p)
    low, top = float(v[:, 2].min()), float(v[:, 2].max())
    result = {"points": head.view.pos + head.view.rays(rows[hit], cols[hit])*depth[hit, None],
              "rows": rows[hit], "cols": cols[hit], "kind": model,
              "vertices": v, "parameters": p, "bottom_z": low, "top_z": top,
              "object_height_m": top-low, "height_m": top-kin.TABLE_Z,
              "views": len(reliable_marks), "match": min(reliable_marks), "fit": info}
    return result, info
