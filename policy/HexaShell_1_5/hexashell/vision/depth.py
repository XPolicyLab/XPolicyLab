"""Depth along a pixel's ray from the pictures themselves.

The few pixels around the picked one are taken to show a level surface (a table, a block's top).
Each height that surface could have carries those pixels into every other camera's picture; at the
right height the carried patch looks like the original (zero-mean normalized correlation of grey
levels, so a change of brightness between cameras does not matter).  A height is accepted when
the match is good, clearly better than at heights a few millimetres away, and the cameras that
can judge it agree.  Plain surfaces carry no texture to match and give no answer.

This needs no outline, so it checks the outline-based answer of :meth:`Scene.surface_spot`
and stands in for it where nothing is outlined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..world import TABLE_Z

#: Half-size of the patch (pixels): an 11 x 11 patch.
HALF = 5
#: Grey-level spread (0..255) a patch needs to be matched at all.
MIN_CONTRAST = 4.0
#: Heights searched, relative to the table top (m), and the step.
LOW, HIGH, STEP = -0.01, 0.40, 0.001
#: Correlation the best height needs, and its lead over any height MARGIN_AWAY (m) or farther away.
GOOD = 0.8
LEAD = 0.08
MARGIN_AWAY = 0.004
#: Share of the patch another camera must see (in its picture, off the robot) at a height.
SEEN = 0.8
#: Cameras' heights within this (m) agree.
AGREE = 0.004


@dataclass
class Depth:
    """The level surface under a pixel, from matching pictures."""

    point: np.ndarray | None  # world point on the pixel's ray (None: no answer)
    height: float | None = None
    cams: list[str] = field(default_factory=list)  # cameras whose pictures agreed
    score: float = 0.0  # weakest of their correlations
    why: str = ""  # why there is no answer, or how the cameras disagree
    per_cam: dict[str, tuple[float, float]] = field(default_factory=dict)  # cam -> (height, correlation)


def _grey(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _correlate(patch: np.ndarray, samples: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero-mean normalized correlation of the patch (N,) with each row of samples (M, N), over the
    valid entries of that row; NaN where too few are valid."""
    w = valid.astype(np.float32)
    n = w.sum(axis=1)
    ok = n >= SEEN * patch.size
    n = np.maximum(n, 1.0)
    a = patch[None, :] * w
    b = samples * w
    ma = a.sum(axis=1) / n
    mb = b.sum(axis=1) / n
    da = (patch[None, :] - ma[:, None]) * w
    db = (samples - mb[:, None]) * w
    num = (da * db).sum(axis=1)
    den = np.sqrt((da * da).sum(axis=1) * (db * db).sum(axis=1))
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    out[~ok] = np.nan
    return out


def level_depth(views: dict, cam: str, xy, *, floor: float = TABLE_Z) -> Depth:
    """The level surface under pixel ``xy`` of ``cam``; ``views`` maps camera names to objects with
    ``camera``, ``image`` and ``robot`` (bool picture of the robot)."""
    view = views[cam]
    camera = view.camera
    x, y = int(round(xy[0])), int(round(xy[1]))
    if not (HALF <= x < camera.width - HALF and HALF <= y < camera.height - HALF):
        return Depth(None, why="too close to the picture's edge to match")
    grey = getattr(view, "grey", None)
    grey = grey if grey is not None else _grey(view.image)
    ys, xs = np.mgrid[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
    xs, ys = xs.ravel(), ys.ravel()
    mine = ~view.robot[ys, xs]
    if mine.mean() < SEEN:
        return Depth(None, why="the pixel is on the robot")
    patch = grey[ys, xs]
    if float(patch[mine].std()) < MIN_CONTRAST:
        return Depth(None, why="the surface there is too plain to match between pictures")
    origin = camera.position
    dirs = camera.rays(np.stack([xs, ys], axis=1).astype(float))
    if np.any(dirs[:, 2] > -0.05):
        return Depth(None, why="the ray runs too flat to search heights")
    heights = floor + np.arange(LOW, HIGH + 1e-9, STEP)
    t = (heights[:, None] - origin[2]) / dirs[None, :, 2]  # (M, N)
    points = origin + dirs[None, :, :] * t[..., None]
    per_cam: dict[str, tuple[float, float]] = {}
    found = []
    for name, other in views.items():
        if name == cam:
            continue
        px, depth = other.camera.project(points.reshape(-1, 3))
        w, h = other.camera.width, other.camera.height
        inside = (depth > 0.02) & (px[:, 0] >= 0) & (px[:, 0] <= w - 1) & (px[:, 1] >= 0) & (px[:, 1] <= h - 1)
        xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        inside &= ~other.robot[yi, xi]
        valid = inside.reshape(points.shape[:2]) & mine[None, :]
        if not valid.any():
            continue
        their = getattr(other, "grey", None)
        their = their if their is not None else _grey(other.image)
        # maps shaped (heights, patch pixels): remap wants fewer than 32767 columns
        mx = px[:, 0].astype(np.float32).reshape(points.shape[:2])
        my = px[:, 1].astype(np.float32).reshape(points.shape[:2])
        samples = cv2.remap(their, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        score = _correlate(patch, samples, valid)
        if np.all(np.isnan(score)):
            continue
        best = int(np.nanargmax(score))
        top = float(score[best])
        away = np.abs(heights - heights[best]) >= MARGIN_AWAY
        rival = float(np.nanmax(np.where(away, score, np.nan))) if np.any(away & ~np.isnan(score)) else -1.0
        if top < GOOD or top - rival < LEAD or best in (0, len(heights) - 1):
            per_cam[name] = (float(heights[best]), top)
            continue
        # refine with a parabola through the best height and its neighbours
        z = float(heights[best])
        s0, s1, s2 = score[best - 1], score[best], score[best + 1]
        if np.isfinite([s0, s1, s2]).all():
            bend = s0 - 2 * s1 + s2
            if bend < -1e-9:
                z += float(np.clip(0.5 * (s0 - s2) / bend, -0.5, 0.5)) * STEP
        per_cam[name] = (z, top)
        found.append((name, z, top))
    if not found:
        tried = ", ".join(f"{c} {s:.2f} at z {z:.3f}" for c, (z, s) in per_cam.items())
        return Depth(None, why=("no other camera's picture matches it clearly" + (f" ({tried})" if tried else
                                                                                    " (no other camera sees it)")),
                     per_cam=per_cam)
    zs = [z for _, z, _ in found]
    if max(zs) - min(zs) > AGREE:
        tried = ", ".join(f"{c}: z {z:.3f}" for c, z, _ in found)
        return Depth(None, why=f"the cameras disagree ({tried})", per_cam=per_cam)
    z = float(np.mean(zs))
    o, d = camera.ray((xy[0], xy[1]))
    point = o + d * ((z - o[2]) / d[2])
    return Depth(point, z, [c for c, _, _ in found], min(s for _, _, s in found), per_cam=per_cam)
