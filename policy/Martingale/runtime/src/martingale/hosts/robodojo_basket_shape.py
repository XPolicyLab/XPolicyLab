# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SCALE = 2
SHAPE = (240, 320)


def dilate(mask, radius=1):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or radius < 0:
        raise ValueError("mask must be 2-D and radius nonnegative")
    return cv2.dilate(mask.astype("uint8"), np.ones((2 * radius + 1,) * 2, "uint8")) > 0


def _basis(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def box_vertices(p, table_z_m):
    x, y, theta, length, width, height, taper = np.asarray(p, float)
    xy = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * np.array([length, width]) / 2
    xy = np.concatenate([xy * taper, xy]) @ _basis(theta).T + [x, y]
    return np.c_[xy, table_z_m + np.repeat([0, height], 4)]


def arch_points(p, n=64):
    x, y, theta, halfspan, bottom, rise, _radius = np.asarray(p, float)
    t = np.linspace(0, np.pi, n)
    xy = np.outer(np.cos(t) * halfspan, [np.cos(theta), np.sin(theta)]) + [x, y]
    return np.c_[xy, bottom + rise * np.sin(t)]


def _projection(p, view, kind, table_z_m):
    points = box_vertices(p, table_z_m) if kind == "body" else arch_points(p)
    rr, cc, depth = view.project(points)
    mask = np.zeros(SHAPE, "uint8")
    if np.any(depth <= .01):
        return mask.astype(bool)
    uv = np.rint(np.c_[cc, rr] / SCALE).astype("int32")
    if kind == "body":
        cv2.fillConvexPoly(mask, cv2.convexHull(uv), 1)
    else:
        for i in range(len(uv) - 1):
            thick = max(1, int(round(2 * p[-1] * view.fx / np.mean(depth[i:i + 2]) / SCALE)))
            cv2.line(mask, tuple(uv[i]), tuple(uv[i + 1]), 1, thick)
    return mask.astype(bool)


def fit_part(views, masks, robot_masks, kind, bounds, initial, table_z_m):
    """Fit one part from synchronized calibrated views and binary masks."""
    if kind not in ("body", "handle"):
        raise ValueError("kind must be body or handle")
    targets, valid, weights = {}, {}, {}
    for camera in CAMERAS:
        if camera not in views or camera not in masks or camera not in robot_masks:
            continue
        whole = np.asarray(masks[camera]["whole"], bool)
        handle = np.asarray(masks[camera]["handle"], bool) & dilate(whole, 3)
        robot = np.asarray(robot_masks[camera], bool)
        if whole.shape != (480, 640) or handle.shape != whole.shape or robot.shape != whole.shape:
            raise ValueError(f"{camera}: masks must be 480x640")
        raw = whole & ~handle if kind == "body" else handle
        target = raw[::SCALE, ::SCALE].astype("uint8")
        if kind == "body" and target.any():
            rc = np.argwhere(target)[:, ::-1].astype("int32")
            target[:] = 0
            cv2.fillConvexPoly(target, cv2.convexHull(rc), 1)
        targets[camera] = target.astype(bool)
        other = handle if kind == "body" else (whole & ~handle)
        unknown = (robot | dilate(other))[::SCALE, ::SCALE]
        valid[camera] = ~unknown | targets[camera]
        weights[camera] = min(1., np.sqrt(target.sum() / 250))
    usable = [c for c in CAMERAS if c in targets and targets[c].sum() >= 12]
    if len(usable) < 2:
        return None, {"accepted": False, "reason": "fewer than two part masks", "supporting_views": usable}

    def cost(params):
        scores = []
        for camera in usable:
            predicted = _projection(params, views[camera], kind, table_z_m) & valid[camera]
            target = targets[camera]
            scores.append(2 * (predicted & target).sum() / max(1, predicted.sum() + target.sum()))
        return 1 - np.average(scores, weights=[weights[c] for c in usable])

    limits = np.asarray(bounds, float)
    init = np.asarray(initial, float)
    if limits.ndim != 2 or limits.shape[1] != 2 or init.shape != (len(limits),):
        raise ValueError("bounds/initial shapes do not agree")
    if not np.isfinite(limits).all() or not np.isfinite(init).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("finite, increasing bounds and finite initial values are required")
    margin = (limits[:, 1] - limits[:, 0]) * 1e-8
    init = np.clip(init, limits[:, 0] + margin, limits[:, 1] - margin)
    fitted = differential_evolution(cost, list(map(tuple, limits)), x0=init, seed=17,
                                    popsize=9, maxiter=85, tol=.001, polish=False)
    refined = minimize(cost, fitted.x, method="Powell", bounds=list(map(tuple, limits)),
                       options={"maxiter": 35, "xtol": .0003, "ftol": .00001})
    best = min((fitted, refined), key=lambda result: result.fun)
    return best.x, {"accepted": True, "parameters": best.x.tolist(),
                    "mask_fit_score": float(1 - best.fun), "supporting_views": usable,
                    "function_evaluations": int(fitted.nfev + refined.nfev)}
