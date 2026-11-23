# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from martingale.hosts import robodojo_kinematics as kin
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
SCALE = 2
SHAPE = (240, 320)

def dilate(mask, radius=1):
    return cv2.dilate(mask.astype('uint8'), np.ones((2 * radius + 1,) * 2, 'uint8')) > 0

def basis(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def box_vertices(p):
    x, y, theta, length, width, height, taper = p
    xy = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * np.array([length, width]) / 2
    xy = np.concatenate([xy * taper, xy]) @ basis(theta).T + [x, y]
    return np.c_[xy, kin.TABLE_Z + np.repeat([0, height], 4)]

def arch_points(p, n=64):
    x, y, theta, halfspan, bottom, rise, radius = p
    t = np.linspace(0, np.pi, n)
    xy = np.outer(np.cos(t) * halfspan, [np.cos(theta), np.sin(theta)]) + [x, y]
    return np.c_[xy, bottom + rise * np.sin(t)]

def projection(p, view, kind):
    points = box_vertices(p) if kind == 'body' else arch_points(p)
    rr, cc, dd = view.project(points)
    mask = np.zeros(SHAPE, 'uint8')
    if np.any(dd <= 0.01):
        return mask.astype(bool)
    uv = np.rint(np.c_[cc, rr] / SCALE).astype('int32')
    if kind == 'body':
        cv2.fillConvexPoly(mask, cv2.convexHull(uv), 1)
    else:
        for i in range(len(uv) - 1):
            thickness = max(1, int(round(2 * p[-1] * view.fx / np.mean(dd[i:i + 2]) / SCALE)))
            cv2.line(mask, tuple(uv[i]), tuple(uv[i + 1]), 1, thickness)
    return mask.astype(bool)

def fit_model(views, masks, robots, kind, bounds, initial):
    targets = {}
    valid = {}
    weights = {}
    for c in CAMERAS:
        handle = masks[c]['handle'] & dilate(masks[c]['whole'], 3)
        raw = masks[c]['whole'] & ~handle if kind == 'body' else handle
        target = raw[::SCALE, ::SCALE].astype('uint8')
        if kind == 'body' and target.any():
            rc = np.argwhere(target)[:, ::-1].astype('int32')
            target[:] = 0
            cv2.fillConvexPoly(target, cv2.convexHull(rc), 1)
        targets[c] = target.astype(bool)
        other = handle if kind == 'body' else masks[c]['whole'] & ~handle
        unknown = (robots[c] | dilate(other))[::SCALE, ::SCALE]
        valid[c] = ~unknown | targets[c]
        weights[c] = min(1.0, np.sqrt(target.sum() / 250))
    usable = [c for c in CAMERAS if targets[c].sum() >= 12]
    if len(usable) < 2:
        return (None, {'accepted': False, 'reason': 'fewer than two part masks'})

    def cost(p):
        scores = []
        for c in usable:
            predicted = projection(p, views[c], kind) & valid[c]
            target = targets[c]
            overlap = (predicted & target).sum()
            scores.append(2 * overlap / max(1, predicted.sum() + target.sum()))
        return 1 - np.average(scores, weights=[weights[c] for c in usable])
    limits = np.asarray(bounds)
    margin = (limits[:, 1] - limits[:, 0]) * 1e-08
    initial = np.clip(initial, limits[:, 0] + margin, limits[:, 1] - margin)
    fitted = differential_evolution(cost, bounds, x0=initial, seed=17, popsize=9, maxiter=85, tol=0.001, polish=False)
    refined = minimize(cost, fitted.x, method='Powell', bounds=bounds, options={'maxiter': 35, 'xtol': 0.0003, 'ftol': 1e-05})
    best = min([fitted, refined], key=lambda r: r.fun)
    info = {'accepted': True, 'parameters': best.x, 'mask_fit_score': 1 - best.fun, 'cameras': usable, 'function_evaluations': int(fitted.nfev + refined.nfev)}
    return (best.x, info)
