# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from scipy.optimize import least_squares
from martingale.hosts import robodojo_sight as sight
from martingale.inference.extensions.mugs.hosts.robodojo_mug_helpers import _support, DIRECTIONS

def triangulate(views, pixels):
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for view, (r, c) in zip(views, pixels):
        d = view.rays(np.array([r]), np.array([c]))[0]
        d = d / np.linalg.norm(d)
        m = np.eye(3) - np.outer(d, d)
        a += m
        b += m @ view.pos
    if np.linalg.cond(a) > 100000.0:
        raise ValueError('Insufficient camera parallax')
    return np.linalg.solve(a, b)

def mesh(root, tip, radius):
    axis = np.asarray(tip) - root
    axis = axis / np.linalg.norm(axis)
    helper = np.eye(3)[np.argmin(abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.arange(64) * 2 * np.pi / 64
    ring = radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    vertices = np.concatenate([root + ring, tip + ring])
    faces = []
    for i in range(64):
        j = (i + 1) % 64
        faces.extend([[i, j, 64 + j], [i, 64 + j, 64 + i]])
    for i in range(1, 63):
        faces.extend([[0, i + 1, i], [64, 64 + i, 64 + i + 1]])
    return (vertices, np.asarray(faces))

def wood_mask(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return (lab[:, :, 0] > 160) & (lab[:, :, 2] > 140) & (rgb[:, :, 0].astype(float) > rgb[:, :, 2] * 1.12)

def peg_mask(rgb, seed):
    """Local RGB segmentation seeded by the selected peg's centreline."""
    root, tip = np.asarray(seed, float)
    d = tip - root
    length = np.linalg.norm(d)
    margin = max(12, 0.3 * length)
    low = np.maximum(0, np.floor(np.minimum(root, tip) - margin).astype(int))
    high = np.minimum(rgb.shape[:2], np.ceil(np.maximum(root, tip) + margin).astype(int))
    patch = rgb[low[0]:high[0], low[1]:high[1]]
    rr, cc = np.indices(patch.shape[:2])
    p = np.stack([rr + low[0], cc + low[1]], axis=-1) - root
    t = p @ d / length ** 2
    across = abs(p @ np.array([-d[1], d[0]])) / length
    labels = np.full(patch.shape[:2], cv2.GC_PR_BGD, np.uint8)
    labels[(t > 0) & (t < 1.1) & (across < max(4, 0.12 * length))] = cv2.GC_PR_FGD
    labels[(t > 0.3) & (t < 0.85) & (across < 1.1)] = cv2.GC_FGD
    labels[(t < -0.1) | (t > 1.3) | (across > max(8, 0.22 * length))] = cv2.GC_BGD
    cv2.grabCut(patch, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    mask = np.zeros(rgb.shape[:2], bool)
    mask[low[0]:high[0], low[1]:high[1]] = (labels == 1) | (labels == 3)
    return mask

def crop_peg(view, mask, seed, occluded=None):
    root, tip = np.asarray(seed, float)
    direction = tip - root
    length = np.linalg.norm(direction)
    if length < 12:
        raise ValueError('Peg is too foreshortened for a reliable fit')
    rr, cc = np.indices(mask.shape)
    p = np.stack([rr, cc], axis=-1) - root
    along = p @ direction / length ** 2
    across = abs(p @ np.array([-direction[1], direction[0]])) / length
    roi = (along > 0.22) & (along < 1.3) & (across < max(7, 0.2 * length))
    unknown = ~roi
    if occluded is not None:
        unknown |= occluded
    selected = mask & ~unknown
    if selected.sum() < 20:
        raise ValueError('Selected peg has too few visible pixels')
    return sight.Sight(view, selected, unknown)

def fit_peg(views, masks, seeds, occluded=None):
    if len(views) < 2:
        raise ValueError('Need two supporting views')
    observations = [crop_peg(v, m, s, None if occluded is None else occluded[i]) for i, (v, m, s) in enumerate(zip(views, masks, seeds))]
    root = triangulate(views, [s[0] for s in seeds])
    tip = triangulate(views, [s[1] for s in seeds])
    length = np.linalg.norm(tip - root)
    if not 0.02 < length < 0.4:
        raise ValueError('Inconsistent endpoint correspondence')
    widths = []
    for ob, seed in zip(observations, seeds):
        rr, cc = np.nonzero(ob.mask)
        direction = np.array(seed[1]) - seed[0]
        n = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        offsets = (np.c_[rr, cc] - seed[0]) @ n
        widths.append(0.5 * np.ptp(offsets) * length / np.linalg.norm(direction))
    radius = float(np.clip(np.median(widths), 0.0015, 0.02))
    p0 = np.r_[root, tip, radius]
    supports = [_support(o) for o in observations]
    strips = []
    for ob, seed in zip(observations, seeds):
        rr, cc = np.nonzero(ob.mask)
        origin = np.array(seed[0])
        d = np.array(seed[1]) - origin
        length_px = np.linalg.norm(d)
        d = d / length_px
        n = np.array([-d[1], d[0]])
        xy = np.c_[rr, cc] - origin
        along = xy @ d
        across = xy @ n
        samples = []
        for fraction in np.linspace(0.35, 0.85, 12):
            at = fraction * length_px
            values = across[abs(along - at) < 1.5]
            if len(values) >= 3:
                samples.append((at, 0.5 * (values.min() + values.max())))
        strips.append((origin, d, n, samples))

    def residual(p):
        vertices, _ = mesh(p[:3], p[3:6], p[6])
        parts = []
        for ob, (target, clipped) in zip(observations, supports):
            r, c, d = ob.view.project(vertices)
            e = (np.c_[c, r] @ DIRECTIONS.T).max(0) - target
            parts.append(np.where(clipped, np.minimum(e, 0), e))
        for ob, (origin, d, n, samples) in zip(observations, strips):
            rr, cc, _ = ob.view.project(np.array([p[:3], p[3:6]]))
            ends = np.c_[rr, cc] - origin
            along = ends @ d
            across = ends @ n
            parts.append(np.array([across[0] + (at - along[0]) * (across[1] - across[0]) / (along[1] - along[0]) - observed for at, observed in samples]))
        parts.append((p[:3] - root) * 5)
        return np.concatenate(parts)
    lo = np.r_[p0[:6] - 0.06, 0.001]
    hi = np.r_[p0[:6] + 0.06, 0.025]
    sol = least_squares(residual, p0, bounds=(lo, hi), x_scale='jac', loss='soft_l1', f_scale=1, max_nfev=150)
    p = sol.x
    vertices, _ = mesh(p[:3], p[3:6], p[6])
    scores = []
    for ob in observations:
        r, c, d = ob.view.project(vertices)
        pred = np.zeros(ob.mask.shape, np.uint8)
        cv2.fillConvexPoly(pred, cv2.convexHull(np.rint(np.c_[c, r]).astype('int32')), 1)
        a = pred.astype(bool) & ~ob.unknown
        b = ob.mask
        scores.append(float((a & b).sum() / max(1, (a | b).sum())))
    axis = p[3:6] - p[:3]
    axis /= np.linalg.norm(axis)
    rmse = float(np.sqrt(np.mean(residual(p)[:-3] ** 2)))
    return dict(valid=bool(min(scores) > 0.65 and rmse < 2), root_xyz=p[:3].tolist(), tip_xyz=p[3:6].tolist(), axis_xyz=axis.tolist(), radius_m=float(p[6]), length_m=float(np.linalg.norm(p[3:6] - p[:3])), per_view_iou=scores, support_rmse_px=rmse, baseline=dict(root_xyz=root.tolist(), tip_xyz=tip.tolist(), radius_m=radius), note='Selected static peg only; refit if the rack moves. Visibility does not certify insertion clearance.')
