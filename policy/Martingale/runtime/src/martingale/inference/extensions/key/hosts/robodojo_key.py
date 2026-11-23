# Copyright © 2026 Martingale. All Rights Reserved.
import cv2
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.ndimage import distance_transform_edt
from martingale.hosts import robodojo_kinematics as kin

def outline(p, side=1):
    x, y, yaw, r, s, w, d, t, g, e, h = p[:11]
    a = np.arcsin(np.clip(w / (2 * r), 0, 0.95))
    join = r * np.cos(a)
    start = join + s
    end = start + 2 * t + g
    tip = end + e
    xy = [[join, w / 2], [start, w / 2], [start, w / 2 + d], [start + t, w / 2 + d], [start + t, w / 2], [start + t + g, w / 2], [start + t + g, w / 2 + d], [end, w / 2 + d], [end, w / 2], [tip, w / 2], [tip, -w / 2]]
    xy += np.c_[r * np.cos(np.linspace(-a, -2 * np.pi + a, 41)), r * np.sin(np.linspace(-a, -2 * np.pi + a, 41))].tolist()
    xy = np.asarray(xy)
    xy[:, 1] *= side
    angle = np.radians(yaw)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return ((xy @ rotation.T + [x, y]) / 1000, tip / 1000)

def solids(p, side=1):
    x, y, yaw, r, s, w, d, t, g, e, h = p[:11]
    ratio = p[11]
    a = np.arcsin(np.clip(w / (2 * r), 0, 0.95))
    join = r * np.cos(a)
    start = join + s
    end = start + 2 * t + g
    tip = end + e
    blade = np.array([[0, w / 2], [start, w / 2], [start, w / 2 + d], [start + t, w / 2 + d], [start + t, w / 2], [start + t + g, w / 2], [start + t + g, w / 2 + d], [end, w / 2 + d], [end, w / 2], [tip, w / 2], [tip, -w / 2], [0, -w / 2]])
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    bow = np.c_[r * np.cos(angle), r * np.sin(angle)]
    tilt = -np.arctan2(h - h * ratio, 2 * (tip - r))
    yaw = np.radians(yaw)
    axis = np.array([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
    across = np.array([-np.sin(yaw), np.cos(yaw), 0])
    normal = np.cross(axis, across)
    center = np.array([x / 1000, y / 1000, kin.TABLE_Z + (h / 2 * np.cos(tilt) + r * abs(np.sin(tilt))) / 1000])
    result = []
    for xy, height in [(bow, h), (blade, h * ratio)]:
        xy = xy.copy()
        xy[:, 1] *= side
        mid = center + (xy[:, 0, None] * axis + xy[:, 1, None] * across) / 1000
        result.append((xy, np.vstack([mid - height / 2000 * normal, mid + height / 2000 * normal])))
    return (result, center, axis, tip / 1000)

def mesh(p, side=1):
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    vertices = []
    faces = []
    for xy, xyz in solids(p, side)[0]:
        poly = Polygon(xy)
        n = len(xy)
        offset = sum((len(v) for v in vertices))
        for triangle in triangulate(poly):
            if not poly.covers(triangle):
                continue
            ids = [int(np.argmin(np.linalg.norm(xy - np.array(c), axis=1))) for c in list(triangle.exterior.coords)[:3]]
            faces.extend([[i + offset for i in ids], [i + n + offset for i in ids]])
        for i in range(n):
            j = (i + 1) % n
            faces.extend([[i + offset, j + offset, j + n + offset], [i + offset, j + n + offset, i + n + offset]])
        vertices.append(xyz)
    return (np.vstack(vertices), np.asarray(faces, int))

def project_mask(p, side, view, roi, scale=3, *, pixel_centres=True):
    x0, y0, x1, y1 = roi
    canvas = np.zeros(((y1 - y0) * scale, (x1 - x0) * scale), 'uint8')
    for xy, xyz in solids(p, side)[0]:
        n = len(xy)
        rr, cc, _ = view.project(xyz)
        pts = np.c_[cc - x0, rr - y0] * scale
        if pixel_centres:
            pts += (scale - 1) / 2 - 0.5 * scale
        pix = np.rint(pts * 256).astype('int32')
        cv2.fillPoly(canvas, [pix[:n]], 255, shift=8)
        cv2.fillPoly(canvas, [pix[n:]], 255, shift=8)
        for i in range(n):
            j = (i + 1) % n
            cv2.fillConvexPoly(canvas, pix[[i, j, j + n, i + n]], 255, shift=8)
    return cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(float) / 255

def _fit_profile(observations):
    usable = []
    for name, view, mask in observations:
        mask = np.asarray(mask, bool)
        rr, cc = np.nonzero(mask)
        if len(rr) < 30:
            continue
        clipped = min(rr.min(), cc.min()) <= 1 or rr.max() >= mask.shape[0] - 2 or cc.max() >= mask.shape[1] - 2
        if name == 'cam_head' and clipped:
            continue
        pad = 12
        roi = (max(0, int(cc.min()) - pad), max(0, int(rr.min()) - pad), min(mask.shape[1], int(cc.max()) + pad + 1), min(mask.shape[0], int(rr.max()) + pad + 1))
        x0, y0, x1, y1 = roi
        target = mask[y0:y1, x0:x1].astype(float)
        usable.append((name, view, mask, roi, target))
    head = next((o for o in usable if o[0] == 'cam_head'), None)
    if head is None:
        return dict(accepted=False, reasons=['Need a complete head mask.'])
    rr, cc = np.nonzero(head[2])
    xy, ok = head[1].on_plane(rr, cc, kin.TABLE_Z + 0.004)
    xy = xy[ok] * 1000
    lo = xy.min(0) - 3
    pix = np.rint(xy - lo).astype('int32')
    canvas = np.zeros(tuple((pix.max(0) + 4)[::-1]), 'uint8')
    canvas[pix[:, 1], pix[:, 0]] = 1
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3, 3), 'uint8'))
    dt = distance_transform_edt(canvas)
    by, bx = np.unravel_index(dt.argmax(), dt.shape)
    bow = np.array([bx, by]) + lo
    r = max(3.0, float(dt.max()))
    _, _, vh = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    u = vh[0]
    if (xy.mean(0) - bow) @ u < 0:
        u = -u
    along = (xy - bow) @ u
    across = (xy - bow) @ np.array([-u[1], u[0]])
    length = float(along.max())
    shaft = across[(along > 0.25 * length) & (along < 0.55 * length)]
    width = float(np.ptp(shaft)) if len(shaft) > 4 else r * 0.5
    width = np.clip(width, 1.5, r)
    yaw = np.degrees(np.arctan2(u[1], u[0]))
    t = max(2.0, length * 0.08)
    gap = t
    e = max(1.0, length * 0.05)
    s = max(2.0, length - r - 2 * t - gap - e)
    initial = np.array([*bow, yaw, r, s, width, max(2.0, r * 0.6), t, gap, e, 6.0, 0.5])
    lower = np.array([bow[0] - 15, bow[1] - 15, yaw - 25, max(2, r * 0.5), max(1, s * 0.4), 0.7, 0.5, 0.7, 0.7, 0.3, 0.3, 0.15])
    upper = np.array([bow[0] + 15, bow[1] + 15, yaw + 25, r * 1.8, s * 1.7 + 5, r * 1.8, r * 2, t * 2.5, gap * 3, e * 4 + 2, 25.0, 1.0])

    def objective(p, side):
        if p[5] > 1.8 * p[3]:
            return 10 + (p[5] - 1.8 * p[3]) ** 2
        losses = []
        for name, view, mask, roi, target in usable:
            pred = project_mask(p, side, view, roi, pixel_centres=False)
            losses.append(np.sum((pred - target) ** 2) / (target.sum() + 1))
        return float(np.mean(losses))
    fits = []
    for side in (-1, 1):
        result = minimize(objective, np.clip(initial, lower, upper), args=(side,), method='Powell', bounds=list(zip(lower, upper)), options=dict(maxiter=45, xtol=0.005, ftol=0.0001))
        fits.append((result.fun, side, result.x))
    loss, side, p = min(fits, key=lambda f: f[0])
    xyz, faces = mesh(p, side)
    _, bow, axis, tip = solids(p, side)
    errors = {}
    for name, view, mask, roi, target in usable:
        pred = project_mask(p, side, view, roi, pixel_centres=False) > 0.5
        gt = target.astype(bool)
        errors[name] = float((pred & gt).sum() / max(1, (pred | gt).sum()))
    reasons = []
    if len(usable) < 2:
        reasons.append('Need a supporting view to resolve thickness.')
    rays = [(bow - o[1].pos) / np.linalg.norm(bow - o[1].pos) for o in usable]
    separation = max([np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for i, a in enumerate(rays) for b in rays[i + 1:]] or [0.0])
    if len(usable) > 1 and separation < 10:
        reasons.append('Insufficient camera separation.')
    if min(errors.values()) < 0.6:
        reasons.append('Key profile disagrees with a silhouette.')
    if p[10] > 24.5:
        reasons.append('Thickness reached its fit bound.')
    return dict(accepted=not reasons, reasons=reasons, kind='thick_bow_thin_toothed_shaft', parameters=p.tolist(), tooth_side=side, bow_center_xyz=bow.tolist(), tip_xyz=(bow + axis * tip).tolist(), shaft_axis_xyz=axis.tolist(), yaw_deg=float(p[2] % 360), bow_radius_m=float(p[3] / 1000), shaft_width_m=float(p[5] / 1000), thickness_m=float(np.ptp(xyz[:, 2])), bow_thickness_m=float(p[10] / 1000), shaft_thickness_m=float(p[10] * p[11] / 1000), top_z=float(xyz[:, 2].max()), length_m=float(tip + p[3] / 1000), vertices_xyz=xyz.tolist(), faces=faces.tolist(), view_iou=errors, loss=loss, supporting_views=[o[0] for o in usable], note='Resting on bow and tip only; tilt follows unequal thicknesses. Bow hole omitted; dimensions inferred from RGB.')

def fit_key(observations, prior=None):
    """Preserve the original shaft estimate while refining the bow's thickness.

    Without a prior, return the established profile fit. After a new view, a second,
    one-dimensional fit uses corrected pixel centres without moving the shaft tip
    in XY or changing its material thickness. All outputs come from one mesh.
    """
    observations = list(observations)
    if prior is None:
        return _fit_profile(observations)
    fit = dict(prior)
    if not fit['accepted']:
        return fit
    usable = []
    for name, view, mask in observations:
        if name not in fit['supporting_views']:
            continue
        mask = np.asarray(mask, bool)
        rr, cc = np.nonzero(mask)
        if len(rr) < 30:
            continue
        roi = (max(0, int(cc.min()) - 12), max(0, int(rr.min()) - 12), min(mask.shape[1], int(cc.max()) + 13), min(mask.shape[0], int(rr.max()) + 13))
        x0, y0, x1, y1 = roi
        usable.append((name, view, roi, mask[y0:y1, x0:x1].astype(float)))
    if len(usable) < 2 or not any((o[0] == 'cam_head' for o in usable)):
        return dict(fit, refinement=dict(applied=False, method='shaft_anchored_bow_thickness', reason='Need the head and an original supporting wrist mask; retaining prior.'))
    original = np.asarray(fit['parameters'])
    side = fit['tooth_side']
    shaft = original[10] * original[11]
    anchor = np.asarray(fit['tip_xyz'])

    def parameters(height):
        p = original.copy()
        p[10] = height
        p[11] = shaft / height
        _, bow, axis, tip = solids(p, side)
        p[:2] += (anchor[:2] - (bow + axis * tip)[:2]) * 1000
        return p

    def losses(height):
        p = parameters(height)
        return [float(np.sum((project_mask(p, side, v, roi) - target) ** 2) / (target.sum() + 1)) for _, v, roi, target in usable]

    def objective(height):
        return float(np.mean(losses(height)))
    grid = np.linspace(shaft, 25.0, 65)
    values = np.asarray([objective(h) for h in grid])
    best = int(values.argmin())
    before = losses(original[10])
    info = dict(applied=False, method='shaft_anchored_bow_thickness', original_bow_thickness_m=fit['bow_thickness_m'], corrected_loss_before=float(np.mean(before)))
    fit = dict(fit, refinement=info)
    if best in (0, len(grid) - 1):
        info['reason'] = 'Bow thickness minimum reached a bound.'
        return fit
    result = minimize_scalar(objective, bounds=(grid[best - 1], grid[best + 1]), method='bounded', options={'xatol': 0.01})
    height = min((float(grid[best]), float(result.x), float(original[10])), key=objective)
    if height > 24.5:
        info['reason'] = 'Refined thickness reached its fit bound.'
        return fit
    after = losses(height)
    if np.mean(before) - np.mean(after) <= 1e-06:
        info['reason'] = 'No resolved silhouette improvement.'
        return fit
    p = parameters(height)
    ious = {}
    for name, view, roi, target in usable:
        pred = project_mask(p, side, view, roi) > 0.5
        mask = target.astype(bool)
        ious[name] = float((pred & mask).sum() / max(1, (pred | mask).sum()))
    if min(ious.values()) < 0.6:
        info['reason'] = 'Refinement disagrees with a supporting silhouette.'
        return fit
    xyz, faces = mesh(p, side)
    _, bow, axis, tip = solids(p, side)
    info.update(applied=True, corrected_loss_after=float(np.mean(after)), view_losses={name: dict(before=b, after=a) for (name, *_), b, a in zip(usable, before, after)}, tip_shift_m=float(np.linalg.norm(bow + axis * tip - anchor)))
    fit.update(parameters=p.tolist(), bow_center_xyz=bow.tolist(), tip_xyz=(bow + axis * tip).tolist(), shaft_axis_xyz=axis.tolist(), thickness_m=float(np.ptp(xyz[:, 2])), bow_thickness_m=float(p[10] / 1000), shaft_thickness_m=float(shaft / 1000), top_z=float(xyz[:, 2].max()), vertices_xyz=xyz.tolist(), faces=faces.tolist(), view_iou=ious, loss=float(np.mean(after)))
    return fit

def measure(primitives, prior=None):
    from martingale.hosts import robodojo_tools as tools
    obs = tools._obs(primitives)
    sam = getattr(primitives, 'sam3_client', None)
    if sam is None:
        return dict(accepted=False, reasons=['No segmentation client.'])
    observations = []
    for name, cam in obs.get('vision', {}).items():
        view = tools._view(name, tools._shown(obs))
        image = cam.get('color')
        if view is None or image is None:
            continue
        result = sam.segment(image, text_prompt='the white key', min_score=0.2)
        if result.found:
            observations.append((name, view, result.mask))
    return fit_key(observations, prior=prior)
