# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from martingale.hosts import robodojo_kinematics as kin
_ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
_RING = np.c_[np.cos(_ANGLES), np.sin(_ANGLES)]
_DIRECTIONS = _RING[::2]

def detect_tokens(image):
    """Closed dark neutral silhouettes; output includes their printed white marks."""
    hsv = cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2HSV)
    dark = ((hsv[:, :, 1] < 25) & (hsv[:, :, 2] < 175)).astype('uint8')
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        xy = contour[:, 0].astype(float)
        if len(xy) < 12 or not 60 < cv2.contourArea(contour) < 18000:
            continue
        body = np.zeros(image.shape[:2], 'uint8')
        cv2.drawContours(body, [contour], -1, 1, cv2.FILLED)
        kernel_size = 5 if cv2.contourArea(contour) > 2000 else 3
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        cleaned, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cleaned:
            continue
        contour = max(cleaned, key=cv2.contourArea)
        xy = contour[:, 0].astype(float)
        if len(xy) < 12:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(x, y) <= 1 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
            continue
        center, axes, angle = cv2.fitEllipse(contour)
        if min(axes) < 5 or min(axes) / max(axes) < 0.16:
            continue
        area_ratio = cv2.contourArea(contour) / (np.pi * axes[0] * axes[1] / 4)
        if not 0.85 < area_ratio < 1.12:
            continue
        filled = np.zeros(image.shape[:2], 'uint8')
        cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
        white = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 190)
        white_share = float(white[filled.astype(bool)].mean())
        if not 0.015 < white_share < 0.6:
            continue
        candidates.append(dict(pixels=xy, pixel_xy=np.asarray(center), mask=filled.astype(bool)))
    return sorted(candidates, key=lambda c: (c['pixel_xy'][0], c['pixel_xy'][1]))

def projected_outline(parameters, view, base_z):
    """Project both circular rims of a disk; their convex hull is its silhouette."""
    x, y, radius, thickness = parameters
    xy = np.array([x, y]) + radius * _RING
    xyz = np.vstack([np.c_[xy, np.full(len(xy), base_z + z)] for z in (0, thickness)])
    rows, cols, depth = view.project(xyz)
    return (np.c_[cols, rows], depth)

def _initial(candidate, view, base_z):
    pixels = candidate['pixels']
    xy, valid = view.on_plane(pixels[:, 1], pixels[:, 0], base_z + 0.006)
    if not np.all(valid):
        return None
    origin = xy.mean(0)
    local = xy - origin
    q = np.linalg.lstsq(np.c_[2 * local, np.ones(len(xy))], (local * local).sum(1), rcond=None)[0]
    radius = np.sqrt(max(0.0, q[2] + q[:2] @ q[:2]))
    return np.r_[origin + q[:2], radius]

def fit_disk(observations, base_z=kin.TABLE_Z):
    """Fit center XY, radius and thickness jointly to calibrated view contours.

    observations contains (camera_name, View, candidate) tuples. All views must
    describe the same complete silhouette. The disk is assumed flat on base_z;
    this is not a pose estimator for carried, tilted, stacked or occluded disks.
    """
    if len(observations) < 2:
        return dict(accepted=False, reasons=['Need two complete supporting views.'])
    guesses = [_initial(c, v, base_z) for _, v, c in observations]
    if any((g is None for g in guesses)):
        return dict(accepted=False, reasons=['Disk is outside the supported plane.'])
    seed = np.median(guesses, axis=0)
    expected = [np.max(c['pixels'] @ _DIRECTIONS.T, axis=0) + 0.5 for _, _, c in observations]

    def residual(p):
        return np.concatenate([np.max(projected_outline(p, v, base_z)[0] @ _DIRECTIONS.T, axis=0) - s for (_, v, _), s in zip(observations, expected)])
    bounds = ([seed[0] - 0.04, seed[1] - 0.04, 0.005, 0.0003], [seed[0] + 0.04, seed[1] + 0.04, 0.1, 0.06])
    if not 0.005 < seed[2] < 0.1:
        return dict(accepted=False, reasons=['Not a supported small disk.'])
    fits = [least_squares(residual, np.r_[seed, height], bounds=bounds, diff_step=0.0001, loss='soft_l1', f_scale=1, max_nfev=100) for height in (0.003, 0.015)]
    best = min(fits, key=lambda f: np.mean(residual(f.x) ** 2))
    x, y, radius, thickness = best.x
    errors = residual(best.x).reshape(len(observations), -1)
    rms = np.sqrt(np.mean(errors ** 2, axis=1))
    center = np.array([x, y, base_z + thickness / 2])
    rays = [center - v.pos for _, v, _ in observations]
    rays = [r / np.linalg.norm(r) for r in rays]
    separation = max((np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in itertools.combinations(rays, 2)))
    reasons = []
    if separation < 10:
        reasons.append('Views lack parallax; obtain an oblique wrist view.')
    if max(rms) > 1.5:
        reasons.append('Silhouettes disagree with a complete flat disk.')
    if thickness >= 0.0595 or thickness <= 0.00035:
        reasons.append('Thickness is at the fitting bound.')
    if thickness > radius:
        reasons.append('Not a shallow disk.')
    uncertainty = np.sqrt(np.maximum(0, np.diag(np.linalg.pinv(best.jac.T @ best.jac))))
    if uncertainty[3] > 0.004:
        reasons.append('Thickness is poorly constrained by these views.')
    return dict(accepted=not reasons, reasons=reasons, kind='disk', center_xyz=center.tolist(), top_z=float(base_z + thickness), base_z=float(base_z), radius_m=float(radius), diameter_m=float(2 * radius), thickness_m=float(thickness), normal_xyz=[0.0, 0.0, 1.0], supporting_views=[n for n, _, _ in observations], silhouette_rms_px=float(np.sqrt(np.mean(errors ** 2))), view_errors_px={n: float(e) for (n, _, _), e in zip(observations, rms)}, view_separation_deg=float(separation), height_sensitivity_m=float(uncertainty[3]))

def fit_tokens(images, views, base_z=kin.TABLE_Z):
    """Match each head token to supporting silhouettes and fit one shared disk."""
    detections = {name: detect_tokens(im) for name, im in images.items() if name in views}
    output = []
    for head in detections.get('cam_head', []):
        seed = _initial(head, views['cam_head'], base_z)
        if seed is None or not 0.005 < seed[2] < 0.1:
            continue
        groups = []
        for name, candidates in detections.items():
            if name == 'cam_head':
                continue
            nearby = []
            for candidate in candidates:
                guess = _initial(candidate, views[name], base_z)
                if guess is not None and np.linalg.norm(seed[:2] - guess[:2]) < 0.035 and (abs(seed[2] - guess[2]) < 0.015):
                    nearby.append((name, views[name], candidate))
            if nearby:
                groups.append(nearby)
        anchor = ('cam_head', views['cam_head'], head)
        fits = []
        for count in range(len(groups), 0, -1):
            for subset in itertools.combinations(groups, count):
                fits.extend((fit_disk([anchor, *others], base_z) for others in itertools.product(*subset)))
        fits = [f for f in fits if 'silhouette_rms_px' in f]
        chosen = min(fits, key=lambda f: (not f['accepted'], -len(f['supporting_views']), f['silhouette_rms_px'])) if fits else dict(accepted=False, reasons=['No complete supporting wrist view; reposition the free wrist.'])
        chosen['excluded_views'] = [g[0][0] for g in groups if g[0][0] not in chosen.get('supporting_views', [])]
        output.append(dict(id=len(output), head_pixel_xy=head['pixel_xy'].tolist(), observation_target_xyz=[float(seed[0]), float(seed[1]), float(base_z)], **chosen))
    return dict(tokens=output, accepted_count=sum((t['accepted'] for t in output)), note='Identify +, -, x or divide visually at head_pixel_xy. Geometry fits the entire disk, not the printed mark. observation_target_xyz is approximate and only for aiming a camera, never a measured grasp height. Only flat unoccluded disks resting on the table are supported. A disk has no geometrically preferred grasp yaw; choose finger clearance. The placed symbol must end in the upright reading direction of the equation: correct any initial skew rather than preserving it.')

def measure(primitives):
    from martingale.hosts import robodojo_tools as tools
    obs = tools._obs(primitives)
    views = {name: tools._view(name, tools._shown(obs)) for name in obs.get('vision', {})}
    views = {name: view for name, view in views.items() if view is not None}
    images = {name: np.asarray(obs['vision'][name]['color']) for name in views if obs['vision'][name].get('color') is not None}
    return fit_tokens(images, views)
