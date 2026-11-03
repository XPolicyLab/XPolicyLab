"""Metric visible-surface geometry. No simulator object truth enters this module.

Rotations map box-local axes into the named world frame. A fitted box is NOT an
object's semantic heading. The equivalent box frames are aligned before differencing.
"""
from __future__ import annotations

import itertools
import numpy as np


def rotation(value):
    r = np.asarray(value, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("rotation must be a finite proper SO(3) matrix")
    return r


def backproject(depth, mask, K, world_from_camera, *, confidence=None, min_confidence=0.):
    """OpenCV optical z-depth in metres; all pixel arrays must already be aligned."""
    z, mask, k, t = np.asarray(depth), np.asarray(mask, bool), np.asarray(K, float), np.asarray(world_from_camera, float)
    if z.ndim != 2 or z.shape != mask.shape or k.shape != (3, 3) or t.shape != (4, 4):
        raise ValueError("unaligned depth/mask or invalid calibration shape")
    if not np.isfinite(k).all() or not np.isfinite(t).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or np.allclose(k, np.eye(3)):
        raise ValueError("missing or placeholder metric camera calibration")
    rotation(t[:3, :3])
    if not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous transform")
    valid = mask & np.isfinite(z) & (z > 0) & (z < 20)
    if confidence is not None:
        confidence = np.asarray(confidence)
        if confidence.shape != z.shape:
            raise ValueError("unaligned depth confidence")
        valid &= np.isfinite(confidence) & (confidence >= min_confidence)
    v, u = np.nonzero(valid)
    values = z[v, u]
    if len(values) < 8:
        raise ValueError("insufficient valid object depth")
    # Robust depth gate; retain a noise floor so near-planar objects survive.
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    keep = np.abs(values - median) <= max(0.015, 4.5 * 1.4826 * mad)
    rays = np.linalg.solve(k, np.stack([u[keep], v[keep], np.ones(keep.sum())]))
    points = (rays * values[keep]).T @ t[:3, :3].T + t[:3, 3]
    return points, float(keep.sum() / max(1, mask.sum()))


def _frames():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            q = np.eye(3)[:, perm] * signs
            if np.linalg.det(q) > 0:
                yield q


BOX_FRAMES = tuple(_frames())


def align_box(R, dimensions, reference):
    """Resolve the 24 equivalent axis/sign conventions; not semantic symmetry."""
    r, reference = rotation(R), rotation(reference)
    q = max(BOX_FRAMES, key=lambda q: np.trace(reference.T @ r @ q))
    return r @ q, np.abs(q).T @ np.asarray(dimensions), q


def angular_velocity(previous, current, dt):
    if dt <= 0:
        return np.zeros(3)
    delta = rotation(current) @ rotation(previous).T
    # Eigenvector construction handles rotations near pi, unlike sin(angle) division.
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    if angle < 1e-7:
        return np.zeros(3)
    skew = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    if np.linalg.norm(skew) < 1e-7:
        vals, vecs = np.linalg.eig(delta)
        axis = np.real(vecs[:, np.argmin(abs(vals - 1))])
    else:
        axis = skew
    return axis / np.linalg.norm(axis) * angle / dt


def fit_box(points, reference=None, upright=False):
    """Robust OBB: hull-edge minimum area upright, SO(3) volume refinement otherwise.

    Upright is an explicitly selected support-plane hypothesis, never inferred just
    from low motion. General 3D PCA fits are estimates, not minimum-volume proofs.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 8 or not np.isfinite(p).all():
        raise ValueError("at least eight finite XYZ points required")
    center = np.median(p, axis=0)
    if upright:
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull = p[ConvexHull(p[:, :2]).vertices, :2]
            edges = np.roll(hull, -1, axis=0) - hull
            angles = np.arctan2(edges[:, 1], edges[:, 0])
        except QhullError:
            angles = [0.]
        candidates = []
        for angle in angles:
            c, s = np.cos(angle), np.sin(angle)
            r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            local = (p - center) @ r
            lo, hi = local.min(axis=0), local.max(axis=0)
            candidates.append(((hi[0]-lo[0])*(hi[1]-lo[1]), r))
        r = min(candidates, key=lambda item: item[0])[1]
    else:
        _, _, vh = np.linalg.svd(p - center, full_matrices=False)
        r = vh.T
        if np.linalg.det(r) < 0:
            r[:, -1] *= -1
        # PCA is only an initializer. Refine the enclosing volume in SO(3), with
        # a bounded iteration count so geometric fitting cannot dominate a turn.
        from scipy.optimize import minimize
        from scipy.spatial.transform import Rotation
        sample = p[np.linspace(0, len(p)-1, min(1024, len(p))).astype(int)] - center
        initial = r.copy()
        def volume(vector):
            local = sample @ (initial @ Rotation.from_rotvec(vector).as_matrix())
            lo, hi = np.quantile(local, [.01, .99], axis=0)
            return float(np.prod(np.maximum(hi-lo, 1e-5)))
        fitted = minimize(volume, np.zeros(3), method="Powell", bounds=[(-.3, .3)]*3,
                          options={"maxiter": 8, "maxfev": 160, "xtol": .002, "ftol": .002})
        if volume(fitted.x) < volume(np.zeros(3)):
            r = initial @ Rotation.from_rotvec(fitted.x).as_matrix()
    local = (p - center) @ r
    lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
    dimensions = hi - lo
    box_center = center + r @ ((lo + hi) / 2)
    if reference is not None:
        r, dimensions, _ = align_box(r, dimensions, reference)
    return {"center": box_center.tolist(), "dimensions": dimensions.tolist(), "R": r.tolist(),
            "extent_kind": "visible_surface", "orientation_kind": "box_axes_not_semantic_heading",
            "fit": "upright_min_area" if upright else "robust_volume_3d", "points": len(p)}


def pattern(image, mask):
    """A cheap masked color/texture descriptor; original crops remain the ReID evidence."""
    from PIL import Image
    image, mask = np.asarray(image), np.asarray(mask, bool)
    v, u = np.nonzero(mask)
    if not len(u):
        raise ValueError("empty instance mask")
    bounds = [int(u.min()), int(v.min()), int(u.max()+1), int(v.max()+1)]
    x0, y0, x1, y1 = bounds
    crop = image[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 0
    grid = np.asarray(Image.fromarray(crop).resize((16, 16)), float) / 255
    hist = np.concatenate([np.histogram(image[..., i][mask], bins=16, range=(0, 256))[0] for i in range(3)]).astype(float)
    hist /= max(1, np.linalg.norm(hist))
    texture = np.concatenate([np.diff(grid, axis=0).ravel(), np.diff(grid, axis=1).ravel()])
    texture /= max(1e-8, np.linalg.norm(texture))
    return bounds, crop, np.concatenate([hist, texture]).tolist()
