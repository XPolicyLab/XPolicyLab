"""Camera and point-set geometry for perception (numpy only).

Pixel (u, v) indexes image[v, u]; integer coordinates are pixel centres. Cameras use
the optical frame (+X right, +Y down, +Z forward) and extrinsics map camera to world.
Optical depth is the camera-frame Z of a point, so a camera ray scaled to Z = 1 and
multiplied by the depth gives the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np


class Points(NamedTuple):
    """Points with the pixel each one came from."""

    xyz: np.ndarray  # N x 3
    rows: np.ndarray  # N, image row (v)
    cols: np.ndarray  # N, image column (u)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def select(self, keep: np.ndarray) -> "Points":
        return Points(self.xyz[keep], self.rows[keep], self.cols[keep])


# ---------------------------------------------------------------- cameras


def check_calibration(intrinsics, extrinsics) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(intrinsics, dtype=float)
    t = np.asarray(extrinsics, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("intrinsics must be a finite 3x3 pinhole matrix")
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("extrinsics must be a finite 4x4 rigid transform")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-3) or abs(float(np.linalg.det(r)) - 1.0) > 1e-3:
        raise ValueError("extrinsics rotation is not orthonormal")
    return k, t


def camera_rays(intrinsics: np.ndarray, u, v) -> np.ndarray:
    """N x 3 camera-frame rays through pixels (u, v), scaled to Z = 1."""
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    pix = np.stack([u, v, np.ones_like(u)])
    return np.linalg.solve(np.asarray(intrinsics, dtype=float), pix).T


def pixel_rays(intrinsics: np.ndarray, extrinsics: np.ndarray, u, v) -> tuple[np.ndarray, np.ndarray]:
    """World ray origin (3,) and unit directions (N x 3) through pixels (u, v)."""
    t = np.asarray(extrinsics, dtype=float)
    dirs = camera_rays(intrinsics, u, v) @ t[:3, :3].T
    return t[:3, 3].copy(), dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def depth_gain_z(intrinsics: np.ndarray, extrinsics: np.ndarray, u, v) -> np.ndarray:
    """World Z change per metre of optical depth at each pixel (negative when looking down)."""
    return camera_rays(intrinsics, u, v) @ np.asarray(extrinsics, dtype=float)[2, :3]


def valid_depth(depth: np.ndarray, near: float = 0.05, far: float = 5.0) -> np.ndarray:
    d = np.asarray(depth, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(d) & (d > near) & (d < far)


def backproject(depth: np.ndarray, intrinsics: np.ndarray, mask: Optional[np.ndarray] = None) -> Points:
    """Camera-frame points of the valid depth pixels (inside mask, if given)."""
    depth = np.asarray(depth, dtype=float)
    keep = valid_depth(depth)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(keep)
    xyz = camera_rays(intrinsics, cols, rows) * depth[rows, cols][:, None]
    return Points(xyz, rows, cols)


def to_world(points_cam: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    t = np.asarray(extrinsics, dtype=float)
    return np.asarray(points_cam, dtype=float).reshape(-1, 3) @ t[:3, :3].T + t[:3, 3]


def depth_to_world(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Points:
    cam = backproject(depth, intrinsics, mask)
    return Points(to_world(cam.xyz, extrinsics), cam.rows, cam.cols)


def project_points(intrinsics: np.ndarray, extrinsics: np.ndarray, xyz) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N x 2) and optical depth (N,) of world points; uv is NaN behind the camera."""
    t = np.asarray(extrinsics, dtype=float)
    p = (np.asarray(xyz, dtype=float).reshape(-1, 3) - t[:3, 3]) @ t[:3, :3]
    z = p[:, 2]
    uvw = p @ np.asarray(intrinsics, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.where((z > 1e-9)[:, None], uvw[:, :2] / uvw[:, 2:3], np.nan)
    return uv, z


def in_view(uv: np.ndarray, depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """True where a projected point lands on a pixel of a (width, height) image in front of the camera."""
    w, h = size
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return ((np.asarray(depth) > 0) & (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5)
                & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5))


def ray_plane_z(origin: np.ndarray, dirs: np.ndarray, z: float) -> np.ndarray:
    """Intersections of rays with the horizontal plane at height z; NaN rows where it is behind or parallel."""
    dirs = np.asarray(dirs, dtype=float).reshape(-1, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - origin[2]) / dirs[:, 2]
    s = np.where(np.abs(dirs[:, 2]) > 1e-9, s, np.nan)
    s = np.where(s > 0, s, np.nan)
    return origin + s[:, None] * dirs


def optical_axis_tilt_deg(extrinsics: np.ndarray) -> float:
    """Angle between the camera's optical axis and straight down."""
    forward = np.asarray(extrinsics, dtype=float)[:3, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(forward[2]) / float(np.linalg.norm(forward))))))


# ---------------------------------------------------------------- planes


@dataclass
class PlaneFit:
    """normal . p + offset = 0, with the normal's Z made non-negative."""

    normal: np.ndarray
    offset: float
    inliers: np.ndarray  # bool, one per input point
    rms: float

    @property
    def inlier_ratio(self) -> float:
        return float(self.inliers.mean()) if self.inliers.size else 0.0

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.normal[2])))))

    def z_at(self, x: float, y: float) -> float:
        n = self.normal
        if abs(n[2]) < 1e-9:
            raise ValueError("vertical plane has no height")
        return float(-(n[0] * x + n[1] * y + self.offset) / n[2])

    def distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float).reshape(-1, 3) @ self.normal + self.offset


def _lsq_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ centroid)


def fit_plane(points: np.ndarray, *, threshold: float = 0.003, iterations: int = 200,
              seed: int = 0, max_points: int = 20000) -> PlaneFit:
    """RANSAC plane (largest consensus) refined by least squares on its inliers."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points_ok = np.isfinite(points).all(axis=1)
    pts = points[points_ok]
    if len(pts) < 3:
        raise ValueError("plane fit needs at least 3 finite points")
    rng = np.random.default_rng(seed)
    scored = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]
    best_count, best = -1, None
    for _ in range(iterations):
        a, b, c = scored[rng.choice(len(scored), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n /= norm
        count = int((np.abs((scored - a) @ n) < threshold).sum())
        if count > best_count:
            best_count, best = count, (n, -float(n @ a))
    if best is None:
        raise ValueError("points are degenerate (collinear)")
    normal, offset = best
    inliers = np.abs(pts @ normal + offset) < threshold
    for _ in range(2):
        if inliers.sum() < 3:
            break
        normal, offset = _lsq_plane(pts[inliers])
        inliers = np.abs(pts @ normal + offset) < threshold
    if normal[2] < 0:
        normal, offset = -normal, -offset
    full = np.zeros(len(points), dtype=bool)
    full[np.nonzero(points_ok)[0]] = inliers
    resid = pts[inliers] @ normal + offset
    rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
    return PlaneFit(normal, float(offset), full, rms)


# ---------------------------------------------------------------- 2-D shapes


def normalize_axis_deg(deg: float) -> float:
    """An undirected axis angle in (-90, 90]."""
    return 90.0 - (90.0 - float(deg)) % 180.0


def convex_hull(xy: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull vertices (Andrew's monotone chain), without repeating the first."""
    pts = np.unique(np.asarray(xy, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts

    def chain(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append((p[0], p[1]))
        return out

    lower, upper = chain(pts), chain(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


@dataclass
class Rect:
    """Oriented rectangle: length >= width, long-axis angle in degrees from +X in (-90, 90]."""

    center: np.ndarray
    length: float
    width: float
    angle_deg: float

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        a = math.radians(self.angle_deg)
        u = np.array([math.cos(a), math.sin(a)])
        return u, np.array([-u[1], u[0]])

    def corners(self) -> np.ndarray:
        u, v = self.axes
        hl, hw = self.length / 2, self.width / 2
        return np.array([self.center + su * hl * u + sv * hw * v for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])

    def grown(self, margin: float) -> "Rect":
        return Rect(self.center, self.length + margin, self.width + margin, self.angle_deg)


def _extents(pts: np.ndarray, deg: float, trim_pct: float):
    a = math.radians(deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    pu, pv = pts @ u, pts @ v
    if trim_pct > 0:
        (ul, uh), (vl, vh) = np.percentile(pu, [trim_pct, 100 - trim_pct]), np.percentile(pv, [trim_pct, 100 - trim_pct])
    else:
        ul, uh, vl, vh = pu.min(), pu.max(), pv.min(), pv.max()
    return float(uh - ul), float(vh - vl), u * (ul + uh) / 2 + v * (vl + vh) / 2


def min_area_rect(xy: np.ndarray, trim_pct: float = 0.0) -> Rect:
    """Minimum-area enclosing rectangle by rotating calipers over the convex hull's edge directions.

    With trim_pct > 0 each candidate's extents are percentiles of all points, so a few outliers on
    the hull neither set the size nor win the orientation.
    """
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError("rectangle fit needs points")
    hull = convex_hull(pts)
    if len(hull) < 3:
        if len(hull) == 1:
            return Rect(hull[0].copy(), 0.0, 0.0, 0.0)
        d = hull[1] - hull[0]
        return Rect(hull.mean(axis=0), float(np.linalg.norm(d)), 0.0,
                    normalize_axis_deg(math.degrees(math.atan2(d[1], d[0]))))
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.round(np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 90.0, 6))
    sample = pts if trim_pct > 0 else hull
    best = None
    for deg in angles:
        lu, lv, centre = _extents(sample, float(deg), trim_pct)
        if best is None or lu * lv < best[0] - 1e-15:
            best = (lu * lv, float(deg), lu, lv, centre)
    _, deg, lu, lv, centre = best
    if lv > lu:
        lu, lv, deg = lv, lu, deg + 90.0
    return Rect(np.asarray(centre), lu, lv, normalize_axis_deg(deg))


@dataclass
class Circle:
    center: np.ndarray
    radius: float
    rms: float
    arc_deg: float  # angular coverage of the supporting points
    inlier_ratio: float


def _kasa(xy: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.c_[2 * xy, np.ones(len(xy))] * w[:, None]
    b = (xy ** 2).sum(axis=1) * w
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = sol[:2]
    return centre, math.sqrt(max(float(sol[2] + centre @ centre), 0.0))


def arc_coverage_deg(xy: np.ndarray, centre: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    angles = np.sort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
    return 360.0 - math.degrees(float(gaps.max()))


def _median_residual(xy: np.ndarray, centre: np.ndarray, radius: float) -> float:
    return float(np.median(np.abs(np.linalg.norm(xy - centre, axis=1) - radius)))


def _lmeds_circle(xy: np.ndarray, trials: int, seed: int, max_points: int = 400) -> tuple[np.ndarray, float]:
    """Least-median-of-squares circle through random point triplets (robust start)."""
    rng = np.random.default_rng(seed)
    if len(xy) > max_points:
        xy = xy[rng.choice(len(xy), max_points, replace=False)]
    best = (np.inf, None, 0.0)
    for _ in range(trials):
        tri = xy[rng.choice(len(xy), 3, replace=False)]
        a = 2 * (tri[1:] - tri[0])
        if abs(float(np.linalg.det(a))) < 1e-15:
            continue
        centre = np.linalg.solve(a, (tri[1:] ** 2).sum(axis=1) - (tri[0] ** 2).sum())
        radius = float(np.linalg.norm(tri[0] - centre))
        score = _median_residual(xy, centre, radius)
        if score < best[0]:
            best = (score, centre, radius)
    return best[1], best[2]


def fit_circle(xy: np.ndarray, *, iterations: int = 20, tukey_c: float = 4.685, trials: int = 200,
               seed: int = 0) -> Circle:
    """Algebraic (Kasa) circle refined by iteratively reweighted least squares on geometric residuals.

    The start is the better of the plain Kasa fit and a least-median-of-squares fit, and Tukey weights
    give gross outliers zero weight, so a minority of stray points does not move the circle.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        raise ValueError("circle fit needs at least 3 points")
    centre, radius = _kasa(xy, np.ones(len(xy)))
    if len(xy) > 3:
        robust_centre, robust_radius = _lmeds_circle(xy, trials, seed)
        if robust_centre is not None and (_median_residual(xy, robust_centre, robust_radius)
                                          < _median_residual(xy, centre, radius)):
            centre, radius = robust_centre, robust_radius
    for _ in range(iterations):
        resid = np.linalg.norm(xy - centre, axis=1) - radius
        scale = 1.4826 * float(np.median(np.abs(resid)))
        if scale < 1e-9:
            break
        w = np.clip(1.0 - (resid / (tukey_c * scale)) ** 2, 0.0, None) ** 2
        if np.count_nonzero(w) < 3:
            break
        new_centre, new_radius = _kasa(xy, np.sqrt(w))
        moved = float(np.linalg.norm(new_centre - centre)) + abs(new_radius - radius)
        centre, radius = new_centre, new_radius
        if moved < 1e-7:
            break
    resid = np.linalg.norm(xy - centre, axis=1) - radius
    scale = max(1.4826 * float(np.median(np.abs(resid))), 1e-6)
    inliers = np.abs(resid) <= 3 * scale
    rms = float(np.sqrt(np.mean(resid[inliers] ** 2))) if inliers.any() else float("nan")
    return Circle(centre, radius, rms, arc_coverage_deg(xy[inliers], centre), float(inliers.mean()))


# ---------------------------------------------------------------- 3-D lines


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid, unit principal direction and descending eigenvalues of a point set."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        raise ValueError("principal axis needs at least 2 points")
    centroid = pts.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((pts - centroid).T))
    order = np.argsort(values)[::-1]
    return centroid, vectors[:, order[0]], values[order]


def perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


@dataclass
class CylinderAxis:
    point: np.ndarray  # a point on the axis
    direction: np.ndarray  # unit
    radius: float
    rms: float
    arc_deg: float


def fit_cylinder_axis(points: np.ndarray, direction: np.ndarray, *, iterations: int = 3,
                      min_arc_deg: float = 90.0, min_points: int = 30) -> Optional[CylinderAxis]:
    """Axis of a cylinder seen from one side (a partial surface), starting from an approximate direction.

    Circles fitted to the cross-sections of the first and last thirds give two axis points; their
    difference is the next direction. None when the cross-sections are not circular arcs.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)

    def section(sel):
        if sel.sum() < min_points:
            return None
        e1, e2 = perpendicular_basis(d)
        p = pts[sel]
        circle = fit_circle(np.c_[p @ e1, p @ e2])
        if circle.arc_deg < min_arc_deg:
            return None
        return circle, e1, e2, float((p @ d).mean())

    for _ in range(iterations):
        t = pts @ d
        lo, hi = np.percentile(t, [2, 98])
        third = (hi - lo) / 3
        ends = [section((t >= lo) & (t < lo + third)), section((t > hi - third) & (t <= hi))]
        if any(e is None for e in ends):
            return None
        axis_pts = [c.center[0] * e1 + c.center[1] * e2 + s * d for c, e1, e2, s in ends]
        step = axis_pts[1] - axis_pts[0]
        if np.linalg.norm(step) < 1e-9:
            return None
        d = step / np.linalg.norm(step)
    whole = section(np.ones(len(pts), dtype=bool))
    if whole is None:
        return None
    circle, e1, e2, s = whole
    point = circle.center[0] * e1 + circle.center[1] * e2 + s * d
    return CylinderAxis(point, d, circle.radius, circle.rms, circle.arc_deg)


# ---------------------------------------------------------------- masks


def box_sum(mask: np.ndarray, radius: int, pad_value: int = 0) -> np.ndarray:
    """Count of set pixels in the (2r+1)^2 window around each pixel (integral image)."""
    a = np.asarray(mask).astype(np.int32)
    if radius <= 0:
        return a
    k = 2 * radius + 1
    c = np.pad(a, radius, constant_values=pad_value).cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return box_sum(mask, radius) > 0


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square erosion; the image border does not erode (as in OpenCV's default)."""
    return box_sum(mask, radius, pad_value=1) == (2 * radius + 1) ** 2


def ring_mask(mask: np.ndarray, inner_px: int, outer_px: int) -> np.ndarray:
    """Pixels between inner_px and outer_px (square distance) outside the mask."""
    return dilate(mask, outer_px) & ~dilate(mask, inner_px)


def _reach_runs(free: np.ndarray, reached: np.ndarray) -> np.ndarray:
    """Spread `reached` along each row's runs of free pixels."""
    h, w = free.shape
    flat = free.reshape(-1)
    start = flat.copy()
    start[1:] &= ~flat[:-1]
    start[::w] = flat[::w]
    ids = np.cumsum(start) * flat
    hit = np.zeros(int(ids.max()) + 1, dtype=bool)
    hit[ids[reached.reshape(-1) & flat]] = True
    hit[0] = False
    return hit[ids].reshape(h, w)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Mask with enclosed background filled: a pixel in a ring's hole belongs to the ring."""
    mask = np.asarray(mask, dtype=bool)
    free = ~mask
    reached = np.zeros_like(free)
    reached[0, :], reached[-1, :], reached[:, 0], reached[:, -1] = free[0, :], free[-1, :], free[:, 0], free[:, -1]
    while True:
        grown = _reach_runs(free, reached)
        grown = _reach_runs(free.T, grown.T).T
        if (grown == reached).all():
            break
        reached = grown
    return mask | (free & ~reached)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty mask")
    return float(cols.mean()), float(rows.mean())


def local_max(values: np.ndarray, radius: int) -> np.ndarray:
    """NaN-aware maximum over the (2r+1)^2 neighbourhood (separable); NaN where all are NaN."""
    a = np.where(np.isfinite(values), np.asarray(values, dtype=float), -np.inf)
    for axis in (0, 1):
        p = np.pad(a, [(radius, radius) if i == axis else (0, 0) for i in (0, 1)], constant_values=-np.inf)
        n = a.shape[axis]
        a = np.max([np.take(p, range(i, i + n), axis=axis) for i in range(2 * radius + 1)], axis=0)
    return np.where(np.isfinite(a), a, np.nan)


# ---------------------------------------------------------------- statistics


def finite_values(values) -> np.ndarray:
    a = np.asarray(values, dtype=float).reshape(-1)
    return a[np.isfinite(a)]


def robust_percentile(values, q):
    """Percentile(s) of the finite values; ValueError when there are none."""
    a = finite_values(values)
    if a.size == 0:
        raise ValueError("no finite values")
    out = np.percentile(a, q)
    return float(out) if np.ndim(out) == 0 else out


def iqr(values) -> float:
    q25, q75 = robust_percentile(values, [25, 75])
    return float(q75 - q25)


def mad(values) -> float:
    a = finite_values(values)
    if a.size == 0:
        raise ValueError("no finite values")
    return float(np.median(np.abs(a - np.median(a))))


def unit3(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n
