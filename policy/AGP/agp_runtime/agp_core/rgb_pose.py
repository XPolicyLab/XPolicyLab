"""Pose query (switch OURS_POSE_QUERY): an object's footprint seen from above, RGB only.

A measure query with `pose` (agp_core/syntax.py token; a query object's pose: true; agp_core/runner.py) is measured as before,
then the region of the query's camera image that has the colour of the picked pixel and is connected to it is
back-projected to the horizontal plane at the measured point's height (the public camera model and the pixel-centre
rays of agp_core/rgb_measure.py, in the query arm's base frame). From those points:
  xy        the region's centroid (m, base frame)
  axis_deg  the direction of its long axis, degrees from +x toward +y, in (-90, 90] (see edge_deg; pca_deg is the
            principal axis of the points alone)
  tail_deg  the same axis pointed toward the region's longer tail (sign of the third moment along it), in (-180, 180]
  length_mm / width_mm  the region's extent along the long and the short axis
  elong     sqrt of the ratio of the two principal variances (near 1: no clear long axis)
  edge_deg  the direction of the region's straight outline edges on the plane, modulo 90 deg (blurred-mask gradients
            mapped to the plane, averaged 4-fold), and `straight` how well they agree (0..1); when it is at least
            STRAIGHT_MIN the long axis is the edge direction (or the one at 90 deg to it) nearest the principal axis,
            which on swap_T's replay truth gave 0.40 deg RMS (max 0.73) against 0.96 (max 2.0) for the principal axis
  px        the pixel count
  status    'ok', or 'uncertain' with the reasons: the region touches the image border ('edge') or the robot's own
            silhouette ('robot'; it may be cut off or include the robot), is very small ('small') or very large
            ('large': it ran into the background)
  notes     'round' when the region has no clear long axis (elong below ROUND_ELONG): axis_deg and tail_deg are then
            arbitrary, while edge_deg (its straight outline edges, modulo 90) still holds
and a review image (the region outlined, the centroid and the axis drawn). The colour rule: a pixel belongs when its
chromaticity is within CHROMA_TOL of the seed colour's and its brightness within BRIGHT_RANGE of it, so the visible
sides of a uniformly coloured object (darker, same hue) join its top face. No object-specific constants; nothing is
read from the simulator. With the switch off this module is not imported.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agp_core import rgb_measure as rm

SEED_RADIUS_PX = 2               # the seed colour is the median of the (2r+1)^2 window around the pick, robot excluded
CHROMA_TOL = 0.06                # max distance of normalised rgb (r, g, b divided by their sum) to the seed's
BRIGHT_RANGE = (0.35, 1.6)       # brightness (r + g + b) relative to the seed's
MIN_SEED_BRIGHT = 30.0           # a darker seed has no usable chromaticity
MAX_ITERATIONS = 600             # region growing steps (one pixel ring each)
ROBOT_GAP_PX = 2                 # a region within this many pixels of the robot silhouette is 'robot'
MIN_PIXELS = 40
MAX_FRACTION = 0.15              # of the image
ROUND_ELONG = 1.08
STRAIGHT_MIN = 0.5              # 4-fold concentration of the outline's edge directions needed to use them
EDGE_SIGMA_PX = 2.0


def _grow(candidates, seed):
    """Pixels of `candidates` (HxW bool) 4-connected to `seed` (row, col)."""
    region = np.zeros_like(candidates)
    region[seed] = True
    for _ in range(MAX_ITERATIONS):
        grown = region.copy()
        grown[1:, :] |= region[:-1, :]
        grown[:-1, :] |= region[1:, :]
        grown[:, 1:] |= region[:, :-1]
        grown[:, :-1] |= region[:, 1:]
        grown &= candidates
        if grown.sum() == region.sum():
            return region, True
        region = grown
    return region, False


def _dilate(mask, steps):
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _wrap(deg, period):
    half = period / 2.0
    out = (deg + half) % period - half
    return half if out == -half else out


def footprint(capture, cam, arm, u, v, z):
    """The pose of the colour region at pixel (u, v) of `cam`, back-projected to height z in `arm`'s base frame."""
    img = capture.image(cam).astype(float)
    h, w = img.shape[:2]
    robot = capture.robot_mask(cam)
    if robot[v, u]:
        return {"status": "error", "why": ["the pixel shows the robot"]}
    v0, v1 = max(0, v - SEED_RADIUS_PX), min(h, v + SEED_RADIUS_PX + 1)
    u0, u1 = max(0, u - SEED_RADIUS_PX), min(w, u + SEED_RADIUS_PX + 1)
    window = img[v0:v1, u0:u1][~robot[v0:v1, u0:u1]]
    seed_rgb = np.median(window, axis=0)
    seed_bright = float(seed_rgb.sum())
    if seed_bright < MIN_SEED_BRIGHT:
        return {"status": "error", "why": ["the pixel is too dark to read its colour"]}
    bright = img.sum(-1)
    chroma = img / np.maximum(bright, 1e-6)[..., None]
    distance = np.linalg.norm(chroma - seed_rgb / seed_bright, axis=-1)
    ratio = bright / seed_bright
    candidates = (distance <= CHROMA_TOL) & (ratio >= BRIGHT_RANGE[0]) & (ratio <= BRIGHT_RANGE[1]) & ~robot
    candidates[v, u] = True
    region, settled = _grow(candidates, (v, u))
    vv, uu = np.nonzero(region)
    n = int(len(uu))
    why = []
    if not settled:
        why.append("large")
    if n < MIN_PIXELS:
        why.append("small")
    if n > MAX_FRACTION * h * w and "large" not in why:
        why.append("large")
    if vv.min() == 0 or uu.min() == 0 or vv.max() == h - 1 or uu.max() == w - 1:
        why.append("edge")
    if (_dilate(region, ROBOT_GAP_PX) & robot).any():
        why.append("robot")
    K, T = capture.K(cam), capture.T(cam, arm)
    rays = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones(n)], axis=1) @ T[:3, :3].T
    origin = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - origin[2]) / rays[:, 2]
    keep = np.isfinite(s) & (s > 0)
    if keep.sum() < 3:
        return {"status": "error", "why": ["the region's rays do not reach that height"], "px": n}
    P = origin[None, :2] + s[keep, None] * rays[keep, :2]
    c = P.mean(0)
    cov = np.cov((P - c).T)
    evals, evecs = np.linalg.eigh(cov)
    major, minor = evecs[:, 1], evecs[:, 0]
    elong = math.sqrt(max(evals[1], 0.0) / max(evals[0], 1e-18))
    notes = ["round"] if elong < ROUND_ELONG else []      # no long axis: axis_deg / tail_deg are arbitrary
    pca_deg = _wrap(math.degrees(math.atan2(major[1], major[0])), 180.0)
    # Refine the axis with the region's outline: the side of its minimum-area bounding rectangle nearest the
    # principal axis (straight outer edges fix a heading far better than the area moments, which visible side
    # faces and shading skew).
    edge_deg, straightness = _edge_axis_deg(region, K, T, z)
    if edge_deg is None or straightness < STRAIGHT_MIN:
        axis_deg = pca_deg
    else:
        axis_deg = min((edge_deg, _wrap(edge_deg + 90.0, 180.0)), key=lambda a: abs(_wrap(a - pca_deg, 180.0)))
    major = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg))])
    minor = np.array([-major[1], major[0]])
    along, across = (P - c) @ major, (P - c) @ minor
    third = float(np.mean(along ** 3)) / max(float(np.mean(along ** 2)) ** 1.5, 1e-18)
    tail = major if third >= 0 else -major
    tail_deg = _wrap(math.degrees(math.atan2(tail[1], tail[0])), 360.0)
    result = {"status": "uncertain" if why else "ok", "why": why, "notes": notes,
              "xy": [float(c[0]), float(c[1])], "z": float(z),
              "axis_deg": float(axis_deg), "tail_deg": float(tail_deg), "skew": third, "elong": elong,
              "pca_deg": float(pca_deg), "edge_deg": None if edge_deg is None else float(edge_deg),
              "straight": float(straightness),
              "length_mm": float(along.max() - along.min()) * 1000.0,
              "width_mm": float(across.max() - across.min()) * 1000.0, "px": n,
              "seed_rgb": [float(x) for x in seed_rgb], "frame": f"{arm}_base", "cam": cam, "pixel": [u, v]}
    result["image"] = _review(capture, cam, arm, region, robot, c, tail, along if third >= 0 else -along, z, u, v, result)
    return result


def _edge_axis_deg(region, K, T, z):
    """Dominant direction of the region's outline edges on the plane, modulo 90 deg, and its concentration (0..1).

    The region mask is blurred (EDGE_SIGMA_PX) so a straight edge has one gradient direction along its staircase;
    each outline pixel's tangent is mapped to the plane with the local pixel -> plane Jacobian of the back-projection,
    and the tangent angles are averaged 4-fold (exp(4i angle), weighted by the gradient), which is the orientation
    of a set of edges meeting at right angles; the length of that mean is how well they agree."""
    from PIL import ImageFilter
    blurred = np.asarray(Image.fromarray((region * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(EDGE_SIGMA_PX)), float) / 255.0
    gy, gx = np.gradient(blurred)
    mag = np.hypot(gx, gy)
    band = (mag > 0.05) & _dilate(region, 2) & _dilate(~region, 2)
    vv, uu = np.nonzero(band)
    if len(uu) < 10:
        return None, 0.0

    def plane(cu, cv):
        rays = np.stack([(cu - K[0, 2]) / K[0, 0], (cv - K[1, 2]) / K[1, 1], np.ones(len(cu))], axis=1) @ T[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - T[2, 3]) / rays[:, 2]
        return T[None, :2, 3] + s[:, None] * rays[:, :2]
    cu, cv = uu + 0.5, vv + 0.5
    base = plane(cu, cv)
    du = plane(cu + 1.0, cv) - base
    dv = plane(cu, cv + 1.0) - base
    tx, ty = -gy[vv, uu], gx[vv, uu]                       # tangent in pixels (perpendicular to the gradient)
    tangent = du * tx[:, None] + dv * ty[:, None]
    weight = mag[vv, uu]
    ok = np.isfinite(tangent).all(1)
    if ok.sum() < 10:
        return None, 0.0
    angle = np.arctan2(tangent[ok, 1], tangent[ok, 0])
    z4 = np.sum(weight[ok] * np.exp(4j * angle)) / np.sum(weight[ok])
    return _wrap(math.degrees(np.angle(z4) / 4.0), 180.0), float(abs(z4))


def _outline_points(region, K, T, z):
    """Plane points (base frame xy) of the corners of the region's outer pixels."""
    edge = region & _dilate(~region, 1)
    vv, uu = np.nonzero(edge)
    if len(uu) < 3:
        return None
    cu = np.concatenate([uu, uu + 1, uu, uu + 1]).astype(float)
    cv = np.concatenate([vv, vv, vv + 1, vv + 1]).astype(float)
    rays = np.stack([(cu - K[0, 2]) / K[0, 0], (cv - K[1, 2]) / K[1, 1], np.ones(len(cu))], axis=1) @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - T[2, 3]) / rays[:, 2]
    keep = np.isfinite(s) & (s > 0)
    if keep.sum() < 3:
        return None
    return T[None, :2, 3] + s[keep, None] * rays[keep, :2]


def _min_area_rect_deg(points):
    """Direction (deg, in (-90, 90]) of one side of the minimum-area rectangle around `points` (N, 2)."""
    hull = rm.convex_hull_2d(points)
    if len(hull) < 3:
        return None
    best = None
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        d = b - a
        if not np.any(d):
            continue
        angle = math.atan2(d[1], d[0])
        c, s = math.cos(angle), math.sin(angle)
        x = hull @ np.array([c, s])
        y = hull @ np.array([-s, c])
        area = (x.max() - x.min()) * (y.max() - y.min())
        if best is None or area < best[0]:
            best = (area, angle)
    return None if best is None else _wrap(math.degrees(best[1]), 180.0)


def _review(capture, cam, arm, region, robot, centre, tail, along, z, u, v, result):
    """Crop around the region, enlarged: region outline green, robot red tint, centroid and tail axis yellow."""
    img = capture.image(cam)
    h, w = region.shape
    vv, uu = np.nonzero(region)
    pad = 16
    r0, r1 = max(0, vv.min() - pad), min(h, vv.max() + pad + 1)
    c0, c1 = max(0, uu.min() - pad), min(w, uu.max() + pad + 1)
    scale = max(1, min(6, int(480 / max(r1 - r0, c1 - c0))))
    crop = img[r0:r1, c0:c1].copy()
    edge = region & _dilate(~region, 1)
    crop[edge[r0:r1, c0:c1]] = (0, 255, 0)
    tint = robot[r0:r1, c0:c1]
    crop[tint] = (crop[tint] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    pil = Image.fromarray(crop).resize(((c1 - c0) * scale, (r1 - r0) * scale), Image.NEAREST)
    draw = ImageDraw.Draw(pil)
    K, T = capture.K(cam), capture.T(cam, arm)
    ends = np.array([[*(centre + tail * along.min()), z], [*(centre + tail * along.max()), z], [*centre, z]])
    if np.isfinite(ends).all():                  # `along` is measured along `tail`: the circle marks the tail's end
        uv, _ = rm.project(K, T, ends)
        pts = [((p[0] - c0) * scale, (p[1] - r0) * scale) for p in uv]
        a, b = pts[0], pts[1]
        draw.line([a, b], fill=(255, 255, 0), width=2)
        draw.ellipse([b[0] - 4, b[1] - 4, b[0] + 4, b[1] + 4], outline=(255, 255, 0), width=2)
        cx, cy = pts[2]
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 0))
    sx, sy = (u + 0.5 - c0) * scale, (v + 0.5 - r0) * scale
    draw.rectangle([sx - 3, sy - 3, sx + 3, sy + 3], outline=(255, 0, 255))
    title = (f"pose {cam} {u},{v}: axis {result['axis_deg']:+.1f} tail {result['tail_deg']:+.1f} deg, "
             f"{result['length_mm']:.0f}x{result['width_mm']:.0f} mm, {result['status']}"
             + (f" ({', '.join(result['why'])})" if result["why"] else ""))
    canvas = Image.new("RGB", (max(pil.width, 6 * len(title) + 4), pil.height + 16), (0, 0, 0))
    canvas.paste(pil, (0, 16))
    ImageDraw.Draw(canvas).text((2, 2), title, fill=(255, 255, 255))
    path = Path(capture.directory) / f"pose_{cam}_{u}_{v}_{arm}.png"
    canvas.save(path)
    return {"label": f"pose {cam} {u},{v}", "path": str(path)}


def pose(session, capture_id, query, z):
    """The footprint for one kept measurement (query: arm, cam, u, v; z: its measured height)."""
    capture = rm.load_capture(session, capture_id)
    return footprint(capture, query["cam"], query["arm"], int(query["u"]), int(query["v"]), float(z))
