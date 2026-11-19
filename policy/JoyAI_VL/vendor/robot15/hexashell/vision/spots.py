"""World points from picture points (``robot.to_world``).

A pixel fixes a ray, not a point; the depth along it has to come from somewhere:

* ``z=``: a horizontal plane at that height (the top of a known object, a rim, the table);
* ``pair=(cam2, (x2, y2))``: the same point picked in a second picture - the two rays are
  intersected, and how far apart they pass says whether the two picks really match;
* neither: the surface under the pixel - the object there when perception can outline it
  (:meth:`Scene.surface_spot`), otherwise the table top.  The result names its source.

Every answer carries how far one pixel of picking error moves the point, so the model can judge
how carefully it has to pick.  A pixel that shows the robot's own arm or gripper in the picture it
was picked from (the latest one of that camera the model has seen) is flagged: what the model meant
lies hidden behind the robot there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..errors import Refused
from ..robot.results import Spot
from ..world import TABLE_Z
from .cameras import Camera, cameras_for, check_cam, triangulate
from .robotmask import robot_mask

#: Rays of a matched pair should pass closer than this (mm); beyond it the picks likely differ.
PAIR_GAP_MM = 5.0
#: Below this angle between the two rays (degrees) depth is poorly determined.
PAIR_MIN_ANGLE_DEG = 8.0
#: A ray within this many degrees of a horizontal plane only grazes it.
GRAZING_DEG = 12.0
#: Table area (with a margin) - points outside are flagged.
TABLE_X = (-0.75, 0.75)
TABLE_Y = (-0.45, 0.55)
#: A picked pixel shows the robot when it, or more than this share of the 5 x 5 patch around it, lies
#: in the robot's outline (drawn without a margin: on the pictures it fits within about a pixel).
ON_ROBOT_PATCH = 0.5


def pixel_of(value, camera: Camera) -> tuple[float, float]:
    try:
        xy = np.asarray(value, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused("a pixel is (x, y): column, row from the top-left corner") from None
    if xy.size != 2 or not np.all(np.isfinite(xy)):
        raise Refused("a pixel is (x, y): column, row from the top-left corner")
    x, y = float(xy[0]), float(xy[1])
    if not (-0.5 <= x <= camera.width - 0.5 and -0.5 <= y <= camera.height - 0.5):
        raise Refused(f"pixel ({x:.0f}, {y:.0f}) is outside the {camera.name} picture "
                          f"(x 0..{camera.width - 1}, y 0..{camera.height - 1})")
    return x, y


def mm_per_pixel_on_plane(camera: Camera, point: np.ndarray, direction: np.ndarray) -> float:
    """How far the point on a horizontal plane moves for one pixel of error (worst direction)."""
    dist = float(np.linalg.norm(point - camera.position))
    sin_elev = max(abs(float(direction[2])), 1e-3)
    return 1000.0 * dist / camera.focal / sin_elev


def on_plane(camera: Camera, xy: tuple[float, float], z: float, source: str, *, table_z: float = TABLE_Z,
             area: tuple | None = None) -> Spot:
    """Where the pixel's ray meets the plane at height z (``table_z`` and ``area``, the table area with
    a margin as (x0, x1), (y0, y1): the table's, where it is not RoboDojo's)."""
    origin, d = camera.ray(xy)
    if abs(d[2]) < 1e-6:
        return Spot(None, camera.name, xy, source, warning="the ray runs parallel to that plane")
    t = (z - origin[2]) / d[2]
    if t <= 0:
        side = "above" if z > origin[2] else "below"
        return Spot(None, camera.name, xy, source,
                    warning=f"the plane z = {z:.3f} is {side} the camera and not in front of it along this ray")
    p = origin + t * d
    warnings = []
    elev = math.degrees(math.asin(min(1.0, abs(float(d[2])))))
    if elev < GRAZING_DEG:
        warnings.append(f"the ray grazes the plane ({elev:.0f}°): small picking errors move the point a lot")
    if z < table_z - 0.005:
        warnings.append(f"z = {z:.3f} is below the table top ({table_z:.3f})")
    (x0, x1), (y0, y1) = area or (TABLE_X, TABLE_Y)
    if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
        warnings.append("the point lies outside the table area")
    return Spot(tuple(round(float(v), 4) for v in p), camera.name, xy, source,
                warning="; ".join(warnings) or None, mm_per_px=round(mm_per_pixel_on_plane(camera, p, d), 1))


def from_pair(cams: dict[str, Camera], cam: str, xy: tuple[float, float], pair) -> Spot:
    try:
        cam2, xy2 = pair
    except (TypeError, ValueError):
        raise Refused("pair is (second camera, (x, y)), e.g. pair=('left_wrist', (310, 222))") from None
    cam2 = check_cam(cam2)
    if cam2 == cam:
        raise Refused("pair needs the point in a different camera")
    first, second = cams[cam], cams[cam2]
    xy2 = pixel_of(xy2, second)
    o1, d1 = first.ray(xy)
    o2, d2 = second.ray(xy2)
    p, gap, angle = triangulate(o1, d1, o2, d2)
    source = f"two views ({cam} + {cam2})"
    if (p - o1) @ d1 <= 0 or (p - o2) @ d2 <= 0:
        return Spot(None, cam, xy, source, gap_mm=round(1000 * gap, 1), angle_deg=round(angle, 1),
                    warning="the rays meet behind a camera: the two picks are not the same point")
    # depth spread for one pixel of error in either picture
    r1 = float(np.linalg.norm(p - o1)) / first.focal
    r2 = float(np.linalg.norm(p - o2)) / second.focal
    per_px = 1000.0 * max(r1, r2) / max(math.sin(math.radians(angle)), 1e-3)
    warnings = []
    if 1000 * gap > PAIR_GAP_MM:
        warnings.append(f"the rays pass {1000 * gap:.0f} mm apart: the two pixels probably do not show the same "
                        "point")
    if angle < PAIR_MIN_ANGLE_DEG:
        warnings.append(f"the views differ by only {angle:.0f}°: the depth is uncertain")
    if p[2] < TABLE_Z - 0.01:
        warnings.append(f"the point is {100 * (TABLE_Z - p[2]):.0f} cm below the table top: the two pixels probably "
                        "do not show the same point")
    return Spot(tuple(round(float(v), 4) for v in p), cam, xy, source, gap_mm=round(1000 * gap, 1),
                angle_deg=round(angle, 1), warning="; ".join(warnings) or None, mm_per_px=round(per_px, 1))


def picked_from(host, cam: str):
    """The snapshot of the picture a pixel of ``cam`` was picked from: the latest one of that camera
    the model has seen (``host.shown``, when the host keeps it), else the current one."""
    shown = getattr(host, "shown", None) or {}
    return shown.get(cam) or host.snap


def on_robot(host, cam: str, xy) -> bool:
    """Whether the pixel shows the robot's own arm or gripper in the picture it was picked from."""
    snap = picked_from(host, cam)
    mask = robot_mask(cameras_for(snap)[cam], snap.arms_in_pictures, margin_px=0)
    h, w = mask.shape
    x, y = min(max(int(round(xy[0])), 0), w - 1), min(max(int(round(xy[1])), 0), h - 1)
    patch = mask[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    return bool(mask[y, x]) or float(patch.mean()) > ON_ROBOT_PATCH


def robot_note(host, picks) -> str | None:
    """A warning when picked pixels ((cam, xy) pairs) show the robot itself."""
    hit = [cam for cam, xy in picks if on_robot(host, cam, xy)]
    if not hit:
        return None
    where = f"the {hit[0]} picture you picked it from" if len(hit) == 1 else "the pictures you picked them from"
    what = "the pixel is" if len(hit) == 1 else "both pixels are"
    return f"{what} on the robot's own arm or gripper in {where}: move the arm aside, look and pick again"


def to_world(host, cam: str, xy, *, z=None, pair=None) -> Spot:
    cam = check_cam(cam)
    cams = cameras_for(host.snap)
    camera = cams[cam]
    xy = pixel_of(xy, camera)
    picks = [(cam, xy)]
    if z is not None and pair is not None:
        raise Refused("give z= or pair=, not both")
    if z is not None:
        try:
            z = float(z)
        except (TypeError, ValueError):
            raise Refused("z is a height in metres (the table top is 0.765)") from None
        if not math.isfinite(z):
            raise Refused("z is a height in metres (the table top is 0.765)")
        spot = on_plane(camera, xy, z, f"plane z = {z:.3f}")
    elif pair is not None:
        spot = from_pair(cams, cam, xy, pair)
        cam2 = check_cam(pair[0])
        picks.append((cam2, pixel_of(pair[1], cams[cam2])))
    elif host.scene is not None:
        spot = host.scene.surface_spot(cam, xy)
    else:
        spot = on_plane(camera, xy, TABLE_Z, "the table top (no perception in this run)")
        note = "assumes the pixel shows the table top; for a point on an object give z= or pair="
        spot.warning = note if spot.warning is None else f"{note}; {spot.warning}"
    note = robot_note(host, picks)
    if note is not None:
        spot.warning = note if spot.warning is None else f"{note}; {spot.warning}"
    return spot


# Estimate surface height by matching RGB patches.
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
