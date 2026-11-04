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


def on_plane(camera: Camera, xy: tuple[float, float], z: float, source: str) -> Spot:
    """Where the pixel's ray meets the plane at height z."""
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
    if z < TABLE_Z - 0.005:
        warnings.append(f"z = {z:.3f} is below the table top ({TABLE_Z:.3f})")
    if not (TABLE_X[0] <= p[0] <= TABLE_X[1] and TABLE_Y[0] <= p[1] <= TABLE_Y[1]):
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
