"""Which pixels of a camera picture show the robot itself.

The two X5 arms are drawn from their meshes (``assets/robot_model.npz``, built by
``bench/build_robot_points.py`` from RoboDojo's STL/glTF files): base_link and links 1-5 are
placed by forward kinematics of the joint readings, link6 with the fingers and the wrist camera by
the reported link6 pose (``flange``; it is what places the wrist cameras too).

Drawing: the triangles of a closed surface that face the camera cover exactly its outline, and
on the picture they all turn the same way.  So a part is filled from the triangle sides between
a triangle that faces the camera and one that does not (its outline and folds): a pixel is
inside when those sides wind around its centre.  The camera housing, whose mesh is not closed,
is drawn as the convex hull of its points.  Triangles reaching behind the near plane are cut
there; a wrist camera leaves out what of its own arm is nearer than ``OWN_NEAR_M``.  The mask
is then widened by a margin of a few pixels.

The fingers sit at ``openings`` when it gives a value for the arm (the 0..1 gripper scale).
Otherwise they are drawn swept from the gripper reading to ``GRIP_SLACK`` further open: RoboDojo
reports the command it applied, which it keeps within that much of the fingers' real position, so
fingers stopped by what they squeeze (or by each other) are up to that much further open.  A
caller that knows the gripper was told to close can pass ``reading + GRIP_SLACK`` (capped at 1):
that is where squeezing fingers are, and the sweep would also cover that much of what they hold.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..geometry import exp_so3, invert, log_so3
from ..motion.chain import arm_chain
from ..state import ArmReading
from ..world import ARMS
from .cameras import Camera

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "robot_model.npz"

#: Widening of the mask (pixels) when the caller gives none.
DEFAULT_MARGIN_PX = {"head": 4, "left_wrist": 7, "right_wrist": 7}
#: Triangles are cut at this depth (m) in front of the camera.
NEAR_M = 0.005
#: A wrist camera leaves out its own arm nearer than this (m): it sits inside that part.
OWN_NEAR_M = 0.03
#: Finger joint (m) at gripper reading 0 and 1, and the joint's limits (0: pads touching).
FINGER_TRAVEL = (-0.01, 0.044)
FINGER_LIMIT = (0.0, 0.044)
#: The fingers may be this much further open than the reading (0..1 scale; RoboDojo's limit).
GRIP_SLACK = 0.2

#: Label groups, in label order.
GROUPS = {
    "base": ("base_link", "link1"),
    "upper_arm": ("link2",),
    "forearm": ("link3",),
    "wrist": ("link4", "link5"),
    "hand": ("link6", "camera_base", "camera"),
    "fingers": ("link7", "link8"),
}
GROUP_NAMES = tuple(GROUPS)
FINGERS = GROUP_NAMES.index("fingers")
#: Label image values: 0 nothing, then one per (arm, group).
LABELS = {1 + a * len(GROUPS) + g: f"{arm} {name}" for a, arm in enumerate(ARMS) for g, name in enumerate(GROUPS)}
#: Parts a wrist camera never sees of its own arm (it sits on them).
OWN_HIDDEN = ("camera_base", "camera")
#: When FK of the joints and the reported link6 pose differ by up to this much (m, rad), links
#: 1-5 are moved part of the way (BLEND_WEIGHTS, base to wrist) to agree with link6.
BLEND_LIMIT = (0.05, 0.35)
BLEND_WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)
#: Outline sides filled per batch (keeps work arrays small; big fresh arrays are slow to touch).
_BATCH = 6000


@dataclass(frozen=True)
class Part:
    name: str
    link: str
    group: int
    hull: bool  # drawn as the convex hull of its points
    verts: np.ndarray  # (3, V) float32, in the link frame
    center: np.ndarray  # bounding sphere
    radius: float
    faces: np.ndarray  # (F, 3)
    corners: tuple  # three (F,) columns of faces
    normals: np.ndarray  # (F, 3) float32, not unit length
    offsets: np.ndarray  # (F,) normal . first corner
    edges: tuple  # (start, end, face, other): each side once, as it runs in face; other is F if none


@dataclass(frozen=True)
class RobotModel:
    parts: tuple[Part, ...]
    finger_origins: dict[str, np.ndarray]  # link -> 4x4 joint origin in link6
    finger_axes: dict[str, np.ndarray]


def _edges(faces: np.ndarray, twins: np.ndarray) -> tuple:
    nf = len(faces)
    face = np.repeat(np.arange(nf), 3)
    k = np.tile(np.arange(3), nf)
    other = twins.reshape(-1)
    once = (other < 0) | (face < other)
    start, end = faces[face, k], faces[face, (k + 1) % 3]
    other = np.where(other < 0, nf, other)
    return tuple(np.ascontiguousarray(a[once]) for a in (start, end, face, other))


@lru_cache(maxsize=2)
def load_model(path: str | None = None) -> RobotModel:
    data = np.load(path or MODEL_PATH, allow_pickle=False)
    group_of = {p: i for i, parts in enumerate(GROUPS.values()) for p in parts}
    verts = data["vertices"].astype(np.float64)
    faces = data["faces"].astype(np.int64)
    twins = data["twins"].astype(np.int64)
    parts = []
    for i, name in enumerate(data["part_names"]):
        name = str(name)
        if name not in group_of:
            continue
        v0, nv = int(data["vertex_start"][i]), int(data["vertex_count"][i])
        f0, nf = int(data["face_start"][i]), int(data["face_count"][i])
        v = verts[v0:v0 + nv]
        f = np.ascontiguousarray(faces[f0:f0 + nf])
        center = (v.min(axis=0) + v.max(axis=0)) / 2
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        normals = np.cross(b - a, c - a)
        parts.append(Part(
            name=name, link=str(data["part_links"][i]), group=group_of[name],
            hull=bool(data["part_kinds"][i] == 1), verts=np.ascontiguousarray(v.T, dtype=np.float32),
            center=center, radius=float(np.sqrt(((v - center) ** 2).sum(axis=1).max())) + 1e-4,
            faces=f, corners=tuple(np.ascontiguousarray(f[:, k]) for k in range(3)),
            normals=np.ascontiguousarray(normals, dtype=np.float32),
            offsets=np.einsum("ij,ij->i", normals, a).astype(np.float32),
            edges=_edges(f, twins[f0:f0 + nf]),
        ))
    links = [str(n) for n in data["finger_links"]]
    return RobotModel(
        parts=tuple(parts),
        finger_origins={n: data["finger_origins"][k] for k, n in enumerate(links)},
        finger_axes={n: data["finger_axes"][k] for k, n in enumerate(links)},
    )


# -- where the parts are -----------------------------------------------------------------------------
def finger_travel(opening: float) -> float:
    """Finger joint position (m) for a gripper opening on the 0..1 scale."""
    lo, hi = FINGER_TRAVEL
    return float(np.clip(lo + (hi - lo) * float(opening), *FINGER_LIMIT))


def link_poses(arm: str, reading: ArmReading, opening: float | None = None) -> dict[str, np.ndarray]:
    """World pose of every link of one arm."""
    chain = arm_chain(arm)
    joints = np.asarray(reading.joints, float).reshape(-1)[:6]
    flange = np.asarray(reading.flange, float)
    grip = float(reading.gripper if opening is None else opening)
    if not (np.isfinite(joints).all() and np.isfinite(flange).all() and np.isfinite(grip)):
        raise ValueError(f"{arm} arm reading is not finite")
    frames = chain.frames(joints)
    poses = {"base_link": chain.base}
    fix = flange @ invert(frames[-1])
    turn = log_so3(fix[:3, :3])
    shift, angle = float(np.linalg.norm(fix[:3, 3])), float(np.linalg.norm(turn))
    blend = (shift > 1e-5 or angle > 1e-5) and shift <= BLEND_LIMIT[0] and angle <= BLEND_LIMIT[1]
    for k, frame in enumerate(frames[:5]):
        if blend:
            # turn the link about its own origin and move that origin, both by a share of fix
            w = BLEND_WEIGHTS[k]
            origin = frame[:3, 3]
            moved = fix[:3, :3] @ origin + fix[:3, 3]
            part = np.eye(4)
            part[:3, :3] = exp_so3(w * turn)
            part[:3, 3] = origin + w * (moved - origin) - part[:3, :3] @ origin
            frame = part @ frame
        poses[f"link{k + 1}"] = frame
    poses["link6"] = flange
    travel = finger_travel(grip)
    model = load_model()
    for link, origin in model.finger_origins.items():
        slide = np.eye(4)
        slide[:3, 3] = model.finger_axes[link] * travel
        poses[link] = flange @ origin @ slide
    return poses


def finger_sweep(arm: str, reading: ArmReading) -> dict[str, np.ndarray]:
    """World displacement of each finger from the reading to GRIP_SLACK further open (none if the
    fingers cannot move that way)."""
    grip = float(reading.gripper)
    reach = finger_travel(min(1.0, grip + GRIP_SLACK)) - finger_travel(grip)
    if not reach > 1e-6:
        return {}
    model = load_model()
    flange = np.asarray(reading.flange, float)
    return {link: (flange @ origin)[:3, :3] @ model.finger_axes[link] * reach
            for link, origin in model.finger_origins.items()}


_recent_poses: dict = {}


def _poses(arm: str, reading: ArmReading, opening: float | None) -> tuple[dict, dict]:
    """link_poses and the finger sweep (without an opening), remembered for the last few readings
    (one call per camera, same arms)."""
    key = (arm, np.asarray(reading.joints, float).tobytes(), np.asarray(reading.flange, float).tobytes(),
           float(reading.gripper), None if opening is None else float(opening))
    hit = _recent_poses.get(key)
    if hit is None:
        hit = (link_poses(arm, reading, opening), finger_sweep(arm, reading) if opening is None else {})
        if len(_recent_poses) >= 16:
            _recent_poses.clear()
        _recent_poses[key] = hit
    return hit


# -- filling -------------------------------------------------------------------------------------------
_local = threading.local()


def _buffers(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Two int16 work arrays of ``size``, kept per thread."""
    held = getattr(_local, "held", None)
    if held is None or len(held[0]) < size:
        held = _local.held = (np.zeros(size, np.int16), np.zeros(size, np.int16))
    return held[0][:size], held[1][:size]


def _crossings(sides, x0: int, y0: int, width: int, height: int, diff: np.ndarray) -> None:
    """Add to ``diff`` ((height, width + 1) flat, region at column x0, row y0) +1 or -1 where a
    side (xa, ya, xb, yb), going down or up the picture, crosses a row of pixel centres: at the
    first pixel whose centre is at or right of the crossing."""
    xa, ya, xb, yb = sides
    r0 = np.ceil(np.minimum(ya, yb) - y0)
    r1 = np.ceil(np.maximum(ya, yb) - y0) - 1
    np.maximum(r0, 0, out=r0)
    np.minimum(r1, height - 1, out=r1)
    keep = np.flatnonzero(r1 >= r0)
    if len(keep) == 0:
        return
    xa, ya, xb, yb = xa[keep], ya[keep], xb[keep], yb[keep]
    r0 = r0[keep].astype(np.intp)
    n = r1[keep].astype(np.intp) - r0 + 1
    t = np.repeat(np.arange(len(n)), n)
    row = np.arange(len(t)) - np.repeat(np.cumsum(n) - n - r0, n)
    # from the side's middle, so that a side and its reverse cross each row at the same x
    xm = (xa + xb) * 0.5 - x0
    ym = (ya + yb) * 0.5 - y0
    slope = (xb - xa) / (yb - ya)
    x = xm[t] + (row.astype(xm.dtype) - ym[t]) * slope[t]
    np.ceil(x, out=x)
    np.clip(x, 0, width, out=x)
    at = row * (width + 1) + x.astype(np.intp)
    turns = (yb > ya).astype(np.int16) * 2 - 1
    np.add.at(diff, at, turns[t])


def _fill(diff: np.ndarray, sums: np.ndarray, out: np.ndarray) -> None:
    """``out`` (h, w) = True where the running sum of ``diff`` along each row is not zero."""
    height, width = out.shape
    sums = sums.reshape(height, width + 1)
    np.cumsum(diff.reshape(height, width + 1), axis=1, dtype=np.int16, out=sums)
    np.not_equal(sums[:, :width], 0, out=out)


def _wind(sides, x0: int, y0: int, out: np.ndarray) -> None:
    """Fill ``out``, a bool (h, w) region of the picture starting at column x0, row y0: True
    where the closed chains of sides (xa, ya, xb, yb) wind around the pixel centre."""
    height, width = out.shape
    diff, sums = _buffers(height * (width + 1))
    diff[:] = 0
    for k in range(0, len(sides[0]), _BATCH):
        _crossings(tuple(s[k:k + _BATCH] for s in sides), x0, y0, width, height, diff)
    _fill(diff, sums, out)


def _spans(x0, y0, x1, y1, x2, y2, width: int, height: int):
    """Row spans (row, first col, last col) of triangles given by corner columns, in pixels.
    A pixel (centre at integer coordinates) counts when the triangle meets its square."""
    # sort the corners by y
    s = y1 < y0
    x0, y0, x1, y1 = np.where(s, x1, x0), np.where(s, y1, y0), np.where(s, x0, x1), np.where(s, y0, y1)
    s = y2 < y1
    x1, y1, x2, y2 = np.where(s, x2, x1), np.where(s, y2, y1), np.where(s, x1, x2), np.where(s, y1, y2)
    s = y1 < y0
    x0, y0, x1, y1 = np.where(s, x1, x0), np.where(s, y1, y0), np.where(s, x0, x1), np.where(s, y0, y1)
    r0 = np.maximum(np.ceil(y0 - 0.5), 0)
    r1 = np.minimum(np.floor(y2 + 0.5), height - 1)
    xmin = np.minimum(np.minimum(x0, x1), x2)
    xmax = np.maximum(np.maximum(x0, x1), x2)
    ok = (r1 >= r0) & (xmax >= -0.5) & (xmin <= width - 0.5)
    if not ok.any():
        e = np.zeros(0, np.int64)
        return e, e, e
    x0, y0, x1, y1, x2, y2 = x0[ok], y0[ok], x1[ok], y1[ok], x2[ok], y2[ok]
    r0 = r0[ok].astype(np.int64)
    n = r1[ok].astype(np.int64) - r0 + 1
    h = y2 - y0
    s02 = (x2 - x0) / np.where(h > 0, h, 1.0)
    h = y1 - y0
    s01 = np.where(h > 0, (x1 - x0) / np.where(h > 0, h, 1.0), 0.0)
    h = y2 - y1
    s12 = np.where(h > 0, (x2 - x1) / np.where(h > 0, h, 1.0), 0.0)
    t = np.repeat(np.arange(len(n)), n)
    row = np.arange(len(t)) - np.repeat(np.cumsum(n) - n - r0, n)
    # the part of the triangle between row - 0.5 and row + 0.5: its x range
    Y0, Y1, X0, X1 = y0[t], y1[t], x0[t], x1[t]
    a = np.maximum(row - 0.5, Y0)
    b = np.minimum(row + 0.5, y2[t])
    S = s02[t]
    la = X0 + (a - Y0) * S
    lb = X0 + (b - Y0) * S
    S01, S12 = s01[t], s12[t]
    sa = np.where(a <= Y1, X0 + (a - Y0) * S01, X1 + (a - Y1) * S12)
    sb = np.where(b <= Y1, X0 + (b - Y0) * S01, X1 + (b - Y1) * S12)
    lo = np.minimum(np.minimum(la, lb), np.minimum(sa, sb))
    hi = np.maximum(np.maximum(la, lb), np.maximum(sa, sb))
    mid = (Y1 >= a) & (Y1 <= b)
    lo = np.maximum(np.ceil(np.where(mid, np.minimum(lo, X1), lo) - 0.5), 0)
    hi = np.minimum(np.floor(np.where(mid, np.maximum(hi, X1), hi) + 0.5), width - 1)
    keep = lo <= hi
    return row[keep], lo[keep].astype(np.int64), hi[keep].astype(np.int64)


def raster_triangles(tri_xy, width: int, height: int) -> np.ndarray:
    """Bool (height, width): the pixels whose square touches any of the 2-D triangles (N, 3, 2)
    given as (x, y).  Slower than the outline fill but takes any triangles."""
    tri = np.asarray(tri_xy, float).reshape(-1, 3, 2)
    rows, first, last = _spans(*[np.ascontiguousarray(tri[:, k, j]) for k in range(3) for j in range(2)],
                               width, height)
    if len(rows) == 0:
        return np.zeros((height, width), bool)
    base = rows * (width + 1)
    size = height * (width + 1)
    diff = np.bincount(base + first, minlength=size) - np.bincount(base + last + 1, minlength=size)
    return np.cumsum(diff.reshape(height, width + 1), axis=1)[:, :width] > 0


# -- drawing the arms ----------------------------------------------------------------------------------
@dataclass
class _Drawn:
    arm: int
    group: int
    depth: float  # depth of the part's centre (m)
    box: tuple  # (x0, y0, x1, y1) on the picture, ends excluded
    sides: tuple | None = None  # outline sides (xa, ya, xb, yb), pixels
    hull: np.ndarray | None = None  # (K, 2) polygon, pixels


def _box(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> tuple | None:
    """Pixels (x0, y0, x1, y1), ends excluded, holding the points xs, ys; None if off the picture."""
    x0, y0 = max(int(np.floor(xs.min())), 0), max(int(np.floor(ys.min())), 0)
    x1, y1 = min(int(np.ceil(xs.max())) + 1, width), min(int(np.ceil(ys.max())) + 1, height)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _union(boxes) -> tuple | None:
    if not boxes:
        return None
    b = np.array(boxes)
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def _clip_near(tri: np.ndarray, near: float) -> np.ndarray:
    """Triangles (N, 3, 3) in camera coordinates, cut at depth ``near`` (depth = -z); the
    pieces keep the turning order of the triangle they come from."""
    inside = -tri[:, :, 2] > near
    count = inside[:, 0].astype(int) + inside[:, 1] + inside[:, 2]
    out = [tri[count == 3]]
    for n_in in (1, 2):
        sel = count == n_in
        if not np.any(sel):
            continue
        t, ins = tri[sel], inside[sel]
        # roll each triangle so that its lone corner (in front or behind) comes first
        lone = np.argmax(ins if n_in == 1 else ~ins, axis=1)
        idx = (lone[:, None] + np.arange(3)[None, :]) % 3
        t = np.take_along_axis(t, idx[:, :, None], axis=1)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        da, db, dc = -a[:, 2], -b[:, 2], -c[:, 2]
        ab = a + (b - a) * ((near - da) / (db - da))[:, None]
        ac = a + (c - a) * ((near - da) / (dc - da))[:, None]
        if n_in == 1:
            out.append(np.stack([a, ab, ac], axis=1))
        else:
            out.append(np.stack([ab, b, c], axis=1))
            out.append(np.stack([ab, c, ac], axis=1))
    return np.concatenate(out)


def _pixels(cam: Camera, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel x, y of camera-frame points (3, ...) at least the near plane in front of the camera."""
    scale = p.dtype.type(cam.focal) / -p[2]
    return p[0] * scale + p.dtype.type(cam.cx), p.dtype.type(cam.cy) - p[1] * scale


def _chain(cam: Camera, p: np.ndarray, part: Part, eye: np.ndarray, near: float, all_ahead: bool):
    """Outline sides of a closed part at camera-frame points p (3, V) seen from ``eye`` (the camera
    in the part's frame), as four lists of columns; and for sweeping: the points' pixels u, v, the
    mesh corners (s, e) of the outline sides and which points are in front (None: all)."""
    front = part.normals @ eye.astype(np.float32) > part.offsets
    cut = ahead = None
    if all_ahead:
        u, v = _pixels(cam, p)
    else:
        # triangles wholly in front as usual; the others are cut at the near plane
        f0, f1, f2 = part.corners
        ahead = -p[2] > near
        all_in = ahead[f0] & ahead[f1] & ahead[f2]
        some_in = (ahead[f0] | ahead[f1] | ahead[f2]) & ~all_in
        front &= all_in
        u, v = _pixels(cam, np.where(ahead, p, np.float32(-1.0)))
        if some_in.any():
            cut = _clip_near(p.T[part.faces[some_in]], near)
    # sides between a triangle that faces the camera and one that does not, run as in the first
    start, end, face, other = part.edges
    ext = np.append(front, False)
    fa, fb = ext[face], ext[other]
    sel = np.flatnonzero(fa != fb)
    back = fb[sel]
    a, b = start[sel], end[sel]
    s, e = np.where(back, b, a), np.where(back, a, b)
    sides = [[u[s]], [v[s]], [u[e]], [v[e]]]
    if cut is not None and len(cut):
        cx, cy = _pixels(cam, cut.transpose(2, 0, 1))
        facing = (cx[:, 1] - cx[:, 0]) * (cy[:, 2] - cy[:, 0]) - (cx[:, 2] - cx[:, 0]) * (cy[:, 1] - cy[:, 0]) < 0
        cx, cy = cx[facing], cy[facing]
        for k in range(3):
            j = (k + 1) % 3
            for out, col in zip(sides, (cx[:, k], cy[:, k], cx[:, j], cy[:, j])):
                out.append(col)
    return sides, (u, v, s, e, ahead)


def _outline(cam: Camera, rot: np.ndarray, shift: np.ndarray, part: Part, near: float,
             all_ahead: bool, move: np.ndarray | None = None) -> tuple | None:
    """Outline sides (xa, ya, xb, yb) of a closed part placed by rot, shift in the camera frame;
    with ``move`` (camera frame), of the part swept along that straight line."""
    p = rot.astype(np.float32) @ part.verts + shift.astype(np.float32)[:, None]
    sides, (u, v, s, e, ahead) = _chain(cam, p, part, -rot.T @ shift, near, all_ahead)
    if move is not None:
        q = p + move.astype(np.float32)[:, None]
        far, (u1, v1, _, _, ahead1) = _chain(cam, q, part, -rot.T @ (shift + move), near, all_ahead)
        for out, cols in zip(sides, far):
            out.extend(cols)
        if ahead is not None:
            keep = np.flatnonzero(ahead[s] & ahead[e] & ahead1[s] & ahead1[e])
            s, e = s[keep], e[keep]
        # what each outline side sweeps on the way: the quad a0 b0 b1 a1, turned like the outline
        quad = [(u[s], v[s]), (u[e], v[e]), (u1[e], v1[e]), (u1[s], v1[s])]
        twice_area = sum(quad[k][0] * quad[(k + 1) % 4][1] - quad[(k + 1) % 4][0] * quad[k][1] for k in range(4))
        flip = twice_area > 0
        for k in range(4):
            (xa, ya), (xb, yb) = quad[k], quad[(k + 1) % 4]
            for out, col in zip(sides, (np.where(flip, xb, xa), np.where(flip, yb, ya),
                                        np.where(flip, xa, xb), np.where(flip, ya, yb))):
                out.append(col)
    sides = tuple(np.concatenate(c) for c in sides)
    return sides if len(sides[0]) else None


def _hull(cam: Camera, p: np.ndarray, near: float) -> np.ndarray | None:
    """Outline polygon (K, 2) of the convex hull of camera-frame points (3, P)."""
    depth = -p[2]
    ahead = depth > near
    if not ahead.any():
        return None
    pts = p[:, ahead]
    if not ahead.all():
        # the hull cut at the near plane: points in front and where the segments cross the plane
        a, b = p[:, ahead], p[:, ~ahead]
        da, db = depth[ahead][:, None], depth[~ahead][None, :]
        w = (near - db) / (da - db)
        cross = b[:, None, :] + (a[:, :, None] - b[:, None, :]) * w[None]
        pts = np.concatenate([pts, cross.reshape(3, -1)], axis=1)
    x, y = _pixels(cam, pts)
    return cv2.convexHull(np.stack([x, y], axis=1).astype(np.float32)).reshape(-1, 2)


def _frustum(cam: Camera, pad: float) -> np.ndarray:
    """Unit normals (4, 3) of the planes through the camera bounding the picture widened by
    ``pad`` pixels; a point p (camera frame) is inside when every normal . p >= 0."""
    f = cam.focal
    n = np.array([
        [f, 0.0, -(cam.cx + 0.5 + pad)],
        [-f, 0.0, -(cam.width - 0.5 + pad - cam.cx)],
        [0.0, -f, -(cam.cy + 0.5 + pad)],
        [0.0, f, -(cam.height - 0.5 + pad - cam.cy)],
    ])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _own_arm(cam: Camera) -> str | None:
    for arm in ARMS:
        if cam.name in (f"{arm}_wrist", f"cam_{arm}_wrist", f"wrist_{arm[0]}"):
            return arm
    return None


def _scene(cam: Camera, arms: dict[str, ArmReading], openings: dict[str, float] | None,
           groups=None, pad: float = 16.0) -> list[_Drawn]:
    model = load_model()
    to_cam = invert(np.asarray(cam.pose, float))
    planes = _frustum(cam, pad)
    own = _own_arm(cam)
    w, h = cam.width, cam.height
    out = []
    for a, arm in enumerate(ARMS):
        reading = (arms or {}).get(arm)
        if reading is None:
            continue
        poses, moves = _poses(arm, reading, None if openings is None else openings.get(arm))
        near = OWN_NEAR_M if arm == own else NEAR_M
        for part in model.parts:
            if (groups is not None and part.group not in groups) or (arm == own and part.name in OWN_HIDDEN):
                continue
            m = to_cam @ poses[part.link]
            rot, shift = m[:3, :3], m[:3, 3]
            c = rot @ part.center + shift
            radius = part.radius
            move = moves.get(part.link)
            if move is not None:
                move = to_cam[:3, :3] @ move
                c = c + move / 2
                radius += float(np.linalg.norm(move)) / 2
            # skip parts wholly behind the near plane or off the picture
            if -c[2] + radius <= near or (planes @ c < -radius).any():
                continue
            if part.hull:
                p = rot.astype(np.float32) @ part.verts + shift.astype(np.float32)[:, None]
                poly = _hull(cam, p, near)
                box = None if poly is None else _box(poly[:, 0], poly[:, 1], w, h)
                if box is not None:
                    out.append(_Drawn(a, part.group, float(-c[2]), box, hull=poly))
            else:
                sides = _outline(cam, rot, shift, part, near, -c[2] - radius > near, move)
                # every corner of the closed chains starts a side
                box = None if sides is None else _box(sides[0], sides[1], w, h)
                if box is not None:
                    out.append(_Drawn(a, part.group, float(-c[2]), box, sides=sides))
    return out


def _paint(drawn: list[_Drawn], width: int, height: int) -> tuple[np.ndarray, tuple | None]:
    """Bool (height, width) of everything drawn, and its bounds (x0, y0, x1, y1) or None."""
    mask = np.zeros((height, width), bool)
    with_sides = [d for d in drawn if d.sides is not None]
    box = _union([d.box for d in with_sides])
    if box is not None:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        diff, sums = _buffers(h * (w + 1))
        diff[:] = 0
        batch, count = [], 0
        for i, d in enumerate(with_sides):
            batch.append(d.sides)
            count += len(d.sides[0])
            if count >= _BATCH or i == len(with_sides) - 1:
                joined = batch[0] if len(batch) == 1 else tuple(np.concatenate(c) for c in zip(*batch))
                _crossings(joined, x0, y0, w, h, diff)
                batch, count = [], 0
        _fill(diff, sums, mask[y0:y1, x0:x1])
    for d in drawn:
        if d.hull is not None:
            cv2.fillConvexPoly(mask.view(np.uint8), np.round(d.hull * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return mask, _union([d.box for d in drawn])


@lru_cache(maxsize=16)
def _kernel(margin: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))


def _dilate(image: np.ndarray, margin: int, box: tuple | None) -> np.ndarray:
    """``image`` (bool or uint8) dilated by a disc of radius ``margin`` around what is in ``box``."""
    if margin <= 0 or box is None:
        return image
    h, w = image.shape
    x0, y0, x1, y1 = max(box[0] - margin, 0), max(box[1] - margin, 0), min(box[2] + margin, w), min(box[3] + margin, h)
    region = np.ascontiguousarray(image[y0:y1, x0:x1])
    grown = cv2.dilate(region.view(np.uint8), _kernel(margin))
    image[y0:y1, x0:x1] = grown.view(image.dtype)
    return image


def default_margin(cam: Camera) -> int:
    if cam.name in DEFAULT_MARGIN_PX:
        return DEFAULT_MARGIN_PX[cam.name]
    return DEFAULT_MARGIN_PX["head" if _own_arm(cam) is None else "left_wrist"]


def robot_mask(cam: Camera, arms: dict[str, ArmReading], openings: dict[str, float] | None = None, *,
               margin_px: int | None = None) -> np.ndarray:
    """Bool (cam.height, cam.width): True where the picture shows either arm, widened by
    ``margin_px`` (default: ``DEFAULT_MARGIN_PX`` for the camera)."""
    margin = default_margin(cam) if margin_px is None else max(0, int(margin_px))
    mask, box = _paint(_scene(cam, arms, openings, pad=margin + 2.0), cam.width, cam.height)
    return _dilate(mask, margin, box)


def fingers_mask(cam: Camera, arms: dict[str, ArmReading], openings: dict[str, float] | None = None, *,
                 arm: str | None = None, margin_px: int | None = None) -> np.ndarray:
    """Like :func:`robot_mask`, for the fingers only (of ``arm``, or of both arms)."""
    if arm is not None:
        arms = {arm: arms[arm]} if arm in arms else {}
    margin = default_margin(cam) if margin_px is None else max(0, int(margin_px))
    mask, box = _paint(_scene(cam, arms, openings, {FINGERS}, pad=margin + 2.0), cam.width, cam.height)
    return _dilate(mask, margin, box)


def robot_labels(cam: Camera, arms: dict[str, ArmReading], openings: dict[str, float] | None = None, *,
                 margin_px: int = 0) -> np.ndarray:
    """uint8 (cam.height, cam.width): which part of which arm each pixel shows, as keys of
    ``LABELS`` (0: none).  Where parts overlap the nearer part (by centre depth) wins.  With a
    margin, the band around the arms takes the label of the part next to it."""
    margin = max(0, int(margin_px))
    out = np.zeros((cam.height, cam.width), np.uint8)
    drawn = sorted(_scene(cam, arms, openings, pad=margin + 2.0), key=lambda d: -d.depth)
    for d in drawn:
        value = 1 + d.arm * len(GROUPS) + d.group
        if d.hull is not None:
            cv2.fillConvexPoly(out, np.round(d.hull * 16).astype(np.int32), value, cv2.LINE_8, 4)
            continue
        x0, y0, x1, y1 = d.box
        inside = np.empty((y1 - y0, x1 - x0), bool)
        _wind(d.sides, x0, y0, inside)
        out[y0:y1, x0:x1][inside] = value
    if margin > 0 and drawn:
        grown = _dilate(out.copy(), margin, _union([d.box for d in drawn]))
        out = np.where(out > 0, out, grown)
    return out


def label_name(value: int) -> str:
    return LABELS.get(int(value), "none")
