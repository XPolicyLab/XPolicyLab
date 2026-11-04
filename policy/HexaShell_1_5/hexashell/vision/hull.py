"""Objects in 3-D from their outlines in several cameras.

A point in space can belong to an object only if every camera that could see the point shows it
inside the object's outline.  Cameras that cannot judge a point (it lies outside their picture,
behind them, or behind the robot's own arm) do not count.  The space this leaves is probed along
the rays of each outline's pixels:

* where a ray first enters the space is a point of the surface that camera sees;
* how far down the ray stays inside it is how low that pixel can still be explained.

With two or three cameras the space is larger than the object: a wedge above the object that no
camera can rule out, and a tail behind it that every camera's view is blocked from.  The object's
top is therefore taken as the lowest height that still explains almost every outline pixel (the
wedge is never needed for that), and its footprint from the surface points each camera sees
below the top (the hidden tail is never seen first).

With one camera the height cannot be measured: the object is assumed to rest on its support, and
its height is either known from an earlier measurement or guessed from a footprint about as deep
as it is wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..world import TABLE_X, TABLE_Y, TABLE_Z
from .cameras import Camera

#: An outline is trusted to half a pixel: a point counts as inside when it lands this close.
EDGE_PX = 0.5
#: Pixels around the robot's own outline that cannot rule anything out.
ROBOT_PAD_PX = 2
#: Outlines are looked up on a grid this many times finer than the picture.
SUB = 2
#: Spacing of the probe points along a ray (m); the two ends of each run are then refined.
STEP = 0.002
#: Rays per outline (its edge pixels first).
RAYS = 1500
#: Highest object top searched for above the support (m).
MAX_HEIGHT = 0.40
#: Share of an outline's pixels the top must explain.
TOP_SHARE = 0.985
#: Cameras looking at an object at least this far from straight down (degrees) read its top; the
#: steeper ones only when no camera is that slanted.
TOP_SLANT_DEG = 20.0
#: The rays that read the top go through the outline's edge pixels moved this far (pixels) outward:
#: an edge pixel's centre lies on average half a pixel inside the true outline, so a ray through it
#: leaves a slanted view's object below its top edge.
TOP_OUTWARD_PX = 0.5
#: Allowed space below the support height (m).
BELOW = 0.008
#: Objects are searched for this far around the table (m).
MARGIN = 0.10
#: Depth under the top at which the footprint is read from rays that pass over the top (m).
SKIN = 0.003

OUT, IN, BLIND = 0, 1, 2


@dataclass
class Sight:
    """One camera's outline of the object, with the robot's own pixels in that picture."""

    cam: str
    camera: Camera
    mask: np.ndarray
    robot: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, bool)
        fine = np.repeat(np.repeat(self.mask.astype(np.uint8), SUB, axis=0), SUB, axis=1)
        grow = int(round(EDGE_PX * SUB))
        if grow:
            fine = cv2.dilate(fine, np.ones((2 * grow + 1,) * 2, np.uint8))
        codes = np.where(fine > 0, IN, OUT).astype(np.uint8)
        if self.robot is not None and np.any(self.robot):
            pad = np.ones((2 * ROBOT_PAD_PX + 1,) * 2, np.uint8)
            self.blind = cv2.dilate(np.asarray(self.robot, np.uint8), pad) > 0
            codes[np.repeat(np.repeat(self.blind, SUB, axis=0), SUB, axis=1)] = BLIND
        else:
            self.blind = np.zeros(self.mask.shape, bool)
        self.codes = codes

    def look(self, points: np.ndarray) -> np.ndarray:
        """IN / OUT / BLIND for world points (N, 3).  Pixel (i, j) covers [i - 0.5, i + 0.5) x [j - 0.5, j + 0.5)."""
        cam = self.camera
        local = (points - cam.pose[:3, 3]) @ cam.pose[:3, :3]
        depth = -local[:, 2]
        ok = depth > 0.02
        safe = np.where(ok, depth, 1.0)
        u = np.floor(SUB * (cam.focal * local[:, 0] / safe + cam.cx + 0.5))
        v = np.floor(SUB * (-cam.focal * local[:, 1] / safe + cam.cy + 0.5))
        ok &= (u >= 0) & (u <= SUB * cam.width - 1) & (v >= 0) & (v <= SUB * cam.height - 1)
        out = np.full(len(points), BLIND, np.uint8)
        out[ok] = self.codes[v[ok].astype(np.intp), u[ok].astype(np.intp)]
        return out

    def mm_per_px(self, point) -> float:
        depth = float(np.dot(np.asarray(point, float) - self.camera.position, self.camera.axis))
        return 1000.0 * max(depth, 0.01) / self.camera.focal

    @property
    def whole(self) -> bool:
        """Nothing of the outline is cut off: it keeps clear of the picture's edge and the robot."""
        m = self.mask
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            return False
        near = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return not bool(np.any(near & self.blind))

    def steepness(self, point) -> float:
        """Angle (degrees) between straight down and this camera's line of sight to the point."""
        d = np.asarray(point, float) - self.camera.position
        return math.degrees(math.acos(min(1.0, max(-1.0, -d[2] / max(float(np.linalg.norm(d)), 1e-9)))))


@dataclass
class Shape:
    """What the outlines say about one object."""

    views: list[str]
    top: float | None = None
    bottom: float | None = None
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None  # length >= width, height (m)
    yaw: float | None = None  # direction of the long side, degrees in [-90, 90)
    round: bool = False
    err_mm: float | None = None
    explained: dict[str, float] = field(default_factory=dict)  # share of each outline the shape explains
    judged: dict[str, float] = field(default_factory=dict)  # share of its rays some other camera could judge
    confirmed: dict[str, float] = field(default_factory=dict)  # share of its rays some other camera's outline meets
    tops: dict[str, float] = field(default_factory=dict)  # the top each camera's outline asks for
    footprint_from: str = ""  # the camera whose surface points gave the footprint ('' for all of them)
    surface: np.ndarray | None = None  # (N, 3) surface points seen by the cameras, below the top
    kind: str = ""  # 'stands' (columns on a support), 'hangs' (widest at the top, e.g. a bowl), 'one view'
    single: bool = False
    guessed: bool = False  # single camera and no known height
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.center is not None

    @property
    def grip_yaw(self) -> float | None:
        """Yaw that closes a top-down grasp across the short side (degrees in (-90, 90])."""
        if self.yaw is None:
            return None
        g = self.yaw + 90.0
        return g - 180.0 if g > 90.0 else g


# -- rays --------------------------------------------------------------------------------------------
def _ray_pixels(sight: Sight, n: int, outward: float = 0.0) -> np.ndarray:
    """Pixels to cast rays through: most of them on the outline's edge, the rest inside.  With
    ``outward`` the edge pixels move that far (pixels) towards the background next to them."""
    usable = sight.mask & ~sight.blind
    if np.count_nonzero(usable) < 6:
        return np.zeros((0, 2))
    inner = cv2.erode(usable.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    edge = usable & ~inner
    ey, ex = np.nonzero(edge)
    iy, ix = np.nonzero(inner)
    n_edge = min(len(ex), int(0.6 * n))
    n_in = min(len(ix), n - n_edge)
    pick_e = np.unique(np.linspace(0, len(ex) - 1, n_edge).round().astype(int)) if n_edge else np.zeros(0, int)
    pick_i = np.unique(np.linspace(0, len(ix) - 1, n_in).round().astype(int)) if n_in else np.zeros(0, int)
    ex, ey = ex[pick_e], ey[pick_e]
    shift = outward * _outward(sight.mask, ex, ey) if outward and len(ex) else np.zeros((len(ex), 2))
    xs = np.concatenate([ex + shift[:, 0], ix[pick_i]]).astype(float)
    ys = np.concatenate([ey + shift[:, 1], iy[pick_i]]).astype(float)
    return np.stack([xs, ys], axis=1)


def _outward(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Unit (x, y) directions from outline pixels towards the background around them; (0, 0) where
    there is none (only the robot, or the picture's edge, next to them)."""
    pad = np.pad(mask, 1, constant_values=True)
    v = np.zeros((len(xs), 2))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                bg = ~pad[ys + 1 + dy, xs + 1 + dx]
                v[:, 0] += dx * bg
                v[:, 1] += dy * bg
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > 0, v / np.maximum(norm, 1e-9), 0.0)


def _box(sights: list[Sight], z_lo: float, z_hi: float) -> np.ndarray:
    """Search box [[x0, y0, z0], [x1, y1, z1]]: the table area cut down to every outline's cone
    (and kept under any camera that looks down on the object)."""
    for s in sights:
        if s.camera.axis[2] < -0.5:
            z_hi = min(z_hi, float(s.camera.position[2]) - 0.02)
    lo = np.array([TABLE_X[0] - MARGIN, TABLE_Y[0] - MARGIN, z_lo])
    hi = np.array([TABLE_X[1] + MARGIN, TABLE_Y[1] + MARGIN, max(z_hi, z_lo + 0.01)])
    for s in sights:
        ys, xs = np.nonzero(s.mask)
        if not len(xs):
            continue
        corners = np.array([[xs.min() - 2, ys.min() - 2], [xs.max() + 2, ys.min() - 2],
                            [xs.min() - 2, ys.max() + 2], [xs.max() + 2, ys.max() + 2]], float)
        pts = np.concatenate([s.camera.on_plane(corners, lo[2]), s.camera.on_plane(corners, hi[2])])
        if np.isnan(pts).any():
            continue
        lo[:2] = np.maximum(lo[:2], pts[:, :2].min(axis=0) - 0.01)
        hi[:2] = np.minimum(hi[:2], pts[:, :2].max(axis=0) + 0.01)
    return np.stack([lo, hi])


def _spans(origin: np.ndarray, dirs: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entry and exit distances of rays through an axis-aligned box (exit < entry: missed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (box[0] - origin) * inv
        t2 = (box[1] - origin) * inv
    near = np.where(np.isnan(t1) | np.isnan(t2), -np.inf, np.minimum(t1, t2))
    far = np.where(np.isnan(t1) | np.isnan(t2), np.inf, np.maximum(t1, t2))
    return np.maximum(near.max(axis=1), 0.03), far.min(axis=1)


class _Space:
    """The space the outlines leave, for probing."""

    def __init__(self, sights: list[Sight], floor: float):
        self.sights = sights
        self.floor = floor

    def probe(self, points: np.ndarray, skip: int) -> tuple[np.ndarray, np.ndarray]:
        """(not ruled out, number of other cameras showing the point inside their outline)."""
        n = len(points)
        ruled_out = points[:, 2] < self.floor - BELOW
        support = np.zeros(n, np.int8)
        for i, s in enumerate(self.sights):
            if i == skip:
                continue
            code = s.look(points)
            ruled_out |= code == OUT
            support += code == IN
        return ~ruled_out, support


@dataclass
class _RayResult:
    pixels: np.ndarray  # (N, 2)
    hit: np.ndarray  # explained: another camera confirms some point of the ray
    judged: np.ndarray  # another camera could judge some point of the ray
    entry: np.ndarray  # (N, 3) first confirmed point (NaN if none)
    entry_ok: np.ndarray  # that point is a real boundary (not the edge of the search box)
    low: np.ndarray  # lowest height of the run that starts there (NaN if none)
    dirs: np.ndarray


def _march(space: _Space, idx: int, pixels: np.ndarray, box: np.ndarray, step: float = STEP) -> _RayResult:
    """Walk the rays of one camera's outline pixels through the space.

    A ray's run starts at its first point another camera shows inside its outline, and goes on
    through points no camera can judge until a camera rules one out.  (Points before the first
    confirmed one are ignored: near the camera the others often cannot see at all.)
    """
    cam = space.sights[idx].camera
    origin = cam.position
    n = len(pixels)
    dirs = cam.rays(pixels) if n else np.zeros((0, 3))
    tmin, tmax = _spans(origin, dirs, box)
    span = np.where(tmax > tmin, tmax - tmin, 0.0)
    k = int(min(700, max(2, math.ceil(float(span.max(initial=0.0)) / step) + 1)))
    t = tmin[:, None] + step * np.arange(k)[None, :]
    valid = t <= tmax[:, None] + 1e-9
    pts = origin + dirs[:, None, :] * t[..., None]
    free = np.zeros(t.shape, bool)
    support = np.zeros(t.shape, np.int8)
    flat = valid.ravel()
    if flat.any():
        ins, sup = space.probe(pts.reshape(-1, 3)[flat], idx)
        free.ravel()[flat] = ins
        support.ravel()[flat] = sup
    # bridge single-sample gaps (an outline pixel lost to noise)
    gap = np.zeros_like(free)
    gap[:, 1:-1] = free[:, :-2] & free[:, 2:] & valid[:, 1:-1]
    free |= gap
    confirmed = free & (support >= 1)
    hit = confirmed.any(axis=1)
    judged = ((support >= 1) | (~free & valid)).any(axis=1)
    first = np.argmax(confirmed, axis=1)
    cols = np.arange(k)[None, :]
    stop = (cols > first[:, None]) & ~free
    end = np.where(stop.any(axis=1), np.argmax(stop, axis=1) - 1, k - 1)
    end = np.where(hit, end, 0)
    rows = np.arange(n)
    entry_ok = hit & (first > 0)

    def refine(sel, lo_t, hi_t, want_at_hi, need_support):
        """Bisect on the selected rays between a sample that fails and one that passes."""
        lo_t, hi_t, d = lo_t.copy(), hi_t.copy(), dirs[sel]
        for _ in range(4):
            mid = 0.5 * (lo_t + hi_t)
            ins, sup = space.probe(origin + d * mid[:, None], idx)
            ok = ins & (sup >= 1) if need_support else ins
            take_hi = ok if want_at_hi else ~ok
            hi_t = np.where(take_hi, mid, hi_t)
            lo_t = np.where(take_hi, lo_t, mid)
        return 0.5 * (lo_t + hi_t)

    t_first = t[rows, first]
    if entry_ok.any():
        t_first[entry_ok] = refine(entry_ok, t[rows, first - 1][entry_ok], t_first[entry_ok], True, True)
    t_end = t[rows, end]
    nxt = np.minimum(end + 1, k - 1)
    can_end = hit & (end < k - 1) & valid[rows, nxt]
    if can_end.any():
        t_end[can_end] = refine(can_end, t_end[can_end], t[rows, nxt][can_end], False, False)
    entry = origin + dirs * t_first[:, None]
    exit_ = origin + dirs * t_end[:, None]
    low = np.minimum(entry[:, 2], exit_[:, 2])
    entry[~hit] = np.nan
    low[~hit] = np.nan
    return _RayResult(pixels, hit, judged, entry, entry_ok, low, dirs)


# -- shape from several cameras ------------------------------------------------------------------------
def _footprint(points: np.ndarray) -> tuple[np.ndarray, float, float, float, bool]:
    """Centre (x, y), length, width, long-side direction (deg) and roundness of xy points."""
    pts = np.asarray(points, np.float32)
    rect = cv2.minAreaRect(pts)
    # trim the outermost 1% along the rectangle's axes and fit again
    ang = math.radians(rect[2])
    axes = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]], np.float32)
    proj = (pts - np.asarray(rect[0], np.float32)) @ axes.T
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    keep = np.all((proj >= lo) & (proj <= hi), axis=1)
    if keep.sum() >= 8:
        pts = pts[keep]
        rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if w >= h:
        length, width, direction = w, h, angle
    else:
        length, width, direction = h, w, angle + 90.0
    direction = (direction + 90.0) % 180.0 - 90.0
    hull_area = cv2.contourArea(cv2.convexHull(pts)) if len(pts) >= 3 else 0.0
    fill = hull_area / max(length * width, 1e-9)
    is_round = length <= 1.15 * width and fill < 0.88
    return np.array([cx, cy]), float(length), float(width), float(direction), bool(is_round)


def _err_mm(sights: list[Sight], point: np.ndarray, used: list[int]) -> float:
    """Rough error from pixel size and the angle between the two best cameras."""
    sizes = sorted((sights[i].mm_per_px(point), i) for i in used)
    if len(sizes) < 2:
        return float("nan")
    best = sizes[0]
    dirs = {i: (point - sights[i].camera.position) / np.linalg.norm(point - sights[i].camera.position) for i in used}
    angle = 0.0
    for _, j in sizes[1:]:
        angle = max(angle, math.degrees(math.acos(min(1.0, abs(float(dirs[best[1]] @ dirs[j]))))))
    spread = math.sqrt(best[0] ** 2 + sizes[1][0] ** 2) / math.sqrt(2.0)
    return float(min(50.0, max(1.0, spread / max(math.sin(math.radians(angle)), 0.15))))


# -- columns: objects that stand on something ------------------------------------------------------------
#: Column grid: cell side and level spacing (m), and the most grid points probed.
CELL = 0.0015
LEVEL = 0.002
GRID_POINTS = 1_500_000


@dataclass
class _Columns:
    """Which points of a grid over the object's region no camera rules out, as vertical runs."""

    x0: float
    y0: float
    cell: float
    z0: float  # height of level 0
    dz: float
    start: np.ndarray  # (levels, ny, nx): lowest level of the free run through this level, or -1
    end: np.ndarray  # (levels, ny, nx): highest level of that run, or -1

    def index(self, points: np.ndarray):
        """(level, iy, ix, on the grid) of points (nearest grid point); points with no position are off it."""
        finite = np.isfinite(points).all(axis=1)
        p = np.where(finite[:, None], points, -1e9)
        lv = np.rint((p[:, 2] - self.z0) / self.dz).astype(np.int64)
        iy = np.rint((p[:, 1] - self.y0) / self.cell).astype(np.int64)
        ix = np.rint((p[:, 0] - self.x0) / self.cell).astype(np.int64)
        nl, ny, nx = self.start.shape
        ok = finite & (lv >= 0) & (lv < nl) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        return lv, iy, ix, ok

    def runs(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(level, lowest and highest level of its free run, or -1) of points."""
        lv, iy, ix, ok = self.index(points)
        st = np.full(len(points), -1, np.int64)
        en = np.full(len(points), -1, np.int64)
        st[ok] = self.start[lv[ok], iy[ok], ix[ok]]
        en[ok] = self.end[lv[ok], iy[ok], ix[ok]]
        return lv, st, en


def _columns(space: _Space, sights: list[Sight], floor: float, top: float) -> _Columns | None:
    box = _box(sights, floor, top + LEVEL)
    if np.any(box[1, :2] <= box[0, :2]):
        return None
    levels = max(1, int(math.floor((top - floor) / LEVEL + 1e-9)) + 1)
    cell = CELL
    while True:
        nx = int(math.ceil((box[1, 0] - box[0, 0]) / cell)) + 1
        ny = int(math.ceil((box[1, 1] - box[0, 1]) / cell)) + 1
        if nx * ny * levels <= GRID_POINTS or cell > 0.006:
            break
        cell *= 1.25
    x0, y0, z0 = box[0, 0], box[0, 1], floor  # level 0 lies on the table itself
    gz, gy, gx = np.meshgrid(z0 + LEVEL * np.arange(levels), y0 + cell * np.arange(ny), x0 + cell * np.arange(nx),
                             indexing="ij")
    free, _ = space.probe(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1), -1)
    free = free.reshape(levels, ny, nx)
    start = np.full(free.shape, -1, np.int32)
    start[0][free[0]] = 0
    for lv in range(1, levels):
        run_on = free[lv] & free[lv - 1]
        start[lv] = np.where(run_on, start[lv - 1], np.where(free[lv], lv, -1))
    end = np.full(free.shape, -1, np.int32)
    end[-1][free[-1]] = levels - 1
    for lv in range(levels - 2, -1, -1):
        run_on = free[lv] & free[lv + 1]
        end[lv] = np.where(run_on, end[lv + 1], np.where(free[lv], lv, -1))
    return _Columns(x0, y0, cell, z0, LEVEL, start, end)


def _column_rays(cols: _Columns, sights: list[Sight], box: np.ndarray, rays: int, outward: float = 0.0) -> list:
    """Each view's outline rays sampled through the column grid:
    (pixels, dirs, t, z, level, run start, run end)."""
    out = []
    step = min(cols.cell, cols.dz) * 0.8
    for s in sights:
        pix = _ray_pixels(s, rays, outward)
        cam = s.camera
        dirs = cam.rays(pix) if len(pix) else np.zeros((0, 3))
        tmin, tmax = _spans(cam.position, dirs, box)
        span = np.where(tmax > tmin, tmax - tmin, 0.0)
        k = int(min(1200, max(2, math.ceil(float(span.max(initial=0.0)) / step) + 1)))
        t = tmin[:, None] + step * np.arange(k)[None, :]
        pts = cam.position + dirs[:, None, :] * t[..., None]
        lv, st, en = cols.runs(pts.reshape(-1, 3))
        lv, st, en = lv.reshape(t.shape), st.reshape(t.shape), en.reshape(t.shape)
        inside = t <= tmax[:, None]
        st, en = np.where(inside, st, -1), np.where(inside, en, -1)
        out.append((pix, dirs, t, pts[..., 2], lv, st, en))
    return out


def _support_share(cols: _Columns, per_view: list) -> np.ndarray:
    """(views, levels): share of each view's rays that meet the columns standing on each level."""
    nl = cols.start.shape[0]
    shares = []
    for pix, dirs, t, z, lv, st, en in per_view:
        if not len(pix):
            shares.append(np.zeros(nl))
            continue
        use = st >= 0
        # a sample at level lv on a run from st backs every support in [st, lv]
        diff = np.zeros((len(pix), nl + 1), np.int32)
        r = np.broadcast_to(np.arange(len(pix))[:, None], t.shape)
        np.add.at(diff, (r[use], st[use]), 1)
        np.add.at(diff, (r[use], np.minimum(lv[use], nl - 1) + 1), -1)
        shares.append((np.cumsum(diff[:, :nl], axis=1) > 0).mean(axis=0))
    return np.array(shares)


def _hang_share(cols: _Columns, per_view: list, sights: list[Sight], top: float) -> dict[str, float]:
    """Share of each view's rays that meet the columns hanging from the top level (free all the
    way up to it): what a rim explains of the outlines."""
    top_lv = int(round((top - cols.z0) / cols.dz))
    out = {}
    for s, (pix, dirs, t, z, lv, st, en) in zip(sights, per_view):
        hung = (en >= top_lv) & (lv <= top_lv)
        out[s.cam] = round(float(hung.any(axis=1).mean()), 3) if len(pix) else 0.0
    return out


#: Exact steps (0.5 mm each) walked past the grid's first hit in search of a standing point.
WALK = 24


def _stands(space: _Space, points: np.ndarray, support_z: float, top: float) -> np.ndarray:
    """Points no camera rules out, at most at the top, whose column down to the support is free."""
    ok = (points[:, 2] <= top + 1e-6) & (points[:, 2] >= support_z - 1e-6)
    if ok.any():
        idx = np.nonzero(ok)[0]
        ins, _ = space.probe(points[idx], -1)
        idx = idx[ins]
        ok[:] = False
        ok[idx] = _column_free(space, points[idx], support_z)
    return ok


def _first_standing(space: _Space, cand: np.ndarray, support_z: float, top: float) -> np.ndarray:
    """Index of the first of each row's candidate points (rows, k, 3) that stands, or -1.  The
    column check is costly, so each row only checks its free candidates in turn until one passes."""
    rows, k = cand.shape[:2]
    flat = cand.reshape(-1, 3)
    free = (flat[:, 2] <= top + 1e-6) & (flat[:, 2] >= support_z - 1e-6)
    if free.any():
        idx = np.nonzero(free)[0]
        free[idx] = space.probe(flat[idx], -1)[0]
    free = free.reshape(rows, k)
    out = np.full(rows, -1)
    nxt = np.zeros(rows, np.int64)
    live = free.any(axis=1)
    order = np.arange(k)[None, :]
    while live.any():
        idx = np.nonzero(live)[0]
        left = free[idx] & (order >= nxt[idx][:, None])
        has = left.any(axis=1)
        live[idx[~has]] = False
        idx, left = idx[has], left[has]
        if not len(idx):
            break
        j = np.argmax(left, axis=1)
        ok = _column_free(space, cand[idx, j], support_z)
        out[idx[ok]] = j[ok]
        live[idx[ok]] = False
        nxt[idx[~ok]] = j[~ok] + 1
    return out


def _column_free(space: _Space, points: np.ndarray, support_z: float) -> np.ndarray:
    """Whether the vertical segment from the support up to each point is free (exact xy)."""
    n = len(points)
    if not n:
        return np.zeros(0, bool)
    k = int(math.ceil(max(float(points[:, 2].max()) - support_z, 0.0) / LEVEL)) + 1
    zs = support_z + LEVEL * np.arange(k)
    zz = np.minimum(zs[None, :], points[:, 2:3])
    probe = np.concatenate([np.repeat(points[:, :2], k, axis=0), zz.reshape(-1, 1)], axis=1)
    free, _ = space.probe(probe, -1)
    return free.reshape(n, k).all(axis=1)


#: The standing reading is doubted when it explains this much less of the outlines than the cones
#: do, and the hanging one must then explain this much more than it.
HANG_MARGIN = 0.03
#: A support this low (m, and at most LOW_SHARE of the height) is the table under a rounded bottom.
LOW_SUPPORT = 0.010
LOW_SHARE = 0.3
#: Supports whose share of explained outline is within this of the best count as equally good.
PLATEAU = 0.005


def shape(sights: list[Sight], floor: float = TABLE_Z, *, rays: int = RAYS) -> Shape:
    """Measure one object from its outlines in two or more cameras (``single`` handles one).

    The outlines' cones overlap in more places than the object fills: in front of it, where two
    cameras' views of it cross, and behind it, where it hides the space from every camera.  So the
    object is read as standing on a support (the table, or what it rests on): only columns of free
    space that reach down to the support count, and the support is the height that lets them explain
    the outlines best.  An object that is widest at the top (a bowl) cannot stand that way; it is
    read as hanging from its rim, whose outline the cameras do pin down.
    """
    names = [s.cam for s in sights]
    out = Shape(views=names)
    if len(sights) < 2:
        out.notes.append("needs two cameras")
        return out
    box = _box(sights, floor - BELOW, floor + MAX_HEIGHT)
    if np.any(box[1] <= box[0]):
        out.notes.append("the outlines' cones do not meet above the table")
        for s in sights:
            out.explained[s.cam] = 0.0
        return out
    space = _Space(sights, floor)
    results = []
    for i, s in enumerate(sights):
        res = _march(space, i, _ray_pixels(s, rays), box)
        results.append(res)
        n = max(len(res.pixels), 1)
        out.explained[s.cam] = out.confirmed[s.cam] = round(float(res.hit.sum()) / n, 3)
        out.judged[s.cam] = round(float(res.judged.sum()) / n, 3)
    # first guess at the top: the lowest height that explains almost every pixel of every outline
    # (a rim's height only from the slanted cameras, as for a standing object's top)
    tops, slanted = {}, []
    for s, res in zip(sights, results):
        if res.hit.sum() >= 12:
            tops[s.cam] = float(np.quantile(res.low[res.hit], TOP_SHARE))
            if _slant_deg(res.dirs[res.hit]) >= TOP_SLANT_DEG:
                slanted.append(s.cam)
    if not tops:
        out.notes.append("the outlines do not agree on any point")
        return out
    top_guess = max(tops.values())
    rim = max(tops[c] for c in (slanted or tops))
    plain = min(out.explained.values())
    z_hi = min(top_guess + max(0.02, 0.3 * (top_guess - floor)), floor + MAX_HEIGHT)
    cols = _columns(space, sights, floor, z_hi)
    fit = None
    if cols is not None:
        reach = _box(sights, floor, z_hi + LEVEL)
        per_view = _column_rays(cols, sights, reach, rays)
        top_view = _column_rays(cols, sights, reach, rays, TOP_OUTWARD_PX) if TOP_OUTWARD_PX else per_view
        fit = _stand(space, sights, floor, cols, per_view, top_view)
        # Not a good prism: perhaps it hangs from its rim (a bowl).  A rounded object neither
        # stands nor hangs well; it keeps the standing reading unless a round rim explains clearly
        # more.  (An overhang alone does not decide it: tiles, mugs and bottles show one too, and
        # read from a rim their centre moved by centimetres.)
        if fit is None or fit["share"] < plain - HANG_MARGIN or fit["overhang"]:
            hung = _hang(space, sights, floor, rim)
            if hung is not None:
                hung["explained"] = _hang_share(cols, per_view, sights, rim)
                hung["share"] = min(hung["explained"].values())
                round_rim = hung["footprint"][4]
                if fit is None or (round_rim and hung["share"] >= fit["share"] + HANG_MARGIN):
                    fit = hung
    if fit is None:
        out.notes.append("no solid fits the outlines")
        return out
    out.kind = fit["kind"]
    out.explained.update(fit.get("explained", {}))
    out.notes.extend(fit.get("notes", []))
    top, bottom = fit["top"], fit["bottom"]
    height = max(top - bottom, 0.002)
    (cx, cy), length, width, direction, is_round = fit["footprint"]
    out.top, out.bottom = top, bottom
    out.center = (float(cx), float(cy), float(bottom + height / 2.0))
    out.size = (length, width, height)
    out.yaw = direction
    out.round = is_round
    out.surface = fit["surface"]
    out.tops = {k: round(v, 4) for k, v in fit.get("tops", {}).items()}
    out.footprint_from = fit.get("footprint_from", "")
    used = [i for i, s in enumerate(sights) if out.explained.get(s.cam, 0.0) > 0.5]
    out.err_mm = _err_mm(sights, np.array(out.center), used or list(range(len(sights))))
    return out


def _slant_deg(dirs: np.ndarray) -> float:
    """How far (degrees) rays run from straight down, typically."""
    return math.degrees(math.acos(min(1.0, float(np.median(-dirs[:, 2])))))


def _stand(space, sights, floor, cols, per_view, top_view=None):
    """Read the object as columns standing on the support that best explains the outlines
    (``top_view``: the rays that read the top, ``per_view`` by default)."""
    shares = _support_share(cols, per_view)
    nl = cols.start.shape[0]
    worst = shares.min(axis=0)[: max(1, nl - 2)]  # the least explained view, per support level
    # The supports that explain the outlines (nearly) best form a run of levels around the real one.
    best = int(np.argmax(worst))
    lo = hi = best
    while lo > 0 and worst[lo - 1] >= worst[best] - PLATEAU:
        lo -= 1
    while hi + 1 < len(worst) and worst[hi + 1] >= worst[best] - PLATEAU:
        hi += 1
    # Among supports that explain the outlines equally well, the real one is where columns begin:
    # the object's own columns all start on it, while the overlaps in front start at all heights.
    # A support between two levels makes its columns begin on either; take the busiest pair of
    # levels, and its upper level only if a fair share of the columns begin there.
    begins = np.array([np.count_nonzero(cols.start[lv] == lv) for lv in range(lo, hi + 1)], float)
    below = np.concatenate([[float(np.count_nonzero(cols.start[lo - 1] == lo - 1)) if lo > 0 else 0.0], begins[:-1]])
    k = int(np.argmax(begins + below))
    support = lo + k if (k == 0 and lo == 0) or begins[k] >= 0.5 * below[k] else lo + k - 1
    support_z = cols.z0 + support * cols.dz
    # top: lowest height at which the standing columns explain almost every outline ray.  A camera
    # looking straight down cannot tell: its rays run down the object's sides and leave the columns
    # wherever the grid happens to step outward, so only slanted cameras count when there are any.
    tops, slanted = {}, []
    inside = [(st >= 0) & (st <= support) & (lv >= support) for pix, dirs, t, z, lv, st, en in per_view]
    for s, (pix, dirs, t, z, lv, st, en) in zip(sights, top_view or per_view):
        solid = (st >= 0) & (st <= support) & (lv >= support)
        meets = solid.any(axis=1)
        if meets.sum() >= 12:
            low = np.where(solid, z, np.inf).min(axis=1)[meets]
            tops[s.cam] = float(np.quantile(low, TOP_SHARE))
            if _slant_deg(dirs[meets]) >= TOP_SLANT_DEG:
                slanted.append(s.cam)
    if not tops:
        return None
    top = max(tops[c] for c in (slanted or tops))
    pts_all, by_view, explained = [], [], {}
    for s, (pix, dirs, t, z, lv, st, en), solid in zip(sights, per_view, inside):
        if not len(pix):
            explained[s.cam] = 0.0
            continue
        solid = solid & (z <= top + 1e-6)
        explained[s.cam] = round(float(solid.any(axis=1).mean()), 3)
        first = np.argmax(solid, axis=1)
        rows = np.nonzero(solid.any(axis=1) & (first > 0))[0]
        if not len(rows):
            continue
        # The grid is coarse: from its first hit, walk on with exact checks to the first point
        # that really stands on the support, then refine between it and the point before.
        origin, d = s.camera.position, dirs[rows]
        t0 = t[rows, first[rows]]
        fine = 0.0005
        steps = t0[:, None] + fine * np.arange(WALK)[None, :]
        cand = origin + d[:, None, :] * steps[..., None]
        j = _first_standing(space, cand, support_z, top)
        found = j >= 0
        rows, d, j, steps = rows[found], d[found], j[found], steps[found]
        hi_t = steps[np.arange(len(j)), j]
        lo_t = np.where(j > 0, hi_t - fine, t[rows, first[rows] - 1])
        for _ in range(3):
            mid = 0.5 * (lo_t + hi_t)
            good = _stands(space, origin + d * mid[:, None], support_z, top)
            hi_t = np.where(good, mid, hi_t)
            lo_t = np.where(good, lo_t, mid)
        pts_all.append(origin + d * hi_t[:, None])
        by_view.append((s, pts_all[-1]))
    if not pts_all:
        return None
    surface = np.concatenate(pts_all)
    if len(surface) < 12:
        return None
    # Is the support the table after all?  Yes when it is just a rounded bottom's lift, or when the
    # free space carries on down to the table about as wide (an overhang, like a bowl's rim).
    bottom, notes, overhang = support_z, [], False
    if support > 0:
        area = (cols.start >= 0).sum(axis=(1, 2)).astype(float)
        if support_z - floor <= min(LOW_SUPPORT, LOW_SHARE * (top - floor)) + 1e-6:
            bottom = floor
        elif area[0] >= 0.8 * area[support]:
            bottom, overhang = floor, True
        else:
            notes.append(f"rests on something {1000 * (support_z - floor):.0f} mm above the table")
    share = float(min(explained.values()))
    seen_by, foot_pts = _footprint_points(by_view, surface)
    footprint = _footprint(foot_pts[:, :2])
    circle = _upright_round(space, sights, support_z if not overhang else floor, top, surface)
    if circle is not None:
        footprint = (circle[0], 2.0 * circle[1], 2.0 * circle[1], 0.0, True)
    else:
        footprint = footprint[:4] + (False,)
    if not slanted:
        notes.append("every camera looks down on it steeply: its height is uncertain")
    return {"kind": "stands", "share": share, "top": top, "bottom": bottom, "surface": surface,
            "footprint": footprint, "explained": explained, "notes": notes, "overhang": overhang,
            "tops": tops, "footprint_from": seen_by}


#: The footprint is read from one camera alone when it looks down on the object this steeply
#: (degrees from straight down) and sees all of it: the coarser, slanted views only blur it.
FOOT_STEEP_DEG = 20.0
#: ...and when that camera gives at least this many surface points.
FOOT_POINTS = 40


def _footprint_points(by_view: list, surface: np.ndarray) -> tuple[str, np.ndarray]:
    """(camera, points) the footprint is read from: the finest camera that looks down on the whole
    object, else every camera's points ('')."""
    centre = np.median(surface, axis=0)
    best = None
    for s, pts in by_view:
        if len(pts) < FOOT_POINTS or s.steepness(centre) > FOOT_STEEP_DEG or not s.whole:
            continue
        scale = s.mm_per_px(centre)
        if best is None or scale < best[0]:
            best = (scale, s.cam, pts)
    return ("", surface) if best is None else (best[1], best[2])


@dataclass
class _Section:
    """The free space's largest connected piece at one height, with each point's room around it
    (distance to the piece's edge, m)."""

    z: float
    xs: np.ndarray
    ys: np.ndarray
    free: np.ndarray  # (len(ys), len(xs)) bool
    room: np.ndarray  # same shape, m
    cell: float

    @property
    def points(self) -> np.ndarray:
        gx, gy = np.meshgrid(self.xs, self.ys)
        return np.stack([gx[self.free], gy[self.free]], axis=1)

    @property
    def area(self) -> float:
        return float(self.free.sum()) * self.cell * self.cell

    @property
    def radius(self) -> float:
        return float(self.room.max())

    @property
    def centre(self) -> np.ndarray:
        iy, ix = np.unravel_index(int(np.argmax(self.room)), self.room.shape)
        return np.array([self.xs[ix], self.ys[iy]])


def _section(space: _Space, sights: list[Sight], z: float, cell: float = 0.001) -> _Section | None:
    box = _box(sights, z - 0.001, z + 0.001)
    if np.any(box[1, :2] <= box[0, :2]):
        return None
    side = max(box[1, 0] - box[0, 0], box[1, 1] - box[0, 1])
    cell = max(cell, side / 400.0)
    xs = np.arange(box[0, 0], box[1, 0] + cell, cell)
    ys = np.arange(box[0, 1], box[1, 1] + cell, cell)
    gx, gy = np.meshgrid(xs, ys)
    free, _ = space.probe(np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1), -1)
    free = free.reshape(gx.shape)
    if free.sum() < 12:
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    free = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    room = cv2.distanceTransform(np.pad(free, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return _Section(z, xs, ys, free, room.astype(float) * cell, cell)


def _ring(centre, radius: float, z: float, n: int = 90) -> np.ndarray:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(ang), centre[1] + radius * np.sin(ang), np.full(n, z)], axis=1)


#: How far (m) the largest circle may be from the largest of all and still be a candidate.
ROOM_SLACK = 0.0015


def _outline_spread(flat: np.ndarray, centre: np.ndarray, sectors: int = 24) -> float | None:
    """Spread (q90 - q10, m) of the outermost point's distance from the centre over the directions
    seen: about zero around a round object, a box's corner to face-middle otherwise."""
    rel = flat - centre
    dist = np.linalg.norm(rel, axis=1)
    which = ((np.arctan2(rel[:, 1], rel[:, 0]) + math.pi) / (2 * math.pi) * sectors).astype(int) % sectors
    far = np.full(sectors, np.nan)
    np.fmax.at(far, which, dist)
    far = far[~np.isnan(far)]
    if len(far) < 6:
        return None
    return float(np.quantile(far, 0.9) - np.quantile(far, 0.1))


def _upright_round(space, sights, base, top, surface):
    """An upright round object's widest cross-section, as (centre, radius, height) or None.

    Each camera's two outline tangents bound the free space's slice, so the largest circle inside
    it is the object's own; the hidden space behind and the overlaps in front lie outside it.  Seen
    from few directions the circle can slide along the slice: of the circles nearly as large, take
    the one the visible surface fits best.  The visible surface must stay inside that circle and,
    at that height, reach about equally far in every direction it is seen in (three slanted views
    of a cube leave a hexagon whose inner circle nearly reaches the cube's corners).
    """
    best = None
    sample = surface
    if len(sample) > 3000:
        sample = sample[np.linspace(0, len(sample) - 1, 3000).astype(int)]
    band = max(0.003, 0.1 * (top - base))
    for f in (0.3, 0.5, 0.7):
        sec = _section(space, sights, base + f * (top - base))
        if sec is None or sec.radius < 0.003:
            continue
        iy, ix = np.nonzero(sec.room >= sec.radius - ROOM_SLACK)
        if len(iy) > 400:
            keep = np.linspace(0, len(iy) - 1, 400).astype(int)
            iy, ix = iy[keep], ix[keep]
        centres = np.stack([sec.xs[ix], sec.ys[iy]], axis=1)
        rooms = sec.room[iy, ix]
        reach = np.quantile(np.linalg.norm(sample[None, :, :2] - centres[:, None, :], axis=2), 0.99, axis=1)
        poke = reach - rooms
        i = int(np.argmin(poke))
        centre, radius = centres[i], float(rooms[i])
        if poke[i] > max(0.003, 0.1 * radius):
            continue
        level = np.abs(surface[:, 2] - sec.z) <= band
        if level.sum() < 12:
            continue
        spread = _outline_spread(surface[level, :2], centre)
        if spread is None or spread > max(0.002, 0.15 * radius):
            continue
        if best is None or radius > best[1]:
            best = (centre, radius, sec.z)
    return best


def _hang(space, sights, floor, top):
    """Read the object as hanging from its rim: its footprint is the free space just under the top
    (where the rim's outline bounds it closely); a round rim is the largest circle that fits."""
    z = top - min(0.0015, 0.1 * max(top - floor, 0.0))
    sec = _section(space, sights, z)
    if sec is None:
        return None
    notes = ["widest at the top: read from the rim"]
    if math.pi * sec.radius ** 2 >= 0.85 * sec.area:
        d = 2.0 * sec.radius
        footprint = (sec.centre, d, d, 0.0, True)
        surface = _ring(sec.centre, sec.radius, z)
    else:
        footprint = _footprint(sec.points)
        surface = np.column_stack([sec.points, np.full(len(sec.points), z)])
    return {"kind": "hangs", "share": 0.0, "top": top, "bottom": floor, "surface": surface,
            "footprint": footprint, "notes": notes}


# -- one camera ----------------------------------------------------------------------------------------
def _grid(sight: Sight, floor: float, top: float, cell: float):
    ys, xs = np.nonzero(sight.mask)
    corners = np.array([[xs.min() - 2, ys.min() - 2], [xs.max() + 2, ys.min() - 2],
                        [xs.min() - 2, ys.max() + 2], [xs.max() + 2, ys.max() + 2]], float)
    pts = np.concatenate([sight.camera.on_plane(corners, floor), sight.camera.on_plane(corners, top)])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return None
    lo = np.maximum(pts[:, :2].min(axis=0), [TABLE_X[0] - MARGIN, TABLE_Y[0] - MARGIN])
    hi = np.minimum(pts[:, :2].max(axis=0), [TABLE_X[1] + MARGIN, TABLE_Y[1] + MARGIN])
    if np.any(hi <= lo):
        return None
    gx = np.arange(lo[0], hi[0], cell) + cell / 2
    gy = np.arange(lo[1], hi[1], cell) + cell / 2
    if len(gx) * len(gy) > 400_000:
        return None
    return np.meshgrid(gx, gy)


def _cast(sight: Sight, gx: np.ndarray, gy: np.ndarray, z: float) -> np.ndarray:
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    return (sight.look(pts) == IN).reshape(gx.shape)


#: A camera whose ray to the object is steeper than this looks at it from straight above.
ABOVE_DEG = 65.0
#: The outline's sides are read for the height only from at least this far above the table plane
#: (degrees): lower down a pixel's stretch on the plane grows long and the reading is noise.
SIDE_MIN_DEG = 35.0
#: ... and only when no pixel of the outline meets the plane within this distance (m) of the point
#: under the camera: an object under the camera has no sides that point away from it.
SIDE_CLEAR = 0.02


def _side_heights(sight: Sight, floor: float) -> list[float]:
    """The height (m) each side of the outline suggests, seen around the point under the camera.

    An upright edge at the side of an object - a box's corner, the flank of a can - lies in a
    vertical plane through the camera, so on the table plane its pixels run straight out from the
    point under the camera: from its foot at distance r to r * A / (A - h), A being the camera's
    height above the plane.  The outermost pixels of each side are read that way.  Caveats: an
    edge lying along the line of sight looks just the same (too tall), and a rounded or flaring
    side ends before its top (too low).  A side cut off by the picture's edge or the robot is not
    read.
    """
    cam = sight.camera
    rows, cols = np.nonzero(sight.mask & ~sight.blind)
    above = float(cam.position[2] - floor)
    if len(rows) < 30 or above < 0.05:
        return []
    on = cam.on_plane(np.column_stack([cols, rows]).astype(float), floor)
    kept = np.isfinite(on).all(axis=1)
    on, rows, cols = on[kept], rows[kept], cols[kept]
    if len(on) < 30:
        return []
    high, wide = sight.mask.shape
    cut = cv2.dilate(sight.blind.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cut = cut[rows, cols] | (rows < 2) | (cols < 2) | (rows >= high - 2) | (cols >= wide - 2)
    out = on[:, :2] - cam.position[:2]
    reach = np.hypot(out[:, 0], out[:, 1])
    if reach.min() < SIDE_CLEAR:
        return []
    heading = math.atan2(out[:, 1].mean(), out[:, 0].mean())
    turn = (np.arctan2(out[:, 1], out[:, 0]) - heading + math.pi) % (2 * math.pi) - math.pi
    # one pixel, as a turn about the point under the camera
    step = float(np.median(np.linalg.norm(on - cam.position, axis=1) / cam.focal / reach))
    skip = max(2, int(0.003 * len(turn)))  # stray pixels beyond the side
    heights = []
    for t in (turn, -turn):
        edge = np.partition(t, skip - 1)[skip - 1]
        band = np.abs(t - edge - 0.25 * step) <= 0.75 * step
        if np.count_nonzero(band) >= 4 and not cut[band].any():
            near, far = np.percentile(reach[band], [2, 98])
            heights.append(above * (1.0 - near / far))
    return heights


def single(sight: Sight, floor: float = TABLE_Z, height: float | None = None, cell: float = 0.002) -> Shape:
    """Measure an object seen by one camera, assuming it rests at ``floor``.

    With ``height`` (known from before) the footprint is where the outline's cones at the bottom
    and at the top overlap.  Without it the height is guessed, from two readings that fail in
    different ways: the height that makes the footprint as deep (along the view) as it is wide, and
    what the outline's sides say (:func:`_side_heights`).  From a slanted view it is their mean
    (the first alone from low down); from straight above the first is taken as the footprint's
    width, and the lower one is kept.
    """
    out = Shape(views=[sight.cam], single=True, guessed=height is None)
    if not sight.mask.any():
        out.notes.append("empty outline")
        return out
    ys, xs = np.nonzero(sight.mask)
    ray = sight.camera.rays(np.array([[xs.mean(), ys.mean()]]))[0]
    from_above = -ray[2] > math.sin(math.radians(ABOVE_DEG))
    grid = _grid(sight, floor, floor + (height or (0.10 if from_above else 0.30)), cell)
    if grid is None:
        out.notes.append("the outline does not map onto the table")
        return out
    gx, gy = grid
    base = _cast(sight, gx, gy, floor)
    if not base.any():
        out.notes.append("the outline does not reach the table")
        return out
    along = np.array([float(gx[base].mean() - sight.camera.position[0]), float(gy[base].mean() - sight.camera.position[1])])
    along = along / max(np.linalg.norm(along), 1e-9)
    across = np.array([-along[1], along[0]])

    def footprint(h):
        both = base & _cast(sight, gx, gy, floor + h)
        if np.count_nonzero(both) < 3:
            return None
        return np.stack([gx[both], gy[both]], axis=1)

    if height is not None:
        h = float(height)
        pts = footprint(h)
        if pts is None:
            out.notes.append("an object of that height does not fit the outline")
            return out
    elif from_above:
        flat = np.stack([gx[base], gy[base]], axis=1)
        width = _footprint(flat)[2] + cell
        h = float(np.clip(min([width] + _side_heights(sight, floor)), 0.003, 0.10))
        pts = footprint(h)
        if pts is None:
            pts = flat
        out.notes.append("seen only from above: its height is not known")
    else:
        pts, h = None, None
        for trial in np.arange(0.005, 0.30, 0.0025):
            got = footprint(trial)
            if got is None:
                break
            pts, h = got, float(trial)
            depth = np.ptp(got @ along) + cell
            wide = np.ptp(got @ across) + cell
            if depth <= wide:
                break
        if pts is None:
            out.notes.append("no footprint fits the outline")
            return out
        sides = _side_heights(sight, floor) if -ray[2] > math.sin(math.radians(SIDE_MIN_DEG)) else []
        if sides:
            mean = float(np.clip(0.5 * (h + max(sides)), 0.003, 0.30))
            got = footprint(mean)
            if got is not None:
                pts, h = got, mean
        out.notes.append("seen by one camera: its height is a guess")
    (cx, cy), length, width, direction, is_round = _footprint(pts)
    out.top, out.bottom = floor + h, floor
    out.center = (float(cx), float(cy), float(floor + h / 2.0))
    out.size = (length + cell, width + cell, h)
    out.yaw = direction
    out.round = is_round
    out.surface = np.column_stack([pts, np.full(len(pts), floor + h)])
    out.explained[sight.cam] = 1.0
    # How far the footprint moves if the height is off by half: along the view by h/2/tan(elevation);
    # from above, radially by (distance from under the camera) * (h/2) / (camera height).
    pos = sight.camera.position
    if from_above:
        reach = math.hypot(cx - pos[0], cy - pos[1])
        slack = reach * 0.5 * h / max(pos[2] - floor, 0.05) if height is None else 0.001
    else:
        elevation = math.atan2(pos[2] - (floor + h / 2), math.hypot(cx - pos[0], cy - pos[1]))
        slack = (0.5 * h if height is None else 0.005) / max(math.tan(elevation), 0.2)
    pixel = 1000.0 * float(np.linalg.norm(np.array([cx, cy, floor + h]) - pos)) / sight.camera.focal
    out.err_mm = float(min(80.0, max(pixel, 1000.0 * slack)))
    return out


def surface_along(sights: list[Sight], idx: int, pixel, floor: float = TABLE_Z, top: float | None = None):
    """Where the ray of ``pixel`` in camera ``idx`` first meets the object's space (below ``top``),
    or None.  Returns (point, judged by another camera)."""
    box = _box(sights, floor - BELOW, floor + MAX_HEIGHT)
    if np.any(box[1] <= box[0]):
        return None, False
    space = _Space(sights, floor)
    res = _march(space, idx, np.asarray([pixel], float), box, step=0.001)
    if not res.hit[0]:
        return None, False
    point = res.entry[0]
    if top is not None and point[2] > top + 0.001:
        d = res.dirs[0]
        if d[2] >= 0:
            return None, False
        point = point + d * ((top - point[2]) / d[2])
        ins, _ = space.probe(point[None, :], -1)
        if not ins[0]:
            return None, False
    return point, bool(res.entry_ok[0])


# -- do two outlines show the same object? -------------------------------------------------------------
@dataclass
class Agreement:
    """How well two outlines in two cameras can show the same object."""

    inside: float  # share of the other outline's rays that pass through the first one's cone ...
    seen: float  # ... counted among this share of them: those the first camera can judge at all
    cover: float  # share of the first outline's rays through the other's cone (of those it can judge)
    depth: float  # median distance from the first camera to where its rays enter both cones (NaN: none)


def agreement(first: Sight, other: Sight, floor: float = TABLE_Z, rays: int = 200) -> Agreement:
    """Rays a camera cannot judge (the robot hides them, or they leave its picture) count for
    neither outline: an object half hidden by the arm in one picture can still match its whole
    outline in another."""
    sights = [first, other]
    box = _box(sights, floor - BELOW, floor + MAX_HEIGHT)
    if np.any(box[1] <= box[0]):
        return Agreement(0.0, 0.0, 0.0, float("nan"))
    space = _Space(sights, floor)
    theirs = _march(space, 1, _ray_pixels(other, rays), box)
    ours = _march(space, 0, _ray_pixels(first, rays), box)

    def share(res: _RayResult) -> tuple[float, float]:
        judged = int(np.count_nonzero(res.judged))
        if judged < 6:
            return 0.0, judged / max(len(res.pixels), 1)
        return float(np.count_nonzero(res.hit & res.judged)) / judged, judged / len(res.pixels)

    inside, seen = share(theirs)
    cover, _ = share(ours)
    entered = ours.hit & ours.entry_ok
    if np.count_nonzero(entered) >= 6:
        depth = float(np.median(np.linalg.norm(ours.entry[entered] - first.camera.position, axis=1)))
    else:
        depth = float("nan")
    return Agreement(inside, seen, cover, depth)


# -- could the object reach further than measured? -------------------------------------------------------
#: Past each side of a measured object the space is checked in slabs this thick (m), up to REACH_MAX.
REACH_STEP = 0.005
REACH_MAX = 0.06
#: A slab is clear when some camera shows at least this share of its points outside the outline.
CLEAR_SHARE = 0.25


@dataclass
class Hidden:
    """A side of a measured object past which no camera can see whether it goes on."""

    normal: np.ndarray  # outward direction (x, y)
    reach: float  # how far (m) past that side it could go on unseen
    why: dict[str, str]  # cam -> 'robot' (behind the robot), 'edge' (outside the picture) or 'outline'
    points: np.ndarray  # sample points just past the side


def hidden_reach(sights: list[Sight], shape: Shape) -> list[Hidden]:
    """Sides of a measured object past which no camera can see whether it goes on.

    The measured box holds what the cameras saw of the object; where every camera's view past one
    side is blocked by the robot, or leaves the picture, more of the object may be hidden there.
    """
    if not shape.ok:
        return []
    cx, cy = shape.center[0], shape.center[1]
    length, width = shape.size[0], shape.size[1]
    lo = shape.bottom + 0.003
    hi = max(shape.top - 0.003, lo)
    zs = np.linspace(lo, hi, 4)
    if shape.round:
        angles = np.radians(np.arange(0, 360, 45))
        sides = [(np.array([math.cos(a), math.sin(a)]), 0.5 * length, 0.35 * length) for a in angles]
    else:
        t = math.radians(shape.yaw or 0.0)
        u = np.array([math.cos(t), math.sin(t)])
        v = np.array([-math.sin(t), math.cos(t)])
        sides = [(u, 0.5 * length, 0.5 * width), (-u, 0.5 * length, 0.5 * width),
                 (v, 0.5 * width, 0.5 * length), (-v, 0.5 * width, 0.5 * length)]
    out = []
    for normal, offset, half in sides:
        along = np.array([-normal[1], normal[0]])
        spread = np.linspace(-0.8 * half, 0.8 * half, 5)
        reach, first = 0.0, None
        why: dict[str, str] = {}
        d = 0.0
        while d < REACH_MAX - 1e-9:
            base = np.array([cx, cy]) + (offset + d + 0.5 * REACH_STEP) * normal
            xy = base[None, :] + spread[:, None] * along[None, :]
            pts = np.column_stack([np.repeat(xy, len(zs), axis=0), np.tile(zs, len(xy))])
            clear = False
            for s in sights:
                code = s.look(pts)
                if np.mean(code == OUT) >= CLEAR_SHARE:
                    clear = True
                    break
                if s.cam not in why:
                    if np.mean(code == BLIND) >= 0.5:
                        px, depth = s.camera.project(pts)
                        seen = (depth > 0.02) & (px[:, 0] >= 0) & (px[:, 0] <= s.camera.width - 1) \
                            & (px[:, 1] >= 0) & (px[:, 1] <= s.camera.height - 1)
                        why[s.cam] = "robot" if seen.mean() >= 0.5 else "edge"
                    else:
                        why[s.cam] = "outline"
            if clear:
                break
            if first is None:
                first = pts
            d += REACH_STEP
            reach = d
        if reach >= REACH_STEP:
            out.append(Hidden(normal, reach, why, first))
    return out
