# Copyright 2026 PyRUA-Lean Contributors.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Selected generic View/Sight/measure closure; see PROVENANCE.json.
# Modified: explicit supporting-plane height required, otherwise ValueError.
# Omitted: SDK camera setup, robot rendering, measured finger geometry, task constants.
from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np

IMAGE_SHAPE = (480, 640)

GRID_CELLS = 40000

CELL_MIN_M = 0.001

LEVEL_M = 0.002

SWEEP_EVERY = 2

SWEEP_MAX_M = 0.36

SECOND_VIEW_MIN = 0.55

SECOND_VIEW_WITHIN = 0.8

SECOND_VIEW_MAX_SHARE = 0.3

SECOND_VIEW_ON_ROBOT = 0.85

LONE_COVER = 0.985

LONE_LOST_PX = 6.0

LONE_LOST_MIN = 0.003

TALLER_GAIN = 0.02

HULL_BELOW = 0.9

CARVE_KEEPS = 0.85

KNOWN_BETTER = 0.03

KNOWN_FITS = 0.9

KNOWN_SLACK = (1.25, 0.01)

EDGE_LEVEL = 0.3

HANG_MARGIN = 0.03

PART_GAP_PX = 12

MIN_PIXELS = 12

TRACE: list[dict[str, Any]] | None = None

MIN_CELLS = 6

class View:
    """A pinhole camera at a known pose (Isaac's convention: it looks down its -z, +y is up)."""

    def __init__(
        self,
        intrinsic: Any,
        extrinsic: Any,
        shape: tuple[int, int] = IMAGE_SHAPE,
    ) -> None:
        k = np.asarray(intrinsic, dtype=np.float64)
        self.pose = np.asarray(extrinsic, dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        self.shape = (int(shape[0]), int(shape[1]))
        self.pos = self.pose[:3, 3]
        self.rot = self.pose[:3, :3]

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, depth along the optical axis) of world points; depth <= 0 is behind."""

        local = (np.asarray(xyz, dtype=np.float64) - self.pos) @ self.rot
        depth = -local[..., 2]
        safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
        return (
            -self.fy * local[..., 1] / safe + self.cy,
            self.fx * local[..., 0] / safe + self.cx,
            depth,
        )

    def rays(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """World direction of each pixel's ray, per unit of depth along the optical axis."""

        local = np.stack(
            [
                (np.asarray(cols, dtype=np.float64) - self.cx) / self.fx,
                -(np.asarray(rows, dtype=np.float64) - self.cy) / self.fy,
                -np.ones(np.shape(rows)),
            ],
            axis=-1,
        )
        return local @ self.rot.T

    def on_plane(
        self, rows: np.ndarray, cols: np.ndarray, z: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Where each pixel's ray meets the horizontal plane ``z`` (N x 2), and which rays do."""

        d = self.rays(rows, cols)
        down = d[..., 2]
        ok = down < -1e-6
        t = np.where(ok, (z - self.pos[2]) / np.where(ok, down, -1.0), 0.0)
        ok &= t > 0
        return self.pos[:2] + d[..., :2] * t[..., None], ok

@dataclass
class Sight:
    """What one camera shows of the object: its mask, and the pixels that cannot tell."""

    view: View
    mask: np.ndarray
    unknown: Any = None
    #: the robot's pixels as given, before the mask was taken out of them
    robot: Any = None

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        self._object = self.mask.astype(np.float32)
        if self.unknown is None:
            self.unknown = np.zeros(self.mask.shape, dtype=bool)
        self.robot = np.asarray(self.unknown, dtype=bool)
        # the segmenter's word stands where it marked the object; the rest cannot tell
        self.unknown = self.robot & ~self.mask
        self._maybe: Any = None

    def _sample(self, field: np.ndarray, xyz: np.ndarray, outside: float) -> np.ndarray:
        r, c, depth = self.view.project(xyz)
        h, w = field.shape
        ok = (depth > 1e-6) & (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
        r0 = np.clip(np.floor(np.where(ok, r, 0.0)).astype(np.int64), 0, h - 2)
        c0 = np.clip(np.floor(np.where(ok, c, 0.0)).astype(np.int64), 0, w - 2)
        fr, fc = np.clip(r - r0, 0.0, 1.0), np.clip(c - c0, 0.0, 1.0)
        val = (
            field[r0, c0] * (1 - fr) * (1 - fc)
            + field[r0 + 1, c0] * fr * (1 - fc)
            + field[r0, c0 + 1] * (1 - fr) * fc
            + field[r0 + 1, c0 + 1] * fr * fc
        )
        return np.where(ok, val, outside)

    def shows(self, xyz: np.ndarray, level: float = 0.5) -> np.ndarray:
        """Points that project onto the object's own pixels.

        ``level`` is where the mask's edge is drawn between the last pixel inside and the first
        one outside: 0.5 half way, lower further out.
        """

        return self._sample(self._object, xyz, 0.0) >= level

    def allows(self, xyz: np.ndarray, level: float = 0.5) -> np.ndarray:
        """Points that may belong to the object: on its mask, on pixels that cannot tell, or
        off the image."""

        if self._maybe is None:
            self._maybe = (self.mask | self.unknown).astype(np.float32)
        return self._sample(self._maybe, xyz, 1.0) >= level

    def in_image(self, xyz: np.ndarray) -> np.ndarray:
        r, c, depth = self.view.project(xyz)
        h, w = self.mask.shape
        return (depth > 1e-6) & (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)

class _Grid:
    """Cells of the supporting plane around an object's shadow."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, base_z: float) -> None:
        area = float(np.prod(np.maximum(hi - lo, 1e-6)))
        self.cell = max(CELL_MIN_M, math.sqrt(area / GRID_CELLS))
        self.lo = lo
        self.nx = max(2, int(math.ceil((hi[0] - lo[0]) / self.cell)) + 1)
        self.ny = max(2, int(math.ceil((hi[1] - lo[1]) / self.cell)) + 1)
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self.xy = lo + np.stack([gx.ravel(), gy.ravel()], axis=1) * self.cell
        self.base_z = base_z

    def at(self, height: float, cells: np.ndarray | None = None) -> np.ndarray:
        xy = self.xy if cells is None else self.xy[cells]
        return np.c_[xy, np.full(len(xy), self.base_z + height)]

    def index(self, xy: np.ndarray) -> np.ndarray:
        """The cell each plane point falls into (-1: off the grid)."""

        ix = np.rint((xy[..., 0] - self.lo[0]) / self.cell).astype(np.int64)
        iy = np.rint((xy[..., 1] - self.lo[1]) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.where(ok, iy * self.nx + ix, -1)

    def holds(self, cells: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Which plane points fall into one of the given cells (a flat boolean per cell)."""

        ix = np.rint((xy[..., 0] - self.lo[0]) / self.cell).astype(np.int64)
        iy = np.rint((xy[..., 1] - self.lo[1]) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.zeros(np.shape(ix), dtype=bool)
        out[ok] = cells[iy[ok] * self.nx + ix[ok]]
        return out

class _Solid:
    """What a reading says the object is: over each cell, a column from ``low`` to ``high``
    (metres above the plane; NaN where there is none)."""

    def __init__(self, grid: _Grid, low: np.ndarray, high: np.ndarray) -> None:
        self.grid, self.low, self.high = grid, low, high
        self.cells = np.isfinite(high)

    def spans(self, xy: np.ndarray, z: np.ndarray, slack: float = 0.0) -> np.ndarray:
        """Which points (plane position and height above the plane) lie in the solid."""

        index = self.grid.index(xy)
        on = index >= 0
        low = np.where(on, self.low[np.where(on, index, 0)], np.nan)
        high = np.where(on, self.high[np.where(on, index, 0)], np.nan)
        with np.errstate(invalid="ignore"):
            return on & (z >= low - slack) & (z <= high + slack)

def _march(view: View, rows: np.ndarray, cols: np.ndarray, solid: _Solid) -> np.ndarray:
    """The depth along each pixel's ray at which it first meets the solid (NaN: never)."""

    grid = solid.grid
    d = view.rays(rows, cols)
    pos = view.pos
    lo_t = np.zeros(len(rows))
    hi_t = np.full(len(rows), 50.0)
    inside = solid.cells
    floor = float(np.nanmin(solid.low[inside])) if inside.any() else 0.0
    roof = float(np.nanmax(solid.high[inside])) if inside.any() else 0.0
    box_lo = np.array([grid.lo[0], grid.lo[1], grid.base_z + floor - 1e-4])
    box_hi = np.array(
        [
            grid.lo[0] + grid.nx * grid.cell,
            grid.lo[1] + grid.ny * grid.cell,
            grid.base_z + roof + 1e-4,
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis in range(3):
            t1 = (box_lo[axis] - pos[axis]) / d[:, axis]
            t2 = (box_hi[axis] - pos[axis]) / d[:, axis]
            flat = np.abs(d[:, axis]) < 1e-12
            within = (pos[axis] >= box_lo[axis]) & (pos[axis] <= box_hi[axis])
            near = np.where(flat, np.where(within, -np.inf, np.inf), np.minimum(t1, t2))
            far = np.where(flat, np.where(within, np.inf, -np.inf), np.maximum(t1, t2))
            lo_t, hi_t = np.maximum(lo_t, near), np.minimum(hi_t, far)
    alive = hi_t >= lo_t
    first = np.full(len(rows), np.nan)
    if not alive.any() or not inside.any():
        return first
    run = np.linalg.norm(d, axis=1) * np.where(alive, hi_t - lo_t, 0.0)
    steps = int(min(400, max(2, math.ceil(float(run.max()) / (grid.cell * 0.8)) + 1)))
    slack = grid.cell * 0.5
    for f in np.linspace(0.0, 1.0, steps):
        t = np.where(alive, lo_t + (hi_t - lo_t) * f, 0.0)
        at = pos + d * t[:, None]
        hit = alive & np.isnan(first) & solid.spans(at[:, :2], at[:, 2] - grid.base_z, slack)
        first[hit] = t[hit]
    return first

def _match(sight: Sight, solid: _Solid, pixels: tuple[np.ndarray, np.ndarray]) -> float | None:
    """How well a solid agrees with the object's mask in another view (F1).

    ``None``: this view cannot say (the solid lies outside its image, or on unknown pixels).
    """

    cells = solid.cells
    if cells.sum() < MIN_CELLS:
        return 0.0
    grid = solid.grid
    low, high = solid.low[cells], solid.high[cells]
    inside = total = 0
    for f in (0.0, 0.5, 1.0):
        pts = np.c_[grid.xy[cells], grid.base_z + low + (high - low) * f]
        seen = sight.in_image(pts)
        known = seen & ~sight.unknown[_rc(sight, pts, seen)]
        total += int(known.sum())
        inside += int(sight.shows(pts[known]).sum())
    if total < MIN_CELLS:
        return None
    precision = inside / total
    if len(pixels[0]) == 0:
        return 0.0
    recall = float(np.isfinite(_march(sight.view, pixels[0], pixels[1], solid)).mean())
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _rc(sight: Sight, pts: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c, _ = sight.view.project(pts)
    h, w = sight.mask.shape
    ri = np.clip(np.rint(np.where(seen, r, 0.0)).astype(np.int64), 0, h - 1)
    ci = np.clip(np.rint(np.where(seen, c, 0.0)).astype(np.int64), 0, w - 1)
    return ri, ci

class _Levels:
    """Which points over the grid the head mask allows, level by level."""

    def __init__(self, head: Sight, grid: _Grid, top_height: float) -> None:
        self.grid = grid
        count = int(top_height / LEVEL_M) + 1
        rows = [head.shows(grid.at(0.0), EDGE_LEVEL)]
        for j in range(1, count):
            row = head.allows(grid.at(j * LEVEL_M), EDGE_LEVEL)
            if not row.any() and not rows[-1].any():
                break
            rows.append(row)
        self.inside = np.array(rows)  # [level, cell]
        self.count = len(rows)
        # how many levels from the plane up are inside without a break (0: the foot is not)
        self.rise = np.logical_and.accumulate(self.inside, axis=0).sum(axis=0)
        # for each level, how many levels down from it are inside without a break
        self.drop = np.zeros(self.inside.shape, dtype=np.int32)
        run = np.zeros(self.inside.shape[1], dtype=np.int32)
        for j in range(self.count):
            run = np.where(self.inside[j], run + 1, 0)
            self.drop[j] = run

    def cone(self) -> _Solid:
        """Everything the head mask allows over the grid: per cell, from the lowest level that
        is inside to the highest."""

        any_in = self.inside.any(axis=0)
        first = self.inside.argmax(axis=0)
        last = self.count - 1 - self.inside[::-1].argmax(axis=0)
        return _Solid(
            self.grid,
            np.where(any_in, first * LEVEL_M, np.nan),
            np.where(any_in, last * LEVEL_M, np.nan),
        )

    def prism(self, level: int, rise: np.ndarray | None = None) -> _Solid:
        """What stands on the plane, no taller than ``level``: over each cell of the shadow a
        column as high as the mask allows it there.  For a box or a can that is the box or the
        can; a bottle keeps its shoulders and a mug its handle, each at its own height.
        ``rise`` replaces the head mask's word on how high each column may go (the least that
        every view allows)."""

        rise = self.rise if rise is None else rise
        on = rise > 0
        top = np.minimum(rise - 1, level) * LEVEL_M
        return _Solid(self.grid, np.where(on, 0.0, np.nan), np.where(on, top, np.nan))

    def block(self, level: int) -> _Solid:
        """The cells whose column reaches ``level``, all cut there: the upright prism proper."""

        on = self.rise > level
        return _Solid(
            self.grid,
            np.where(on, 0.0, np.nan),
            np.where(on, level * LEVEL_M, np.nan),
        )

    def rise_in(self, sight: Sight, level: int) -> np.ndarray:
        """How many levels from the plane up another view allows over each cell, counted up to
        ``level`` and only where the head mask has a column at all."""

        out = np.zeros(len(self.grid.xy), dtype=np.int64)
        alive = self.rise > 0
        for j in range(min(level, self.count - 1) + 1):
            if not alive.any():
                break
            ok = sight.allows(self.grid.at(j * LEVEL_M, alive), EDGE_LEVEL)
            index = np.flatnonzero(alive)
            out[index[ok]] = j + 1
            alive[index[~ok]] = False
        return out

    def hang(self, level: int) -> _Solid:
        on = self.inside[level]
        low = (level - self.drop[level] + 1) * LEVEL_M
        return _Solid(
            self.grid,
            np.where(on, low, np.nan),
            np.where(on, level * LEVEL_M, np.nan),
        )

    def solid(self, kind: str, level: int) -> _Solid:
        """The solid a reading is judged by in another view."""

        return self.block(level) if kind == "prism" else self.hang(level)

def _head_cover(head: Sight, solid: _Solid, shadow: np.ndarray) -> float:
    """The share of the head mask's shadow that a prism casts."""

    cells = solid.cells
    if cells.sum() < MIN_CELLS:
        return 0.0
    grid = solid.grid
    height = float(np.nanmax(solid.high[cells]))
    foot = head.view.pos[:2]
    above = head.view.pos[2] - grid.base_z
    scale = above / max(above - height, 1e-6)  # a point at `height` shadows this much further out
    hit = np.zeros(len(shadow), dtype=bool)
    reach = float(np.linalg.norm(shadow - foot, axis=1).max()) * (1.0 - 1.0 / scale)
    for s in np.linspace(1.0 / scale, 1.0, max(2, int(reach / grid.cell) + 2)):
        hit |= grid.holds(cells, foot + (shadow - foot) * s)
    return float(hit.mean())

def largest_part(mask: np.ndarray, gap_px: int = PART_GAP_PX) -> np.ndarray:
    """A mask's largest connected part, with the parts that lie within ``gap_px`` of it.

    A text prompt that fits two objects comes back as one mask over both; a finger across an
    object cuts its mask in two.  The first is two objects, the second one.
    """

    mask = np.asarray(mask, dtype=bool)
    rows = np.nonzero(mask.any(axis=1))[0]
    runs: list[tuple[int, int, int]] = []  # row, first col, last col
    for r in rows:
        line = np.flatnonzero(np.diff(np.r_[0, mask[r].astype(np.int8), 0]))
        runs.extend(
            (int(r), int(a), int(b) - 1) for a, b in zip(line[::2], line[1::2], strict=True)
        )
    if len(runs) <= 1:
        return mask
    parent = list(range(len(runs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_row: dict[int, list[int]] = {}
    for i, (r, _, _) in enumerate(runs):
        by_row.setdefault(r, []).append(i)
    for i, (r, a, b) in enumerate(runs):
        for j in by_row.get(r - 1, ()):
            if runs[j][1] <= b + 1 and runs[j][2] >= a - 1:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(runs))]
    size: dict[int, int] = {}
    box: dict[int, list[int]] = {}
    for root, (r, a, b) in zip(roots, runs, strict=True):
        size[root] = size.get(root, 0) + b - a + 1
        bb = box.setdefault(root, [r, r, a, b])
        bb[0], bb[1], bb[2], bb[3] = min(bb[0], r), max(bb[1], r), min(bb[2], a), max(bb[3], b)
    if len(size) == 1:
        return mask
    main = max(size, key=size.get)  # type: ignore[arg-type]
    m = box[main]
    keep = {
        root
        for root, bb in box.items()
        if max(bb[0] - m[1], m[0] - bb[1], 0) <= gap_px
        and max(bb[2] - m[3], m[2] - bb[3], 0) <= gap_px
    }
    out = np.zeros_like(mask)
    for root, (r, a, b) in zip(roots, runs, strict=True):
        if root in keep:
            out[r, a : b + 1] = True
    return out

def _smear_height(head: Sight, shadow: np.ndarray, base_z: float) -> float | None:
    """An upright object's height from one mask alone.

    Seen from the point under the camera, the mask's shadow on the plane ends to either side in a
    stretch that runs straight away from that point: the shadow of an upright edge (or of the
    side of a can), whose near end is on the plane and whose far end is that edge's top.
    """

    rel = shadow - head.view.pos[:2]
    rho = np.hypot(rel[:, 0], rel[:, 1])
    theta = np.arctan2(rel[:, 0], rel[:, 1])
    if len(rho) < 30 or rho.min() <= 1e-6:
        return None
    width = 1.5 / head.view.fx  # what a pixel and a half spans, seen from the camera
    scales = []
    for side in (np.percentile(theta, 1.5), np.percentile(theta, 98.5)):
        near = np.abs(theta - side) <= width
        if near.sum() >= 4:
            scales.append(np.percentile(rho[near], 95) / np.percentile(rho[near], 5))
    if not scales:
        return None
    above = float(head.view.pos[2] - base_z)
    return above * (1.0 - 1.0 / float(np.median(scales)))

def _seen(sights: Sequence[tuple[Sight, Any]], solid: _Solid) -> float:
    """The share of the other views' mask pixels whose ray meets the solid."""

    marks = [float(np.isfinite(_march(s.view, px[0], px[1], solid)).mean()) for s, px in sights]
    return float(np.mean(marks)) if marks else 0.0

def _taller(
    levels: _Levels, rise: np.ndarray, level: int, trusted: Sequence[tuple[Sight, Any]]
) -> int:
    """The level an object's top is at, given the level of its body.

    A prism is judged at one height, and for a bottle the best one is its shoulder: the neck
    above it is thin.  The columns every view allows go higher there.  They also go higher
    along the edge of a box that faces the cameras, where nothing shows whether the box ends:
    so the top is raised only as far as it lets the solid account for more of the other views'
    masks (a neck does, that edge does not).
    """

    last = int(rise.max()) - 1
    if last <= level:
        return level
    most = _seen(trusted, levels.prism(last, rise))
    if most - _seen(trusted, levels.prism(level, rise)) < TALLER_GAIN:
        return level
    low, high = level, last  # the lowest top that accounts for nearly all the tallest does
    while high - low > 1:
        mid = (low + high) // 2
        if most - _seen(trusted, levels.prism(mid, rise)) < TALLER_GAIN / 2:
            high = mid
        else:
            low = mid
    return high

def measure(
    head: Sight,
    others: Sequence[Sight | Sequence[Sight]] = (),
    base_z: float | None = None,
    known: Sequence[Any] | None = None,
) -> dict[str, Any] | None:
    """Where an object's surface is, from its mask in one view and, if any, in others.

    ``head`` is the view the object is read from (a camera that looks down on it).  ``others``
    see the same object from elsewhere and settle its height; an entry may be a list of masks
    from one camera, of which the one that fits is taken.  ``known`` is the object's height,
    reading and match from an earlier measurement that two views agreed on (``(height_m, kind)``
    or ``(height_m, kind, match)``): it is used when no other view sees the object now, and
    when the views that do agree on it less well than those did; never when a prism that tall
    no longer fits the mask (the object has fallen over).  Returns ``points`` (N x 3), the world
    point each pixel of the head mask shows on the solid the reading describes (what a depth
    image would have given, for an object of that shape) with those pixels (``rows``,
    ``cols``), its ``height_m`` above ``base_z``, the reading taken (``kind``: ``"prism"`` or
    ``"hang"``), the number of ``views`` that agreed and their ``match``, whether the height is
    the ``remembered`` one, and whether a known height was ``outgrown`` (it no longer fits the
    mask); or ``None`` when the mask gives no footprint.
    """

    if base_z is None or not np.isfinite(base_z):
        raise ValueError("Explicit public supporting-plane height required")
    head = Sight(head.view, largest_part(head.mask))
    rows, cols = np.nonzero(head.mask)
    if len(rows) < MIN_PIXELS:
        return None
    shadow, ok = head.view.on_plane(rows, cols, base_z)
    shadow = shadow[ok]
    above = float(head.view.pos[2] - base_z)
    if len(shadow) < MIN_PIXELS or above <= 0.05:
        return None
    foot = head.view.pos[:2]
    top_height = min(SWEEP_MAX_M, above - 0.05)
    # every reading lies between the shadow and the shadow drawn in towards the camera's foot
    shrunk = foot + (shadow - foot) * (1.0 - top_height / above)
    lo = np.minimum(shadow.min(axis=0), shrunk.min(axis=0)) - 0.006
    hi = np.maximum(shadow.max(axis=0), shrunk.max(axis=0)) + 0.006
    if np.any(hi - lo > 1.6):
        return None
    grid = _Grid(lo, hi, base_z)
    levels = _Levels(head, grid, top_height)
    if levels.inside[0].sum() < MIN_CELLS:
        return None
    full_shadow = shadow
    if len(shadow) > 2500:
        shadow = shadow[:: len(shadow) // 2500 + 1]

    cone = levels.cone()
    seconds: list[list[tuple[Sight, tuple[np.ndarray, np.ndarray]]]] = []
    for entry in others:
        candidates = []
        for sight in entry if isinstance(entry, (list, tuple)) else [entry]:
            r2, c2 = np.nonzero(sight.mask)
            if len(r2) < MIN_PIXELS or len(r2) > SECOND_VIEW_MAX_SHARE * sight.mask.size:
                continue
            if sight.robot is not None and sight.robot[r2, c2].mean() > SECOND_VIEW_ON_ROBOT:
                continue
            if len(r2) > 1500:
                pick = slice(None, None, len(r2) // 1500 + 1)
                r2, c2 = r2[pick], c2[pick]
            # the object's mask in another view lies within what the head mask allows
            within = float(np.isfinite(_march(sight.view, r2, c2, cone)).mean())
            if within >= SECOND_VIEW_WITHIN:
                candidates.append((sight, (r2, c2)))
        if candidates:
            seconds.append(candidates)

    kinds = ("prism", "hang")
    tried = list(range(0, levels.count, SWEEP_EVERY))
    solids: dict[tuple[str, int], _Solid] = {}

    def solid_of(kind: str, level: int) -> _Solid:
        if (kind, level) not in solids:
            solids[kind, level] = levels.solid(kind, level)
        return solids[kind, level]

    # each other view by itself first: a mask of something else matches no reading well.  Of
    # several masks from one camera the one that matches best is the object's.
    trusted = []
    coarse = tried[::2] if len(tried) > 12 else tried
    for candidates in seconds:
        ranked = []
        for sight, px in candidates:
            marks = [
                m
                for kind in kinds
                for j in coarse
                if (m := _match(sight, solid_of(kind, j), px)) is not None
            ]
            if marks:
                ranked.append((max(marks), len(ranked), sight, px))
        if TRACE is not None:
            TRACE.append({"ranked": [(round(m, 3), i, int(s.mask.sum())) for m, i, s, _ in ranked]})
        if ranked and max(ranked)[0] >= SECOND_VIEW_MIN:
            _, _, sight, px = max(ranked)
            trusted.append((sight, px))

    def score(kind: str, level: int) -> float | None:
        marks = [_match(s, solid_of(kind, level), px) for s, px in trusted]
        marks = [m for m in marks if m is not None]
        return float(np.mean(marks)) if marks else None

    best: dict[str, tuple[float, int]] = {}
    if trusted:
        for kind in kinds:
            for j in tried:
                mark = score(kind, j)
                if mark is not None and (kind not in best or mark > best[kind][0] + 1e-9):
                    best[kind] = (mark, j)
        for kind, (_, coarse) in list(best.items()):
            for j in range(
                max(0, coarse - SWEEP_EVERY + 1), min(levels.count, coarse + SWEEP_EVERY)
            ):
                mark = score(kind, j)
                if mark is not None and mark > best[kind][0] + 1e-9:
                    best[kind] = (mark, j)
    if TRACE is not None:
        TRACE.append(
            {"best": {k: (round(m, 3), j) for k, (m, j) in best.items()}, "levels": levels.count}
        )
        if trusted:
            TRACE.append(
                {
                    "sweep": {
                        kind: [(j, round(m, 3)) for j in tried if (m := score(kind, j)) is not None]
                        for kind in kinds
                    }
                }
            )
    # the height measured before, if a solid that tall still fits this mask
    recalled: tuple[str, int] | None = None
    if known is not None:
        k_kind = "hang" if known[1] == "hang" else "prism"
        k_level = int(min(levels.count - 1, max(0, round(float(known[0]) / LEVEL_M))))
        whole = _head_cover(head, levels.block(0), shadow)
        above = int(round((float(known[0]) * KNOWN_SLACK[0] + KNOWN_SLACK[1]) / LEVEL_M))
        if k_kind == "hang":
            # every height plane cuts the cone of the mask's rays: nothing to check a rim's by
            fits = int(levels.inside[k_level].sum()) >= MIN_CELLS
        else:
            fits = _head_cover(head, levels.block(k_level), shadow) >= KNOWN_FITS * whole
            if fits and above < levels.count:
                fits = _head_cover(head, levels.block(above), shadow) < KNOWN_FITS * whole
        if fits:
            recalled = (k_kind, k_level)
    remembered = False
    if best:
        # the hanging reading only where it explains the other views clearly better: anything
        # that stands on its own footprint is a prism
        kind = "prism" if "prism" in best else "hang"
        if "hang" in best and (
            "prism" not in best or best["hang"][0] > best["prism"][0] + HANG_MARGIN
        ):
            kind = "hang"
        mark, level = best[kind]
        views = 1 + len(trusted)
        before = known[2] if known is not None and len(known) > 2 else None
        if recalled is not None and before is not None and mark < float(before) - KNOWN_BETTER:
            # the cameras agreed better then than they do now (a view cut by the picture's
            # edge, a finger in the way): what they said then stands
            (kind, level), mark, views, trusted, remembered = recalled, None, 1, [], True
    elif recalled is not None:
        (kind, level), mark, views, remembered = recalled, None, 1, True
    else:
        # one view: the tallest prism that still casts the whole shadow, no taller than the
        # shadow's straight sides say
        kind, mark, views = "prism", None, 1
        level = 0
        limit = _smear_height(head, full_shadow, base_z)
        limit = top_height if limit is None else min(top_height, max(limit, 0.0) + 0.004)
        full = _head_cover(head, levels.block(0), shadow)
        # what a prism too tall loses is a few pixels at the shadow's corners, whatever its size
        lost = min(1.0 - LONE_COVER, max(LONE_LOST_MIN, LONE_LOST_PX / len(rows)))
        for j in range(1, min(levels.count, int(limit / LEVEL_M) + 1)):
            trial = levels.block(j)
            if trial.cells.sum() < MIN_CELLS:
                break
            if _head_cover(head, trial, shadow) < (1.0 - lost) * full:
                break
            level = j
    if kind == "prism":
        rise = levels.rise
        body = levels.block(level)
        if trusted and _seen(trusted, body) < HULL_BELOW:
            # the plain prism leaves part of what the other views show unexplained: take the
            # columns every view allows instead (a bottle's shoulders and a mug's handle stand
            # at their own height beside what reaches the top), and look for a top above
            carved = rise
            for sight, _px in trusted:
                carved = np.minimum(carved, levels.rise_in(sight, levels.count - 1))
            if body.cells.any() and float((carved[body.cells] > 0).mean()) >= CARVE_KEEPS:
                rise = carved
                level = _taller(levels, rise, level, trusted)
        solid = levels.prism(level, rise)
    else:
        solid = solid_of(kind, level)
    if solid.cells.sum() < MIN_CELLS:
        return None
    depth = _march(head.view, rows, cols, solid)
    got = np.isfinite(depth)
    if got.sum() < MIN_PIXELS:
        return None
    points = head.view.pos + head.view.rays(rows[got], cols[got]) * depth[got, None]
    return {
        "points": points,
        "rows": rows[got],
        "cols": cols[got],
        "height_m": float(level * LEVEL_M),
        "kind": kind,
        "views": views,
        "match": mark,
        "remembered": remembered,
        # a height was known and a solid that tall no longer fits the mask
        "outgrown": known is not None and recalled is None,
        "cell_m": grid.cell,
    }
