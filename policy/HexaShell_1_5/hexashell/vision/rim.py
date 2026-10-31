"""The rim of a round open dish (a bowl, a deep plate, a pot) from the depth points around it.

A pinch across the dish's wall needs the outer edge of its rim where the hand comes in: an object's
outline (:mod:`.rgbd`) leaves its outermost pixels out, so its size runs a few centimetres short of
the rim.  Here every depth point near the dish counts: the surface it stands on is the lowest level
well covered with points, the dish is the connected part rising above that surface (taller
neighbours, a box or an appliance beside it, left out; of the parts left, the widest whose outline
from above holds the point given, or that reaches it), its outline from above is the circle through
most of the part's outermost points around that point (something low joined to it, a tray or a
second bowl touching it, falls off that circle), and the rim's edge on the near side is the
outermost of its highest points in a strip along the line from the hand's side to the dish's middle (``outer``:
of its points next to the outermost one, for the bottom dish of a stack, whose rim stands lower than the rim of
the dish set in it, a little further in).

World frame, metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: the surface under the dish: points within this (m, across) of the point given, in levels of LEVEL_M
SURFACE_R = 0.22
LEVEL_M = 0.006
LEVEL_MIN = 25
#: the dish: points from OVER_M to RISE_M over that surface within REACH_M (across) of the point given (of its
#: middle, when what is found there reaches that far: a point given near its rim)
OVER_M = 0.012
RISE_M = 0.22
REACH_M = 0.24
#: on a grid of CELL_M; a cell whose points rise more than TALL_M over the surface is a neighbour; parts
#: (8-connected cells of 2 points or more) of fewer than PART_MIN points are specks
CELL_M = 0.02
TALL_M = 0.15
PART_MIN = 30
#: the near side: points within STRIP_M of the line through the middle, within EDGE_M of the outermost
#: one, and the rim there: within TOP_M of the highest of them
STRIP_M = 0.02
EDGE_M = 0.025
TOP_M = 0.01
#: outer: the rim's top from the strip's points within OUTER_M of the outermost one (a smaller dish set in this one
#: shows its rim higher, from the gap between the two rims in: in StackBowlsCabinet scenes 1.3-5 cm)
OUTER_M = 0.01
#: dishes are this wide across the top (m) and this tall over what they stand on
RADIUS_RANGE = (0.04, 0.22)
HEIGHT_RANGE = (0.015, 0.15)
#: the dish's outline from above: the part's outermost point in each SECTOR_DEG around the point given, and the circle
#: (of RING_TRIES drawn through three of them) that the most of them lie within RING_M of and that holds that point.
#: A neighbour joined to the dish in the part (a machine's low tray, a second bowl touching it) only adds points off
#: that circle; with fewer than RING_SHARE of them on it, the part's extent stands for the outline
SECTOR_DEG = 5.0
RING_M = 0.01
RING_SHARE = 0.4
RING_TRIES = 300


@dataclass
class Rim:
    center: tuple[float, float]  # the middle of the dish's outline from above
    radius: float  # half its width across, at the widest (the rim, for a dish that flares)
    top: float  # the rim's height on the near side
    on_z: float  # the surface it stands on
    edge: tuple[float, float, float]  # the outer edge of the rim on the near side
    toward: tuple[float, float]  # the unit direction from the near side to the middle (across)
    points: int

    @property
    def height(self) -> float:
        return self.top - self.on_z


def surface_under(points: np.ndarray, p) -> float | None:
    """Height of the surface under ``p``: the lowest level within SURFACE_R of it, up to 25 cm under
    it, that holds at least LEVEL_MIN points and a quarter of the fullest level's."""
    p = np.asarray(p, float)
    near = points[(np.hypot(points[:, 0] - p[0], points[:, 1] - p[1]) < SURFACE_R)
                  & (points[:, 2] > p[2] - 0.25) & (points[:, 2] < p[2] + 0.01)]
    if len(near) < LEVEL_MIN:
        return None
    hist, edges = np.histogram(near[:, 2], bins=np.arange(p[2] - 0.25, p[2] + 0.01 + LEVEL_M, LEVEL_M))
    if hist.max() < LEVEL_MIN:
        return None
    k = int(np.nonzero(hist >= max(LEVEL_MIN, 0.25 * hist.max()))[0][0])
    band = near[(near[:, 2] >= edges[k] - LEVEL_M) & (near[:, 2] <= edges[k + 1] + LEVEL_M)]
    return float(np.median(band[:, 2]))


def rim_of(points: np.ndarray, p, side, outer: bool = False) -> tuple[Rim | None, str]:
    """The rim of the dish at ``p`` (x, y, z: a point on it, e.g. a measured centre at its top), its
    near side the one facing ``side`` (x, y: where the hand comes from); ``outer``: the outermost rim
    there, not the highest (a stack's bottom dish).  Returns (Rim, '') or (None, why)."""
    points = np.asarray(points, float).reshape(-1, 3)
    p = np.asarray(p, float)
    zs = surface_under(points, p)
    if zs is None:
        return None, "no surface found under it (too few depth points around it)"
    at = p
    for _ in range(2):
        q, why = _dish_at(points, at, zs)
        if q is None:
            return None, why
        c, radius = _outline(q, at)
        if float(np.max(np.hypot(q[:, 0] - at[0], q[:, 1] - at[1]))) < REACH_M - CELL_M:
            break
        at = np.array([c[0], c[1], p[2]])  # it reaches the edge of what was looked at (a point near its rim)
    if not RADIUS_RANGE[0] < radius < RADIUS_RANGE[1]:
        return None, f"what stands there is {200 * radius:.0f} cm across: not a dish"
    q = q[np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) < radius + EDGE_M]  # without what stands beside it
    u = c - np.asarray(side, float)[:2]
    if float(np.linalg.norm(u)) < 0.05:
        return None, "the hand's side is over the dish's middle"
    u = u / float(np.linalg.norm(u))
    rel = q[:, :2] - c
    along, across = rel @ u, rel @ np.array([-u[1], u[0]])
    strip = q[np.abs(across) < STRIP_M]
    a_strip = along[np.abs(across) < STRIP_M]
    if len(strip) < 10:
        return None, ("the near side of its rim is not seen (if the arm hides it from the cameras, move the hand "
                      "away first)")
    outmost = float(np.percentile(a_strip, 1))
    near = a_strip < outmost + (OUTER_M if outer else EDGE_M)
    top = float(np.percentile(strip[near, 2], 98))
    at_top = a_strip[near & (strip[:, 2] > top - TOP_M)]
    edge_along = float(np.percentile(at_top, 1)) if len(at_top) else outmost
    height = top - zs
    if not HEIGHT_RANGE[0] < height < HEIGHT_RANGE[1]:
        return None, f"its near rim stands {100 * height:.1f} cm over the surface: not a dish"
    e = c + u * edge_along
    return Rim(center=(float(c[0]), float(c[1])), radius=radius, top=top, on_z=zs,
               edge=(float(e[0]), float(e[1]), top), toward=(float(u[0]), float(u[1])), points=int(len(q))), ""


def _outline(q: np.ndarray, p) -> tuple[np.ndarray, float]:
    """Middle (x, y) and radius of the dish's outline from above in its part ``q`` (see RING_M), else of the part's
    extent (1-99 percentiles)."""
    rel = q[:, :2] - np.asarray(p, float)[:2]
    dist = np.hypot(rel[:, 0], rel[:, 1])
    sector = np.floor((np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) + 180.0) / SECTOR_DEG).astype(int)
    order = np.lexsort((-dist, sector))  # by sector, the farthest first
    first = np.r_[True, sector[order][1:] != sector[order][:-1]]
    rim = q[order[first], :2]
    best = (0, None)
    if len(rim) >= 12:
        rng = np.random.default_rng(0)
        for _ in range(RING_TRIES):
            fit = _circle(rim[rng.choice(len(rim), 3, replace=False)])
            if fit is None or not RADIUS_RANGE[0] < fit[1] < RADIUS_RANGE[1]:
                continue
            if float(np.hypot(*(np.asarray(p, float)[:2] - fit[0]))) > fit[1] + RING_M:
                continue
            on = int(np.sum(np.abs(np.hypot(*(rim - fit[0]).T) - fit[1]) < RING_M))
            if on > best[0]:
                best = (on, fit)
    if best[1] is not None and best[0] >= RING_SHARE * len(rim):
        c, r = best[1]
        fit = _circle(rim[np.abs(np.hypot(*(rim - c).T) - r) < RING_M])
        if fit is not None:
            return fit
    lo, hi = np.percentile(q[:, :2], 1, axis=0), np.percentile(q[:, :2], 99, axis=0)
    return (lo + hi) / 2.0, float(np.mean(hi - lo) / 2.0)


def _circle(xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The circle through points (x, y) by least squares: (middle, radius), or None (in a line)."""
    a = np.c_[2.0 * xy, np.ones(len(xy))]
    sol, _, rank, _ = np.linalg.lstsq(a, (xy ** 2).sum(axis=1), rcond=None)
    if rank < 3:
        return None
    c = sol[:2]
    r2 = float(sol[2] + c @ c)
    return (c, float(np.sqrt(r2))) if r2 > 0.0 else None


def _dish_at(points: np.ndarray, p, zs: float) -> tuple[np.ndarray | None, str]:
    """The points of the dish at ``p`` standing on the surface at ``zs``, or (None, why)."""
    dish = points[(points[:, 2] > zs + OVER_M) & (points[:, 2] < zs + RISE_M)
                  & (np.hypot(points[:, 0] - p[0], points[:, 1] - p[1]) < REACH_M)]
    if len(dish) < 60:
        return None, f"too few depth points over the surface at z {zs:.3f} around it"
    cells: dict[tuple[int, int], list[int]] = {}
    for k, key in enumerate(map(tuple, np.floor(dish[:, :2] / CELL_M).astype(int))):
        cells.setdefault(key, []).append(k)
    low = {key: idx for key, idx in cells.items() if float(np.max(dish[idx, 2])) <= zs + TALL_M}
    if sum(len(idx) for idx in low.values()) < 60:
        return None, "only something taller than a dish there"
    q = _part_at(dish, low, p)
    return (None, "only specks of depth points over the surface there") if q is None else (q, "")


def _part_at(dish: np.ndarray, cells: dict, p) -> np.ndarray | None:
    """The points of the part of ``cells`` that is the dish at ``p``: of the parts of PART_MIN points or more,
    the widest whose outline from above (1-99 percentiles, a cell's margin) holds ``p`` or that has a cell
    next to its cell; else the nearest.  None: only specks."""
    here = tuple(np.floor(np.asarray(p, float)[:2] / CELL_M).astype(int))
    best, width, near = None, 0.0, None
    seen: set = set()
    for start in cells:
        if start in seen or len(cells[start]) < 2:
            continue
        part, stack = {start}, [start]
        seen.add(start)
        while stack:
            a = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    b = (a[0] + di, a[1] + dj)
                    if b in cells and b not in seen and len(cells[b]) >= 2:
                        seen.add(b)
                        part.add(b)
                        stack.append(b)
        q = dish[sorted(k for c in part for k in cells[c])]
        if len(q) < PART_MIN:
            continue
        lo, hi = np.percentile(q[:, :2], 1, axis=0), np.percentile(q[:, :2], 99, axis=0)
        around = bool(np.all(lo - CELL_M <= p[:2]) and np.all(p[:2] <= hi + CELL_M))
        at = any(abs(c[0] - here[0]) <= 1 and abs(c[1] - here[1]) <= 1 for c in part)
        if (around or at) and float(np.mean(hi - lo)) > width:
            best, width = q, float(np.mean(hi - lo))
        d = float(np.min(np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])))
        if near is None or d < near[0]:
            near = (d, q)
    if best is not None:
        return best
    return None if near is None else near[1]
