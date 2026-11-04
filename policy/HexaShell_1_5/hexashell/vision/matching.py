"""The same object in the other cameras' pictures.

An object outlined in one picture lies inside that outline's cone: the rays of its pixels, down
to the table.  In another camera the same object's outline must fall inside the picture of that
cone, and its own cone must in turn hold the first outline's rays.  Candidate outlines come from
SAM3: the same words, when the object was found by words, and clicks at points of the first
outline's middle ray a few heights above the table (whatever the object's height, one of them is
in or near it).  Each candidate is scored by how well the two cones explain each other.  Colour
settles near ties, and of two candidates that still tie the one nearer the first camera wins: the
first camera sees the front one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..world import TABLE_Z
from . import hull
from .hull import Sight

#: Heights above the table (m) of the clicks along the first outline's middle ray, in the order tried.
SEED_HEIGHTS = (0.02, 0.05, 0.004, 0.10, 0.18)
#: Outlines smaller than this (pixels) are noise.
MIN_PIXELS = 12
#: Outlines covering more of the picture than this are the table or the background.
MAX_PICTURE_SHARE = 0.30
#: Outlines lying this much on the robot (or on what it holds) are not the object.
MAX_ROBOT_SHARE = 0.85
#: Share of a candidate's rays that must pass through the first outline's cone (of those the first
#: camera can judge) ...
INSIDE_MIN = 0.8
#: ... and the share of them it must be able to judge.
SEEN_MIN = 0.3
#: Agreement (harmonic mean of the two shares) needed to use a candidate ...
TRUST = 0.6
#: ... and at which no more clicks are tried.
GOOD = 0.88
#: Candidates within this much agreement of the best are near ties.
TIE = 0.04
#: Of near ties, those whose cone the first camera's rays enter this much farther away (m) are behind.
BEHIND = 0.01
#: Colour difference (OpenCV 8-bit Lab, lightness counted a third) a candidate may have for free,
#: and the agreement it loses per unit beyond that.
COLOUR_FREE = 15.0
COLOUR_COST = 0.005
#: Clicks closer than this (|dx| + |dy| pixels) to one already tried are skipped.
CLICK_SPACING = 6
#: Two outlines overlapping more than this (intersection over union) are one.
SAME_IOU = 0.85


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / float(np.count_nonzero(a | b))


def parts(mask: np.ndarray, keep: float = 0.15) -> np.ndarray:
    """An outline without specks: its largest connected part, and other parts at least ``keep``
    of its size (an object cut in two by the arm in front of it stays whole)."""
    m = np.asarray(mask, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    good = 1 + np.nonzero(areas >= max(keep * float(areas.max()), MIN_PIXELS))[0]
    return np.isin(labels, good)


def middle_pixel(mask: np.ndarray) -> tuple[float, float] | None:
    """The pixel deepest inside the outline (nearest the middle of its box among equals)."""
    m = np.asarray(mask, np.uint8)
    if not m.any():
        return None
    depth = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    ys, xs = np.nonzero(depth >= depth.max() - 0.5)
    by, bx = np.nonzero(m)
    cx, cy = (bx.min() + bx.max()) / 2.0, (by.min() + by.max()) / 2.0
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return float(xs[i]), float(ys[i])


def seed_points(camera, pixel, floor: float = TABLE_Z) -> list[np.ndarray]:
    """Points of a pixel's ray at the seed heights above the table (those in front of the camera)."""
    origin, d = camera.ray(pixel)
    if d[2] >= -1e-6:
        return []
    out = []
    for h in SEED_HEIGHTS:
        t = (floor + h - origin[2]) / d[2]
        if t > 0.03:
            out.append(origin + t * d)
    return out


def colour_of(lab: np.ndarray, mask: np.ndarray, robot: np.ndarray | None = None) -> np.ndarray | None:
    """Median Lab colour of an outline's inner pixels (not the robot's)."""
    m = np.asarray(mask, bool)
    if robot is not None:
        m = m & ~robot
    inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(inner) >= 30:
        m = inner
    if np.count_nonzero(m) < 6:
        return None
    return np.median(lab[m].astype(float), axis=0)


def colour_gap(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return math.nan
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(math.sqrt((d[0] / 3.0) ** 2 + d[1] ** 2 + d[2] ** 2))


@dataclass(eq=False)
class Candidate:
    """One outline in another camera that might show the object."""

    cam: str
    mask: np.ndarray
    score: float  # SAM3's
    how: str  # 'words' or 'click (x, y)'
    inside: float = 0.0  # share of its rays through the first outline's cone (of those the first camera judges)
    seen: float = 0.0  # share of its rays the first camera can judge
    cover: float = 0.0  # share of the first outline's rays through its cone (of those it can judge)
    depth: float = math.nan  # distance from the first camera to where those rays enter both cones
    colour_gap: float = math.nan

    @property
    def fit(self) -> float:
        s = self.inside + self.cover
        return 0.0 if s <= 0 else 2.0 * self.inside * self.cover / s

    @property
    def rank(self) -> float:
        gap = self.colour_gap if math.isfinite(self.colour_gap) else 0.0
        return self.fit - COLOUR_COST * max(0.0, gap - COLOUR_FREE)

    @property
    def usable(self) -> bool:
        return self.inside >= INSIDE_MIN and self.seen >= SEEN_MIN and self.fit >= TRUST

    def describe(self) -> str:
        return (f"{self.cam} {self.how}: inside {self.inside:.2f} (seen {self.seen:.2f}) cover {self.cover:.2f} "
                f"colour {self.colour_gap:.0f} depth {self.depth:.3f}")


def choose(found: list[Candidate]) -> Candidate | None:
    usable = [c for c in found if c.usable]
    if not usable:
        return None
    best = max(c.rank for c in usable)
    near = [c for c in usable if c.rank >= best - TIE]
    depths = [c.depth for c in near if math.isfinite(c.depth)]
    if depths:
        front = min(depths)
        near = [c for c in near if not math.isfinite(c.depth) or c.depth <= front + BEHIND]
    return max(near, key=lambda c: c.rank)


class Matcher:
    """Finds an object outlined in one picture in other pictures.

    ``segment(view, **prompt)`` asks SAM3 (text= or points=); views need ``cam``, ``camera``,
    ``blind`` (bool picture of the pixels that hide what is behind them: the robot, and what its
    gripper holds) and ``lab`` (the picture in Lab).
    """

    def __init__(self, segment: Callable, first: Sight, colour: np.ndarray | None, *, text: str | None = None,
                 floor: float = TABLE_Z, text_min_score: float = 0.1):
        self.segment = segment
        self.first = first
        self.colour = colour
        self.text = text
        self.floor = floor
        self.text_min_score = text_min_score
        pixel = middle_pixel(first.mask)
        self.seeds = seed_points(first.camera, pixel, floor) if pixel is not None else []
        self.tried: dict[str, list[Candidate]] = {}

    def _consider(self, view, segments, how: str, found: list[Candidate]) -> None:
        blind = view.blind
        for seg in segments:
            m = parts(seg.mask)
            area = int(np.count_nonzero(m))
            if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size:
                continue
            if np.count_nonzero(m & blind) > MAX_ROBOT_SHARE * area:
                continue
            if any(iou(m, c.mask) > SAME_IOU for c in found):
                continue
            c = Candidate(view.cam, m, float(seg.score), how)
            a = hull.agreement(self.first, Sight(view.cam, view.camera, m, blind), self.floor)
            c.inside, c.seen, c.cover, c.depth = a.inside, a.seen, a.cover, a.depth
            c.colour_gap = colour_gap(self.colour, colour_of(view.lab, m, blind))
            found.append(c)

    def find_in(self, view, *, text_found=None, clicks=()) -> Candidate | None:
        """The outline in ``view`` that best shows the same object, or None.

        text_found: SAM3's answer to the words in this view, if already asked.  clicks: world
        points to click at first (where the object was last seen, say).
        """
        found: list[Candidate] = []
        if self.text:
            segments = text_found if text_found is not None else self.segment(
                view, text=self.text, min_score=self.text_min_score, max_masks=8)
            self._consider(view, segments, "words", found)
        clicked: list[tuple[float, float]] = []
        for point in list(clicks) + list(self.seeds):
            if any(c.usable and c.fit >= GOOD for c in found):
                break
            px = view.camera.pixel(point)
            if px is None:
                continue
            x, y = px
            xi = int(min(max(round(x), 0), view.camera.width - 1))
            yi = int(min(max(round(y), 0), view.camera.height - 1))
            if view.blind[yi, xi]:
                continue
            if any(abs(x - a) + abs(y - b) < CLICK_SPACING for a, b in clicked):
                continue
            if any(c.usable and c.mask[yi, xi] for c in found):
                continue
            clicked.append((x, y))
            self._consider(view, self.segment(view, points=[(x, y)]), f"click ({x:.0f}, {y:.0f})", found)
        self.tried[view.cam] = found
        return choose(found)
