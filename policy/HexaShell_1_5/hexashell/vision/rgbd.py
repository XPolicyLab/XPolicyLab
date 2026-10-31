"""Perception from RGB-D pictures: SAM3 outlines plus the depth image give an object's 3-D points.

Benchmarks other than RoboDojo render a depth image with every camera, so an outline's pixels
back-project to the surface the camera sees (no shape fitting as in :mod:`.scene`).  A measurement
merges the outlines of one object in every camera that sees it: centre and size from the merged
points (an oriented box on the floor plan, height from the lowest to the highest point), the yaw
of a top-down grasp across the short side, the views it was seen in.

Only the visible surface is measured: the far side of an object stays hidden, so the centre lies
toward the cameras by up to half its depth when they see it from one side only.

Everything is in the benchmark's world frame; cameras are :class:`.cameras.Camera` models (looking
along -z, +y up the picture, pixel (0, 0) the top-left one).
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from ..errors import Refused
from ..robot.results import Mask, MaskList, Spot, Thing
from .cameras import Camera, triangulate
from .matching import MIN_PIXELS, iou, parts
from .sam3_client import Sam3Client, Sam3Error, Sam3Unavailable
from .scene import _describe, _prompt

TEXT_MIN_SCORE = 0.25
CANDIDATE_MIN_SCORE = 0.10
TEXT_MAX = 8
SAME_OUTLINE_IOU = 0.85
#: An outline covering more of the picture than this is a wall, the floor or the counter, not an object.
MAX_PICTURE_SHARE = 0.85
#: An outline lying this much on the robot, with a score below ROBOT_SCORE, shows the robot.
ON_ROBOT = 0.5
ROBOT_SCORE = 0.5
KEEP_MASKS = 400
#: Depth readings (m) outside this range are none.
NEAR, FAR = 0.02, 8.0
#: Points of one outline farther than this (m, or OUTLIER_MADS deviations) from its median depth are
#: background seen past the object's edge.
OUTLIER_M = 0.04
OUTLIER_MADS = 4.0
#: Outlines of one object in two cameras: their point sets' middles agree within this (m), or
#: within SAME_SHARE of the object's size.
SAME_M = 0.06
SAME_SHARE = 0.5
#: Other instances are listed when they score at least OTHER_SCORE (at most OTHER_MAX).
OTHER_SCORE = 0.4
OTHER_MAX = 4
#: Footprints this close to square are round (any grasp yaw).
ROUND_RATIO = 1.15
#: Percentiles of the points taken as the extent (the outermost points are often edge noise).
LOW_Q, HIGH_Q = 3.0, 97.0
#: A re-measured object this far (m) from where it was has moved.
MOVED = 0.015
#: A close that holds carries the measured thing nearest the grasp point (within CARRY_GAP_M of its
#: extent: half its length across, its height; not longer than CARRY_MAX_M, which is a fixture), and
#: the other measurements of it (centres within SAME_M, about its size).
CARRY_GAP_M = 0.03
CARRY_MAX_M = 0.6
#: Depth readings of a pixel are the median of this window (pixels) that lie on the pixel's surface.
WINDOW = 5
#: An object whose points lie on the surface around its outlines (FLUSH_SHARE of them within FLUSH_M
#: of it, and 95 % within THIN_M), fitted to the band RING_GAP..RING_GAP + RING_PX pixels out
#: (SURFACE_SHARE of it on that surface), does not stand out in depth: it is flat, or the depth
#: picture sees through it (clear plastic, glass).
RING_GAP, RING_PX = 2, 6
FLUSH_M = 0.004
FLUSH_SHARE = 0.6
SURFACE_SHARE = 0.5
#: An object measured less than THIN_M tall (only its top shows) whose level surroundings lie
#: TOP_ONLY_M (from, to) below it: the note says how much lower.
THIN_M = 0.006
TOP_ONLY_M = (0.006, 0.15)
#: Two rays of one point pass closer than this (m); views less than PAIR_MIN_DEG apart leave the
#: distance along them uncertain; a ray this flat (degrees) grazes a plane.
PAIR_GAP_M = 0.005
PAIR_MIN_DEG = 3.0
GRAZING_DEG = 10.0


def closing_yaw_robocasa(direction_xy) -> float:
    """Yaw at which a gripper pointing down closes along ``direction_xy`` (RoboCasa: at yaw 0 the
    fingers close along the world y axis), in (-90, 90]."""
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    yaw = math.degrees(math.atan2(-dx, dy))
    while yaw <= -90.0:
        yaw += 180.0
    while yaw > 90.0:
        yaw -= 180.0
    return round(yaw, 1)


@dataclass
class DepthView:
    """One camera's pictures at one step."""

    cam: str
    image: np.ndarray  # RGB uint8 (H, W, 3)
    depth: np.ndarray  # float32 (H, W): metres along the optical axis
    camera: Camera
    robot: np.ndarray  # bool (H, W): the robot's own pixels
    step: int = 0

    def valid(self) -> np.ndarray:
        d = self.depth
        return np.isfinite(d) & (d > NEAR) & (d < FAR)

    def back_project(self, xs: np.ndarray, ys: np.ndarray, depth: np.ndarray) -> np.ndarray:
        cam = self.camera
        local = np.stack([(xs - cam.cx) / cam.focal * depth, -(ys - cam.cy) / cam.focal * depth, -depth], axis=1)
        return local @ cam.pose[:3, :3].T + cam.pose[:3, 3]

    def points(self, pixels: np.ndarray, *, robot_too: bool = False) -> np.ndarray:
        """World points (N, 3) of an outline's pixels that have depth, background past the edge left out."""
        m = np.asarray(pixels, bool) & self.valid()
        if not robot_too:
            m &= ~self.robot
        inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if np.count_nonzero(inner) >= max(MIN_PIXELS, 0.3 * np.count_nonzero(m)):
            m = inner
        ys, xs = np.nonzero(m)
        if not len(xs):
            return np.zeros((0, 3))
        d = self.depth[ys, xs].astype(float)
        mid = float(np.median(d))
        mad = 1.4826 * float(np.median(np.abs(d - mid)))
        keep = np.abs(d - mid) <= max(OUTLIER_M, OUTLIER_MADS * mad)
        return self.back_project(xs[keep].astype(float), ys[keep].astype(float), d[keep])

    def ring(self, pixels: np.ndarray) -> np.ndarray:
        """World points (N, 3) of the band of pixels around an outline (RING_GAP to RING_GAP + RING_PX
        pixels out), the robot left out."""
        m = np.asarray(pixels, bool).astype(np.uint8)
        near = cv2.dilate(m, np.ones((2 * RING_GAP + 1,) * 2, np.uint8)).astype(bool)
        far = cv2.dilate(m, np.ones((2 * (RING_GAP + RING_PX) + 1,) * 2, np.uint8)).astype(bool)
        ys, xs = np.nonzero(far & ~near & self.valid() & ~self.robot)
        return self.back_project(xs.astype(float), ys.astype(float), self.depth[ys, xs].astype(float))

    def depth_at(self, xy, *, robot_too: bool = False) -> float | None:
        """Depth (m) at a pixel: the median of the readings in a small window that lie within 2 cm of
        the pixel's own (an edge does not mix two surfaces).  Robot pixels count only when the pixel
        itself shows the robot."""
        h, w = self.depth.shape
        xi, yi = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= xi < w and 0 <= yi < h):
            return None
        r = WINDOW // 2
        rows, cols = slice(max(0, yi - r), min(h, yi + r + 1)), slice(max(0, xi - r), min(w, xi + r + 1))
        d = self.depth[rows, cols]
        ok = self.valid()[rows, cols]
        if not (robot_too or self.robot[yi, xi]):
            ok &= ~self.robot[rows, cols]
        if not np.count_nonzero(ok):
            return None
        if ok[yi - rows.start, xi - cols.start]:
            ok &= np.abs(d - self.depth[yi, xi]) < 0.02
        return float(np.median(d[ok]))


@dataclass
class _Record:
    thing: Thing
    points: np.ndarray
    text: str | None
    moved: np.ndarray | None = None  # where the hand has taken it since it was measured (4x4)


@dataclass
class _Candidate:
    view: DepthView
    pixels: np.ndarray
    score: float
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def middle(self) -> np.ndarray | None:
        return None if len(self.points) == 0 else np.median(self.points, axis=0)

    @property
    def span(self) -> float:
        if len(self.points) < 2:
            return 0.0
        lo, hi = np.percentile(self.points, [LOW_Q, HIGH_Q], axis=0)
        return float(np.linalg.norm(hi - lo))


class DepthScene:
    def __init__(self, url: str, *, cams: list[str], check_cam: Callable[[str], str], log=None,
                 client: Sam3Client | None = None, closing_yaw: Callable = closing_yaw_robocasa):
        self.log = log or (lambda event: None)
        self.sam = client or Sam3Client(url)
        try:
            health = self.sam.health()
        except Sam3Unavailable as exc:
            raise RuntimeError(f"the SAM3 service at {url} is not answering ({exc})") from None
        if not health.get("ok"):
            raise RuntimeError(f"the SAM3 service at {url} is not ready: {health}")
        self.cams = list(cams)
        self.check_cam = check_cam
        self.closing_yaw = closing_yaw
        self.get_views: Callable[..., dict[str, DepthView]] | None = None
        self.masks: OrderedDict[int, tuple[Mask, DepthView]] = OrderedDict()
        self.records: dict[int, _Record] = {}
        self.carried: dict[int, np.ndarray] = {}  # Thing id -> the grasp pose it has moved with since
        self.hand: Callable[[], np.ndarray] | None = None
        self.next_mask = 1
        self.next_thing = 1
        self._pool = ThreadPoolExecutor(max_workers=len(self.cams), thread_name_prefix="sam3")

    def bind(self, get_views: Callable[..., dict[str, DepthView]],
             hand: Callable[[], np.ndarray] | None = None) -> None:
        """``get_views(cams)``: the pictures of the cameras named (a host whose pictures are dear
        renders only those); ``hand()``: the grasp pose now, for a host that says when the fingers
        close on something (``grasped``) and let go (``released``)."""
        self.get_views = get_views
        self.hand = hand

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self.sam.close()

    # -- outlines --------------------------------------------------------------------------------
    def views(self, cams) -> dict[str, DepthView]:
        return self.get_views(list(cams))

    def view(self, cam: str) -> DepthView:
        views = self.views([cam])
        if cam not in views:
            raise Refused(f"no {cam} picture now")
        return views[cam]

    def segment(self, view: DepthView, **prompt) -> list:
        t0 = time.monotonic()
        if prompt.get("points"):
            h, w = view.image.shape[:2]
            prompt["points"] = [(min(max(float(x), 0.0), w - 1.0), min(max(float(y), 0.0), h - 1.0))
                                for x, y in prompt["points"]]
        try:
            found = self.sam.segment(view.image, **prompt)
        except Sam3Unavailable as exc:
            raise Refused(f"the SAM3 service is not answering right now ({exc}); try again in a moment") from None
        except Sam3Error as exc:
            raise Refused(f"SAM3 refused the request: {exc}") from None
        self.log({"kind": "sam3", "cam": view.cam, "prompt": {k: v for k, v in prompt.items() if v is not None},
                  "found": len(found), "s": round(time.monotonic() - t0, 3)})
        return found

    def register(self, view: DepthView, pixels: np.ndarray, label: str, score: float, prompt: dict) -> Mask:
        area = int(np.count_nonzero(pixels))
        ys, xs = np.nonzero(pixels)
        share = float(np.count_nonzero(pixels & view.robot)) / max(area, 1)
        mask = Mask(id=self.next_mask, cam=view.cam, label=label, score=round(float(score), 3), pixels=pixels,
                    box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    centroid=(round(float(xs.mean()), 1), round(float(ys.mean()), 1)), area=area, seq=view.step,
                    robot_share=round(share, 3), prompt=prompt)
        self.next_mask += 1
        self.masks[mask.id] = (mask, view)
        while len(self.masks) > KEEP_MASKS:
            self.masks.popitem(last=False)
        return mask

    def find(self, cam: str, text=None, *, at=None, points=None, labels=None, box=None) -> MaskList:
        prompt, label = _prompt(text, at, points, labels, box)
        view = self.view(cam)
        found = self.segment(view, **prompt)
        if "text" in prompt:
            found = [s for s in found if s.score >= TEXT_MIN_SCORE]
        kept: list = []
        for seg in sorted(found, key=lambda s: -s.score):
            if seg.area < MIN_PIXELS or any(iou(seg.mask, k.mask) > SAME_OUTLINE_IOU for k in kept):
                continue
            kept.append(seg)
        if "text" in prompt:
            kept = kept[:TEXT_MAX]
        shown = {k: v for k, v in prompt.items() if v is not None}
        return MaskList(self.register(view, seg.mask.astype(bool), label, seg.score, shown) for seg in kept)

    def known(self, mask: Mask) -> tuple[Mask, DepthView]:
        entry = self.masks.get(getattr(mask, "id", None))
        if entry is None:
            raise Refused(f"Mask#{getattr(mask, 'id', '?')} is not known (too old, or not from find)")
        return entry

    # -- measuring -------------------------------------------------------------------------------
    def measure(self, target, *, cams="auto") -> Thing:
        allowed = self._cams(cams)
        t0 = time.monotonic()
        prior = None
        if isinstance(target, Thing):
            prior = self.records.get(target.id)
            near = np.asarray(target.center_xyz, float) if target.center_xyz is not None else None
            if prior is not None and (prior.moved is not None or target.id in self.carried):
                near = self._place(target.id, prior)  # the hand has carried it since it was measured
            keep = target.id if prior else None
            text = prior.text if prior else (None if target.label.startswith("at (") else target.label)
            if text:
                thing = self._by_words(text, allowed, near=near, keep_id=keep)
                if not thing.found and near is not None:
                    thing = self._by_place_instead(thing, near, allowed, target, text, keep)
            elif near is not None:
                thing = self._by_place(near, allowed, target.label, keep_id=keep)
            else:
                raise Refused("measure: that Thing was not found; measure words, a Mask or a pixel instead")
        elif isinstance(target, str):
            thing = self._by_words(target, allowed)
        elif isinstance(target, Mask):
            mask, view = self.known(target)
            first = _Candidate(view, mask.pixels, mask.score)
            first.points = view.points(mask.pixels)
            cands, masks = self._more_views([first], allowed, mask.label)
            thing = self._from_candidates(cands, mask.label, text=mask.prompt.get("text"),
                                          masks={view.cam: mask, **masks})
        elif isinstance(target, (tuple, list)) and len(target) == 2 and isinstance(target[0], str):
            thing = self._by_pixel(self.check_cam(target[0]), target[1], allowed)
        elif isinstance(target, (tuple, list)) and target and all(isinstance(m, Mask) for m in target):
            pairs = [self.known(m) for m in target]
            if len({m.cam for m, _ in pairs}) != len(pairs):
                raise Refused("measure: give at most one Mask per camera (all of the same object)")
            thing = self._from_candidates([_Candidate(v, m.pixels, m.score) for m, v in pairs], pairs[0][0].label,
                                          text=pairs[0][0].prompt.get("text"), masks={v.cam: m for m, v in pairs})
        else:
            raise Refused("measure: target is words ('red mug'), a Mask, (cam, (x, y)), a list of Masks of one "
                          "object in different cameras, or a Thing measured before")
        if prior is not None and thing.found and prior.thing.center_xyz is not None:
            moved = float(np.linalg.norm(np.asarray(thing.center_xyz) - np.asarray(prior.thing.center_xyz)))
            if moved > MOVED:
                thing.notes.insert(0, f"moved {100 * moved:.1f} cm since it was measured")
        if thing.found and thing.id in self.carried and self.hand is not None:
            self.carried[thing.id] = np.asarray(self.hand(), float).copy()  # measured where the hand has it
        self.log({"kind": "measure", "target": _describe(target), "id": thing.id, "found": thing.found,
                  "views": thing.views, "center": thing.center_xyz, "size": thing.size,
                  "s": round(time.monotonic() - t0, 3)})
        return thing

    def things(self) -> list[Thing]:
        return [self.records[i].thing for i in sorted(self.records)]

    # -- what the hand carries -------------------------------------------------------------------
    def grasped(self, tcp) -> None:
        """The fingers closed and hold something at grasp pose ``tcp``: the measured thing nearest the
        grasp point (and the other measurements of it) moves with the hand till ``released``."""
        tcp = np.asarray(tcp, float)
        self.released(tcp)
        p, best = tcp[:3, 3], None
        for thing_id, rec in self.records.items():
            c, size = self._place(thing_id, rec), rec.thing.size
            if c is None or size is None or size[0] > CARRY_MAX_M:
                continue
            reach = 0.5 * size[0] + CARRY_GAP_M
            d = math.hypot(p[0] - c[0], p[1] - c[1])
            if d <= reach and abs(p[2] - c[2]) <= 0.5 * size[2] + CARRY_GAP_M and (best is None or d / reach < best[0]):
                best = (d / reach, c, size[0])
        if best is None:
            return
        _, c, length = best
        for thing_id, rec in self.records.items():
            other = self._place(thing_id, rec)
            if other is not None and float(np.linalg.norm(other - c)) <= SAME_M \
                    and 0.5 * length - 0.02 <= rec.thing.size[0] <= 1.5 * length + 0.02:
                self.carried[thing_id] = tcp.copy()
        self.log({"kind": "carry", "held": sorted(self.carried)})

    def released(self, tcp) -> None:
        """The hand let go at grasp pose ``tcp`` (or what it held slipped out): what it carried is taken
        to be where the hand brought it."""
        if not self.carried:
            return
        tcp = np.asarray(tcp, float)
        for thing_id, grasp in self.carried.items():
            rec = self.records.get(thing_id)
            if rec is not None:
                move = tcp @ np.linalg.inv(grasp)
                rec.moved = move if rec.moved is None else move @ rec.moved
        self.log({"kind": "carry", "set": sorted(self.carried)})
        self.carried = {}

    def _place(self, thing_id: int, rec: _Record) -> np.ndarray | None:
        """Where a measured thing is taken to be now: where it was measured, or where the hand has
        taken it since (and, while it holds it, where the hand is now)."""
        if not rec.thing.found or rec.thing.center_xyz is None:
            return None
        c, move = np.asarray(rec.thing.center_xyz, float), rec.moved
        if thing_id in self.carried and self.hand is not None:
            now = np.asarray(self.hand(), float) @ np.linalg.inv(self.carried[thing_id])
            move = now if move is None else now @ move
        return c if move is None else move[:3, :3] @ c + move[:3, 3]

    def _cams(self, cams) -> list[str]:
        if cams is None or cams in ("auto", "all"):
            return list(self.cams)
        names = [cams] if isinstance(cams, str) else list(cams)
        out: list[str] = []
        for c in names:
            name = self.check_cam(c)
            if name not in out:
                out.append(name)
        if not out:
            raise Refused("measure: cams is 'auto' or a list of camera names")
        return out

    def _instances(self, view: DepthView, segments) -> list[_Candidate]:
        kept: list[_Candidate] = []
        for seg in sorted(segments, key=lambda s: -s.score):
            if seg.score < TEXT_MIN_SCORE:
                continue
            m = parts(seg.mask)
            area = int(np.count_nonzero(m))
            if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size:
                continue
            if any(iou(m, k.pixels) > SAME_OUTLINE_IOU for k in kept):
                continue
            if np.count_nonzero(m & view.robot) >= ON_ROBOT * area and seg.score < ROBOT_SCORE:
                continue
            cand = _Candidate(view, m, float(seg.score))
            cand.points = view.points(m)
            if len(cand.points) >= MIN_PIXELS // 2:
                kept.append(cand)
        return kept[:TEXT_MAX]

    def _by_words(self, text: str, allowed: list[str], *, near: np.ndarray | None = None,
                  keep_id: int | None = None) -> Thing:
        text = (text or "").strip()
        if not text:
            raise Refused("measure: the words name the object, e.g. 'red mug'")
        views = self.views(allowed)
        present = [views[c] for c in allowed if c in views]
        if not present:
            raise Refused("no pictures from those cameras now")
        jobs = {v.cam: self._pool.submit(self.segment, v, text=text, min_score=CANDIDATE_MIN_SCORE, max_masks=TEXT_MAX)
                for v in present}
        answers = {cam: job.result() for cam, job in jobs.items()}
        found = {v.cam: self._instances(v, answers[v.cam]) for v in present}
        groups = _group([c for cam in found for c in found[cam]])
        if not groups:
            return Thing(id=0, label=text, found=False,
                         why_missing=f"no '{text}' outlined in {', '.join(v.cam for v in present)}",
                         notes=["the words find by appearance: name how it looks (colour, shape, what it is), or "
                                "point at it: robot.measure((cam, (x, y)))"])
        if near is not None:
            pick = min(groups, key=lambda g: float(np.linalg.norm(_middle(g) - near)))
        else:
            pick = max(groups, key=lambda g: (max(c.score for c in g), len(g)))
        masks = {c.view.cam: self.register(c.view, c.pixels, text, c.score, {"text": text}) for c in pick}
        others = [g for g in groups if g is not pick and max(c.score for c in g) >= OTHER_SCORE]
        notes = []
        if others:
            listed = "; ".join(f"at ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}) score {max(c.score for c in g):.2f}"
                               for g in others[:OTHER_MAX] for m in [_middle(g)])
            more = f" and {len(others) - OTHER_MAX} more" if len(others) > OTHER_MAX else ""
            notes.append(f"other '{text}' seen: {listed}{more}")
        return self._from_candidates(pick, text, text=text, masks=masks, notes=notes, keep_id=keep_id)

    def _outline_at(self, view: DepthView, px) -> _Candidate | None:
        """The best outline SAM3 gives for a point prompt that covers the pixel itself."""
        h, w = view.depth.shape
        xi, yi = int(round(float(px[0]))), int(round(float(px[1])))
        if not (0 <= xi < w and 0 <= yi < h):
            return None
        best = None
        for seg in self.segment(view, points=[(float(px[0]), float(px[1]))]):
            m = parts(seg.mask)
            area = int(np.count_nonzero(m))
            if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size or not seg.mask[yi, xi]:
                continue
            if best is None or seg.score > best.score:
                best = _Candidate(view, m, float(seg.score))
        if best is not None:
            best.points = view.points(best.pixels)
        return best

    def _more_views(self, cands: list[_Candidate], allowed: list[str], label: str, *,
                    center: np.ndarray | None = None) -> tuple[list[_Candidate], dict[str, Mask]]:
        """The same object in the other cameras: SAM3 prompted at the pixel where its middle (or
        ``center``) appears there, kept when the outline's points lie around the same place."""
        if center is None:
            if not cands or cands[0].middle is None:
                return cands, {}
            center = np.median(np.concatenate([c.points for c in cands if len(c.points)]), axis=0)
        views = self.views(allowed)
        have = {c.view.cam for c in cands}
        span = max([c.span for c in cands] + [0.0])
        found, masks = list(cands), {}
        todo = [views[c] for c in allowed if c in views and c not in have]
        pixels = {v.cam: v.camera.pixel(center) for v in todo}
        jobs = {v.cam: self._pool.submit(self._outline_at, v, pixels[v.cam]) for v in todo if pixels[v.cam] is not None}
        for cam, job in jobs.items():
            cand = job.result()
            if cand is None or cand.middle is None:
                continue
            if float(np.linalg.norm(cand.middle - center)) > max(SAME_M, SAME_SHARE * max(span, cand.span)):
                continue
            found.append(cand)
            masks[cam] = self.register(cand.view, cand.pixels, label, cand.score,
                                       {"points": [tuple(round(v, 1) for v in pixels[cam])], "labels": [1]})
        return found, masks

    def _by_place(self, center: np.ndarray, allowed: list[str], label: str, *, keep_id: int | None,
                  text: str | None = None) -> Thing:
        cands, masks = self._more_views([], allowed, label, center=center)
        if not cands:
            return Thing(id=0, label=label, found=False, why_missing="nothing is outlined where it was (it may have "
                         "moved): measure it by words or at a pixel")
        return self._from_candidates(cands, label, text=text, masks=masks, keep_id=keep_id)

    def _by_place_instead(self, missed: Thing, near: np.ndarray, allowed: list[str], target: Thing, text: str,
                          keep: int | None) -> Thing:
        """A Thing the words no longer outline (seen from close by or from above it can look different):
        outlined where it was instead, if what is there has about its size."""
        before = self.records.get(keep) if keep else None
        placed = self._by_place(near, allowed, target.label, keep_id=keep, text=text)
        if not placed.found:
            return missed
        was, now = max(target.size or (0.0,)), max(placed.size or (0.0,))
        if was and not (0.5 * was - 0.02 <= now <= 1.5 * was + 0.02):
            if before is not None:
                self.records[keep] = before
            else:
                self.records.pop(placed.id, None)
            missed.notes.append(f"where it was, something {100 * now:.0f} cm across is outlined (it was "
                                f"{100 * was:.0f} cm): it may have moved")
            return missed
        placed.notes.insert(0, f"the words '{text}' outlined nothing now: this is what is outlined where it was "
                               "measured - check the outline")
        return placed

    def _by_pixel(self, cam: str, xy, allowed: list[str]) -> Thing:
        view = self.view(cam)
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError, IndexError):
            raise Refused("measure((cam, (x, y))): (x, y) is a pixel") from None
        xi, yi = int(round(x)), int(round(y))
        best = None
        for seg in self.segment(view, points=[(x, y)]):
            m = parts(seg.mask)
            area = int(np.count_nonzero(m))
            h, w = m.shape
            if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size or not (0 <= yi < h and 0 <= xi < w) \
                    or not seg.mask[yi, xi]:
                continue
            if best is None or seg.score > best[1]:
                best = (m, float(seg.score))
        label = f"at ({x:.0f}, {y:.0f})"
        if best is None:
            return Thing(id=0, label=label, found=False, why_missing=f"SAM3 outlines no object at ({x:.0f}, {y:.0f}) "
                         f"in {cam}")
        mask = self.register(view, best[0], label, best[1], {"points": [(x, y)], "labels": [1]})
        first = _Candidate(view, best[0], best[1])
        first.points = view.points(best[0])
        cands, masks = self._more_views([first], allowed, label)
        return self._from_candidates(cands, label, masks={cam: mask, **masks})

    def _from_candidates(self, cands: list[_Candidate], label: str, *, text: str | None = None,
                         masks: dict[str, Mask], notes: list[str] | None = None, keep_id: int | None = None) -> Thing:
        clouds = []
        for c in cands:
            if not len(c.points):
                c.points = c.view.points(c.pixels)
            if len(c.points):
                clouds.append(c.points)
        notes = list(notes or [])
        if not clouds:
            return Thing(id=0, label=label, found=False, masks=masks,
                         why_missing="the outline has no depth readings off the robot (it may be on the robot or "
                         "too near the camera)")
        pts = np.concatenate(clouds)
        thing = self._thing(pts, label, masks, cands, notes, keep_id)
        self.records[thing.id] = _Record(thing, pts, text)
        return thing

    def _thing(self, pts: np.ndarray, label: str, masks: dict[str, Mask], cands, notes, keep_id) -> Thing:
        lo_z, hi_z = np.percentile(pts[:, 2], [LOW_Q, HIGH_Q])
        xy = pts[:, :2]
        mid = np.median(xy, axis=0)
        cov = np.cov((xy - mid).T) if len(xy) > 2 else np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        major, minor = vecs[:, 1], vecs[:, 0]
        a = (xy - mid) @ major
        b = (xy - mid) @ minor
        a_lo, a_hi = np.percentile(a, [LOW_Q, HIGH_Q])
        b_lo, b_hi = np.percentile(b, [LOW_Q, HIGH_Q])
        centre_xy = mid + major * (a_lo + a_hi) / 2.0 + minor * (b_lo + b_hi) / 2.0
        length, width = float(a_hi - a_lo), float(b_hi - b_lo)
        height = float(hi_z - lo_z)
        is_round = width > 0 and length / max(width, 1e-6) < ROUND_RATIO
        grip_yaw = None if is_round else self.closing_yaw(minor)
        step = max(c.view.step for c in cands)
        warning = None
        if len(pts) < 30:
            warning = f"only {len(pts)} depth points: measure again closer, or from another camera"
        around = surroundings([c.view.ring(c.pixels) for c in cands], pts)
        if around is not None and around.flush:
            level, z = around.level, around.z
            if level:
                said = (f"its depth points lie level with the surface around it (z {z:.3f}): it is flat, or the depth "
                        "picture sees through it (clear plastic, glass) - then its walls and top are not measured: "
                        "judge them from the colour pictures")
            else:
                said = ("its depth points lie on the surface behind it: it is flat on it, or the depth picture sees "
                        "through it (clear plastic, glass) - then it is not where it is measured: judge it from the "
                        "colour pictures")
            if warning is None:
                warning = said
            else:
                notes.append(said)
        elif around is not None and around.level and height < THIN_M and \
                TOP_ONLY_M[0] <= around.above <= TOP_ONLY_M[1]:
            notes.append(f"only its top shows in depth: the surface around it lies {100.0 * around.above:.1f} cm lower "
                         f"(z {around.z:.3f}), so it may be that thick (a board, a box seen from above) or stand on "
                         "something")
        if len(cands) == 1 and len(self.cams) > 1:
            notes.append(f"seen by {cands[0].view.cam} only: its far side is hidden, so the centre may lie up to "
                         "half its depth toward that camera")
        thing_id = keep_id or self.next_thing
        if keep_id is None:
            self.next_thing += 1
        return Thing(id=thing_id, label=label, found=True,
                     center_xyz=tuple(round(float(v), 4) for v in (centre_xy[0], centre_xy[1], (lo_z + hi_z) / 2.0)),
                     top=round(float(hi_z), 4), size=(round(length, 4), round(width, 4), round(height, 4)),
                     grip_yaw=grip_yaw, on=f"z {lo_z:.3f}", on_z=round(float(lo_z), 4), views=sorted(masks),
                     warning=warning, masks=masks, seq=step, round=is_round, notes=notes, step=step)

    # -- points ------------------------------------------------------------------------------------
    def to_world(self, cam: str, xy, *, z=None, pair=None) -> Spot:
        need = [self.check_cam(cam)]
        if pair is not None and isinstance(pair, (tuple, list)) and pair and isinstance(pair[0], str):
            need.append(self.check_cam(pair[0]))
        return spot(self.views(need), cam, xy, z=z, pair=pair, check_cam=self.check_cam)


def spot(views: dict[str, DepthView], cam: str, xy, *, z=None, pair=None, check_cam=lambda c: c) -> Spot:
    """The world point seen at pixel ``xy`` of ``cam``: from the depth image, where the pixel's ray meets
    the plane at height ``z``, or triangulated with the same point in a second camera (``pair``)."""
    if cam not in views:
        raise Refused(f"no {cam} picture now")
    view = views[cam]
    x, y = _pixel(xy, "to_world: xy is a pixel (x, y)")
    camera = view.camera
    if pair is not None:
        try:
            cam2, xy2 = check_cam(pair[0]), _pixel(pair[1], "")
        except (TypeError, IndexError, Refused):
            raise Refused("to_world(pair=(cam2, (x2, y2))): a pixel of the same point in another camera") from None
        if cam2 == cam:
            raise Refused("to_world(pair=...): the second pixel must be in a different camera")
        if cam2 not in views:
            raise Refused(f"no {cam2} picture now")
        other = views[cam2].camera
        (o1, d1), (o2, d2) = camera.ray((x, y)), other.ray(xy2)
        p, gap, angle = triangulate(o1, d1, o2, d2)
        source = f"two views ({cam} + {cam2})"
        if (p - o1) @ d1 <= 0 or (p - o2) @ d2 <= 0:
            return Spot(xyz=None, cam=cam, px=(x, y), source=source, gap_mm=round(1000 * gap, 1),
                        angle_deg=round(angle, 1), warning="the rays meet behind a camera: the two pixels are not "
                        "the same point")
        per_px = 1000.0 * max(float(np.linalg.norm(p - o1)) / camera.focal,
                              float(np.linalg.norm(p - o2)) / other.focal) / max(math.sin(math.radians(angle)), 1e-3)
        warnings = []
        if gap > PAIR_GAP_M:
            warnings.append(f"the rays pass {1000 * gap:.0f} mm apart: the two pixels probably do not show the same "
                            "point")
        if angle < PAIR_MIN_DEG:
            warnings.append(f"the views differ by only {angle:.0f}°: the distance along the rays is uncertain")
        return Spot(xyz=tuple(round(float(v), 4) for v in p), cam=cam, px=(x, y), source=source,
                    gap_mm=round(1000 * gap, 1), angle_deg=round(angle, 1), warning="; ".join(warnings) or None,
                    mm_per_px=round(per_px, 1))
    if z is not None:
        try:
            z = float(z)
        except (TypeError, ValueError):
            raise Refused("to_world(z=...): a height in metres") from None
        if not math.isfinite(z):
            raise Refused("to_world(z=...): a height in metres")
        origin, d = camera.ray((x, y))
        source = f"plane z {z:.3f}"
        t = (z - origin[2]) / d[2] if abs(d[2]) > 1e-6 else -1.0
        if t <= 0:
            return Spot(xyz=None, cam=cam, px=(x, y), source=source,
                        warning="the ray does not meet that plane in front of the camera")
        p = origin + t * d
        elev = math.degrees(math.asin(min(1.0, abs(float(d[2])))))
        warning = (f"the ray grazes the plane ({elev:.0f}°): small picking errors move the point a lot"
                   if elev < GRAZING_DEG else None)
        per_px = 1000.0 * float(np.linalg.norm(p - origin)) / camera.focal / max(abs(float(d[2])), 1e-3)
        return Spot(xyz=tuple(round(float(v), 4) for v in p), cam=cam, px=(x, y), source=source, warning=warning,
                    mm_per_px=round(per_px, 1))
    d = view.depth_at((x, y))
    if d is None:
        return Spot(xyz=None, cam=cam, px=(x, y), source="the depth picture",
                    warning="no depth reading there (off the picture, too far, or only the robot)")
    p = view.back_project(np.array([x]), np.array([y]), np.array([d]))[0]
    h, w = view.robot.shape
    xi, yi = min(max(int(round(x)), 0), w - 1), min(max(int(round(y)), 0), h - 1)
    warning = "that pixel shows the robot" if view.robot[yi, xi] else None
    return Spot(xyz=tuple(round(float(v), 4) for v in p), cam=cam, px=(x, y), source="the depth picture",
                warning=warning, mm_per_px=round(1000 * d / camera.focal, 2))


@dataclass
class Around:
    """The surface around an object's outlines (see :func:`surroundings`)."""

    level: bool  # it is level (its normal within about 18 degrees of up)
    z: float  # its height under the object's middle (level), else the object's middle height
    flush: bool  # the object's points lie on it (none stands out)
    above: float  # how far (m) the object's points lie above it (the median; level only)


def surroundings(rings: list[np.ndarray], pts: np.ndarray) -> Around | None:
    """The surface around an object's outlines (``rings``: the points of their bands, see
    :meth:`DepthView.ring`): the plane most of the band lies on; None when the band does not lie on
    one surface."""
    rings = [r for r in rings if len(r)]
    if not rings or len(pts) < MIN_PIXELS:
        return None
    ring = np.concatenate(rings)
    if len(ring) < MIN_PIXELS:
        return None
    plane = _plane(ring)
    if plane is None:
        return None
    mid, normal = plane
    if np.mean(np.abs((ring - mid) @ normal) <= FLUSH_M) < SURFACE_SHARE:
        return None
    normal = normal if normal[2] >= 0 else -normal
    off = (pts - mid) @ normal
    flush = bool(np.mean(np.abs(off) <= FLUSH_M) >= FLUSH_SHARE and np.percentile(np.abs(off), 95) < THIN_M)
    centre = np.median(pts, axis=0)
    if normal[2] <= 0.95:
        return Around(level=False, z=float(centre[2]), flush=flush, above=0.0)
    z = float(mid[2] - (normal[:2] @ (centre[:2] - mid[:2])) / normal[2])
    return Around(level=True, z=z, flush=flush, above=float(np.median(off)))


def _plane(points: np.ndarray, tries: int = 200, most: int = 4000) -> tuple[np.ndarray, np.ndarray] | None:
    """(a point on it, its unit normal): the plane through the most ``points`` within FLUSH_M - the
    best of ``tries`` planes through three of them, refitted to those near it. Things standing on a
    surface pull a least-squares plane off it; this one stays on it. The draws are fixed, so the same
    points give the same plane."""
    rng = np.random.default_rng(0)
    probe = points[:: max(1, len(points) // most)]
    a, b, c = (probe[rng.integers(0, len(probe), tries)] for _ in range(3))
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-9
    if not np.any(ok):
        return None
    normals, a = normals[ok] / lengths[ok, None], a[ok]
    near = np.abs(probe @ normals.T - np.sum(a * normals, axis=1)) <= FLUSH_M
    best = int(np.argmax(np.count_nonzero(near, axis=0)))
    mid, normal = a[best], normals[best]
    for _ in range(2):
        keep = np.abs((points - mid) @ normal) <= FLUSH_M
        if np.count_nonzero(keep) < MIN_PIXELS:
            return None
        mid = points[keep].mean(axis=0)
        normal = np.linalg.svd(points[keep] - mid, full_matrices=False)[2][2]
    return mid, normal


def _pixel(xy, message: str) -> tuple[float, float]:
    try:
        x, y = float(xy[0]), float(xy[1])
    except (TypeError, ValueError, IndexError):
        raise Refused(message) from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise Refused(message)
    return x, y


def _middle(group: list[_Candidate]) -> np.ndarray:
    return np.median(np.concatenate([c.points for c in group]), axis=0)


def _group(cands: list[_Candidate]) -> list[list[_Candidate]]:
    """Outlines of the same object in different cameras, together: best first, one per camera."""
    groups: list[list[_Candidate]] = []
    for c in sorted(cands, key=lambda c: -c.score):
        m = c.middle
        if m is None:
            continue
        home = None
        for g in groups:
            if any(o.view.cam == c.view.cam for o in g):
                continue
            gm = _middle(g)
            limit = max(SAME_M, SAME_SHARE * max(c.span, max(o.span for o in g)))
            if float(np.linalg.norm(gm - m)) <= limit:
                home = g
                break
        if home is None:
            groups.append([c])
        else:
            home.append(c)
    return groups
