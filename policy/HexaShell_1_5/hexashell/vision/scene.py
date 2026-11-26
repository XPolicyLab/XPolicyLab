"""Perception for one episode: outlines (``find``), 3-D measurement (``measure``), the surface
under a pixel (``to_world`` without z/pair) and the memory of measured objects.

Outlines come from the machine's shared SAM3 service.  Every outline gets an episode-wide id and
is kept here with the camera model and arm readings of its picture, so a Mask that comes back
from a cell can be used even after the robot has moved (the cell may also have edited its
pixels: they are taken as given when they still fit the picture).

A measurement starts from one outline, finds the same object in the other cameras
(:mod:`.matching`) and intersects what they see (:mod:`.hull`).  Measured objects are remembered
by id: measuring a Thing again looks for it where it was last seen (or, when it was held, where
the gripper has taken it), keeps its id, and falls back on its earlier height when only one
camera sees it now.
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
from ..geometry import homogeneous
from ..motion import ik
from ..motion.chain import arm_chain
from ..motion.pose import rotation_of
from ..robot.results import Mask, MaskList, Spot, Thing
from ..state import CAMERAS, ArmReading, Snapshot
from ..world import ARM_BASES, ARMS, GRASP_REACH, TABLE_Z
from . import hull
from .cameras import HEIGHT, WIDTH, WRIST_FOCAL, WRIST_MOUNT, Camera, cameras_for, check_cam, wrist_pose
from .depth import level_depth
from .matching import (MAX_PICTURE_SHARE, MIN_PIXELS, Matcher, colour_gap, colour_of, iou, parts)
from .robotmask import fingers_mask, label_name, robot_labels, robot_mask
from .sam3_client import Sam3Client, Sam3Error, Sam3Unavailable
from .spots import ON_ROBOT_PATCH, mm_per_pixel_on_plane, on_plane, pixel_of

#: Text prompts: outlines scoring below this are dropped.
TEXT_MIN_SCORE = 0.25
#: Outlines of the words in the other cameras are candidates down to this score.
CANDIDATE_MIN_SCORE = 0.10
#: At most this many outlines per text prompt.
TEXT_MAX = 8
#: Other matches of the words are listed when they score at least OTHER_SCORE, or OTHER_SHARE of the
#: measured one's score when that is lower (weaker ones are mostly the table or the wall); at most
#: OTHER_MAX of them.
OTHER_SCORE = 0.4
OTHER_SHARE = 0.8
OTHER_MAX = 4
#: Two outlines overlapping more than this (intersection over union) are one.
SAME_OUTLINE_IOU = 0.85
#: Outlines kept for reuse by id (the oldest are forgotten).
KEEP_MASKS = 400
#: The log keeps the best this many outlines of each SAM3 answer (score, box, area), weak ones too.
LOG_TOP = 3
#: An outline lying this much on the robot, with a score below ROBOT_SCORE, shows the robot.
ON_ROBOT = 0.5
ROBOT_SCORE = 0.5
# An object clicked at a pixel the robot covers (spots.ON_ROBOT_PATCH: the pixel, or more than half of the
# 5 x 5 pixels around it, in the robot's outline drawn without a margin) and not measured is reported as
# covered by that arm.  (R1: 33 of the 48 pixel measures that found nothing had the pixel on an arm in
# the head picture measured - in 12 the arm had moved there in the same cell - and printed "the outline
# does not reach the table" (26), "lies on the robot's own arm or gripper" (6) or "no footprint fits"
# (1); the 44 found with the pixel on an arm print as before.  Open R1: a block the arm had just moved
# over.)
#: With three views, one whose outline the shape explains less than this is left out; with two, flagged.
DISAGREE = 0.75
#: When the first camera sees all of an object and every other one only part of it (cut off by the
#: picture's edge or the robot), the others may not rule out the space the object hides from the first
#: camera: the standing reading runs along it, and an object standing up reads as one lying low and long
#: down the first camera's view.  Such a box is flagged when the others' outlines meet fewer than
#: CONFIRMED_MIN of the first one's rays, its footprint is LONG_ALONG times longer down that camera's
#: view than across it, and the first camera alone reads the object at least TALLER_RATIO times and
#: TALLER_BY (m) taller.  (Open R0/R1/R0-A: a 213 mm bottle standing in front of the head camera, seen
#: at the edge of the start-pose wrist pictures, read 18-94 mm high and 0.17-0.30 m long down the
#: head's view in 21 of 24 episodes, 7 of them with no warning; the head alone read it 170 mm high.
#: Colour outlines of those start pictures: 15-23% of the head's rays met.  R1, the printed boxes
#: replayed: 5 of 2880 measures from two or more cameras flagged.  Measuring from the first camera
#: alone instead read worse than the box in 10 of the 17 R1 cases a later clean measure could judge,
#: better in 3, so the box stays and the warning gives the other reading.)
CONFIRMED_MIN = 0.5
LONG_ALONG = 1.5
TALLER_RATIO = 1.5
TALLER_BY = 0.05
#: A measured object whose outline overlaps a remembered one's place this much (and has its colour) is it.
SAME_THING_IOU = 0.5
SAME_THING_COLOUR = 25.0
#: Looking for a remembered object: the outline must overlap its old place at least this much, or,
#: when that place was a one-camera guess, cover this share of it (or it this share of the outline)
#: and have the remembered colour.
REFIND_IOU = 0.25
REFIND_COVER = 0.6
#: A remembered object not found again, whose old place the robot covers more than REFIND_HIDDEN of in a
#: picture (drawn without a margin), is reported with the arm that covers it.  (R1: 18 of 839 re-measures
#: found nothing.  9 printed "nothing is where it was last measured (it moved?)" or "not where it was":
#: in 4 an arm covered 95-100% of the place in the head picture, in the other 5 at most 9%.  9 re-found
#: an outline that measured badly - "lies on the robot's own arm or gripper" (5), "an object of that
#: height does not fit" (3), "does not reach the table" (1): an arm covered 54-82% of the place in 4,
#: 0-43% in 5.  5 of the 8 covered ones were measured again later within 7 mm of where they had been.)
REFIND_HIDDEN = 0.5
#: A re-measured object this far (m) from where it was has moved.
MOVED = 0.015
#: An object read as hanging from its rim gets the note on how its top errs only from this height (m):
#: thinner ones are tiles, coins and blocks lying flat or tilted.  (R1: 20 of 98 such notes were for
#: things 2-22 mm high - tiles, coins, blocks, a bottle lying down, two 'bowls' measured 2 mm high;
#: bowls, cups and mugs read 30 mm high and more, in the Open tasks too.)
HANG_NOTE_MIN = 0.025
#: An outline whose footprint is longer than this (m) is a surface, not an object (the largest
#: objects on these tables are about 0.42 m long); one seen by one camera and narrower than THIN is
#: an edge or a line.
MAX_OBJECT = 0.5
THIN = 0.005
#: A spot from matching pictures and one from the outline there agree within this (m).
SPOT_AGREE = 0.005
#: SAM3's outline for these words is the bare table top (with some flat things lying on it): a pixel
#: inside it is on the table, unless matching two cameras' pictures puts the surface there more than
#: RAISED (m) higher.  (Over the benchmark pictures, 2% of such pixels were on an object; outlining
#: and measuring whatever is under the pixel instead was wrong by over 1 cm for half of them.)
TABLE_WORDS = "floor"
TABLE_MIN_SCORE = 0.3
RAISED = 0.005
#: Bottom this close above the table top (m): it stands on the table.
ON_TABLE = 0.006
#: A stacked object's bottom is this close (m) to the top of what it rests on.
STACK_GAP = 0.012
#: A closed gripper whose grasp point is this close (m) to an object holds it.
HELD_GAP = 0.02
#: A close that holds carries the measured object it caught (see Scene.grasped): the motion driver's pick
#: when the grasp point is about there, else the one whose middle is nearest it - within CARRY_GAP (m) of
#: its extent (half its length across, half its height up or down).  A re-measure looks where the hand set
#: it down, and when nothing is there, takes the nearest outline of its words that shows no other Thing.
#: (Base test 10-28, before this: re-measures looked where the object had been measured; 181 found
#: nothing there and took the nearest outline of the words - 56 near where the hand had set it down, and
#: about 30 an object another Thing was measured at.)
CARRY_GAP = 0.03
#: What a gripper holds fills the middle of its own wrist camera's picture and hides what lies under
#: it, so those pixels say nothing about other objects.  SAM3 outlines it from a click at the grasp
#: point; an outline covering more than HELD_MAX_SHARE of the picture is something else (the gripper
#: with it, or the table), and without an outline the space between the fingers stands in for it.
HELD_MAX_SHARE = 0.4
HELD_PAD_PX = 3
#: Hover advice: a wrist camera looking at the object's top from about 23 cm.  Best is the grasp point
#: over the object at pitch 60, HOVER_ABOVE (m) above its top and at least HOVER_MIN_Z: the camera then
#: looks straight down (measured 1.4 mm median, against 2.0 from 10 cm aside and 1.6 pointing down 9 cm
#: above it, whose pictures often cut the object off).  Poses are tried best first - over it, then
#: LOOK_ASIDE (m) to the arm's own side, then the lower LOOK_PITCHES, which reach farther - and the
#: first that the arm reaches while leaving LOOK_CLEAR of the object free in both the head and the
#: wrist picture is advised.  (Over an object near the robot's middle, the arm hides it from the head.)
HOVER_ABOVE = 0.15
HOVER_MIN_Z = 0.95
#: A big object is looked at from farther, where its top fills at most HOVER_SHARE of the wrist picture,
#: when an arm reaches that: from closer its outline covers more than MAX_PICTURE_SHARE, is dropped as the
#: table, and the object is matched by a part of it.  (Open R0/R1/R0-A: a box 0.22 m square, advised from
#: 23 cm, filled about half the picture; the next measure of it read a slice 10-38 mm wide in 8 of 11.
#: R1: 48 of 1017 advices were for objects that big, 37 of them over the cut; 31 now advise a farther
#: pose, and the other 17 are as before - no arm reaches farther, or it is within 5 mm.)
HOVER_SHARE = 0.25
LOOK_PITCHES = (60, 45, 30)
LOOK_ASIDE = 0.08
LOOK_CLEAR = 0.7
#: No arm reached a pose with its flange farther than 0.63 m from its base (600 random poses): farther
#: ones are not tried (a failing IK solve takes about 0.25 s).
ARM_SPAN = 0.64
#: An object that could reach this far (m) past a side no camera sees past gets a note, and a warning
#: from HIDDEN_WARN.
HIDDEN_NOTE = 0.01
HIDDEN_WARN = 0.02
#: Wrist cameras' readings run up to 0.2 below the real opening while squeezing.
SQUEEZE_LAG = 0.2
ORDER = ("head", "left_wrist", "right_wrist")


@dataclass
class View:
    """One camera picture with what is needed to reason about it in 3-D."""

    cam: str
    seq: int
    camera: Camera
    image: np.ndarray
    arms: dict[str, ArmReading]
    openings: dict[str, float] | None = None
    #: outlines what this wrist camera's own gripper holds (set while it holds something)
    outline_held: Callable[["View"], np.ndarray] | None = field(default=None, repr=False)
    _robot: np.ndarray | None = field(default=None, repr=False)
    _held: np.ndarray | None = field(default=None, repr=False)
    _blind: np.ndarray | None = field(default=None, repr=False)
    _lab: np.ndarray | None = field(default=None, repr=False)
    _table: np.ndarray | None = field(default=None, repr=False)  # bare table top (see TABLE_WORDS)

    @property
    def robot(self) -> np.ndarray:
        """Pixels showing the robot itself (bool H x W)."""
        if self._robot is None:
            self._robot = robot_mask(self.camera, self.arms, self.openings)
        return self._robot

    @property
    def held(self) -> np.ndarray | None:
        """Pixels showing what this wrist camera's own gripper holds (None when it holds nothing)."""
        if self.outline_held is None:
            return None
        if self._held is None:
            self._held = self.outline_held(self)
        return self._held

    @property
    def blind(self) -> np.ndarray:
        """Pixels that hide whatever lies behind them: the robot, and what its gripper holds."""
        if self._blind is None:
            held = self.held
            self._blind = self.robot if held is None else self.robot | held
        return self._blind

    @property
    def lab(self) -> np.ndarray:
        if self._lab is None:
            self._lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        return self._lab


def views_of(snap: Snapshot, openings: dict[str, float] | None = None) -> dict[str, View]:
    cams = cameras_for(snap)
    return {name: View(name, snap.seq, cams[name], snap.images[name], snap.arms_in_pictures, openings)
            for name in cams if snap.images.get(name) is not None}


def _box_of(pixels: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(pixels)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def solid_points(center, size, yaw, is_round: bool, bottom: float, top: float) -> np.ndarray:
    """Corners of an object's bounding solid (box, or cylinder for a round one)."""
    cx, cy = float(center[0]), float(center[1])
    length, width = float(size[0]), float(size[1])
    if is_round:
        a = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        ring = np.stack([cx + 0.5 * length * np.cos(a), cy + 0.5 * length * np.sin(a)], axis=1)
    else:
        t = math.radians(yaw or 0.0)
        u = np.array([math.cos(t), math.sin(t)]) * 0.5 * length
        v = np.array([-math.sin(t), math.cos(t)]) * 0.5 * width
        ring = np.array([[cx, cy]]) + np.array([u + v, u - v, -u - v, -u + v])
    return np.concatenate([np.column_stack([ring, np.full(len(ring), bottom)]),
                           np.column_stack([ring, np.full(len(ring), top)])])


def silhouette(camera: Camera, points: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Where points' convex hull appears in a camera's picture, and the share of them in the picture."""
    px, depth = camera.project(points)
    ahead = depth > 0.02
    if np.count_nonzero(ahead) < 3:
        return None, 0.0
    inside = ahead & (px[:, 0] >= -0.5) & (px[:, 0] <= camera.width - 0.5) & (px[:, 1] >= -0.5) \
        & (px[:, 1] <= camera.height - 0.5)
    region = np.zeros((camera.height, camera.width), np.uint8)
    poly = cv2.convexHull(np.round(px[ahead] * 16).astype(np.int32))
    cv2.fillConvexPoly(region, poly, 1, cv2.LINE_8, 4)
    return region > 0, float(inside.mean())


def _free_share(camera: Camera, points: np.ndarray, robot: np.ndarray) -> float:
    """Share of the points' convex hull that a picture shows clear of the robot (0 when some are behind
    the camera); the part outside the picture counts as hidden."""
    px, depth = camera.project(points)
    region, _ = silhouette(camera, points)
    if region is None or (depth <= 0.02).any():
        return 0.0
    area = cv2.contourArea(cv2.convexHull(px.astype(np.float32)))
    return min(1.0, np.count_nonzero(region & ~robot) / max(area, 1.0))


def _wrist_offset(pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """(where a wrist camera sits from its grasp point, the way it looks), as world vectors, for a pose
    pointing away from the robot at this pitch."""
    turn = rotation_of(pitch, 0.0, 0.0)
    return turn @ (WRIST_MOUNT[:3, 3] - np.array([GRASP_REACH, 0.0, 0.0])), -(turn @ WRIST_MOUNT[:3, 2])


def _reaching(point, center, size) -> float | None:
    """How near a grasp point is to an object's middle against its extent (0: there, 1: as far as CARRY_GAP
    allows), or None when it is farther."""
    across, up = 0.5 * float(size[0]) + CARRY_GAP, 0.5 * float(size[2]) + CARRY_GAP
    d, dz = math.hypot(point[0] - center[0], point[1] - center[1]), abs(point[2] - center[2])
    return None if d > across or dz > up else math.hypot(d / across, dz / up)


def _box_gap(point, center, size, yaw) -> float:
    """Distance (m) from a point to an object's bounding box (0 inside)."""
    d = np.asarray(point, float) - np.asarray(center, float)
    t = math.radians(yaw or 0.0)
    u = d[0] * math.cos(t) + d[1] * math.sin(t)
    v = -d[0] * math.sin(t) + d[1] * math.cos(t)
    out = np.maximum(np.abs([u, v, d[2]]) - 0.5 * np.asarray(size, float), 0.0)
    return float(np.linalg.norm(out))


@dataclass
class _Target:
    """Where a measurement starts: one outline, and what else is known about the object."""

    mask: Mask
    view: View
    label: str
    text: str | None = None  # words that find it
    given: list[tuple[Mask, View]] | None = None  # outlines in other cameras chosen by the caller
    text_found: dict[str, list] = field(default_factory=dict)  # cam -> SAM3's answer to the words
    clicks: list[np.ndarray] = field(default_factory=list)  # world points to click at in other cameras
    notes: list[str] = field(default_factory=list)
    click: tuple[float, float] | None = None  # the pixel the outline was clicked at, if it was
    place: np.ndarray | None = None  # a remembered object's solid where it was, when measuring it again


@dataclass
class _Record:
    """A measured object, as remembered for measuring it again."""

    thing: Thing
    text: str | None
    solid: np.ndarray  # corners of its bounding solid
    bottom: float
    height: float | None  # measured (two or more views), or remembered from such a measurement
    colour: np.ndarray | None
    yaw: float = 0.0  # direction of its long side (degrees)
    held_by: str | None = None
    set_down: bool = False  # a hand has carried it since it was measured: solid is where it let go


class Scene:
    def __init__(self, url: str, *, log: Callable[[dict], None] | None = None, client: Sam3Client | None = None):
        self.log = log or (lambda event: None)
        self.sam = client or Sam3Client(url)
        try:
            health = self.sam.health()
        except Sam3Unavailable as exc:
            raise RuntimeError(f"the SAM3 service at {url} is not answering ({exc})") from None
        if not health.get("ok"):
            raise RuntimeError(f"the SAM3 service at {url} is not ready: {health}")
        self.host = None
        self.masks: OrderedDict[int, tuple[Mask, View]] = OrderedDict()
        self._views: tuple[int, dict[str, View]] | None = None
        self.next_mask = 1
        self.next_thing = 1
        self.records: dict[int, _Record] = {}
        # per hand: Thing id -> the grasp pose its record's solid goes with (see grasped)
        self.carried: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in ARMS}
        self._chains: dict[str, object] = {}
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="sam3")

    def bind(self, host) -> None:
        self.host = host

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self.sam.close()

    # -- pictures ----------------------------------------------------------------------------------
    def views(self) -> dict[str, View]:
        """The current pictures (cached per observation, so robot masks are drawn once)."""
        snap = self.host.snap
        if self._views is None or self._views[0] != snap.seq:
            views = views_of(snap, self._openings(snap))
            for arm in ARMS:
                view = views.get(f"{arm}_wrist")
                if view is not None and self._holding(arm):
                    view.outline_held = lambda v, arm=arm: self._held_region(arm, v)
            self._views = (snap.seq, views)
        return self._views[1]

    def _held_region(self, arm: str, view: View) -> np.ndarray:
        """What the gripper holds, in its own wrist camera's picture (see HELD_MAX_SHARE)."""
        from ..motion.chain import slide

        w, h = view.camera.width, view.camera.height
        reading = view.arms.get(arm)
        if reading is None:
            return np.zeros((h, w), bool)
        best, how = None, "fingers"
        px = view.camera.pixel(slide(reading.flange, GRASP_REACH)[:3, 3])
        if px is not None:
            xi, yi = min(max(int(round(px[0])), 0), w - 1), min(max(int(round(px[1])), 0), h - 1)
            try:
                segments = self.segment(view, points=[px])
            except Refused:
                segments = []
            for seg in segments:
                if seg.mask[yi, xi] and np.count_nonzero(seg.mask) <= HELD_MAX_SHARE * seg.mask.size \
                        and (best is None or seg.score > best[1]):
                    best = (seg.mask, float(seg.score))
        if best is not None:
            region, how = best[0].astype(np.uint8), "sam3"
        else:
            region = np.zeros((h, w), np.uint8)
            ys, xs = np.nonzero(fingers_mask(view.camera, view.arms, view.openings, arm=arm))
            if len(xs) >= 3:
                cv2.fillConvexPoly(region, cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32)), 1)
        if HELD_PAD_PX:
            region = cv2.dilate(region, np.ones((2 * HELD_PAD_PX + 1,) * 2, np.uint8))
        region = region > 0
        self.log({"kind": "held", "cam": view.cam, "by": how, "share": round(float(region.mean()), 3)})
        return region

    def view(self, cam: str) -> View:
        view = self.views().get(cam)
        if view is None:
            raise Refused(f"no {cam} picture in the latest observation")
        return view

    def _openings(self, snap: Snapshot) -> dict[str, float] | None:
        """Finger openings for squeezing grippers (their readings lag the fingers)."""
        driver = getattr(self.host, "driver", None)
        if driver is None:
            return None
        arms = snap.arms_in_pictures
        out = {arm: min(1.0, float(arms[arm].gripper) + SQUEEZE_LAG) for arm in ARMS if driver.squeezing(arm)}
        return out or None

    # -- outlines ----------------------------------------------------------------------------------
    def segment(self, view: View, **prompt) -> list:
        t0 = time.monotonic()
        if prompt.get("points"):
            # a pixel may lie up to half a pixel past the edge centres; SAM3 takes only 0..size-1
            w, h = view.camera.width - 1, view.camera.height - 1
            prompt["points"] = [(min(max(float(x), 0.0), w), min(max(float(y), 0.0), h))
                                for x, y in prompt["points"]]
        try:
            found = self.sam.segment(view.image, **prompt)
        except Sam3Unavailable as exc:
            raise Refused(f"the SAM3 service is not answering right now ({exc}); try again in a moment") from None
        except Sam3Error as exc:
            raise Refused(f"SAM3 refused the request: {exc}") from None
        top = sorted(found, key=lambda s: -s.score)[:LOG_TOP]
        self.log({"kind": "sam3", "cam": view.cam, "prompt": {k: v for k, v in prompt.items() if v is not None},
                  "found": len(found), "s": round(time.monotonic() - t0, 3),
                  "top": [[round(float(s.score), 3), [int(v) for v in s.box], int(s.area)] for s in top]})
        return found

    def _segment_each(self, views: list[View], **prompt) -> dict[str, list]:
        """The same prompt in several pictures at once."""
        jobs = {v.cam: self._pool.submit(self.segment, v, **prompt) for v in views}
        return {cam: job.result() for cam, job in jobs.items()}

    def register(self, view: View, pixels: np.ndarray, label: str, score: float, prompt: dict) -> Mask:
        area = int(np.count_nonzero(pixels))
        ys, xs = np.nonzero(pixels)
        share = float(np.count_nonzero(pixels & view.robot)) / max(area, 1)
        mask = Mask(id=self.next_mask, cam=view.cam, label=label, score=round(float(score), 3), pixels=pixels,
                    box=_box_of(pixels), centroid=(round(float(xs.mean()), 1), round(float(ys.mean()), 1)),
                    area=area, seq=view.seq, robot_share=round(share, 3), prompt=prompt)
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

    def mask_and_view(self, mask: Mask) -> tuple[Mask, View]:
        """The registered outline behind a Mask from a cell, with the cell's pixels if they still fit."""
        entry = self.masks.get(getattr(mask, "id", None))
        if entry is None:
            raise Refused(f"Mask#{getattr(mask, 'id', '?')} is not known (too old, or not from find)")
        known, view = entry
        pixels = getattr(mask, "pixels", None)
        if (isinstance(pixels, np.ndarray) and pixels.shape == known.pixels.shape and pixels.dtype == bool
                and pixels.any() and not np.array_equal(pixels, known.pixels)):
            known = Mask(**{**known.__dict__, "pixels": pixels, "area": int(np.count_nonzero(pixels)),
                            "box": _box_of(pixels)})
        return known, view

    # -- measuring ---------------------------------------------------------------------------------
    def measure(self, target, *, cams="auto") -> Thing:
        allowed = _cams(cams)
        t0 = time.monotonic()
        if isinstance(target, Thing):
            thing = self._again(target, allowed)
        else:
            start = self._target(target, allowed)
            thing = start if isinstance(start, Thing) else self._measure(start, allowed)[0]
        self.log({"kind": "measure", "target": _describe(target), "id": thing.id, "found": thing.found,
                  "views": thing.views, "center": thing.center_xyz, "size": thing.size, "warning": thing.warning,
                  "s": round(time.monotonic() - t0, 3)})
        return thing

    def things(self) -> list[Thing]:
        return [self.records[i].thing for i in sorted(self.records)]

    # -- what the hands carry ----------------------------------------------------------------------
    def grasped(self, tcp, hand: str, thing: int | None = None) -> None:
        """A close of ``hand`` caught something at grasp pose ``tcp``: the measured object it caught (see
        CARRY_GAP; ``thing``: the motion driver's pick) moves with the hand till :meth:`released`, so that
        measuring it again looks where the hand has it, or set it down."""
        tcp = np.asarray(tcp, float)
        self.released(tcp, hand)
        best = None
        for thing_id, rec in self.records.items():
            near = _reaching(tcp[:3, 3], self._solid_now(thing_id, rec).mean(axis=0), rec.thing.size)
            if near is not None and (best is None or (thing_id != thing, near) < best[0]):
                best = ((thing_id != thing, near), thing_id)
        if best is None:
            return
        caught = best[1]
        for arm, held in self.carried.items():
            if arm != hand and caught in held:  # taken from the other hand: where that hand has it now
                self._set_down(caught, self._grasp_pose(arm) @ np.linalg.inv(held.pop(caught)))
        self.carried[hand][caught] = tcp.copy()
        self.log({"kind": "carry", "arm": hand, "held": [caught], "pick": thing})

    def released(self, tcp, hand: str) -> None:
        """``hand`` let go at grasp pose ``tcp`` (or what it held slipped out there): what it carried is
        taken to be where the hand brought it."""
        held = self.carried.get(hand)
        if not held:
            return
        tcp = np.asarray(tcp, float)
        for thing_id, grasp in held.items():
            self._set_down(thing_id, tcp @ np.linalg.inv(grasp))
        self.log({"kind": "carry", "arm": hand, "set": sorted(held)})
        self.carried[hand] = {}

    def _set_down(self, thing_id: int, move: np.ndarray) -> None:
        """Its record moved as the hand moved (``move``) between taking it and letting go."""
        rec = self.records.get(thing_id)
        if rec is None:
            return
        rec.solid = rec.solid @ move[:3, :3].T + move[:3, 3]
        rec.bottom = float(rec.solid[:, 2].min())
        rec.yaw = float(rec.yaw + math.degrees(math.atan2(move[1, 0], move[0, 0])))
        rec.held_by = None
        rec.set_down = True

    def _carrier(self, thing_id: int) -> str | None:
        """The hand that carries the object, while it holds something."""
        return next((arm for arm, held in self.carried.items() if thing_id in held and self._holding(arm)), None)

    def _solid_now(self, thing_id: int, rec: _Record) -> np.ndarray:
        """A remembered object's solid where it is taken to be now: as measured, or set down, or moved
        with the hand that carries it."""
        arm = self._carrier(thing_id)
        if arm is None:
            return rec.solid
        move = self._grasp_pose(arm) @ np.linalg.inv(self.carried[arm][thing_id])
        return rec.solid @ move[:3, :3].T + move[:3, 3]

    def _target(self, target, allowed: list[str]):
        if isinstance(target, str):
            return self._by_words(target, allowed)
        if isinstance(target, Mask):
            mask, view = self.mask_and_view(target)
            return _Target(mask, view, mask.label, text=mask.prompt.get("text"), click=_clicked(mask.prompt))
        if isinstance(target, (tuple, list)) and len(target) == 2 and isinstance(target[0], str):
            return self._by_pixel(check_cam(target[0]), target[1])
        if isinstance(target, (tuple, list)) and target and all(isinstance(m, Mask) for m in target):
            pairs = [self.mask_and_view(m) for m in target]
            cams = [m.cam for m, _ in pairs]
            if len(set(cams)) != len(cams):
                raise Refused("measure: give at most one Mask per camera (all of the same object)")
            first, view = pairs[0]
            return _Target(first, view, first.label, text=first.prompt.get("text"), given=pairs[1:],
                           click=_clicked(first.prompt))
        raise Refused("measure: target is words ('red cup'), a Mask, (cam, (x, y)), a list of Masks of one "
                      "object in different cameras, or a Thing measured before")

    def _by_words(self, text: str, allowed: list[str], near: np.ndarray | None = None, me: int | None = None):
        text = text.strip()
        if not text:
            raise Refused("measure: the words name the object, e.g. 'red cup'")
        views = self.views()
        present = [views[c] for c in allowed if c in views]
        if not present:
            raise Refused("no pictures from those cameras in the latest observation")
        answers = self._segment_each(present, text=text, min_score=CANDIDATE_MIN_SCORE, max_masks=TEXT_MAX)
        instances = {v.cam: self._instances(v, answers[v.cam]) for v in present}
        cams = [v.cam for v in present if instances[v.cam]]
        if not cams:
            candidates = []
            for v in present:
                for seg in sorted(answers[v.cam], key=lambda s: -s.score)[:2]:
                    m = parts(seg.mask)
                    if np.count_nonzero(m) >= MIN_PIXELS:
                        mask = self.register(v, m, text, seg.score, {"text": text})
                        candidates.append(f"Mask#{mask.id} {v.cam} score {mask.score:.2f} at "
                                          f"({mask.centroid[0]:.0f}, {mask.centroid[1]:.0f})")
            return Thing(id=0, label=text, found=False, candidates=candidates,
                         why_missing=f"no '{text}' outlined in {', '.join(v.cam for v in present)}"
                         + (" (only weak matches, listed below)" if candidates else ""),
                         notes=["the words find by appearance: name how it looks (colour, shape), or point "
                                "at it with robot.find(cam, at=(x, y))"])
        # the head sees the whole table: start there when it shows the object
        cam = cams[0] if cams[0] == "head" else max(cams, key=lambda c: instances[c][0][1])
        view = views[cam]
        masks = [self.register(view, m, text, s, {"text": text}) for m, s in instances[cam]]
        pick = 0
        if near is not None:
            px = view.camera.pixel(near)
            if px is not None:
                # the nearest that does not show where another remembered Thing is
                order = sorted(range(len(masks)),
                               key=lambda i: math.hypot(masks[i].centroid[0] - px[0], masks[i].centroid[1] - px[1]))
                claims = {i: self._claimed(view, masks[i], me) for i in order}
                free = [i for i in order if claims[i] is None]
                if not free:
                    taken = ", ".join(f"Mask#{masks[i].id} (Thing#{claims[i]})" for i in order[:OTHER_MAX])
                    more = f" and {len(order) - OTHER_MAX} more" if len(order) > OTHER_MAX else ""
                    return Thing(id=0, label=text, found=False,
                                 why_missing=f"each '{text}' outlined in {cam} is where another Thing is: {taken}{more}",
                                 notes=["measure one of those Masks if it is the one you mean"])
                pick = free[0]
        chosen = masks[pick]
        notes = []
        floor = min(OTHER_SCORE, OTHER_SHARE * chosen.score)
        others = [m for m in masks if m is not chosen and m.score >= floor]
        if others:
            listed = ", ".join(f"Mask#{m.id} at ({m.centroid[0]:.0f}, {m.centroid[1]:.0f}) score {m.score:.2f}"
                               for m in others[:OTHER_MAX])
            more = f" and {len(others) - OTHER_MAX} more" if len(others) > OTHER_MAX else ""
            notes.append(f"measured Mask#{chosen.id} (score {chosen.score:.2f}); other '{text}' in {cam}: {listed}{more}"
                         " - measure one of those Masks if it is the one you mean")
        return _Target(chosen, view, text, text=text, text_found={c: answers[c] for c in answers if c != cam},
                       notes=notes)

    def _instances(self, view: View, segments) -> list[tuple[np.ndarray, float]]:
        """Distinct outlines worth measuring among SAM3's answer to words, best first."""
        kept: list[tuple[np.ndarray, float]] = []
        for seg in sorted(segments, key=lambda s: -s.score):
            if seg.score < TEXT_MIN_SCORE:
                continue
            m = parts(seg.mask)
            area = int(np.count_nonzero(m))
            if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size:
                continue
            if any(iou(m, k) > SAME_OUTLINE_IOU for k, _ in kept):
                continue
            if np.count_nonzero(m & view.robot) >= ON_ROBOT * area and seg.score < ROBOT_SCORE:
                continue
            kept.append((m, float(seg.score)))
        return kept[:TEXT_MAX]

    def _by_pixel(self, cam: str, xy):
        view = self.view(cam)
        x, y = pixel_of(xy, view.camera)
        xi, yi = int(round(x)), int(round(y))
        best = None
        for seg in self.segment(view, points=[(x, y)]):
            m = parts(seg.mask)
            area = int(np.count_nonzero(m))
            if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size or not seg.mask[yi, xi]:
                continue
            if best is None or seg.score > best[1]:
                best = (m, float(seg.score))
        if best is None:
            return Thing(id=0, label=f"at ({x:.0f}, {y:.0f})", found=False,
                         why_missing=self._covered(view, (x, y)) or f"SAM3 outlines no object at ({x:.0f}, {y:.0f}) "
                         f"in {cam} (only the table, or nothing)")
        mask = self.register(view, best[0], f"at ({x:.0f}, {y:.0f})", best[1], {"points": [(x, y)], "labels": [1]})
        return _Target(mask, view, mask.label, click=(x, y))

    def _measure(self, target: _Target, allowed: list[str], *, prior: _Record | None = None,
                 remember: bool = True) -> tuple[Thing, hull.Shape | None, list[hull.Sight]]:
        view = target.view
        first_mask = parts(target.mask.pixels)
        area = int(np.count_nonzero(first_mask))
        if area < MIN_PIXELS:
            return self._missing(target, prior, f"Mask#{target.mask.id} is too small to measure"), None, []
        if view.held is not None and np.count_nonzero(first_mask & view.held) >= ON_ROBOT * area:
            arm = view.cam.removesuffix("_wrist")
            return self._missing(target, prior, f"Mask#{target.mask.id} shows what the {arm} gripper holds, too close "
                                 "to its own wrist camera to measure: measure it in the head camera, or work out "
                                 "where it is from the gripper", covered=False), None, []
        if np.count_nonzero(first_mask & view.robot) >= ON_ROBOT * area and target.mask.score < ROBOT_SCORE:
            return self._missing(target, prior, f"Mask#{target.mask.id} lies on the robot's own arm or gripper"), \
                None, []
        first = hull.Sight(view.cam, view.camera, first_mask, view.blind)
        colour = colour_of(view.lab, first_mask, view.blind)
        used: dict[str, tuple[Mask, hull.Sight]] = {view.cam: (target.mask, first)}
        notes = list(target.notes)
        if view.seq != self.host.snap.seq:
            notes.append(f"Mask#{target.mask.id} is from an earlier picture: if the object moved since, find it again")
        if target.given is not None:
            for m, v in target.given:
                used[m.cam] = (m, hull.Sight(v.cam, v.camera, parts(m.pixels), v.blind))
        else:
            views = self.views()
            matcher = Matcher(self.segment, first, colour, text=target.text, text_min_score=CANDIDATE_MIN_SCORE)
            for cam in allowed:
                other = views.get(cam)
                if cam in used or other is None:
                    continue
                cand = matcher.find_in(other, text_found=target.text_found.get(cam), clicks=target.clicks)
                if cand is not None:
                    m = self.register(other, cand.mask, target.label, cand.score,
                                      {"same as": f"Mask#{target.mask.id}", "by": cand.how})
                    used[cam] = (m, hull.Sight(cam, other.camera, cand.mask, other.blind))
        sights = [s for _, s in used.values()]
        height = prior.height if prior is not None else None
        floor = TABLE_Z
        if prior is not None and prior.held_by is None and not prior.set_down and prior.height is not None \
                and target.clicks:
            floor = prior.bottom  # found where it was: still on the same support
        shape, sights = self._solve(sights, height, floor, notes)
        if not shape.ok:
            why = shape.notes[-1] if shape.notes else "no solid fits the outlines"
            return self._missing(target, prior, why), shape, sights
        thing = self._thing(target, shape, sights, {s.cam: used[s.cam][0] for s in sights}, colour, prior, notes,
                            remember=remember, floor=floor, height_known=height is not None)
        return thing, shape, sights

    def _solve(self, sights: list[hull.Sight], height: float | None, floor: float,
               notes: list[str]) -> tuple[hull.Shape, list[hull.Sight]]:
        if len(sights) >= 2:
            shape = hull.shape(sights)
            while shape.ok and len(sights) >= 3:
                worst = min(sights[1:], key=lambda s: shape.explained.get(s.cam, 0.0))
                if shape.explained.get(worst.cam, 0.0) >= DISAGREE:
                    break
                notes.append(f"left out the {worst.cam} outline: it does not fit the others")
                sights = [s for s in sights if s is not worst]
                shape = hull.shape(sights)
            if shape.ok:
                return shape, sights
            notes.append(f"the {', '.join(s.cam for s in sights[1:])} outline does not fit the {sights[0].cam} one: "
                         f"measured from {sights[0].cam} alone")
            sights = sights[:1]
        return hull.single(sights[0], floor, height=height), sights

    def _thing(self, target: _Target, shape: hull.Shape, sights: list[hull.Sight], masks: dict[str, Mask], colour,
               prior: _Record | None, notes: list[str], *, remember: bool, floor: float, height_known: bool) -> Thing:
        snap = self.host.snap
        center = tuple(round(float(v), 4) for v in shape.center)
        size = tuple(round(float(v), 4) for v in shape.size)
        top = round(float(shape.top), 4)
        rec = prior
        if rec is None:
            rec = self._identify(target.view, masks[target.view.cam].pixels, colour)
        thing_id = rec.thing.id if rec is not None else 0
        on, on_z, held_by = self._support(shape, thing_id)
        warnings = []
        hull_notes = [n for n in shape.notes if not n.startswith(("seen by one camera", "seen only from above"))]
        advised = False
        if shape.single:
            cam = shape.views[0]
            above = any(n.startswith("seen only from above") for n in shape.notes)
            height = "unknown" if above else "guessed"
            blocker = self._hiding_from_head(center) if cam != "head" else None
            advised = not height_known
            if height_known:
                warnings.append(f"1 view ({cam}): height from its earlier measurement")
            elif blocker is not None:
                warnings.append(f"1 view ({cam}{', from above' if above else ''}): height {height} - the {blocker} "
                                f"arm hides it from the head camera; {self._hover_advice(shape, held_by)}")
            elif above:
                warnings.append(f"1 view ({cam}, looking straight down on it): height unknown - "
                                f"{self._hover_advice(shape, held_by)}")
            else:
                warnings.append(f"1 view ({cam}): height guessed - {self._hover_advice(shape, held_by)}")
        else:
            worst = min(shape.explained.get(c, 0.0) for c in shape.views)
            if worst < DISAGREE:
                warnings.append(f"the cameras agree only roughly ({100 * worst:.0f}% of an outline explained): "
                                "check with robot.look('all', masks=[thing], tile=True)")
            standing = self._standing_instead(shape, sights, floor) if held_by is None else None
            if standing is not None:
                others = [s.cam for s in sights[1:]]
                warnings.append(f"the {' and '.join(others)} camera{'s see' if len(others) > 1 else ' sees'} only "
                                f"part of it: it may stand {1000 * standing.size[2]:.0f} mm tall at "
                                f"({standing.center[0]:.3f}, {standing.center[1]:.3f}), as the {sights[0].cam} "
                                f"picture alone suggests, not lie low - {self._hover_advice(standing, held_by)}")
                advised = True
        if held_by is not None:
            warnings.insert(0, f"held by {held_by}: the fingers hide part of it")
        else:
            hidden, more = self._hidden(sights, shape, advised=advised)
            warnings += hidden
            notes += more
        warning = "; ".join(warnings) or None
        if shape.kind == "hangs":
            hull_notes = [n for n in hull_notes if not n.startswith("widest at the top")]
            if shape.size[2] >= HANG_NOTE_MIN:
                notes.append("widest at the top (a bowl or cup, or a tall can or bottle): top is its rim, size the "
                             "rim's; the top can read high, not low - of several alike, the lowest top is closest")
        notes.extend(hull_notes)
        if prior is not None and prior.thing.found and prior.thing.center_xyz is not None:
            moved = math.dist(center[:2], prior.thing.center_xyz[:2])
            if moved > MOVED and prior.height is not None:
                notes.append(f"moved {100 * moved:.1f} cm since it was last measured")
            elif moved > MOVED:
                notes.append(f"{100 * moved:.1f} cm from where the earlier one-camera measurement put it")
        if not remember:
            return Thing(id=thing_id, label=target.label, found=True, center_xyz=center, top=top, size=size,
                         grip_yaw=None if shape.round else round(float(shape.grip_yaw), 1), on=on, on_z=on_z,
                         views=list(shape.views), err_mm=_round_err(shape.err_mm), warning=warning, masks=masks,
                         seq=snap.seq, round=shape.round, notes=notes, step=snap.step)
        if thing_id == 0:
            thing_id = self.next_thing
            self.next_thing += 1
        label = target.label
        if rec is not None and target.text is None and rec.text is not None:
            label = rec.thing.label  # keep the words over a pixel label
        thing = Thing(id=thing_id, label=label, found=True, center_xyz=center, top=top, size=size,
                      grip_yaw=None if shape.round else round(float(shape.grip_yaw), 1), on=on, on_z=on_z,
                      views=list(shape.views), err_mm=_round_err(shape.err_mm), warning=warning, masks=masks,
                      seq=snap.seq, round=shape.round, notes=notes, step=snap.step)
        bottom = float(shape.bottom)
        if not shape.single:
            height = float(shape.top - shape.bottom)
        elif height_known and prior is not None:
            height = prior.height
        else:
            height = None
        text = target.text or (rec.text if rec is not None else None)
        self.records[thing_id] = _Record(thing=thing, text=text,
                                         solid=solid_points(center, size, shape.yaw, shape.round, bottom, top),
                                         bottom=bottom, height=height, colour=colour, yaw=float(shape.yaw or 0.0),
                                         held_by=held_by)
        # measured in a hand, or about where the hand that carries it has it: it goes on with that hand from
        # here; measured anywhere else, no hand carries it
        for arm, held in self.carried.items():
            if arm == held_by or (held_by is None and thing_id in held and self._holding(arm)
                                  and _reaching(self._grasp_pose(arm)[:3, 3], center, size) is not None):
                held[thing_id] = self._grasp_pose(arm)
            else:
                held.pop(thing_id, None)
        return thing

    @staticmethod
    def _standing_instead(shape: hull.Shape, sights: list[hull.Sight], floor: float) -> hull.Shape | None:
        """The first camera's own reading of an object the other cameras see only part of, when that
        reads it standing much taller than the box lying low and long down its view (see CONFIRMED_MIN)."""
        first = sights[0]
        if len(sights) < 2 or not first.whole or any(s.whole for s in sights[1:]):
            return None
        if shape.confirmed.get(first.cam, 1.0) >= CONFIRMED_MIN:
            return None
        down = np.asarray(shape.center[:2], float) - first.camera.position[:2]
        length, width = float(shape.size[0]), float(shape.size[1])
        if shape.round or np.linalg.norm(down) < 1e-6:
            return None
        phi = math.radians(shape.yaw or 0.0) - math.atan2(down[1], down[0])
        along = length * abs(math.cos(phi)) + width * abs(math.sin(phi))
        across = length * abs(math.sin(phi)) + width * abs(math.cos(phi))
        if along < LONG_ALONG * across:
            return None
        one = hull.single(first, floor)
        tall = float(shape.size[2])
        if not one.ok or one.size[2] < max(TALLER_RATIO * tall, tall + TALLER_BY):
            return None
        return one

    def _missing(self, target: _Target, prior: _Record | None, why: str, *, covered: bool = True) -> Thing:
        if covered and target.click is not None:
            why = self._covered(target.view, target.click) or why
        elif covered and target.place is not None:
            views = self.views()
            hidden = self._arm_over([views[c] for c in ORDER if c in views], target.place)
            if hidden is not None:
                why = f"{why} - {hidden}: move the arm aside and measure again"
        return Thing(id=prior.thing.id if prior is not None else 0, label=target.label, found=False,
                     why_missing=why, notes=list(target.notes), masks={target.view.cam: target.mask},
                     seq=self.host.snap.seq, step=self.host.snap.step)

    @staticmethod
    def _covered(view: View, xy) -> str | None:
        """Why an object clicked at this pixel is not measured, when the robot covers the pixel in the
        picture (see ON_ROBOT_PATCH): which arm, and what to do."""
        labels = robot_labels(view.camera, view.arms, view.openings)
        h, w = labels.shape
        x, y = min(max(int(round(xy[0])), 0), w - 1), min(max(int(round(xy[1])), 0), h - 1)
        value = int(labels[y, x])
        if value == 0:
            patch = labels[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            if np.count_nonzero(patch) <= ON_ROBOT_PATCH * patch.size:
                return None
            values, counts = np.unique(patch[patch > 0], return_counts=True)
            value = int(values[np.argmax(counts)])
        arm = label_name(value).split()[0]
        said = _covering(arm, view.cam, f"({xy[0]:.0f}, {xy[1]:.0f})")
        if view.cam == f"{arm}_wrist":
            return f"{said}: measure it in another camera's picture"
        return f"{said}: move the arm aside and measure again, or measure it in another camera's picture"

    def _identify(self, view: View, pixels: np.ndarray, colour) -> _Record | None:
        """The remembered object this outline shows, if any."""
        best = None
        for thing_id, rec in self.records.items():
            region, share = silhouette(view.camera, self._solid_now(thing_id, rec))
            if region is None or share < 0.5:
                continue
            overlap = iou(region, pixels)
            if overlap < SAME_THING_IOU:
                continue
            gap = colour_gap(rec.colour, colour)
            if math.isfinite(gap) and gap > SAME_THING_COLOUR:
                continue
            if best is None or overlap > best[0]:
                best = (overlap, rec)
        return best[1] if best is not None else None

    def _claimed(self, view: View, mask: Mask, me: int | None) -> int | None:
        """The number of another remembered Thing this outline shows (where it is now), if any."""
        pixels = parts(mask.pixels)
        rec = self._identify(view, pixels, colour_of(view.lab, pixels, view.blind))
        return rec.thing.id if rec is not None and rec.thing.id != me else None

    def _grasp_pose(self, arm: str) -> np.ndarray:
        from ..motion.chain import slide

        return slide(self.host.snap.arms[arm].flange, GRASP_REACH)

    def _holding(self, arm: str) -> bool:
        driver = getattr(self.host, "driver", None)
        return bool(driver is not None and driver.holding(arm))

    def _support(self, shape: hull.Shape, me: int) -> tuple[str, float | None, str | None]:
        """What the object rests on: (on, on_z, arm holding it)."""
        for arm in ARMS:
            if self._holding(arm):
                grip = self._grasp_pose(arm)[:3, 3]
                if _box_gap(grip, shape.center, shape.size, shape.yaw) <= HELD_GAP:
                    return f"held by {arm}", None, arm
        bottom = float(shape.bottom)
        if bottom <= TABLE_Z + ON_TABLE:
            return "table", TABLE_Z, None
        best = None
        for rec in self.records.values():
            t = rec.thing
            if t.id == me or not t.found or t.top is None or rec.held_by is not None or rec.set_down \
                    or any(t.id in held for held in self.carried.values()):
                continue  # (not where it was measured)
            if abs(t.top - bottom) > STACK_GAP:
                continue
            gap = _box_gap((shape.center[0], shape.center[1], t.top - 0.5 * t.size[2]),
                           t.center_xyz, t.size, rec.yaw)
            if gap > 0.01:
                continue
            if best is None or abs(t.top - bottom) < abs(best.top - bottom):
                best = t
        if best is not None:
            return f"Thing#{best.id}", best.top, None
        return f"something at z {bottom:.3f}", round(bottom, 4), None

    def _hover_advice(self, shape: hull.Shape, held_by: str | None) -> str:
        busy = {arm for arm in ARMS if self._holding(arm)} | ({held_by} if held_by else set())
        got = self._look_pose(shape, busy)
        if got is None:
            alt = self._look_pose(shape, set()) if busy else None
            if alt is not None and alt[0] in busy:
                return (f"only the {alt[0]} wrist camera could look down on it, and what the {alt[0]} gripper holds "
                        "would hide it: take its size as rough, or put that down first")
            return "no wrist camera can look down on it from close by: take its size as rough"
        arm, point, pitch, clear = got
        text = (f"for a position within a few mm, look down on it with a wrist camera: robot.goto('{arm}', ({point[0]:.3f}, {point[1]:.3f}, "
                f"{point[2]:.3f}), pitch={pitch}, yaw=0), then measure this Thing again")
        worst = min(clear, key=clear.get)
        if clear[worst] < LOOK_CLEAR:
            text += f" (the robot will still hide about {100 * (1 - clear[worst]):.0f}% of it in {worst})"
        return text

    def _look_pose(self, shape: hull.Shape, busy: set[str]):
        """(arm, grasp point, pitch, {camera: share of the object left free}) of the best pose for a wrist
        camera to look down on an object (see HOVER_ABOVE), or None when no arm reaches any."""
        snap = getattr(self.host, "snap", None)
        if snap is None or not shape.ok or shape.size is None or shape.top is None:
            return None
        views = self.views()
        head = views.get("head")
        openings = head.openings if head is not None else None
        bottom = shape.bottom if shape.bottom is not None else shape.top - shape.size[2]
        corners = solid_points(shape.center, shape.size, shape.yaw, shape.round, bottom, shape.top)
        target = np.array([shape.center[0], shape.center[1], shape.top])
        near = "left" if shape.center[0] < 0 else "right"
        arms = [a for a in (near, "right" if near == "left" else "left") if a not in busy] or [near]
        # the camera's distance to the top: the one the best pose (over it, pitch 60) gives, or for a big
        # object first the one from which it fills HOVER_SHARE of the picture
        above = max(HOVER_MIN_Z, shape.top + HOVER_ABOVE)
        lens, _ = _wrist_offset(60)
        close = above + lens[2] - shape.top
        fits = WRIST_FOCAL * math.sqrt(float(shape.size[0] * shape.size[1]) / (HOVER_SHARE * WIDTH * HEIGHT))
        best = None
        for distance in ([fits] if fits > close else []) + [close]:
            for pitch in LOOK_PITCHES:
                lens, axis = _wrist_offset(pitch)
                centred = target - distance * axis - lens
                back = GRASP_REACH * rotation_of(pitch, 0.0, 0.0)[:, 0]
                for aside in (0.0, LOOK_ASIDE):
                    for arm in arms:
                        point = centred + np.array([-aside if arm == "left" else aside, 0.0, 0.0])
                        if np.linalg.norm(point - back - np.asarray(ARM_BASES[arm][0])) > ARM_SPAN:
                            continue
                        if arm not in self._chains:
                            self._chains[arm] = arm_chain(arm)
                        chain = self._chains[arm]
                        sol, _ = ik.solve(chain, homogeneous(rotation_of(pitch, 0.0, 0.0), point),
                                          snap.arms[arm].joints)
                        if not (sol.reached and sol.clear):
                            continue
                        there = dict(snap.arms)
                        flange = chain.flange(sol.q)
                        there[arm] = ArmReading(joints=sol.q, flange=flange, gripper=snap.arms[arm].gripper)
                        cams = [Camera(f"{arm}_wrist", WRIST_FOCAL, wrist_pose(flange))]
                        if head is not None:
                            cams.append(head.camera)
                        clear = {cam.name: _free_share(cam, corners, robot_mask(cam, there, openings))
                                 for cam in cams}
                        if min(clear.values()) >= LOOK_CLEAR:
                            return arm, point, pitch, clear
                        if best is None or min(clear.values()) > min(best[3].values()):
                            best = (arm, point, pitch, clear)
            if best is not None:
                return best  # from closer the object would fill too much of the picture
        return best

    def _arm_hiding(self, view: View, points: np.ndarray) -> str | None:
        """The arm whose picture covers most of these points' pixels in a view, if any."""
        px, depth = view.camera.project(points)
        ok = (depth > 0.02) & (px[:, 0] >= 0) & (px[:, 0] <= view.camera.width - 1) & (px[:, 1] >= 0) \
            & (px[:, 1] <= view.camera.height - 1)
        if not ok.any():
            return None
        xi, yi = np.round(px[ok, 0]).astype(int), np.round(px[ok, 1]).astype(int)
        best = None
        for arm in ARMS:
            if arm not in view.arms:
                continue
            m = robot_mask(view.camera, {arm: view.arms[arm]}, view.openings)
            if view.held is not None and view.cam == f"{arm}_wrist":
                m = m | view.held  # with what it holds
            share = float(m[yi, xi].mean())
            if share >= 0.3 and (best is None or share > best[1]):
                best = (arm, share)
        return best[0] if best else None

    def _hiding_from_head(self, center) -> str | None:
        """The arm that hides this point from the head camera, if one does."""
        head = self.views().get("head")
        if head is None:
            return None
        c = np.asarray(center, float)
        ring = np.array([c + [dx, dy, 0.0] for dx in (-0.01, 0.0, 0.01) for dy in (-0.01, 0.0, 0.01)])
        return self._arm_hiding(head, ring)

    def _hidden(self, sights: list[hull.Sight], shape: hull.Shape, *, advised: bool) -> tuple[list[str], list[str]]:
        """(warnings, notes) about sides of the object that no camera sees past.  advised: the
        warnings already say where to look from."""
        views = self.views()
        sides: dict[str, tuple[float, list[str], list[str]]] = {}  # side -> (reach, reasons, arms)
        for h in hull.hidden_reach(sights, shape):
            if h.reach < HIDDEN_NOTE:
                continue
            reasons, arms = [], []
            for cam, why in h.why.items():
                if why == "robot":
                    view = views.get(cam)
                    arm = self._arm_hiding(view, h.points) if view is not None else None
                    reasons.append(f"behind the {arm + ' arm' if arm else 'robot'} in {cam}")
                    arms += [arm] if arm else []
                elif why == "edge":
                    reasons.append(f"outside the {cam} picture")
            if not reasons:
                continue
            side = _side(h.normal)  # a round object's eight directions share four names
            reach, why0, arms0 = sides.get(side, (0.0, [], []))
            sides[side] = (max(reach, h.reach), list(dict.fromkeys(why0 + reasons)),
                           list(dict.fromkeys(arms0 + arms)))
        ranked = sorted(sides.items(), key=lambda kv: -kv[1][0])
        notes = [f"may reach up to {1000 * reach:.0f} mm further on its {side} side ({', '.join(why)})"
                 for side, (reach, why, _) in ranked if reach < HIDDEN_WARN]
        warned = [(side, entry) for side, entry in ranked if entry[0] >= HIDDEN_WARN]
        if not warned:
            return [], notes
        reach_text = " and ".join(f"{1000 * reach:.0f} mm further on its {side} side ({', '.join(why)})"
                                  for side, (reach, why, _) in warned)
        arms = list(dict.fromkeys(arm for _, (_, _, those) in warned for arm in those))
        if arms:
            fix = f" - move the {' and '.join(arms)} arm{'s' if len(arms) > 1 else ''} out of the way"
        else:
            fix = "" if advised else f" - {self._hover_advice(shape, None)}"
        return [f"partly hidden: may reach up to {reach_text}; center and size may be off{fix}"], notes

    # -- measuring a remembered object again ---------------------------------------------------------
    def _again(self, thing: Thing, allowed: list[str]) -> Thing:
        rec = self.records.get(thing.id)
        if rec is None:
            raise Refused(f"Thing#{thing.id} is not one this episode measured: measure it from words, a Mask or a "
                          "pixel")
        views = self.views()
        held = self._carrier(thing.id) is not None
        points = self._solid_now(thing.id, rec)
        middle = points.mean(axis=0)
        present = [views[c] for c in allowed if c in views]
        place = None if held else points  # where an arm may cover it (a held one is in the fingers)
        text_found = self._segment_each(present, text=rec.text, min_score=CANDIDATE_MIN_SCORE,
                                        max_masks=TEXT_MAX) if rec.text else {}
        best: dict[str, tuple[float, np.ndarray, float]] = {}
        # A one-camera measurement's solid is a guess (its height above all), so its outline may
        # hardly overlap the object's: then an outline that covers it, in the remembered colour, will do.
        guessed = rec.height is None
        loose: dict[str, tuple[float, np.ndarray, float]] = {}
        for view in present:
            region, share = silhouette(view.camera, points)
            if region is None or share < 0.5:
                continue
            segments = list(text_found.get(view.cam, []))
            click = self._click_for(view, region, middle)
            if click is not None:
                segments += self.segment(view, points=[click])
            for seg in segments:
                m = parts(seg.mask)
                area = int(np.count_nonzero(m))
                if area < MIN_PIXELS or area > MAX_PICTURE_SHARE * m.size:
                    continue
                if view.held is not None and np.count_nonzero(m & view.held) >= ON_ROBOT * area:
                    continue  # what the gripper holds, over the remembered object
                overlap = iou(m, region)
                if overlap >= REFIND_IOU:
                    if view.cam not in best or overlap > best[view.cam][0]:
                        best[view.cam] = (overlap, m, float(seg.score))
                    continue
                if not guessed:
                    continue
                cover = np.count_nonzero(m & region) / max(1, min(area, int(np.count_nonzero(region))))
                gap = colour_gap(rec.colour, colour_of(view.lab, m, view.blind))
                if cover < REFIND_COVER or not math.isfinite(gap) or gap > SAME_THING_COLOUR:
                    continue
                rank = cover * float(seg.score)
                if view.cam not in loose or rank > loose[view.cam][0]:
                    loose[view.cam] = (rank, m, float(seg.score))
        label = rec.thing.label
        if not best and loose:
            best = loose
        if best:
            top_iou = max(b[0] for b in best.values())
            cam = next(c for c in ORDER if c in best)
            if best[cam][0] < top_iou - 0.15:
                cam = max(best, key=lambda c: best[c][0])
            view = views[cam]
            overlap, m, score = best[cam]
            mask = self.register(view, m, label, score, {"again": f"Thing#{thing.id}"})
            target = _Target(mask, view, label, text=rec.text,
                             text_found={c: a for c, a in text_found.items() if c != cam}, clicks=[middle],
                             place=place)
            return self._measure(target, allowed, prior=rec)[0]
        there = "where the hand set it down" if rec.set_down else "where it was last measured"
        if rec.text:
            start = self._by_words(rec.text, allowed, near=middle, me=thing.id)
            if isinstance(start, Thing):
                start.id = thing.id
                start.why_missing = f"not {there if rec.set_down else 'where it was'}, and {start.why_missing}"
                hidden = self._arm_over(present, place) if place is not None else None
                if hidden is not None:
                    start.why_missing += f" - {hidden}: move the arm aside and measure again"
                return start
            start.notes.insert(0, f"not found {there}: measured the '{rec.text}' nearest to there")
            return self._measure(start, allowed, prior=rec)[0]
        why = f"nothing is {there} (it moved?)"
        hidden = self._arm_over(present, place) if place is not None else None
        if hidden is not None:
            why += f" - {hidden}: move the arm aside and measure again, or find it again in the pictures"
        else:
            why += ": find it again in the pictures"
        return Thing(id=thing.id, label=label, found=False, why_missing=why,
                     seq=self.host.snap.seq, step=self.host.snap.step)

    @staticmethod
    def _arm_over(views: list[View], points: np.ndarray) -> str | None:
        """Which arm covers the place of these points in the first picture where one covers more than
        REFIND_HIDDEN of it (see there), said as 'the left arm covers that place in the head picture'."""
        for view in views:
            region, share = silhouette(view.camera, points)
            if region is None or share < 0.5 or not region.any():
                continue
            on = robot_labels(view.camera, view.arms, view.openings)[region]
            if np.count_nonzero(on) <= REFIND_HIDDEN * on.size:
                continue
            return _covering(label_name(int(np.bincount(on[on > 0]).argmax())).split()[0], view.cam, "that place")
        return None

    @staticmethod
    def _click_for(view: View, region: np.ndarray, middle: np.ndarray) -> tuple[float, float] | None:
        """A pixel to click at: the remembered middle, or the region's pixel nearest it off the robot (and
        off what the gripper holds)."""
        free = region & ~view.blind
        if not free.any():
            return None
        px = view.camera.pixel(middle)
        if px is not None:
            xi, yi = int(round(px[0])), int(round(px[1]))
            if 0 <= xi < view.camera.width and 0 <= yi < view.camera.height and free[yi, xi]:
                return float(xi), float(yi)
        ys, xs = np.nonzero(free)
        if px is None:
            cx, cy = xs.mean(), ys.mean()
        else:
            cx, cy = px
        i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        return float(xs[i]), float(ys[i])

    # -- the surface under a pixel -----------------------------------------------------------------
    def surface_spot(self, cam: str, xy) -> Spot:
        """The surface under a pixel: the table top where the picture shows bare table, else where the
        other pictures show the same patch of surface (exact on textured surfaces), else where the
        pixel's ray enters the object outlined there, else the table top.  When the last two both
        exist and differ, the spot says so and lists the other."""
        view = self.view(cam)
        camera = view.camera
        x, y = pixel_of(xy, camera)
        match = level_depth(self.views(), cam, (x, y))
        if self._bare_table(view, (x, y)) and (match.point is None or match.point[2] <= TABLE_Z + RAISED):
            return on_plane(camera, (x, y), TABLE_Z, "the table top (bare table in the picture)")
        on_object, why = self._spot_on_object(view, (x, y))
        if match.point is None:
            return on_object if on_object is not None else _on_table(camera, (x, y), why)
        dist = float(np.linalg.norm(match.point - camera.position))
        spot = Spot(tuple(round(float(v), 4) for v in match.point), cam, (x, y),
                    f"the same surface in {' and '.join(match.cams)}", mm_per_px=round(1000.0 * dist / camera.focal, 1))
        if on_object is not None:
            gap = float(np.linalg.norm(np.array(on_object.xyz) - match.point))
            if gap > SPOT_AGREE:
                spot.warning = (f"the outline there ({on_object.source}) puts the point {1000 * gap:.0f} mm away; "
                                "check with robot.look(marks=...)")
                spot.others.append(f"{on_object.source}: ({', '.join(f'{v:.3f}' for v in on_object.xyz)})")
            else:
                spot.source += f", on {on_object.source}"
        return spot

    def _bare_table(self, view: View, xy) -> bool:
        """Whether the pixel shows bare table top (see TABLE_WORDS)."""
        if view._table is None:
            try:
                found = self.segment(view, text=TABLE_WORDS, min_score=TABLE_MIN_SCORE, max_masks=8)
            except Refused:
                return False
            table = np.zeros((view.camera.height, view.camera.width), bool)
            for seg in found:
                table |= np.asarray(seg.mask, bool)
            view._table = table
        xi = min(max(int(round(xy[0])), 0), view.camera.width - 1)
        yi = min(max(int(round(xy[1])), 0), view.camera.height - 1)
        return bool(view._table[yi, xi]) and not view.blind[yi, xi]

    def _spot_on_object(self, view: View, xy) -> tuple[Spot | None, str]:
        """Where the pixel's ray enters the object outlined there (or None, and why not)."""
        camera = view.camera
        start = self._by_pixel(view.cam, xy)
        if isinstance(start, Thing):
            return None, "nothing is outlined there"
        thing, shape, sights = self._measure(start, list(ORDER), remember=False)
        if not thing.found or shape is None:
            return None, f"the object there could not be measured ({thing.why_missing})"
        if max(shape.size[:2]) > MAX_OBJECT:
            return None, "the outline there is larger than any object (the table, or a mat on it)"
        if shape.single and min(shape.size[:2]) < THIN:
            return None, "the outline there is only a thin strip (an edge or a line on the table)"
        point = None
        if len(sights) >= 2:
            point, _ = hull.surface_along(sights, 0, xy, top=shape.top)
        if point is None:
            point = _ray_into_box(camera, xy, shape)
        if point is None:
            return None, "the pixel's ray misses the measured object"
        name = f"Thing#{thing.id} '{thing.label}'" if thing.id else f"the object outlined there ({start.mask.label})"
        dist = float(np.linalg.norm(point - camera.position))
        return Spot(tuple(round(float(v), 4) for v in point), view.cam, xy,
                    f"{name}, {len(thing.views)} view{'s' if len(thing.views) != 1 else ''}",
                    warning=thing.warning or None, mm_per_px=round(1000.0 * dist / camera.focal, 1)), ""


def _ray_into_box(camera: Camera, xy, shape: hull.Shape) -> np.ndarray | None:
    """Where a pixel's ray first meets a one-view object's box (top, then sides)."""
    origin, d = camera.ray(xy)
    t = math.radians(shape.yaw or 0.0)
    rot = np.array([[math.cos(t), math.sin(t), 0.0], [-math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
    c = np.array([shape.center[0], shape.center[1], 0.5 * (shape.top + shape.bottom)])
    o = rot @ (origin - c)
    v = rot @ d
    half = 0.5 * np.array([shape.size[0], shape.size[1], shape.top - shape.bottom])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / v
        t2 = (half - o) / v
    near = np.nanmax(np.minimum(t1, t2))
    far = np.nanmin(np.maximum(t1, t2))
    if not (far >= near > 0):
        return None
    return origin + near * d


def _on_table(camera: Camera, xy, why: str) -> Spot:
    spot = on_plane(camera, xy, TABLE_Z, "the table top")
    note = f"{why}: assumed the table top; for a point on an object give z= or pair="
    spot.warning = note if spot.warning is None else f"{note}; {spot.warning}"
    return spot


def _side(normal) -> str:
    """Name of the world direction nearest an outward xy direction."""
    x, y = float(normal[0]), float(normal[1])
    if abs(x) >= abs(y):
        return "+x (right)" if x > 0 else "-x (left)"
    return "+y (far)" if y > 0 else "-y (near)"


def _round_err(err) -> float | None:
    if err is None or not math.isfinite(err):
        return None
    return round(float(err), 1)


def _covering(arm: str, cam: str, what: str) -> str:
    """'the left arm covers <what> in the head picture' (the gripper, in its own wrist camera's)."""
    if cam == f"{arm}_wrist":
        return f"the {arm} gripper covers {what} in its own {cam} picture"
    return f"the {arm} arm covers {what} in the {cam} picture"


def _clicked(prompt: dict) -> tuple[float, float] | None:
    """The one pixel an outline was clicked at (find's at=, or measure's (cam, (x, y))), if it was."""
    points = prompt.get("points") or []
    if len(points) != 1 or list(prompt.get("labels") or [1]) != [1] or prompt.get("box") is not None:
        return None
    return float(points[0][0]), float(points[0][1])


def _cams(cams) -> list[str]:
    if cams is None or cams in ("auto", "all"):
        return list(ORDER)
    names = [cams] if isinstance(cams, str) else list(cams)
    out: list[str] = []
    for c in names:
        name = check_cam(c)
        if name not in out:
            out.append(name)
    if not out:
        raise Refused("measure: cams is 'auto' or a list of camera names")
    return out


def _describe(target) -> str:
    if isinstance(target, str):
        return repr(target)
    if isinstance(target, Thing):
        return f"Thing#{target.id}"
    if isinstance(target, Mask):
        return f"Mask#{target.id}"
    if isinstance(target, (list, tuple)) and target and all(isinstance(m, Mask) for m in target):
        return "[" + ", ".join(f"Mask#{m.id}" for m in target) + "]"
    return repr(target)[:80]


def _prompt(text, at, points, labels, box) -> tuple[dict, str]:
    """SAM3 prompt arguments and a short label for the outlines they give."""
    if text is not None and any(v is not None for v in (at, points, box)):
        raise Refused("find: give text, or at= / points= / box= (not text together with them)")
    if at is not None:
        if points is not None:
            raise Refused("find: give at= or points=, not both")
        points, labels = [at], [1]
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            raise Refused("find: text is a short noun phrase, e.g. 'red cup'")
        return {"text": text.strip()}, text.strip()
    if points is None and box is None:
        raise Refused("find: give text, at=(x, y), points=[(x, y), ...] or box=(x0, y0, x1, y1)")
    out: dict = {}
    parts_ = []
    if points is not None:
        pts = np.asarray(points, float).reshape(-1, 2) if np.size(points) else np.zeros((0, 2))
        if not len(pts) or not np.all(np.isfinite(pts)):
            raise Refused("find: points are [(x, y), ...] pixels")
        labs = [1] * len(pts) if labels is None else [int(v) for v in labels]
        if len(labs) != len(pts) or any(v not in (0, 1) for v in labs):
            raise Refused("find: labels are one 1 (on the object) or 0 (not on it) per point")
        out["points"] = [(float(x), float(y)) for x, y in pts]
        out["labels"] = labs
        on = [p for p, v in zip(out["points"], labs) if v == 1]
        parts_.append(f"at ({on[0][0]:.0f}, {on[0][1]:.0f})" if len(pts) == 1 else
                      f"{len(on)}+/{len(pts) - len(on)}- points")
    if box is not None:
        b = np.asarray(box, float).reshape(-1)
        if b.size != 4 or not np.all(np.isfinite(b)) or b[2] <= b[0] or b[3] <= b[1]:
            raise Refused("find: box is (x0, y0, x1, y1) with x1 > x0 and y1 > y0")
        out["box"] = tuple(int(round(v)) for v in b)
        parts_.append("box ({}, {})-({}, {})".format(*out["box"]))
    return out, " + ".join(parts_)
