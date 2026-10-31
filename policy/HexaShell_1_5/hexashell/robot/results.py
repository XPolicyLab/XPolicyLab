"""Result types the ``robot.*`` calls return.

They are plain data (they cross the process boundary between the host and the sandboxed worker
by pickle) and they print as one compact line or a few: the model mostly reads what is printed,
so the important facts come first and warnings lead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _xyz(v) -> str:
    return "(" + ", ".join(f"{float(a):.3f}" for a in v) + ")"


def _deg(v) -> str:
    v = float(v)
    return f"{v:.0f}" if v != v or v in (float("inf"), float("-inf")) else str(round(v))  # no "-0"


class NotFound(Exception):
    """Raised when a missing measurement's position is used."""


# -- motion ----------------------------------------------------------------------------------------
@dataclass
class Move:
    arm: str
    reached: bool
    moved: bool
    target: tuple[float, float, float]
    target_angles: tuple[float, float, float]
    at: tuple[float, float, float]
    angles: tuple[float, float, float]
    err_mm: float = 0.0
    err_deg: float = 0.0
    steps: int = 0
    path: str = ""
    blocked: bool = False
    dropped: bool = False  # what the gripper held slipped out on the way
    lag_mm: float = 0.0  # how far the grasp point ended behind its command (BLOCKED above 5 mm)
    skipped: int = 0  # actions of the plan not sent: it stopped early (stalled, or what it held slipped out)
    halt: str = ""  # 'Blocked' or 'Dropped': the error the call raises so that the rest of the cell does not run
    dry_run: bool = False
    notes: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    steps_left: int | None = None

    @property
    def ok(self) -> bool:
        return self.reached and not self.blocked and not self.dropped

    def __repr__(self) -> str:
        p, y, r = self.angles
        pose = f"{_xyz(self.at)} pitch {_deg(p)} yaw {_deg(y)} roll {_deg(r)}"
        tp, ty, tr = self.target_angles
        goal = f"{_xyz(self.target)} pitch {_deg(tp)} yaw {_deg(ty)} roll {_deg(tr)}"
        left = "" if self.steps_left is None else f"  [{self.steps_left} steps left]"
        if self.dry_run:
            head = f"Move {self.arm} (dry run): " + (
                f"reachable, {self.steps} steps by {self.path}" if self.reached else "NOT reachable"
            )
            lines = [head + f"  target {goal}"]
        elif not self.moved:
            lines = [f"Move {self.arm}: NOT executed — target {goal}"]
        elif self.reached and not self.blocked:
            lines = [f"Move {self.arm}: at {pose}  ±{self.err_mm:.0f} mm {self.err_deg:.0f}°  "
                     f"{self.steps} steps ({self.path}){left}"]
        else:
            what = "BLOCKED" if self.blocked else "short of target"
            lines = [f"Move {self.arm}: {what} — at {pose}, {self.err_mm:.0f} mm / {self.err_deg:.0f}° "
                     f"from {goal}  {self.steps} steps{left}"]
        lines += [f"  note: {n}" for n in self.notes]
        if self.alternatives:
            lines.append("  reachable instead: " + "; ".join(self.alternatives))
        return "\n".join(lines)

    __str__ = __repr__


@dataclass
class Grip:
    arm: str
    opening: float
    width_m: float
    holding: bool | None
    steps: int = 0
    goal: float = 1.0
    steps_left: int | None = None
    notes: list[str] = field(default_factory=list)
    # 'Blocked': an open that left the fingers on what they held, or a close not done; the call raises it
    halt: str = ""
    refused: bool = False  # a close not done (the fingertips rest on something): the fingers did not move

    def __repr__(self) -> str:
        left = "" if self.steps_left is None else f"  [{self.steps_left} steps left]"
        if self.refused:
            state = f"NOT closed (still open {self.opening:.2f})"
        elif self.goal <= 0.05:
            if self.holding:
                state = f"closed on something ~{100 * self.width_m:.1f} cm wide (holding)"
            elif self.holding is None:
                state = f"the fingers did not close (still ~{100 * self.width_m:.1f} cm apart)"
            else:
                state = "closed fully (empty - unless it is something thin like cloth or paper: check a picture)"
        else:
            state = f"open {self.opening:.2f} (gap ~{100 * self.width_m:.1f} cm)"
        lines = [f"Grip {self.arm}: {state}  {self.steps} steps{left}"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__


@dataclass
class Homing:
    arms: dict[str, tuple[float, float]]  # arm -> (distance mm, turn deg) from its start pose
    steps: int = 0
    steps_left: int | None = None
    notes: list[str] = field(default_factory=list)
    dropped: bool = False  # what a gripper held slipped out on the way

    @property
    def ok(self) -> bool:
        return all(d <= 10.0 and t <= 3.0 for d, t in self.arms.values()) and not self.dropped

    def __repr__(self) -> str:
        left = "" if self.steps_left is None else f"  [{self.steps_left} steps left]"
        parts = []
        for arm, (d, t) in self.arms.items():
            parts.append(f"{arm} home" if d <= 10.0 and t <= 3.0 else f"{arm} {d:.0f} mm / {t:.0f}° from home")
        lines = ["Home: " + ", ".join(parts) + f"  {self.steps} steps{left}"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__


# -- perception ------------------------------------------------------------------------------------
@dataclass
class Mask:
    """One object outline in one camera picture."""

    id: int
    cam: str
    label: str
    score: float
    pixels: np.ndarray  # bool (H, W)
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    centroid: tuple[float, float]  # (x, y)
    area: int
    seq: int = 0  # observation the picture belongs to
    robot_share: float = 0.0  # share of the outline on the robot's own pixels
    prompt: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        x0, y0, x1, y1 = self.box
        cx, cy = self.centroid
        warn = f"  ⚠ {100 * self.robot_share:.0f}% on the robot" if self.robot_share > 0.2 else ""
        return (f"Mask#{self.id} {self.cam} '{self.label}' score {self.score:.2f}  centroid ({cx:.0f}, {cy:.0f})  "
                f"box ({x0}, {y0})-({x1}, {y1})  {self.area} px{warn}")

    __str__ = __repr__


class MaskList(list):
    """A list of masks that prints one per line."""

    def __repr__(self) -> str:
        if not self:
            return "[] (nothing found: name how it looks - colour, shape - or point at it with at=(x, y))"
        return "\n".join(repr(m) for m in self)

    __str__ = __repr__


def mask_items(masks) -> list:
    """What ``look(masks=...)`` was given, as a flat list: one Mask or Thing, or a list, tuple or dict
    of them (a Thing's ``.masks``), nested in any way."""
    if masks is None:
        return []
    if isinstance(masks, dict):
        masks = list(masks.values())
    elif not isinstance(masks, (list, tuple)):
        return [masks]
    out = []
    for item in masks:
        out.extend(mask_items(item) if isinstance(item, (list, tuple, dict)) else [item])
    return out


class Views(dict):
    """A Thing's outlines by camera (cam -> Mask).  Adding two of them, or one and a list, gives a list
    of Masks: ``look(masks=a.masks + b.masks)`` draws both."""

    def __add__(self, other):
        return list(self.values()) + mask_items(other)

    def __radd__(self, other):
        return mask_items(other) + list(self.values())


@dataclass
class Thing:
    """An object measured in 3-D."""

    id: int
    label: str
    found: bool
    center_xyz: tuple[float, float, float] | None = None
    top: float | None = None
    size: tuple[float, float, float] | None = None  # length (long side), width, height (m)
    grip_yaw: float | None = None  # yaw for a top-down grasp closing across the short side (None: round)
    on: str = "table"  # what it rests on: "table", "Thing#2", "held by left", "z 0.812"
    on_z: float | None = None
    views: list[str] = field(default_factory=list)
    err_mm: float | None = None
    warning: str | None = None
    why_missing: str | None = None
    candidates: list[str] = field(default_factory=list)
    masks: dict[str, Mask] = field(default_factory=Views)
    seq: int = 0
    round: bool = False
    notes: list[str] = field(default_factory=list)
    step: int | None = None  # episode step of the pictures it was measured in

    def __post_init__(self):
        if not isinstance(self.masks, Views):
            self.masks = Views(self.masks or {})

    @property
    def center(self) -> tuple[float, float, float]:
        if not self.found or self.center_xyz is None:
            name = f"Thing#{self.id} '{self.label}'" if self.id else f"'{self.label}'"
            raise NotFound(f"{name} was not found: {self.why_missing or 'no outline'}")
        return self.center_xyz

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.center

    def __repr__(self) -> str:
        name = f"Thing#{self.id} '{self.label}'" if self.id else f"Thing '{self.label}'"
        if not self.found:
            lines = [f"{name}  NOT FOUND: {self.why_missing or 'no outline'}"]
            lines += [f"  candidate: {c}" for c in self.candidates]
            lines += [f"  note: {n}" for n in self.notes]
            return "\n".join(lines)
        head = name
        if self.warning:
            head += f"  ⚠ {self.warning}"
        size = "×".join(f"{v:.3f}" for v in self.size) if self.size else "?"
        on = self.on if self.on_z is None or self.on == "table" or self.on.startswith(("something", "z ")) \
            else f"{self.on} (z {self.on_z:.3f})"
        shape = " round" if self.round else ""
        yaw = "any" if self.grip_yaw is None else f"{self.grip_yaw:.0f}°"
        err = f" ±{self.err_mm:.0f}mm" if self.err_mm is not None else ""
        lines = [f"{head}  center {_xyz(self.center_xyz)}  top {self.top:.3f}  size {size}{shape}  "
                 f"grip_yaw {yaw}  on {on}  {len(self.views)} view{'s' if len(self.views) != 1 else ''}"
                 f" ({', '.join(self.views)}){err}"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__


@dataclass
class Spot:
    """A world point recovered from a picture point."""

    xyz: tuple[float, float, float] | None
    cam: str
    px: tuple[float, float]
    source: str
    gap_mm: float | None = None
    angle_deg: float | None = None
    warning: str | None = None
    others: list[str] = field(default_factory=list)
    mm_per_px: float | None = None  # how far one pixel of picking error moves the point

    def __iter__(self):
        if self.xyz is None:
            raise NotFound(f"no world point for {self.cam} {self.px}: {self.warning}")
        return iter(self.xyz)

    def __getitem__(self, i):
        if self.xyz is None:
            raise NotFound(f"no world point for {self.cam} {self.px}: {self.warning}")
        return self.xyz[i]

    def __len__(self):
        return 3

    def __repr__(self) -> str:
        if self.xyz is None:
            return f"Spot {self.cam} ({self.px[0]:.0f}, {self.px[1]:.0f}): NONE — {self.warning}"
        warn = f"  ⚠ {self.warning}" if self.warning else ""
        extra = ""
        if self.gap_mm is not None:
            extra = f"  rays {self.gap_mm:.1f} mm apart, {self.angle_deg:.0f}° between them"
        if self.mm_per_px is not None:
            extra += f"  (~{self.mm_per_px:.1f} mm per pixel)"
        lines = [f"Spot {_xyz(self.xyz)} from {self.cam} ({self.px[0]:.0f}, {self.px[1]:.0f}) on {self.source}{extra}{warn}"]
        lines += [f"  also: {o}" for o in self.others]
        return "\n".join(lines)

    __str__ = __repr__


# -- reading ---------------------------------------------------------------------------------------
@dataclass
class ArmStatus:
    arm: str
    at: tuple[float, float, float]
    angles: tuple[float, float, float]
    gripper: float
    goal: float
    holding: bool | None
    joints: tuple[float, ...]
    home_dist_m: float
    home_turn_deg: float

    @property
    def xyz(self):
        return self.at

    def __repr__(self) -> str:
        p, y, r = self.angles
        if self.goal <= 0.05:
            grip = ("closed, holding" if self.holding else "closed fully (empty, or something thin)"
                    if self.holding is not None else "closed, but the fingers stayed about fully open (blocked?)")
        else:
            grip = f"open {self.gripper:.2f}"
        home = "home" if self.home_dist_m < 0.01 and self.home_turn_deg < 3 else \
            f"{100 * self.home_dist_m:.0f} cm / {self.home_turn_deg:.0f}° from home"
        return (f"{self.arm}: grasp point {_xyz(self.at)} pitch {_deg(p)} yaw {_deg(y)} roll {_deg(r)}  "
                f"gripper {grip}  ({home})")

    __str__ = __repr__


@dataclass
class Status:
    step: int | None
    step_limit: int | None
    arms: dict[str, ArmStatus]

    @property
    def steps_left(self) -> int | None:
        if self.step is None or self.step_limit is None:
            return None
        return max(0, self.step_limit - self.step)

    def __getitem__(self, arm: str) -> ArmStatus:
        return self.arms[arm]

    @property
    def left(self) -> ArmStatus:
        return self.arms["left"]

    @property
    def right(self) -> ArmStatus:
        return self.arms["right"]

    def __repr__(self) -> str:
        budget = "" if self.step is None else f"step {self.step} of {self.step_limit} ({self.steps_left} left)"
        return "\n".join([budget] + [repr(a) for a in self.arms.values()])

    __str__ = __repr__


@dataclass
class CameraInfo:
    name: str
    width: int
    height: int
    focal_px: float
    center_px: tuple[float, float]
    position: tuple[float, float, float]
    rotation: list[list[float]]  # camera-to-world; the camera looks along its -z, image up is +y

    def __repr__(self) -> str:
        return (f"Camera {self.name}: {self.width}x{self.height}, focal {self.focal_px:.1f} px, "
                f"at {_xyz(self.position)}, looking {_xyz([-v[2] for v in self.rotation])}")

    __str__ = __repr__


@dataclass
class Waited:
    steps: int
    steps_left: int | None = None
    #: simulated seconds of a step in which nothing moves (RoboDojo: 0.04)
    step_s: float = 0.04

    def __repr__(self) -> str:
        left = "" if self.steps_left is None else f"  [{self.steps_left} steps left]"
        return f"Waited {self.steps} steps ({self.step_s * self.steps:.2f} s){left}"

    __str__ = __repr__


@dataclass
class Picture:
    """What ``look`` attached to the cell's result."""

    cams: list[str]
    what: str
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [f"[picture attached: {self.what}]"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    __str__ = __repr__


@dataclass
class TaskInfo:
    instruction: str
    step: int | None
    step_limit: int | None
    start: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    task_name: str | None = None

    @property
    def steps_left(self) -> int | None:
        if self.step is None or self.step_limit is None:
            return None
        return max(0, self.step_limit - self.step)

    def __repr__(self) -> str:
        lines = [f"Task: {self.instruction}"]
        if self.step_limit is not None:
            lines.append(f"Budget: {self.step_limit} steps; used {self.step}, {self.steps_left} left")
        for arm, (xyz, ang) in self.start.items():
            lines.append(f"Start pose {arm}: grasp point {_xyz(xyz)} pitch {_deg(ang[0])} yaw {_deg(ang[1])} "
                         f"roll {_deg(ang[2])}")
        return "\n".join(lines)

    __str__ = __repr__


class ThingList(list):
    """Things that print one per line."""

    def __repr__(self) -> str:
        if not self:
            return "[] (nothing measured yet)"
        return "\n".join(repr(t) for t in self)

    __str__ = __repr__
