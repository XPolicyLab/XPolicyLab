# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ._robot import RobotBase
from .robots import RobotAdapter

MOVE_TOL_M = 0.01
MOVE_TOL_DEG = 5.0
# Same value as PyRUA-Lean (robodojo.py), unchanged.
TCP_OFFSET = 0.145
HOME_MAX_ACTIONS = 25
HOME_TOL_RAD = 0.01


@dataclass(frozen=True)
class ArmState:
    """One arm at a recorded step.

    tcp_pos: the grasp point [x, y, z] in the world frame, metres: the middle of
        the finger pads, 0.145 m out along the pointing direction from the
        end-effector link.  ``move_to`` drives this point.
    pitch / yaw / roll: the gripper's orientation, degrees (see the class
        docstring): ``pitch`` 0 points level and 90 straight down, ``yaw`` is
        the heading of the pointing direction on the table (0 away from the
        robot, +90 towards the robot's left), ``roll`` the turn of the fingers
        about the pointing direction (0 as at the start).  Pointing straight
        down, ``yaw`` alone says how the fingers lie and ``roll`` reads 0.
    eef_pos: position of the end-effector link (the wrist flange), metres.
    eef_quat: end-effector orientation as a unit quaternion [w, x, y, z].
    gripper: measured gripper opening, normalised: 1.0 = open, 0.0 = closed on
        nothing; closed on an object 0.002 + 0.109 x this reading is the
        object's width in metres.
        (Gripper COMMANDS use the opposite sign convention: see ``set_gripper``.)
    joints: the six arm joint angles, radians (the last turns the fingers about
        the pointing direction).
    """

    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class State:
    """Snapshot of both arms at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call.
    task: the task instruction given by the benchmark (authoritative).
    left / right: the two arms (``state.arm("left")`` picks one by name).
    truncated: the benchmark has ended the episode; no
        further motion is possible.
    """

    step: int
    task: str
    left: ArmState
    right: ArmState
    truncated: bool

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to``.

    target_xyz: the requested grasp-point position.
    start_xyz / final_xyz: where the grasp point was before and after.
    dist_to_target_m: distance from ``final_xyz`` to ``target_xyz``.
    orientation_error_deg: how far the final orientation is from the one asked
        (the angles passed, the others as they were).
    pitch / yaw / roll: the orientation the gripper ended with, degrees.
    reached: within 0.01 m and 5 degrees of the target.
    path: ``"line"`` (the grasp point moved along a straight line),
        ``"joint"`` (the joints swung straight to the solution) or None
        (nothing moved).
    ik_error: why nothing moved (the pose is out of reach, or every way there
        goes through the table), or None.  An out-of-reach target names the
        nearby poses that ARE reachable (a less steep pitch here, or this
        orientation higher or lower).
    note: set when the exact pose was out of reach but one within 2 cm and
        8 degrees was not: the arm went there instead (``dist_to_target_m``
        and ``orientation_error_deg`` say how far off it is).
    blocked: the arm stopped short of a planned pose it could reach: something
        is in the way, or a held object is pushing against something.
    steps_used: actions consumed.
    truncated: the benchmark has ended the episode.
    """

    arm: str
    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    path: str | None
    ik_error: str | None
    note: str | None
    blocked: bool
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Home:
    """Outcome of ``home``.

    arm: ``"left"``, ``"right"`` or ``"both"``, as requested.
    reached: every joint of the returned arm(s) ended within 0.01 rad of the
        angle it had when the episode started.
    max_joint_error_rad: the largest difference that remains.
    steps_used: actions consumed.
    truncated: the benchmark has ended the episode.
    """

    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    commanded: ``"closed"`` or ``"open"``.
    opening: the opening once the fingers stopped (1.0 = open, 0.0 = closed on
        nothing).
    holding: after a close, the fingers stopped apart (an opening above 0.01,
        about 3 mm): something is between them.  It does not say what: check
        that the intended object moves with the arm.
    width_m: after a close, the width of what is held, across the fingers
        (0.0 when nothing is), else None.
    steps_used: actions consumed.
    step: actions executed so far
        (None if the runtime did not report the count).
    truncated: the benchmark has ended the episode.
    """

    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    step: int | None
    truncated: bool


@dataclass(frozen=True)
class Segment:
    """Outcome of ``segment``.

    found: SAM3 produced a mask scoring at least ``min_score``.
    centroid: (row, col) of the mask centroid (mean of the mask pixels, floored),
        or None.  Pass exactly this pixel, with the same camera, to
        ``back_project``.
    area_px: number of mask pixels (0 when not found).
    score: SAM3 confidence in [0, 1], None when not found.
    box: mask bounding box as reported by SAM3, or None.
    contours_rc: optional external mask boundaries, each an ordered sequence of
        (row, col) pixels. Holes are excluded. None means not requested/available;
        an empty tuple means no boundary. No depth or free-space claim.
    """

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str
    contours_rc: tuple[tuple[tuple[int, int], ...], ...] | None = None


@dataclass(frozen=True)
class WorldPoint:
    """Outcome of ``back_project``: the surface point seen at one pixel.

    xyz: world [x, y, z], metres.  It is a visible SURFACE point (the top of
        an object as the camera sees it), not the object's centre.
    depth_m: distance of that point along the camera's optical axis.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str


@dataclass(frozen=True)
class ObjectPose:
    """Outcome of ``locate``: an object measured from its mask and the depth image.

    found: SAM3 produced a mask scoring at least ``min_score``.
    center_xyz: [x, y, z] of the middle of the smallest rectangle around the
        object's footprint, at half its height above the table, metres.  A
        handle, a spout or a tail is part of the footprint and moves the
        middle away from the body's.  None when no 3D measurement was
        possible (see ``note``).
    top_z: height of the object's top surface (the highest points seen).
    height_m: ``top_z`` above the table surface (0.765 m).
    length_m / width_m: the long and the short side of the smallest rectangle
        around the footprint.
    long_axis_deg: direction of the long side on the table, degrees from +x
        (-90 .. 90); of a square, one of its sides.
    grasp_yaw: the ``yaw`` for ``move_to(..., pitch=90, yaw=grasp_yaw)``:
        pointing straight down, the fingers then close across the short side
        (a cube's: across two faces).
    score: SAM3 confidence.
    centroid: (row, col) of the mask in the image, as ``Segment.centroid``.
    area_px: mask pixels.
    camera: the camera measured from.
    note: why there is no 3D measurement (a flat object, no depth), else None.

    The camera sees the top and the near side.  The far side of an object that
    is round about an upright axis (a bottle, a cup) is filled in from the arc
    that is seen; the far side of anything else, and what other things or the
    robot's own gripper hide, is not measured, so such a footprint can be
    shorter than the object.  A held object is usually not found from the head
    camera.
    """

    found: bool
    center_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None


STATEFUL_TOOLS = frozenset({"move_pose", "grip", "home"})


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _tuple3(value: Any) -> tuple[float, float, float]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        vals = ()
    if len(vals) != 3:
        return (math.nan, math.nan, math.nan)
    return vals  # type: ignore[return-value]


def _arm(name: Any) -> str:
    if name not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {name!r}")
    return str(name)


def _camera(name: Any, cameras: tuple[str, ...]) -> str:
    if name not in cameras:
        raise ValueError(f"camera must be one of {cameras}, got {name!r}")
    return str(name)


def _command(value: Any, *, allow_keep: bool) -> int:
    """A gripper command: 1 closes, -1 opens, 0 keeps (``move_to`` only)."""
    allowed = (-1, 0, 1) if allow_keep else (-1, 1)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if number not in allowed:
        words = "1 (close), -1 (open) or 0 (keep)" if allow_keep else "1 (close) or -1 (open)"
        raise ValueError(f"gripper must be {words}, got {value!r}")
    return int(number)


def _scalar(value: Any, default: float = 0.0) -> float:
    """First element of a possibly nested one-value sequence (``[0.97]``, a numpy array)."""
    for _ in range(4):
        if value is None or isinstance(value, (int, float)):
            break
        try:
            value = next(iter(value))
        except (TypeError, StopIteration):
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence(value: Any) -> tuple[float, ...]:
    """A list or numpy array as a tuple of floats; ``()`` when it is neither."""
    if value is None:
        return ()
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return ()


def _rotation(quat: Sequence[float]) -> list[list[float]]:
    """[w, x, y, z] to a 3x3 rotation (rows)."""
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def gripper_angles(quat: Sequence[float]) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of an end-effector quaternion [w, x, y, z].

    The end-effector link's x axis is the pointing direction and its y axis the
    line the fingers close along; the orientation is ``Rz(90 + yaw) Ry(pitch)
    Rx(roll)`` (the starting pose is 0, 0, 0: pointing away from the robot,
    level, fingers side by side).  Within a degree of straight down, yaw and
    roll turn about the same axis: yaw carries their difference, roll reads 0.
    """
    r = _rotation(quat)
    point = (r[0][0], r[1][0], r[2][0])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):
        jaw = (r[0][1], r[1][1])
        return pitch, _wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    a, p = math.radians(90.0 + yaw), math.radians(pitch)
    ca, sa, cp, sp = math.cos(a), math.sin(a), math.cos(p), math.sin(p)
    base = [[ca * cp, -sa, ca * sp], [sa * cp, ca, sa * sp], [-sp, 0.0, cp]]
    r21 = sum(base[k][2] * r[k][1] for k in range(3))
    r11 = sum(base[k][1] * r[k][1] for k in range(3))
    return pitch, _wrap(yaw), _wrap(math.degrees(math.atan2(r21, r11)))


def _arm_state(raw: dict[str, Any], name: str) -> ArmState:
    pose = _sequence(raw.get(f"{name}_ee_pose"))
    if len(pose) != 7:
        pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    r = _rotation(pose[3:])
    tcp = tuple(pose[i] + r[i][0] * TCP_OFFSET for i in range(3))
    pitch, yaw, roll = gripper_angles(pose[3:])
    return ArmState(
        tcp_pos=tcp,  # type: ignore[arg-type]
        pitch=round(pitch, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
        eef_pos=pose[:3],  # type: ignore[arg-type]
        eef_quat=pose[3:],  # type: ignore[arg-type]
        gripper=_scalar(raw.get(f"{name}_ee_joint_state")),
        joints=_sequence(raw.get(f"{name}_arm_joint_state")),
    )


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number of degrees or None, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


class RobodojoRobot(RobotBase):
    """The RoboDojo dual-arm station: two 6-DoF ARX-X5 arms with parallel grippers at a table.

    World frame: x runs across the table (the LEFT arm is at negative x and
    appears on the LEFT of the head image), y points away from the robot
    towards the far edge of the table, z is up; metres.  Orientations are unit
    quaternions ordered [w, x, y, z].  The arms are ``"left"`` and ``"right"``
    (``robo.LEFT`` / ``robo.RIGHT``); a motion call names one arm and holds
    the other where it is.

    ``move_to`` drives an arm's GRASP POINT, the middle of the finger pads
    (``state().left.tcp_pos``), to a world position, with an orientation given
    in degrees: ``pitch`` is how far the gripper points below the horizontal
    (0 level, 90 straight down), ``yaw`` the heading of the pointing direction
    on the table (0 away from the robot, +90 towards the robot's left, -90
    towards its right) and ``roll`` the turn of the fingers about the pointing
    direction (0: side by side, as at the start).  Pointing straight down, the
    fingers close along the line at ``yaw`` degrees from +x (counter-clockwise
    seen from above): ``locate``'s ``grasp_yaw`` is that line across an
    object's narrow side.  The table
    surface is at z = 0.765; the arms stand on it at x = -0.30 and +0.30,
    y = -0.45.

    Two gripper conventions, do not mix them up.  COMMANDS are signed:
    ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1) opens, ``robo.KEEP`` (0)
    leaves the gripper as it is.  MEASUREMENTS (``state().left.gripper``,
    ``Gripper.opening``) are openings: 1.0 = open, 0.0 = closed on nothing.

    do not waste actions. When the benchmark ends the episode, the state
    reports ``truncated`` and motion calls raise ``EpisodeFinished``.  Motion calls block until the
    primitive finishes and return a typed result; nothing is queued.

    Perception is camera based: a fixed head camera (``cam_head``, the whole
    table) and one wrist camera per arm (``cam_left_wrist``,
    ``cam_right_wrist``), 640x480 each.  ``locate`` finds an object by a text
    prompt and measures it in 3D (centre, top, footprint, heading);
    ``segment`` / ``back_project`` give a mask centroid and the world point at
    one pixel.  Object poses are never given.
    """

    ARMS: tuple[str, str] = ("left", "right")
    LEFT: str = "left"
    RIGHT: str = "right"
    CLOSE: int = 1
    OPEN: int = -1
    KEEP: int = 0

    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        ObjectPose,
        Segment,
        WorldPoint,
    )
    NAME: ClassVar[str] = "robodojo"
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset({"done", "world_map"})
    CAMERAS: ClassVar[tuple[str, ...]] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("cam_head", "high"): "cam_head.png",
        ("cam_left_wrist", "high"): "cam_left_wrist.png",
        ("cam_right_wrist", "high"): "cam_right_wrist.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = (
        "cam_head.png",
        "cam_left_wrist.png",
        "cam_right_wrist.png",
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "success",
        "reached",
        "dist_to_target_m",
        "orientation_error_deg",
        "final_xyz",
        "path",
        "blocked",
        "steps_used",
        "max_joint_error_rad",
        "holding",
        "opening",
        "width_m",
        "ik_error",
        "found",
        "score",
        "centroid_rc",
        "center_xyz",
        "top_z",
        "grasp_yaw",
        "world_xyz",
        "truncated",
        "step",
    )


    @property
    def task(self) -> str:
        """The task instruction the benchmark generated for this layout (authoritative).

        Read it before planning.
        """
        return super().task

    @property
    def done(self) -> bool:
        """Absent (not in the API reference): there is no success flag to read.

        It raises instead of returning False, which a ``while not robo.done``
        loop carried over from another robot could spin indefinitely.  ``AttributeError`` also makes ``hasattr(robo, "done")`` false.
        """
        raise AttributeError(
            "RoboDojo reports no task success while the episode runs: there is no robo.done; "
            "judge progress from the images and the robot state"
        )

    def _reported_success(self) -> bool:
        return False

    def state(self, step: int = -1) -> State:
        """Read the recorded state of both arms.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with ``left`` / ``right`` end-effector poses, measured
            gripper openings and joint angles, and the ``truncated`` flag.
        """
        return super().state(step)

    def image(self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (480, 640, 3), RGB.

        Args:
            camera: ``"cam_head"`` (fixed view of the whole table: identify
                objects, destinations and the global layout),
                ``"cam_left_wrist"`` or ``"cam_right_wrist"`` (on the gripper:
                what that arm is about to grasp or is holding).
            resolution: there is one resolution per camera, ``"high"``.  Pixel
                coordinates are (row, col) with row 0 at the top and col 0 at
                the left.
            step: recorded step, ``-1`` = latest.  ``segment`` and
                ``back_project`` always read the CURRENT observation, which is
                the latest recorded step.
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(
        self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Absent (not in the API reference): no per-pixel world maps are recorded."""
        raise ValueError(
            "RoboDojo records no world maps; use back_project(row, col, camera) per pixel"
        )

    def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape`` (image shape), ``intrinsic_matrix`` (3x3),
        ``extrinsic_matrix`` (4x4 camera-to-world; the camera looks down its
        own -z axis with +y up) and ``has_depth``.  The wrist cameras move
        with their arm.
        """
        camera = _camera(camera, self.CAMERAS)
        payload = self._readonly(self.STATE_TOOL, {"step": int(step)})
        cameras = (payload.get("state") or {}).get("cameras") or {}
        return dict(cameras.get(camera) or {})

    def show(self, camera: str = "cam_head") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically; call ``show("cam_head")``,
        ``show("cam_left_wrist")`` or ``show("cam_right_wrist")`` when *you*
        (the author) need to look at the scene before writing the next turn.
        Costs tokens: look only when a decision depends on it.
        """
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``cam_head.png``, ``cam_left_wrist.png`` and
        ``cam_right_wrist.png``.
        """
        return super().artifact(name, step=step)


    def segment(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2,
        *, include_contours: bool = False,
    ) -> Segment:
        """Segment an object in the current image of ``camera`` by a text prompt (SAM3).

        Args:
            text_prompt: what to find, e.g. ``"the green bottle"``.  Name one
                object by its visible properties.
            camera: ``"cam_head"`` (default), ``"cam_left_wrist"`` or
                ``"cam_right_wrist"``.
            min_score: masks scoring below this are reported as not found.
            include_contours: return external pixel outlines from the same SAM3
                result, without another inference. Default omits outline payload.
        Returns:
            Segment; when ``found``, pass ``centroid`` (row, col) to
            ``back_project`` with the same camera to get a world point.  The
            mask is the best match for the text, not a guarantee of identity:
            check where it is against the image.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        if include_contours:
            kwargs["include_contours"] = True
        p = self._readonly("segment", kwargs)
        if not p.get("found"):
            return Segment(False, None, 0, None, None, str(p.get("camera", camera)))
        centroid = p.get("centroid_rc") or ()
        moments = p.get("mask_moments") or ()
        return Segment(
            found=True,
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(moments[0]) if moments else 0,
            score=float(p["score"]) if p.get("score") is not None else None,
            box=p.get("box_px"),
            camera=str(p.get("camera", camera)),
            contours_rc=tuple(tuple((int(r), int(c)) for r, c in outline)
                              for outline in p["contours_rc"])
                        if "contours_rc" in p else None,
        )

    def back_project(self, row: int, col: int, camera: str = "cam_head") -> WorldPoint:
        """Return the world point seen at one pixel of the current image of ``camera``.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left (images are
                480 rows x 640 cols).
            camera: the camera the pixel was read from (default ``"cam_head"``).
        Returns:
            WorldPoint; ``xyz`` is where the camera ray meets the first
            surface.  A pixel on an object's edge or a thin part can land on
            the table or the background behind it: prefer ``Segment.centroid``.
        Raises:
            ToolError: pixel out of range, no valid depth there, or the
            camera calibration is missing.
        """
        camera = _camera(camera, self.CAMERAS)
        kwargs: dict[str, Any] = {"row": int(row), "col": int(col), "camera": camera}
        p = self._readonly("back_project", kwargs)
        pixel = p.get("pixel") or (int(row), int(col))
        return WorldPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
        )

    def locate(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
    ) -> ObjectPose:
        """Find an object by a text prompt and measure it in 3D.

        SAM3 segments the current image of ``camera`` (as ``segment``); every
        pixel of the mask is then projected into the world through the depth
        image, and the points above the table give the object's centre, top,
        footprint and heading.  Nothing moves.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                small red cube on the left"``.
            camera: ``"cam_head"`` (default; the whole table),
                ``"cam_left_wrist"`` or ``"cam_right_wrist"`` (close up, from
                the gripper: re-measure before a precise grasp).
            min_score: masks scoring below this are reported as not found.
        Returns:
            ObjectPose.  For a top grasp: ``move_to([x, y, top_z + 0.08],
            pitch=90, yaw=grasp_yaw)`` above it, then down to the height to
            hold it at.  The mask is the best match for the text, not proof of
            identity: check it against the image.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        p = self._readonly("locate", kwargs)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        return ObjectPose(
            found=bool(p.get("found")),
            center_xyz=_tuple3(center) if center is not None else None,
            top_z=num("top_z"),
            height_m=num("height_m"),
            length_m=num("length_m"),
            width_m=num("width_m"),
            long_axis_deg=num("long_axis_deg"),
            grasp_yaw=num("grasp_yaw"),
            score=num("score"),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(p["geometry_error"]) if p.get("geometry_error") else None,
        )


    @classmethod
    def _api_helpers(cls):
        from .geometry import Geometry
        return {"geometry": Geometry}

    @property
    def geometry(self):
        """Pure geometry on supplied points; no sensing or robot action.

        Points must share the same coordinate frame and units. Geometry does
        not certify collision clearance, current object pose or task success.
        """
        from .geometry import Geometry
        return Geometry()

    def move_to(
            self,
            xyz: Sequence[float],
            arm: str = "right",
            gripper: int = 0,
            pitch: float | None = None,
            yaw: float | None = None,
            roll: float | None = None,
            path: str = "auto",
            *,
            gap_m: float | None = None,
        ) -> Move:
            """Move one arm's grasp point to a world position, with an orientation.

            The grasp point is the middle of the finger pads (``tcp_pos``).  The
            library solves the arm's inverse kinematics for the full pose and,
            with ``path="auto"``, moves the grasp point along a straight line to
            it, about 2.5 cm or 8 degrees per action; when the line leaves
            the arm's reach it swings the joints straight to the solution instead.
            Nothing moves when the pose is out of reach or every way there takes
            the arm into the table: ``ik_error`` says why.

            Args:
                xyz: target grasp-point position [x, y, z], metres.
                arm: ``"left"`` or ``"right"``.  The other arm holds its joints.
                gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1) to command the
                    moving arm's gripper on every action of the motion,
                    ``robo.KEEP`` (0) to keep commanding what it was last told (a
                    held object stays squeezed).  Keep it closed (1) while
                    carrying an object.
                pitch / yaw / roll: the orientation to arrive with, degrees (the
                    class docstring defines them).  An angle left None keeps its
                    current value, so a plain position move does not turn the
                    gripper.  With an EMPTY open gripper the fingers are
                    interchangeable, so the pose turned half a turn about the
                    pointing direction may be taken when it needs less joint
                    travel (``note`` says so; yaw / roll then read 180 off);
                    holding something, the pose is taken exactly as asked.
                path: ``"auto"`` (default), ``"line"`` (straight line or nothing)
                    or ``"joint"`` (swing the joints: fewer actions, but the grasp
                    point does not follow a line: not near objects).
                gap_m: optional calibrated opening during a CLEAR empty-hand move,
                    with explicit ``gripper=robo.OPEN`` only. Uses the same gap
                    mapping as ``set_gripper``. Commands it for at least 10
                    actions in total, overlapping travel, including at least two
                    final actions holding both arms still after successful arrival.
                    All actions count. On failed motion it stops without padding
                    the opening time; do not descend unless ``reached`` is True.
                    This is not for descent, contact or releasing a held object.
                    Verify finger clearance in the wrist image before approaching;
                    command duration does not establish the physical gap under contact.
            Returns:
                Move; test ``reached`` (within 0.01 m and 5 degrees).  ``blocked``
                means the arm was stopped by something on the way.
            Raises:
                ValueError: a malformed argument (nothing is sent).
            """
            arm = _arm(arm)
            target = _floats(xyz, 3, "xyz")
            command = _command(gripper, allow_keep=True)
            if path not in ("auto", "line", "joint"):
                raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
            kwargs: dict[str, Any] = {"xyz": list(target), "arm": arm, "gripper": command, "path": path}
            if gap_m is not None:
                if command != -1:
                    raise ValueError("gap_m requires explicit robo.OPEN")
                _opening_for_gap(gap_m)
                kwargs["gap_m"] = float(gap_m)
            for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
                angle = _angle(value, name)
                if angle is not None:
                    kwargs[name] = angle
            return self._stateful("move_pose", kwargs, lambda p: self._move(arm, target, p))

    def approach_to(
            self, xyz: Sequence[float], arm: str = "right", *, clearance_m: float = 0.0,
            empty_hand: bool = False, vertical: bool = False, max_waypoints: int = 80,
            transport: bool = False,
        ) -> dict[str, Any]:
            """Approach in bounded straight steps, keeping orientation and gripper command.

            Use this instead of writing fixed-step approach loops for the LAST few
            centimetres before contact. It is slow by design: with a held object,
            or with clearance unknown, it moves 2 mm per action (about five
            actions per centimetre). Use move_to for travel, lifts and strokes.
            Set orientation and explicitly open/settle the empty gripper BEFORE calling. This
            helper never opens, closes, rotates, retreats, or retries on failure.

            Args:
                xyz: target TCP position, metres.
                arm: moving arm; the other must stay stationary.
                clearance_m: conservative initial minimum clearance of the WHOLE
                    moving hand/fingers from the object, table and other hand,
                    estimated from images, fitted geometry and hand footprint.
                    NOT simply distance from TCP to target. Omit if unknown.
                    The straight approach corridor must also be clear of the
                    forearm and other obstacles; this is not a collision planner.
                    A 5 mm uncertainty margin is subtracted automatically and the
                    bound decreases with actual travel and orientation drift.
                empty_hand: explicitly True only when this hand holds no object.
                    With known clearance, the default spacing is 10 mm above 30 mm
                    clearance, 5 mm at 10-30 mm, and 2 mm within 10 mm. Steps stop
                    at each boundary; remaining target distance also caps the band.
                    Empty approaches follow a fixed desired path, with target lead
                    over measured TCP capped at spacing + 0.75 mm and the next
                    slower-band boundary. Unknown clearance keeps 2 mm spacing.
                    False (default), outside transport mode, keeps steps at most 2 mm.
                vertical: initial pickup only. Requires pitch within 3 degrees of
                    straight down and x/y aligned within 1 mm; holds starting x/y
                    fixed and lowers only z. Otherwise raises before moving.
                max_waypoints: hard loop limit, 1-200; default 80. Every simulator
                    motion action is still charged; batching does not hide costs.
                transport: opt in ONLY after a successful test lift, for a loaded
                    vertical upward lift with downward-facing gripper. Keeps x/y
                    and orientation fixed. Uses anchored 2 mm waypoints to avoid
                    repeatedly losing tracking lag; 8 mm spacing is permitted
                    only with sufficient whole-hand AND held-object clearance.
                    Target lead over measured TCP is capped at spacing + 0.75 mm.
                    Never use for pickup, lowering, receiver contact or rotation.
                    Re-estimate clearance from geometry between short lift segments:
                    initial table clearance may be small even if the destination
                    is high. Larger spacing must not use destination distance as
                    evidence of clearance. Default False preserves contact motion.

            Returns:
                Dict with reached, reason, steps_used, waypoints, final_xyz and
                remaining_m. Stops on obstruction, >2.5 mm tracking error, lack
                of progress, orientation drift, donor movement or episode termination.
                For initial pickup, require result['reached'] before closing or
                lifting. During handoff, a stop means pause and inspect, not an
                automatic retreat. A receiver may close at its current pose only
                when both pads visibly straddle solid object material, the donor
                still supports it, and closing clears the donor fingers. Otherwise
                relieve interference with the smallest necessary backoff before
                replanning; never blindly retry a blocked target.
            """
            from .approach import approach_to

            return approach_to(self, xyz, arm, clearance_m=clearance_m, empty_hand=empty_hand,
                               vertical=vertical, max_waypoints=max_waypoints, transport=transport)

    def set_gripper(self, arm: str, gripper: int, opening: float | None=None, *, gap_m: float | None=None) -> Gripper:
        """Close or open one gripper, without moving the arms.

            The command is repeated until the reported opening settles (2 to 10
            actions). The official simulator reports a control target,
            not measured finger separation: verify clearance in the wrist image.
            ``holding`` and ``width_m`` are not reliable contact measurements on
            that backend. Later motions with
            ``robo.KEEP`` keep commanding what was commanded last, so a held
            object stays squeezed and a partial opening stays as it is.

            Args:
                arm: ``"left"`` or ``"right"``.
                gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
                opening: with ``robo.OPEN`` only: normalized control target
                    (0 < opening <= 1). On the X5, values below about 0.185
                    command the closed stop. Prefer ``gap_m`` for a physical gap.
                gap_m: with ``robo.OPEN`` only, desired free-space distance between
                    the inner finger pads in metres (0.0008 to 0.0888). Use the
                    estimated object width across the fingers plus clearance,
                    e.g. ``gap_m=shaft_width_m + 0.008``. Cannot combine with
                    ``opening``. Uses the X5 robot geometry and controller scale.
                    Holds the target for 10 actions
                    to allow physical settling. Contact can prevent reaching the gap.
            Returns:
                Gripper.  ``holding`` is not proof that the INTENDED object is
                held: confirm it moves with the arm (wrist camera, or ``locate``
                it again after a lift).
            """
        arm = _arm(arm)
        command = _command(gripper, allow_keep=False)
        kwargs: dict[str, Any] = {'arm': arm, 'gripper': command}
        if gap_m is not None:
            if opening is not None or command != -1:
                raise ValueError('gap_m requires robo.OPEN and cannot be combined with opening')
            opening = _opening_for_gap(gap_m)
            kwargs['settle_steps'] = 10
        if opening is not None:
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'opening must be a number in (0, 1], got {opening!r}') from exc
            if command != -1 or not 0.0 < fraction <= 1.0:
                raise ValueError('opening goes with robo.OPEN and must be in (0, 1]')
            kwargs['opening'] = fraction

        def build(p: dict[str, Any]) -> Gripper:
            status = p.get('status') or {}
            state = self._latest_state()
            width = p.get('width_m')
            return Gripper(arm=arm, commanded=str(p.get('gripper') or ('closed' if command > 0 else 'open')), opening=float(p.get('opening', state.arm(arm).gripper)), holding=None if p.get('holding') is None else bool(p['holding']), width_m=float(width) if width is not None else None, steps_used=int(p.get('steps_used', 0)), step=int(status['step']) if status.get('step') is not None else None, truncated=state.truncated)
        return self._stateful('grip', kwargs, build)

    def home(self, arm: str = "both") -> Home:
        """Return an arm to the joint angles it had when the episode started.

        A grasp point lower than 1.0 m first goes straight up by up to 6 cm;
        then each action commands the starting angles directly, so the
        arm swings straight back: no IK and no path planning beyond that lift
        (clear tall things yourself).  The call returns when every joint is
        within 0.01 rad or after 25 actions of the swing.  The gripper keeps
        the opening it has; an arm that is not named holds its joints.  Finish
        most tasks with both arms back where they started.

        Args:
            arm: ``"left"``, ``"right"`` or ``"both"`` (default).
        Returns:
            Home; test ``reached`` and call again when it is False.
        """
        if arm not in (*self.ARMS, "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        which = str(arm)

        def build(p: dict[str, Any]) -> Home:
            return Home(
                arm=str(p.get("arm", which)),
                reached=bool(p.get("reached")),
                max_joint_error_rad=float(p.get("max_joint_error_rad", math.inf)),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("home", {"arm": which}, build)


    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _move(self, arm: str, target: tuple[float, ...], p: dict[str, Any]) -> Move:
        commanded = p.get("target_xyz")
        angles = p.get("final_angles") or {}
        return Move(
            arm=str(p.get("arm", arm)),
            target_xyz=_tuple3(commanded if commanded is not None else target),
            start_xyz=_tuple3(p.get("start_xyz")),
            final_xyz=_tuple3(p.get("final_xyz")),
            dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
            orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            reached=bool(p.get("reached")),
            path=str(p["path"]) if p.get("path") else None,
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            blocked=bool(p.get("blocked")),
            steps_used=int(p.get("steps_used", 0)),
            truncated=self._latest_state().truncated,
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or summary.get("instruction") or ""),
            left=_arm_state(raw, "left"),
            right=_arm_state(raw, "right"),
            truncated=bool(envelope.get("truncated")),
        )

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        """Latch ``truncated`` only.

        The host records ``terminated=False`` on purpose (official success is read
        by its runner after the planner); nothing an envelope says may end
        this episode as a success either.
        """
        if envelope is None:
            return
        state = self._state_from_envelope(envelope)
        self._last_state = state
        if state.truncated:
            self._truncated = True

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        out: dict[str, Any] = {"step": state.step}
        for name in self.ARMS:
            arm = state.arm(name)
            out[f"{name}_tcp"] = [round(v, 4) for v in arm.tcp_pos]
            out[f"{name}_pitch_yaw_roll"] = [
                round(arm.pitch, 1),
                round(arm.yaw, 1),
                round(arm.roll, 1),
            ]
            out[f"{name}_gripper"] = round(arm.gripper, 3)
        out["truncated"] = state.truncated
        return out


ADAPTER = RobotAdapter(
    name="robodojo",
    robot_cls=RobodojoRobot,
    host="martingale.hosts.robodojo_host",
    knowledge="robodojo",
    blurb="a dual-arm ARX-X5 robot in the RoboDojo benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a SAM3 segmentation service",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "Gripper",
    "HOME_MAX_ACTIONS",
    "HOME_TOL_RAD",
    "Home",
    "MOVE_TOL_DEG",
    "MOVE_TOL_M",
    "Move",
    "ObjectPose",
    "RobodojoRobot",
    "STATEFUL_TOOLS",
    "Segment",
    "State",
    "TCP_OFFSET",
    "WorldPoint",
    "gripper_angles",
]

def _opening_for_gap(gap_m: Any) -> float:
    """X5 inner-pad gap to controller command; uses robot geometry."""
    try:
        gap = float(gap_m)
    except (TypeError, ValueError) as exc:
        raise ValueError('gap_m must be a number from 0.0008 to 0.0888 metres') from exc
    if not 0.0008 <= gap <= 0.0888:
        raise ValueError('gap_m must be from 0.0008 to 0.0888 metres')
    return min(1.0, ((gap - 0.0008) / 2.0 + 0.01) / 0.054)
