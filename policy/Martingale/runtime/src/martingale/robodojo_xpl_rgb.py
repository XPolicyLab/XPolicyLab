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
from dataclasses import dataclass
from typing import Any, ClassVar

from .robodojo import (
    ArmState,
    Gripper,
    Home,
    Move,
    ObjectPose,
    RobodojoRobot,
    Segment,
    State,
    WorldPoint,
    _camera,
    _tuple3,
)
from .robots import RobotAdapter


@dataclass(frozen=True)
class ObjectSight(ObjectPose):
    """Outcome of ``locate``: an object measured from its masks in the colour cameras.

    found: SAM3 produced a mask scoring at least ``min_score``.
    center_xyz: [x, y, z] of the middle of the smallest rectangle around the
        object's footprint, at half its height above the table, metres.  A
        handle, a spout or a tail is part of the footprint and moves the
        middle away from the body's.  None when no 3D measurement was
        possible (see ``note``).
    top_z: height of the object's top surface.
    height_m: ``top_z`` above the table surface (0.765 m).
    length_m / width_m: the long and the short side of the smallest rectangle
        around the footprint.
    long_axis_deg: direction of the long side on the table, degrees from +x
        (-90 .. 90); of a square, one of its sides.
    grasp_yaw: the ``yaw`` for ``move_to(..., pitch=90, yaw=grasp_yaw)``:
        pointing straight down, the fingers then close across the short side
        (a cube's: across two faces).
    views: how many cameras agreed on the measurement.  1: nothing but the
        camera asked saw the object, and its height is a guess (see
        ``locate``); ``note`` says which guess.
    score: SAM3 confidence.
    centroid: (row, col) of the mask in the image, as ``Segment.centroid``.
    area_px: mask pixels.
    camera: the camera measured from.
    note: what limits the measurement (one camera only; two that agree only
        roughly; the mask is the robot's own arm, or not on the table; too
        flat or too far to measure), else None.

    Explicit shape fits additionally return bottom_z, object_height_m, and
    geometry_shape, and do not assume table contact. Without an explicit shape,
    the object is taken to stand on the table. What other things or the
    robot's own arm hide of it is not measured, so a partly hidden object
    comes out smaller and off-centre.  A held object is usually not found from
    the head camera.
    """

    views: int = 1
    bottom_z: float | None = None
    object_height_m: float | None = None
    geometry_shape: str | None = None


@dataclass(frozen=True)
class SeenPoint(WorldPoint):
    """Outcome of ``back_project``: the point seen at one pixel.

    xyz: world [x, y, z], metres: where the pixel's ray meets the object under
        the pixel, or the table.
    depth_m: distance of that point along the camera's optical axis.
    surface: ``"object"`` when the pixel lies on something that could be
        measured, ``"table"`` when the point is on the table top (z = 0.765):
        the pixel shows the table, or something too flat or too far to measure.
    """

    surface: str = "table"


from .inference.recipe_tools import RecipeTools

class RobodojoOfficialColourRobot(RecipeTools, RobodojoRobot):
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

    Perception is from colour images alone; there is no depth image.  A fixed
    head camera (``cam_head``, the whole table) and one wrist camera per arm
    (``cam_left_wrist``, ``cam_right_wrist``), 640x480 each, and where each
    one is, is known from the robot's state.  ``locate`` finds an object by a
    text prompt and measures it in 3D (centre, top, footprint, heading) from
    the cameras that see it: it takes two of them to measure a height.
    ``segment`` / ``back_project`` give a mask centroid and the world point at
    one pixel.  Object poses are never given.
    """

    NAME: ClassVar[str] = "robodojo_xpl_rgb"
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        ObjectSight,
        Segment,
        SeenPoint,
    )
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (*RobodojoRobot.SUMMARY_KEYS, "views", "surface")

    def _reported_success(self) -> bool:
        return bool(self._backend.solved())

    def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape`` (image shape), ``intrinsic_matrix`` (3x3) and
        ``extrinsic_matrix`` (4x4 camera-to-world; the camera looks down its
        own -z axis with +y up).  The head camera is fixed; a wrist camera
        moves with its arm, and its pose is worked out from that arm's state.
        """
        from .hosts import robodojo_sight as sight

        camera = _camera(camera, self.CAMERAS)
        if camera == "cam_head":
            view = sight.head_view()
        else:
            arm = self.state(step).arm("left" if camera == "cam_left_wrist" else "right")
            view = sight.wrist_view([*arm.eef_pos, *arm.eef_quat])
        k = [[view.fx, 0.0, view.cx], [0.0, view.fy, view.cy], [0.0, 0.0, 1.0]]
        return {
            "shape": [view.shape[0], view.shape[1], 3],
            "intrinsic_matrix": k,
            "extrinsic_matrix": [[float(v) for v in row] for row in view.pose],
        }

    def back_project(self, row: int, col: int, camera: str = "cam_head") -> SeenPoint:
        """Return the world point seen at one pixel of the current image of ``camera``.

        There is no depth image, so a pixel alone is only a ray.  The object
        under the pixel is segmented from that pixel (SAM3) and measured as
        ``locate`` measures one, and the answer is where the pixel's ray
        meets it.  A pixel on the table top itself gives the point of the
        table it shows, exactly.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left (images are
                480 rows x 640 cols).
            camera: the camera the pixel was read from (default ``"cam_head"``).
        Returns:
            SeenPoint; check ``surface``.  ``"object"``: ``xyz`` is on the
            object under the pixel, as well as ``locate`` would measure that
            object from this camera and the others that see it.
            ``"table"``: ``xyz`` is on the table top; a pixel that shows
            something too flat or too far to measure also answers so.  A thin
            part in mid-air (a peg, a handle, a cable) is not measured
            reliably: reason from the image and from ``locate`` of the whole
            object instead.
        Raises:
            ToolError: pixel out of range, or it looks past the table.
        """
        camera = _camera(camera, self.CAMERAS)
        kwargs: dict[str, Any] = {"row": int(row), "col": int(col), "camera": camera}
        p = self._readonly("back_project", kwargs)
        pixel = p.get("pixel") or (int(row), int(col))
        return SeenPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
            surface=str(p.get("surface") or "table"),
        )

    def fit_pan(self, text_prompt="dustpan", yaw_hint=None) -> dict[str, Any]:
        """Fit an open pan resting flat on the table using current multi-view RGB/SAM masks. yaw_hint is radians; generic 3mm shell is a declared prior. Returns entrance/floor geometry and residuals, not task success.

        Read accepted/valid, assumptions and residuals before planning. Reobserve
        after object/camera motion, contact or occlusion. No action is sent.
        """
        return self._readonly("fit_pan", {'text_prompt': text_prompt, 'yaw_hint': yaw_hint})

    def fit_peg_landmarks(self, root_pixels, tip_pixels, radius_m=None, measured_step=None) -> dict[str, Any]:
        """Triangulate corresponding peg root/tip (row,column) pixels from two or more current calibrated views. measured_step must match image step. radius_m is an optional caller prior, not estimated. Returns world-meter axis; no insertion clearance claim.

        Read accepted/valid, assumptions and residuals before planning. Reobserve
        after object/camera motion, contact or occlusion. No action is sent.
        """
        return self._readonly("fit_peg_landmarks", {'root_pixels': root_pixels, 'tip_pixels': tip_pixels, 'radius_m': radius_m, 'measured_step': measured_step})

    def fit_peg(self, seeds, measured_step=None) -> dict[str, Any]:
        """Fit a cylinder to current RGB-derived peg masks. seeds maps camera names to two shaft endpoint (row,column) pixels. measured_step must match current image step. Uses midpoint SAM prompts and calibrated views, labels ambiguity; no movement or insertion verification.

        Read accepted/valid, assumptions and residuals before planning. Reobserve
        after object/camera motion, contact or occlusion. No action is sent.
        """
        return self._readonly("fit_peg", {'seeds': seeds, 'measured_step': measured_step})

    def fit_mugs(self) -> dict[str, Any]:
        """Fit initially upright, table-supported mugs and handles from current RGB. Refit after motion; indices are local, not persistent IDs.

        Returns accepted/valid geometry with assumptions and uncertainty in world
        meters/radians as named. Inspect the result before planning; a fit or
        motion reply is not task success. Refresh after motion/contact.
        """
        return self._readonly("fit_mugs", {})

    def measure_disks(self) -> dict[str, Any]:
        """Fit visible flat table-supported disks from RGB; printed symbol identity must be read visually.

        Returns accepted/valid geometry with assumptions and uncertainty in world
        meters/radians as named. Inspect the result before planning; a fit or
        motion reply is not task success. Refresh after motion/contact.
        """
        return self._readonly("measure_disks", {})

    def measure_key(self, *, detail=False) -> dict[str, Any]:
        """Fit the visible white table-supported key from RGB; detail=True includes mesh arrays. Not insertion or grasp verification.

        Returns accepted/valid geometry with assumptions and uncertainty in world
        meters/radians as named. Inspect the result before planning; a fit or
        motion reply is not task success. Refresh after motion/contact.
        """
        return self._readonly("measure_key", {'detail': detail})

    def measure_wedge(self, *, detail=False) -> dict[str, Any]:
        """Fit a visible white ruler-like wedge and ridge from RGB on the table. Long-axis yaw has 180 degree ambiguity. detail=True includes mesh.

        Returns accepted/valid geometry with assumptions and uncertainty in world
        meters/radians as named. Inspect the result before planning; a fit or
        motion reply is not task success. Refresh after motion/contact.
        """
        return self._readonly("measure_wedge", {'detail': detail})

    def fit_basket(self, *, table_z_m, body_bounds, body_initial, handle_bounds, handle_initial, min_score=.2) -> dict[str, Any]:
        """Fit a light-blue table-supported basket body and elliptical handle using current RGB. Explicit metric/radian shape priors required; up to six SAM calls, no motion.

        Returns accepted/valid geometry with assumptions and uncertainty in world
        meters/radians as named. Inspect the result before planning; a fit or
        motion reply is not task success. Refresh after motion/contact.
        """
        return self._readonly("fit_basket", {'table_z_m': table_z_m, 'body_bounds': body_bounds, 'body_initial': body_initial, 'handle_bounds': handle_bounds, 'handle_initial': handle_initial, 'min_score': min_score})


    def fit_slots(self, features: dict, *, step: int = -1) -> dict[str, Any]:
        """Fit two rectangular slot rims from explicit current image corners.

        features.rectangles maps slot_left/slot_right (or outer plus divider_edges)
        to per-camera four-corner (column,row) arrays in original 640x480 RGB.
        Requires head plus wrist views. step binds image/calibration/proprioception;
        outputs world-meter rims, fit residuals and RGB hashes. Does not infer
        bottom/contact or insertion success. No motion; correct mismatched corners
        or acquire a better view when rejected.
        """
        from .hosts.robodojo_slots import fit_slots
        return fit_slots(self, features, step=step)

    def fit_blocks(self) -> dict[str, Any]:
        """Fit magenta upright cubes resting on the calibrated table plane.

        Returns current local IDs, center/size/yaw (90-degree symmetry), outline
        errors and uncertain row alignment. Color and upright-cube assumptions
        are explicit; this is not a detector for arbitrary blocks or held cubes.
        Reobserve after motion. No actions are used;
        accepted geometry does not imply task success.
        """
        return self._readonly("fit_blocks", {})

    def fit_tube_openings(self) -> dict[str, Any]:
        """Measure empty circular holes in the blue rack without moving.

        Fits one shared horizontal rim plane and matches the same hole pattern
        across head and wrist views. Returns accepted/reasons, rim_z,
        normal_xyz, holes, fit_error_m and supporting-view information. Each
        hole has id, center_xyz, radius_m, head_pixel_xy and supporting_views.
        Hole IDs are local to this fit; after refitting identify the desired hole
        from its image/location again. Initial clipped or ambiguous views are
        rejected: reposition a free wrist for a better view before retrying.
        Does not measure a held tube, rack bottom, insertion depth or success.
        No simulator actions are used.
        """
        return self._readonly("fit_tube_openings", {})

    def check_tube_alignment(self, openings: dict, hole_id: int, tip_pixels: dict,
                             shaft_pixels: dict, tube_radius_m: float, *, measured_step: int,
                             clearance_m: float = .002, insertion_m: float = .03) -> dict[str, Any]:
        """Check the actual held tube against an accepted circular-opening fit.

        openings: accepted fit_tube_openings() result for a stationary rack.
        tip_pixels and shaft_pixels: dicts camera -> (column,row), measured in
        the current ORIGINAL 640x480 images. Observe each physical landmark in
        at least two cameras. Tip is the centre of the tube's lower end; shaft
        is the same centreline point at least 25 mm farther up in each view.
        tube_radius_m: observed cylindrical BODY radius, not the cap or gripper.
        measured_step: state().step of those images; stale pixels are rejected.
        clearance_m: margin from the hole rim, default 2 mm.
        insertion_m: planned downward stroke for the conservative tilted-tube
            envelope, default 30 mm; this is not a measured seating depth.

        Returns ready_to_lower/reasons, reconstructed tip/axis, tilt_deg,
        lateral_error_m, remaining_clearance_m and translation_to_entry_m.
        Translation aligns the tip 15 mm above the rim with the current hand
        orientation. Remeasure after rotating. A ready result is not evidence
        of clear fingers, actual contact or successful insertion. Nothing moves.
        """
        if measured_step != self.state().step:
            return {"ready_to_lower":False,"reasons":["Landmark pixels are stale; reobserve the current step."]}
        return self._readonly("check_tube_alignment", dict(openings=openings,hole_id=int(hole_id),
            tip_pixels=tip_pixels,shaft_pixels=shaft_pixels,tube_radius_m=float(tube_radius_m),
            clearance_m=float(clearance_m),insertion_m=float(insertion_m)))


    def locate(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2,
        shape: str | None = None,
    ) -> ObjectSight:
        """Find an object by a text prompt and measure it in 3D from the colour cameras.

        SAM3 segments the current image of ``camera`` (as ``segment``); the
        same object is then looked for in the other two cameras, and its
        height and footprint are those that fit every camera that shows it.
        One camera cannot measure a height: a low deep thing and a tall
        shallow one look alike from it.  ``views`` says how many cameras
        agreed.  Nothing moves.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                small red cube on the left"``.
            camera: ``"cam_head"`` (default; the whole table),
                ``"cam_left_wrist"`` or ``"cam_right_wrist"`` (close up, from
                the gripper).
            min_score: masks scoring below this are reported as not found.
            shape: optional ``"slab"``, ``"box"`` or ``"pyramid"``. Choose a shape explicitly for a geometrically suitable object,
                in any task. Omitting it preserves the original estimator.
                There is no task-name-based geometry selection.
                The pyramid option compares apex and triangular-prism models.
                These fits need
                at least two current views and estimate position, orientation,
                dimensions and bottom elevation afresh on every call. After
                placement or lifting, locate again: ``object_height_m`` is the
                piece's own vertical extent and ``bottom_z`` is its elevation.
                ``height_m`` is the top's height above the table.
        Returns:
            ObjectSight.  For a top grasp: ``move_to([x, y, top_z + 0.09],
            pitch=90)`` above it, ``locate`` it again from there (the wrist
            camera then looks down on it and the reading is good to 1-2 mm),
            turn to ``yaw=grasp_yaw`` and go down to the height to hold it
            at.  With ``views >= 2`` and no ``note`` a block, a can, a bowl
            or the body of a bottle is measured to 1-3 mm; a ``note`` that
            the cameras agree only roughly leaves the height uncertain.  With
            ``views == 1`` the height is a guess, the tallest upright shape
            that fits the one image: close for a box or a can, wrong for a
            bowl, a cup or anything wider above than below, and for a tall
            thing that tapers; ``center_xyz`` is then off, along the line
            away from the robot, by about half of what the height is wrong
            by. These single-view guesses and height recall apply only to the
            original estimator for other objects. Shape-specific fits return
            no position if current views cannot constrain the shape and its
            elevation. Other objects once measured with ``views >= 2`` keep
            that height in later calls with the same prompt. No position comes back
            (``center_xyz`` None, ``note`` says why) for a mask on the robot's
            own arm or on something that is not on the table.  The mask is
            the best match for the text, not proof of identity: check it
            against the image.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        if shape is not None and shape not in ("slab", "box", "pyramid"):
            raise ValueError("shape must be slab, box, pyramid, or None")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        if shape is not None:
            kwargs["shape"] = shape
        p = self._readonly("locate", kwargs)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        note = p.get("geometry_error") or p.get("height_guess")
        return ObjectSight(
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
            note=str(note) if note else None,
            views=int(p.get("views") or 1),
            bottom_z=num("bottom_z"),
            object_height_m=num("object_height_m"),
            geometry_shape=p.get("geometry_shape"),
        )


ADAPTER = RobotAdapter(
    name="robodojo_xpl_rgb",
    robot_cls=RobodojoOfficialColourRobot,
    host="martingale.hosts.xpolicylab_robodojo_rgb",
    knowledge="robodojo_xpl_rgb",
    blurb="a dual-arm ARX-X5 robot in the RoboDojo benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a SAM3 segmentation service",
)


__all__ = ["ADAPTER", "ObjectSight", "RobodojoOfficialColourRobot", "SeenPoint"]
