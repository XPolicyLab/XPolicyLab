"""The ``robot`` object a cell sees.  Every call travels to the host process, which owns the arms,
the cameras and the perception models; results come back as the plain types of
:mod:`hexashell.robot.results`.

Units: metres, degrees.  World frame: x to the robot's right, y away from the robot, z up; the
table top is at z = 0.765.  Pixels are (x, y) = (column, row) from the top-left corner.
"""

from __future__ import annotations

from typing import Any, Callable


class RobotError(Exception):
    """A robot call could not be carried out; the message says why."""


class NotExecuted(RobotError):
    """A motion that cannot be done as asked (out of reach, below the table top, no clear path):
    nothing moved; the message gives the reason and nearby poses that are reachable."""


class Blocked(RobotError):
    """A goto that ran but was stopped by something in the way (it stalled, or ended BLOCKED by more
    than 25 mm - 10 mm when it went down with the gripper open and empty), or an open that left the
    fingers less than 2 cm wider than what they held: the message is the Move or the Grip; the rest
    of the cell does not run, so you can look first.  goto(..., contact=True), for a press, push or
    insertion, never raises it."""


class Dropped(RobotError):
    """What the gripper held slipped out while a goto carried it (not in the last 2 cm of a descent):
    the rest of the motion and of the cell were skipped; the message is the Move."""


class Robot:
    def __init__(self, call: Callable[[str, tuple, dict], Any]):
        self._call = call

    # -- motion (uses the step budget) -----------------------------------------------------------
    def goto(self, arm, xyz=None, *, by=None, pitch=None, yaw=None, roll=None, grip=None,
             path="auto", dry_run=False, contact=False, gentle=False):
        """Move an arm's grasp point (between the finger pads) to ``xyz`` with the given angles.

        arm: 'left' or 'right'.  xyz: world point; ``by=(dx, dy, dz)`` moves relative to it (or to
        the current point when xyz is None).  pitch/yaw/roll (degrees) left out keep the ones last
        asked for: pitch 90 points the fingers straight down; yaw turns them about the vertical (at
        pitch 90, yaw 0 closes the fingers along world x, yaw 90 along y); roll turns them about the
        pointing direction (pointing level, +roll tips the top of an upright held object to the
        hand's right).  grip: an opening ('open', 'close' or 0..1) the gripper starts moving to as
        the motion starts - open on the approach with it, but let go of a held object with
        gripper() once there (with grip= it would let go at the start).  path: 'auto' (a straight
        line, else a joint swing that keeps the grasp point within 3 cm of it), 'line', or 'swing'
        (any joint swing).  dry_run=True only checks reachability (no motion, no steps).
        Returns a Move (.reached, .blocked, .lag_mm, .at, .angles, .steps).  A pose that cannot be
        reached raises NotExecuted (nothing moves); a dry run reports it instead; a height up to 5 mm
        too low for the fingertips to clear the table is raised to the lowest that clears it.  A move
        that stalls against something stops early, and it or one that ends BLOCKED by more than 25 mm
        (10 mm going down with the gripper open and empty) raises Blocked; what it held slipping out
        while carried (not in the last 2 cm of a descent) raises Dropped.  contact=True, for a press,
        push or insertion meant to meet resistance: no early stop and neither error.
        gentle=True: no faster middle on long moves, for loads that slip or swing when moved fast.
        """
        return self._call("goto", (arm, xyz), dict(by=by, pitch=pitch, yaw=yaw, roll=roll, grip=grip,
                                                    path=path, dry_run=dry_run, contact=contact, gentle=gentle))

    def gripper(self, arm, opening):
        """Open or close a gripper: 'open', 'close' or 0..1 (1 = fully open, ~8.9 cm gap).
        Returns a Grip (.opening, .width_m, .holding: True, False, or None when the fingers did not
        close).  Closing keeps squeezing until changed.  An open of a gripper that holds something
        that stops well short, less than 2 cm wider than what it held, raises Blocked (it may still
        be caught between the fingers)."""
        return self._call("gripper", (arm, opening), {})

    def home(self, arm="both"):
        """Lift (if low) and return an arm, or both, to the starting pose.  Returns a Homing."""
        return self._call("home", (arm,), {})

    def wait(self, n):
        """Hold everything still for n steps (one step = 0.04 s of simulated time), n <= 200."""
        return self._call("wait", (n,), {})

    # -- perception (free: no steps) -------------------------------------------------------------
    def find(self, cam="head", text=None, *, at=None, points=None, labels=None, box=None):
        """Outline objects in one camera's current picture with SAM3.

        text: a short noun phrase ('red cup'); returns every instance found, best first.
        at=(x, y): the object under that pixel.  points=[(x, y), ...] with labels=[1, 0, ...]
        (1 = on the object, 0 = not on it).  box=(x0, y0, x1, y1).  Returns a list of Mask.
        """
        return self._call("find", (cam, text), dict(at=at, points=points, labels=labels, box=box))

    def measure(self, target, *, cams="auto"):
        """Measure an object in 3-D from the cameras that see it.

        target: text ('red cup'), a Mask, (cam, (x, y)), a list of Masks of the same object in
        different cameras, or a Thing measured before (re-measured where it is now).
        cams: 'auto' (all cameras that see it) or a list of camera names.  Returns a Thing
        (.center, .top, .size, .grip_yaw, .on_z, .views).
        """
        return self._call("measure", (target,), dict(cams=cams))

    def to_world(self, cam, xy, *, z=None, pair=None):
        """The world point seen at pixel ``xy`` of camera ``cam``.

        Default: the surface under the pixel - matched between two cameras' pictures where it has
        texture, else on the measured object outlined there, else on the table; the Spot says which.
        z=0.80: where the pixel's ray meets the plane z = 0.80.
        pair=(cam2, (x2, y2)): triangulate with the same point seen in a second camera.
        Returns a Spot (.xyz, .source, .warning).
        """
        return self._call("to_world", (cam, xy), dict(z=z, pair=pair))

    def to_pixel(self, xyz, cam="head"):
        """Pixel (x, y) where world point ``xyz`` appears in ``cam``, or None if out of view."""
        return self._call("to_pixel", (xyz, cam), {})

    # -- reading ---------------------------------------------------------------------------------
    @property
    def task(self):
        """The task: instruction, step budget (used / left), start pose of each arm."""
        return self._call("task", (), {})

    def state(self):
        """Both arms now: grasp point, angles, gripper opening and holding, distance from home;
        the step budget."""
        return self._call("state", (), {})

    def image(self, cam="head"):
        """The current RGB picture of a camera as a numpy array (H, W, 3), uint8."""
        return self._call("image", (cam,), {})

    def look(self, cams="head", *, grid=False, marks=None, masks=None, crop=None, zoom=None,
             tile=False, note=None):
        """Attach the current picture(s) to this cell's result, so you can see them.

        cams: a camera name or a list ('head', 'left_wrist', 'right_wrist'); 'all' = the three.
        grid=True: coordinate grid with pixel labels.  marks: points to draw, each (x, y) or
        (x, y, 'label'), or world points as ('world', (x, y, z), 'label').  masks: Masks or Things
        to outline.  crop=(x0, y0, x1, y1) and zoom (e.g. 3): enlarge a region; labels keep the
        full picture's pixel coordinates.  tile=True: the cameras side by side in one image (one
        image of the budget).  Pictures are taken at the moment of the call.
        """
        return self._call("look", (cams,), dict(grid=grid, marks=marks, masks=masks, crop=crop, zoom=zoom,
                                                tile=tile, note=note))

    def camera(self, cam="head"):
        """Camera model: focal length (px), image centre, pose in the world now."""
        return self._call("camera", (cam,), {})

    @property
    def things(self):
        """Every object measured so far in this episode (latest measurement of each)."""
        return self._call("things", (), {})

    def __repr__(self) -> str:
        return ("robot: goto, gripper, home, wait | find, measure, to_world, to_pixel | "
                "task, state, image, look, camera, things")
