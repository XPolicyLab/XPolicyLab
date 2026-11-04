"""The host side of ``robot.*``: owns the motion driver, the cameras and the perception tools.

``dispatch`` is what the kernel calls for every robot call a cell makes.  Arguments come from the
sandbox: they are checked here, and errors the model should see come back as
:class:`~hexashell.kernel.host.CallFailed`.  When the episode ends during a call, the call answers
:class:`~hexashell.kernel.host.Stop` and so does every later one.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..errors import Refused
from ..kernel.host import CallFailed, Stop
from ..motion.driver import Driver, EpisodeOver, MotionError, Solid, display_angles
from ..motion.chain import slide
from ..motion.rig import ROBODOJO, Rig
from ..state import CAMERAS, Snapshot, parse_obs
from ..xpl.bridge import Bridge
from . import results as R
from ..vision import overlay
from ..vision.cameras import cameras_for, check_cam

PUBLIC = ("goto", "gripper", "home", "wait", "find", "measure", "to_world", "to_pixel", "task", "state",
          "image", "look", "camera", "things")
#: The calls that may close a hand on something or let go of it (see RobotHost._carry).
MOTIONS = ("goto", "gripper", "home", "wait")


def describe_verdict(verdict: dict | None) -> str:
    v = verdict or {}
    if v.get("abandoned"):
        return "the evaluation stopped this episode"
    if v.get("success"):
        return "the environment reports SUCCESS: the task is complete"
    if v.get("step_limit") is not None and v.get("step") is not None and v["step"] >= v["step_limit"]:
        return f"the step budget is used up ({v['step']} of {v['step_limit']}); the episode is over"
    return "the episode is over"


class RobotHost:
    def __init__(self, bridge: Bridge | None, first: Snapshot, *, scene=None,
                 log: Callable[[dict], None] | None = None,
                 attach: Callable[[str, np.ndarray, str], None] | None = None,
                 on_step: Callable[[int, dict, dict], None] | None = None, speed=None, rig: Rig = ROBODOJO):
        self.bridge = bridge
        self.snap = first
        #: per camera, the snapshot of the latest picture the model has seen (the start pictures, then
        #: robot.look's from earlier turns): to_world checks picked pixels against it
        self.shown: dict[str, Snapshot] = dict.fromkeys(first.images, first)
        self.looked: dict[str, Snapshot] = {}
        self.log = log or (lambda event: None)
        self.attach = attach or (lambda kind, image, caption: None)
        self.on_step = on_step or (lambda seq, obs, action: None)
        self.rig = rig
        self.driver = Driver(self, first, log=self.log, speed=speed, rig=rig)
        self.scene = scene
        self.verdict: dict | None = None
        #: the perception told when a hand closes on something and lets go (see _carry), if any
        self.carry = None
        if scene is not None:
            self.bind_scene(scene)

    def bind_scene(self, scene) -> None:
        """Give the perception its pictures (RoboDojo's Scene reads them from this host), and tell it what
        the hands carry."""
        scene.bind(self)
        self.carry = scene

    # -- what a benchmark other than RoboDojo answers differently --------------------------------
    def over(self) -> bool:
        """Whether the episode has ended."""
        return self.verdict is not None or self.bridge.over

    def over_reason(self) -> str:
        if self.verdict is None:
            self.verdict = self.bridge.verdict or {}
        return describe_verdict(self.verdict)

    def pictures_now(self, cams=None) -> dict[str, np.ndarray]:
        """The cameras' latest pictures (``cams``: those needed, when not all)."""
        return self.snap.images

    def cameras_now(self, cams=None) -> dict:
        """The camera models of those pictures."""
        return cameras_for(self.snap)

    # -- the driver's link to the simulator ------------------------------------------------------
    def act(self, action: dict) -> Snapshot:
        seq, obs = self.bridge.act(action)
        self.snap = parse_obs(obs, seq, previous=self.snap)
        self.on_step(seq, obs, action)
        return self.snap

    # -- dispatch ----------------------------------------------------------------------------------
    def dispatch(self, name: str, args: tuple, kwargs: dict) -> Any:
        if self.over():
            raise Stop(self.over_reason())
        if name not in PUBLIC:
            raise CallFailed("RobotError", f"robot has no '{name}'")
        method = getattr(self, "_" + name)
        start_step, start_actions = self.snap.step, self.driver.actions
        hands = self._hands() if self.carry is not None and name in MOTIONS else None
        try:
            return method(*args, **kwargs)
        except EpisodeOver as exc:
            # name the call it ended in: R0 and R1 ended 6 of 6 successes inside home(), which printed
            # nothing, and three analyses took them for successes "at the lift before homing"
            self.verdict = exc.verdict or {}
            end = self.verdict.get("step")
            into = (end - start_step if end is not None and start_step is not None
                    else self.driver.actions - start_actions)
            self.log({"kind": "verdict", "during": name, "step": end, "into": into})
            raise Stop(f"{self.over_reason()} (during robot.{name}(), {into} steps into it)") from None
        except R.NotFound as exc:
            raise CallFailed("NotFound", str(exc)) from None
        except (MotionError, Refused) as exc:
            raise CallFailed("ValueError", str(exc)) from None
        except TypeError as exc:
            if "argument" in str(exc):
                raise CallFailed("TypeError", f"robot.{name}: {exc}") from None
            raise
        finally:
            if hands is not None:
                self._carry(hands)

    def _hands(self) -> dict:
        """Per arm, what tells a close or a let-go: (grasp pose, whether the fingers close, whether they
        hold, the latest close that caught something)."""
        d = self.driver
        return {arm: (d.grasp_pose(arm), d.squeezing(arm), d.holding(arm) is True, d.caught[arm])
                for arm in self.rig.arms}

    def _carry(self, before: dict) -> None:
        """Tell the perception what a motion did with the hands: let go (the fingers opened - as the motion
        started -, what they held slipped out, or they closed anew) and closed on something."""
        d = self.driver
        for arm, (start, squeezing, holding, caught) in before.items():
            opened = squeezing and not d.squeezing(arm)
            anew = d.caught[arm] is not caught
            if opened or anew or (holding and d.holding(arm) is not True):
                self.carry.released(start if opened else d.grasp_pose(arm), arm)
            if anew and d.grasped[arm]:
                held = d.held_thing[arm]
                self.carry.grasped(d.grasp_pose(arm), arm, held[0] if held is not None else None)

    # -- motion ------------------------------------------------------------------------------------
    def _goto(self, arm, xyz=None, *, by=None, pitch=None, yaw=None, roll=None, grip=None, path="auto",
              dry_run=False, contact=False, gentle=False):
        move = self.driver.goto(arm, None if xyz is None else point_of(xyz), by=by, pitch=_num(pitch, "pitch"),
                                yaw=_num(yaw, "yaw"), roll=_num(roll, "roll"), grip=grip, path=path,
                                dry_run=bool(dry_run), things=self._solids(), contact=bool(contact),
                                gentle=bool(gentle))
        if not move.moved and not move.dry_run:
            # refused: the cell stops here (unless it catches NotExecuted), so that its next motions
            # do not run from the wrong place
            raise CallFailed("NotExecuted", str(move))
        # the same when the move ran but something stopped it, or what it held slipped out (R1: cells
        # went on to open, place or press from there)
        if move.halt == "Dropped":
            rest = ("the rest of the motion and the cell were skipped - look where it went" if move.skipped else
                    "the rest of the cell was skipped so you can look")
            raise CallFailed("Dropped", f"{move}\n{rest}")
        if move.halt == "Blocked":
            raise CallFailed("Blocked", f"{move}\nthe move ran; the rest of the cell was skipped so you can look")
        return move

    def _gripper(self, arm, opening):
        grip = self.driver.gripper(arm, opening, things=self._solids())
        # an open that left the fingers on what they held (the lift that follows would take it along), or a
        # close on the open fingertips' rest
        if grip.halt == "Blocked":
            raise CallFailed("Blocked", f"{grip}\nthe rest of the cell was skipped so you can look")
        return grip

    def _home(self, arm="both"):
        return self.driver.home(arm)

    def _wait(self, n):
        try:
            n = int(n)
        except (TypeError, ValueError):
            raise Refused("wait(n): n is a whole number of steps, 1..200") from None
        steps = self.driver.wait(n)
        return R.Waited(steps=steps, steps_left=self.snap.steps_left, step_s=self.rig.step_s)

    # -- perception ------------------------------------------------------------------------------
    def _need_scene(self):
        if self.scene is None:
            raise Refused("perception (find / measure / surface points) is not available in this run")
        return self.scene

    def _find(self, cam="head", text=None, *, at=None, points=None, labels=None, box=None):
        return self._need_scene().find(check_cam(cam), text, at=at, points=points, labels=labels, box=box)

    def _measure(self, target, *, cams="auto"):
        return self._need_scene().measure(target, cams=cams)

    def _to_world(self, cam, xy, *, z=None, pair=None):
        from ..vision.spots import to_world

        return to_world(self, check_cam(cam), xy, z=z, pair=pair)

    def _to_pixel(self, xyz, cam="head"):
        cam = check_cam(cam)
        px = self.cameras_now([cam])[cam].pixel(point_of(xyz))
        return None if px is None else (round(px[0], 1), round(px[1], 1))

    def _things(self):
        if self.scene is None:
            return R.ThingList()
        return R.ThingList(self.scene.things())

    def _solids(self) -> list[Solid]:
        """The measured Things as upright boxes, for goto to check its path against (not those measured
        in a hand, which have moved with it)."""
        if self.scene is None:
            return []
        solids = []
        for t in self.scene.things():
            if not t.found or t.center_xyz is None or t.size is None or t.top is None or t.on.startswith("held"):
                continue
            length, width, height = (float(v) for v in t.size)
            ring = t.round or t.grip_yaw is None
            solids.append(Solid(id=t.id, label=t.label, center=(float(t.center_xyz[0]), float(t.center_xyz[1])),
                                half=(length / 2, (length if ring else width) / 2),
                                heading=0.0 if t.grip_yaw is None else float(t.grip_yaw) - 90.0, round=ring,
                                bottom=float(t.on_z) if t.on_z is not None else float(t.top) - height,
                                top=float(t.top), views=len(t.views), step=t.step))
        return solids

    # -- reading ---------------------------------------------------------------------------------
    def _task(self):
        start = {}
        for arm in self.rig.arms:
            pose = self.driver.start_pose[arm]
            start[arm] = (tuple(round(float(v), 3) for v in pose[:3, 3]), display_angles(pose[:3, :3]))
        return R.TaskInfo(instruction=self.snap.instruction, step=self.snap.step, step_limit=self.snap.step_limit,
                          start=start, task_name=self.snap.info.get("task_name"))

    def _state(self):
        arms = {}
        for arm in self.rig.arms:
            reading = self.snap.arms[arm]
            pose = slide(reading.flange, self.rig.grasp_reach)
            dist, turn = self.driver.home_offset(arm)
            arms[arm] = R.ArmStatus(arm=arm, at=tuple(round(float(v), 4) for v in pose[:3, 3]),
                                    angles=display_angles(pose[:3, :3]), gripper=round(float(reading.gripper), 3),
                                    goal=float(self.driver.grip_goal[arm]), holding=self.driver.holding(arm),
                                    joints=tuple(round(float(v), 4) for v in reading.joints),
                                    home_dist_m=dist, home_turn_deg=turn)
        return R.Status(step=self.snap.step, step_limit=self.snap.step_limit, arms=arms)

    def _image(self, cam="head"):
        cam = check_cam(cam)
        image = self.pictures_now([cam]).get(cam)
        if image is None:
            raise Refused(f"no {cam} picture in the latest observation")
        return image.copy()

    def _camera(self, cam="head"):
        cam = check_cam(cam)
        return self.cameras_now([cam])[cam].info()

    def _look(self, cams="head", *, grid=False, marks=None, masks=None, crop=None, zoom=None, tile=False, note=None):
        names = list(CAMERAS) if cams in ("all", None) else [cams] if isinstance(cams, str) else list(cams)
        names = [check_cam(c) for c in names]
        if not names:
            raise Refused("look: name at least one camera")
        if crop is not None and len(names) > 1:
            raise Refused("look: crop applies to one camera at a time")
        cams_now = self.cameras_now(names)
        pictures = self.pictures_now(names)
        rendered, notes = [], []
        for cam in names:
            image = pictures.get(cam)
            if image is None:
                notes.append(f"no {cam} picture")
                continue
            outlines = self._outlines(masks, cam)
            camera = cams_now[cam]
            img, more = overlay.render(image, cam=cam, grid=bool(grid), marks=marks, outlines=outlines, crop=crop,
                                       zoom=zoom, project=lambda p, c=camera: c.pixel(point_of(p)))
            rendered.append((cam, img))
            notes += more
        if not rendered:
            raise Refused("look: none of those pictures is available")
        self.looked.update(dict.fromkeys((c for c, _ in rendered), self.snap))
        extras = [w for w, on in (("grid", grid), ("marks", marks), ("outlines", masks), ("crop", crop)) if on]
        what = ", ".join(c for c, _ in rendered) + (f" ({', '.join(extras)})" if extras else "")
        caption = f"look: {what}" + (f" — {note}" if note else "")
        if tile and len(rendered) > 1:
            self.attach("look", overlay.side_by_side([im for _, im in rendered]), caption + " (side by side)")
        else:
            for cam, img in rendered:
                self.attach("look", img, f"look: {cam}" + (f" ({', '.join(extras)})" if extras else "")
                            + (f" — {note}" if note else ""))
        return R.Picture(cams=[c for c, _ in rendered], what=what + (", side by side" if tile and len(rendered) > 1
                                                                    else ""), notes=notes)

    def pictures_shown(self) -> None:
        """The pictures robot.look took this turn, and the turn's results, have reached the model."""
        self.shown.update(self.looked)
        self.looked = {}
        self.driver.new_turn()

    def _outlines(self, masks, cam: str) -> list[tuple[np.ndarray, str]]:
        if masks is None:
            return []
        items = R.mask_items(masks)
        out = []
        for item in items:
            if isinstance(item, R.Mask):
                if item.cam == cam:
                    out.append((item.pixels, f"#{item.id} {item.label}"))
            elif isinstance(item, R.Thing):
                m = item.masks.get(cam)
                if m is not None:
                    out.append((m.pixels, f"T{item.id} {item.label}"))
            else:
                raise Refused("look(masks=...): give Masks (from find), Things (from measure) or a Thing's .masks")
        return out


def point_of(value) -> np.ndarray:
    """A world point from a tuple, an array, a Spot or a Thing (its centre)."""
    if isinstance(value, R.Thing):
        value = value.center
    elif isinstance(value, R.Spot):
        if value.xyz is None:
            raise R.NotFound(f"the Spot has no world point: {value.warning}")
        value = value.xyz
    elif isinstance(value, R.ArmStatus):
        value = value.at
    try:
        p = np.asarray(value, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused("a world point is three numbers (x, y, z) in metres") from None
    if p.size != 3 or not np.all(np.isfinite(p)):
        raise Refused("a world point is three numbers (x, y, z) in metres")
    return p


def _num(v, name: str):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise Refused(f"{name} is a number of degrees") from None
    if not math.isfinite(f):
        raise Refused(f"{name} is a number of degrees")
    return f
