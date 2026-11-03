"""Robot Runtime: the robot APIs both agents call.

Perception and geometric queries: locate, inspect, grasp_candidates, check_relation, reach_query.
Motion and view acquisition: move_to, shift, rotate, advance, gripper, observe, wait.
Bounded geometric controllers: orient_held, carry_tip_to, center_over, advance_until.

Every call returns a ToolOutcome: a JSON-able receipt plus images the caller
should see. Measurements become references in the shared store; motions run as
executor tracks, so `wait=False` returns at once and the completion arrives as
an event in the next decision's context. While the Local Interaction Agent is
active, every motion is checked against the interaction contract first.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from knowin_agent.core.events import EventBus
from knowin_agent.core.geometry import Pose, axis_angle_matrix, flange_for_point, matrix_to_quat, quat_to_matrix, unit, wrap180
from knowin_agent.core.memory import TaskContext
from knowin_agent.runtime.backend import ARMS, Reach, RobotBackend, other_arm
from knowin_agent.runtime.contract import InteractionContract
from knowin_agent.runtime.executor import ArmBusy, Executor, Track
from knowin_agent.runtime.budget import BudgetGovernor, Checkpoints
from knowin_agent.runtime.motion import DEFAULT_LIMITS, MotionLimits, follow, limits_with, move_flange, plan_move, set_gripper

CAMERA_ALIASES = {
    "head": "head", "cam_head": "head",
    "left_wrist": "left_wrist", "wrist_l": "left_wrist", "cam_left_wrist": "left_wrist", "left": "left_wrist",
    "right_wrist": "right_wrist", "wrist_r": "right_wrist", "cam_right_wrist": "right_wrist", "right": "right_wrist",
}
TASK_MAX_ADVANCE_M = 0.15
FINGER_TABLE_CLEARANCE_M = 0.004
THIN_FINGER_TABLE_CLEARANCE_M = 0.001  # as program/library/manipulation.py grasp_height


class ApiError(RuntimeError):
    """A request the runtime refuses or cannot measure; the message tells the agent why."""


@dataclass
class ToolOutcome:
    data: dict
    images: list = field(default_factory=list)


def _r(value, digits: int = 4):
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in value]
    return value


class RobotRuntime:
    def __init__(self, backend: RobotBackend, *, context: TaskContext, perceiver=None,
                 bus: Optional[EventBus] = None, limits: MotionLimits = DEFAULT_LIMITS, budget_mode: str = "enforce"):
        self.backend = backend
        self.profile = backend.profile
        self.bus = bus or EventBus()
        self.executor = Executor(backend, self.bus)
        self.perceiver = perceiver
        self.ctx = context
        self.limits = limits
        self.contract: Optional[InteractionContract] = None
        self.active_scope = "task"
        self.track_owner = "task"
        self.tip_offsets: dict[str, dict] = {}
        self.governor = BudgetGovernor(self, budget_mode)
        self.default_path = "auto"
        self.async_enabled = True
        self.wall_window: Optional[tuple] = None  # (start, seconds) of the episode's wall clock
        self.checkpoints = Checkpoints(self)
        self._failed_moves: dict[str, tuple] = {}

    # -- shared helpers ----------------------------------------------------------------

    @property
    def obs(self):
        return self.executor.observation

    @property
    def tick(self) -> int:
        return self.executor.tick_count

    def camera_name(self, name: str) -> str:
        key = CAMERA_ALIASES.get(str(name).strip().lower())
        if key is None or key not in self.obs.cameras:
            raise ApiError(f"unknown camera {name!r}; use one of {sorted(self.obs.cameras)}")
        return key

    def _arm(self, arm: str) -> str:
        if arm not in ARMS:
            raise ApiError(f"arm must be 'left' or 'right' (got {arm!r})")
        return arm

    def _need_perceiver(self):
        if self.perceiver is None:
            raise ApiError("no perception service is configured for this robot")
        return self.perceiver

    def _authorize(self, action: str, arm: Optional[str], **kwargs) -> None:
        if self.active_scope == "local" and self.contract is not None:
            self.contract.authorize(action, arm, **kwargs)

    def _rotation(self, arm: str, yaw=None, pitch=None, roll=None, quat=None) -> np.ndarray:
        current = self.obs.arms[arm].ee_pose.rotation
        if quat is not None:
            if any(v is not None for v in (yaw, pitch, roll)):
                raise ApiError("give quat or yaw/pitch/roll, not both")
            return quat_to_matrix(quat)
        if yaw is None and pitch is None and roll is None:
            return current
        p = 90.0 if pitch is None else float(pitch)
        y = self.profile.yaw_of(current, p) if yaw is None else float(yaw)
        return self.profile.orientation(y, p, 0.0 if roll is None else float(roll))

    def _point(self, target: Any) -> np.ndarray:
        try:
            return self.ctx.references.resolve_point(target)
        except (KeyError, ValueError) as exc:
            raise ApiError(str(exc)) from None

    def _flange_for(self, arm: str, point: np.ndarray, rotation: np.ndarray, frame: str) -> Pose:
        if frame == "grasp":
            return flange_for_point(point, rotation, self.profile.grasp_offset)
        if frame == "flange":
            return Pose.make(point, matrix_to_quat(rotation))
        if frame == "tip":
            tip = self.tip_offsets.get(arm)
            if tip is None:
                raise ApiError(f"no held tip measured for the {arm} arm; run inspect(mode='axis_tip', arm=...) first")
            return flange_for_point(point, rotation, tip["tip_in_flange"])
        raise ApiError("frame must be 'grasp', 'flange' or 'tip'")

    def _reach(self, arm: str, flange: Pose) -> Reach:
        kin = self.profile.kinematics
        if kin is None:
            return Reach(None, "no kinematics model")
        try:
            return kin.reach(arm, flange, self.obs.arms[arm].joints)
        except Exception as exc:  # noqa: BLE001 - reachability is advisory
            return Reach(None, f"kinematics error: {exc}")

    def _run(self, arms: Sequence[str], factory, *, label: str, wait: bool, queue: bool) -> dict:
        if not wait and not self.async_enabled:
            raise ApiError("asynchronous motion is disabled in this configuration; use wait=true")
        try:
            track = self.executor.submit(arms, factory, label=label, owner=self.track_owner, queue=queue)
        except ArmBusy as exc:
            raise ApiError(str(exc)) from None
        if not wait:
            return {"track": track.id, "status": track.status.value,
                    "note": "running asynchronously; its result arrives as an event, or call wait(tracks=[...])"}
        self.executor.run([track])
        return self._track_result(track)

    @staticmethod
    def _track_result(track: Track) -> dict:
        result = dict(track.result or {})
        result.setdefault("status", track.status.value)
        result["track"] = track.id
        return result

    def grasp_point(self, arm: str) -> np.ndarray:
        return self.profile.grasp_point(self.obs.arms[arm].ee_pose)

    def arm_status(self) -> dict:
        busy = self.executor.busy_arms()
        out = {}
        for arm, state in self.obs.arms.items():
            rotation = state.ee_pose.rotation
            tool = rotation @ np.asarray(self.profile.tool_axis, dtype=float)
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -tool[2]))))
            out[arm] = {
                "grasp_point": _r(self.grasp_point(arm).tolist()),
                "flange_xyz": _r(list(state.ee_pose.xyz)),
                "yaw_deg": round(self.profile.yaw_of(rotation, 90.0), 1),
                "tool_pitch_deg": round(pitch, 1),
                "gripper_command": round(float(state.gripper_command), 3),
                "gripper_opening": "unknown" if state.gripper_opening is None else round(state.gripper_opening, 3),
                "busy_with": busy[arm].id + " " + busy[arm].label if arm in busy else None,
            }
            if arm in self.tip_offsets:
                tip = state.ee_pose.transform(self.tip_offsets[arm]["tip_in_flange"])
                out[arm]["held_tip_xyz"] = _r(tip.tolist())
        return out

    def budget(self) -> dict:
        return self.backend.budget().as_dict()

    def note_start_poses(self) -> None:
        """Shared note with each arm's pose at episode start (tasks may require returning there)."""
        parts = []
        for arm, state in self.obs.arms.items():
            rotation = state.ee_pose.rotation
            tool = rotation @ np.asarray(self.profile.tool_axis, dtype=float)
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -tool[2]))))
            parts.append(f"{arm}: grasp point {_r(self.grasp_point(arm).tolist(), 3)}, flange "
                         f"{_r(list(state.ee_pose.xyz), 3)}, quat_wxyz {_r(list(state.ee_pose.quat), 4)} "
                         f"(tool pitch {pitch:.0f} deg)")
        self.ctx.notes.write("start_poses", "Arm poses at episode start — return here with move_to(frame='flange', "
                             "quat=...) when the task asks for the robot back at origin. " + "; ".join(parts),
                             "runtime", self.tick)

    # -- perception and geometric queries ----------------------------------------------------

    def locate(self, text: str, camera: str = "head", hint_px: Optional[Sequence[float]] = None,
               max_objects: int = 1, label: Optional[str] = None) -> ToolOutcome:
        perceiver = self._need_perceiver()
        cam = self.camera_name(camera)
        obs = self.executor.refresh()
        try:
            estimates = perceiver.locate(obs, text, cam, hint_px=hint_px, max_objects=max(1, min(8, int(max_objects))))
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f"locate failed: {exc}") from None
        objects = []
        for est in estimates:
            values = est.as_dict()
            ref = self.ctx.references.add("object", label or text, values,
                                          {"camera": cam, "obs_id": obs.obs_id, "method": values.get("method"),
                                           "tick": obs.tick}, obs.tick)
            objects.append({"ref": ref.ref_id, **{k: values[k] for k in (
                "center_xyz", "top_z", "height_m", "length_m", "width_m", "long_axis_deg", "grasp_yaw_deg",
                "score") if k in values}, "warnings": values.get("quality", {}).get("warnings", [])})
        return ToolOutcome({"camera": cam, "obs_id": obs.obs_id, "objects": objects})

    def inspect(self, mode: str, target: Optional[str] = None, camera: str = "head", arm: Optional[str] = None,
                radius_m: float = 0.08) -> ToolOutcome:
        perceiver = self._need_perceiver()
        cam = self.camera_name(camera)
        obs = self.executor.refresh()
        provenance = {"camera": cam, "obs_id": obs.obs_id, "tick": obs.tick}
        text = self._label_of(target)
        try:
            if mode == "shape":
                est = perceiver.inspect_shape(obs, self._need(text, "shape"), cam)
                values = est.as_dict()
                ref = self.ctx.references.add("object", text, values, {**provenance, "method": est.method}, obs.tick)
            elif mode == "axis_tip":
                grip = self.grasp_point(self._arm(arm)) if arm else None
                est = perceiver.inspect_axis_tip(obs, self._need(text, "axis_tip"), cam, gripper_point=grip)
                values = est.as_dict()
                if arm:
                    flange = obs.arms[arm].ee_pose
                    rt = flange.rotation.T
                    values["tip_in_flange"] = _r((rt @ (np.asarray(est.tip_xyz) - flange.position)).tolist())
                    values["axis_in_flange"] = _r((rt @ np.asarray(est.axis_dir)).tolist())
                    values["arm"] = arm
                    self.tip_offsets[arm] = {"tip_in_flange": values["tip_in_flange"],
                                             "axis_in_flange": values["axis_in_flange"], "label": text}
                ref = self.ctx.references.add("axis_tip", text, values, {**provenance, "method": est.method}, obs.tick)
            elif mode == "support_plane":
                around = None
                if target is not None and (not isinstance(target, str) or target in self._ref_ids()):
                    around = self._point(target)
                est = perceiver.inspect_support_plane(obs, cam, around_xyz=around, radius_m=radius_m,
                                                      exclude_text=None if around is not None else text)
                values = est.as_dict()
                ref = self.ctx.references.add("support_plane", text or "support", values,
                                              {**provenance, "method": est.method}, obs.tick)
            elif mode == "opening":
                est = perceiver.inspect_opening(obs, self._need(text, "opening"), cam)
                values = est.as_dict()
                ref = self.ctx.references.add("opening", text, values, {**provenance, "method": est.method}, obs.tick)
            else:
                raise ApiError("mode must be shape, axis_tip, support_plane or opening")
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f"inspect({mode}) failed: {exc}") from None
        if mode != "shape":
            values.pop("quality", None)
        return ToolOutcome({"ref": ref.ref_id, "mode": mode, "camera": cam, **values})

    def _ref_ids(self) -> set:
        return {s["ref"] for s in self.ctx.references.summaries(self.tick, limit=10_000)}

    def _label_of(self, target) -> Optional[str]:
        if isinstance(target, str) and target in self._ref_ids():
            return self.ctx.references.get(target).label
        return target if isinstance(target, str) else None

    @staticmethod
    def _need(text: Optional[str], mode: str) -> str:
        if not text:
            raise ApiError(f"inspect({mode}) needs a target description or an object reference (got none: a "
                           "locate that returned status 'not_found' has ref None; check its status first)")
        return text

    def grasp_candidates(self, target: str, arm: str = "auto", pitch: float = 90.0,
                         max_candidates: int = 4, depth_m: Optional[float] = None) -> ToolOutcome:
        ref = self.ctx.references.get(target) if isinstance(target, str) else None
        if ref is None or ref.kind != "object":
            raise ApiError("grasp_candidates needs an object reference from locate/inspect(shape)")
        v = ref.values
        center = np.asarray(v["center_xyz"], dtype=float)
        top, height = float(v["top_z"]), max(0.005, float(v.get("height_m") or 0.02))
        depth = min(0.02, 0.5 * height) if depth_m is None else float(depth_m)
        thin = height < self.profile.fingertip_beyond_grasp_m + FINGER_TABLE_CLEARANCE_M
        floor = top - height + self.profile.fingertip_beyond_grasp_m + (
            THIN_FINGER_TABLE_CLEARANCE_M if thin else FINGER_TABLE_CLEARANCE_M)
        grasp_z = max(top - depth, floor)
        point = np.array([center[0], center[1], grasp_z])
        long_axis = v.get("long_axis_deg")
        headings = [v["grasp_yaw_deg"]] if v.get("grasp_yaw_deg") is not None else [
            (long_axis or 0.0) + 90.0, (long_axis or 0.0)]
        arms = list(ARMS) if arm in ("auto", None) else [self._arm(arm)]
        busy = self.executor.busy_arms()
        rows = []
        for a in arms:
            for heading in headings:
                best = None
                for yaw in (self.profile.yaw_for_heading(heading, pitch), self.profile.yaw_for_heading(heading + 180, pitch)):
                    rotation = self.profile.orientation(yaw, pitch, 0.0)
                    reach = self._reach(a, flange_for_point(point, rotation, self.profile.grasp_offset))
                    key = ({True: 0, None: 1, False: 2}[reach.reachable], reach.joint_travel_deg or 999.0)
                    if best is None or key < best[0]:
                        best = (key, yaw, reach)
                side_bonus = 0 if (center[0] < 0) == (a == "left") else 1
                rows.append((best[0][0], a in busy, side_bonus, best[0][1], a, heading, best[1], best[2]))
        rows.sort(key=lambda r: r[:4])
        out = []
        for rank, (_, is_busy, _, _, a, heading, yaw, reach) in enumerate(rows[: max(1, int(max_candidates))]):
            values = {"grasp_xyz": _r(point.tolist()), "yaw_deg": round(wrap180(yaw), 1), "pitch_deg": float(pitch),
                      "arm": a, "closing_heading_deg": round(wrap180(heading), 1), "object_ref": ref.ref_id,
                      "approach": "top_down" if abs(pitch - 90) < 1 else f"pitch_{pitch:g}"}
            gref = self.ctx.references.add("grasp", f"grasp {ref.label}", values,
                                           {"from": ref.ref_id, "method": "object_footprint"}, self.tick)
            out.append({"ref": gref.ref_id, **values, **reach.as_dict(), **({"arm_busy": True} if is_busy else {})})
        warnings = [] if not ref.possibly_moved else [f"object reference may be stale: {ref.moved_reason}"]
        return ToolOutcome({"object": ref.ref_id, "candidates": out, "warnings": warnings})

    def check_relation(self, relation: str, a: str, b: Optional[str] = None, tolerance_m: float = 0.01) -> ToolOutcome:
        """Geometric evidence about two references; 'unknown' when a measurement is missing."""
        ra = self.ctx.references.get(a)
        rb = self.ctx.references.get(b) if b else None
        pa = ra.position()
        out: dict = {"relation": relation, "a": a, "b": b}
        stale = [r.ref_id for r in (ra, rb) if r is not None and (r.possibly_moved or r.invalidated)]
        if stale:
            out["warnings"] = [f"stale reference(s) {stale}: refresh with locate/inspect first"]
        if pa is None or (rb is not None and rb.position() is None):
            out["status"] = "unknown"
            return ToolOutcome(out)
        if relation == "distance":
            pb = rb.position()
            d = pa - pb
            out.update(status="measured", dxyz=_r(d.tolist()), xy_m=_r(float(np.linalg.norm(d[:2]))),
                       xyz_m=_r(float(np.linalg.norm(d))))
        elif relation in ("on_top", "inside", "aligned"):
            if rb is None:
                raise ApiError(f"{relation} needs two references")
            vb, va = rb.values, ra.values
            pb = rb.position()
            dxy = float(np.linalg.norm(pa[:2] - pb[:2]))
            half = 0.5 * float(vb.get("width_m") or vb.get("radius_m", 0.0) * 2 or 0.0)
            out.update(xy_offset_m=_r(dxy), xy_vector=_r((pa[:2] - pb[:2]).tolist()))
            if relation == "aligned":
                out["holds"] = dxy <= tolerance_m
                ya, yb = va.get("long_axis_deg"), vb.get("long_axis_deg")
                if ya is not None and yb is not None:
                    diff = abs(wrap180(2 * (float(ya) - float(yb)))) / 2
                    out["axis_difference_deg"] = round(diff, 1)
            elif relation == "on_top":
                bottom_a = float(va.get("top_z", pa[2])) - float(va.get("height_m", 0.0))
                top_b = float(vb.get("top_z", vb.get("rim_z", pb[2])))
                gap = bottom_a - top_b
                out.update(vertical_gap_m=_r(gap), holds=bool(dxy <= max(half, tolerance_m) and abs(gap) <= max(0.01, tolerance_m)))
            else:
                radius = float(vb.get("radius_m") or half)
                top_b = float(vb.get("rim_z", vb.get("top_z", pb[2])))
                top_a = float(va.get("top_z", pa[2]))
                out.update(radius_m=_r(radius), holds=bool(dxy <= max(radius, tolerance_m) and top_a <= top_b + 0.05))
            out["status"] = "measured"
        else:
            raise ApiError("relation must be distance, on_top, inside or aligned")
        return ToolOutcome(out)

    def reach_query(self, arm: str, target: Any, yaw=None, pitch=None, roll=None, quat=None,
                    frame: str = "grasp", variants: bool = True) -> ToolOutcome:
        arms = list(ARMS) if arm in ("auto", "both", None) else [self._arm(arm)]
        point = self._point(target)
        out = {}
        for a in arms:
            rotation = self._rotation(a, yaw, pitch, roll, quat)
            reach = self._reach(a, self._flange_for(a, point, rotation, frame))
            row = reach.as_dict()
            if variants and reach.reachable is not True and quat is None:
                alts = []
                base_pitch = 90.0 if pitch is None else float(pitch)
                for dp in (-15.0, -30.0, -45.0):
                    if base_pitch + dp < 0:
                        continue
                    alt = self._rotation(a, yaw, base_pitch + dp, roll)
                    if self._reach(a, self._flange_for(a, point, alt, frame)).reachable is True:
                        alts.append({"pitch": base_pitch + dp})
                for dz in (0.02, 0.05):
                    if self._reach(a, self._flange_for(a, point + [0, 0, dz], rotation, frame)).reachable is True:
                        alts.append({"higher_m": dz})
                row["reachable_variants"] = alts
            out[a] = row
        return ToolOutcome({"target": _r(point.tolist()), "frame": frame, "arms": out})

    # -- motion and view acquisition -----------------------------------------------------------

    def move_to(self, arm: str, target: Any, yaw=None, pitch=None, roll=None, quat=None, frame: str = "grasp",
                offset: Optional[Sequence[float]] = None, precision: bool = False, stop_on_contact: bool = False,
                gripper: Optional[float] = None, wait: bool = True, queue: bool = False,
                max_ticks: Optional[int] = None, speed: str = "normal", path: Optional[str] = None, final: bool = False,
                translation_step: Optional[float] = None, limits: Optional[MotionLimits] = None) -> ToolOutcome:
        """Plan, check reach and budget, then move. Nothing moves when the result is unreachable, path_blocked,
        over_budget or repeat_refused."""
        grasp_ref = None
        if isinstance(target, str):
            ref = self.ctx.references.get(target)
            if ref.kind == "grasp":
                grasp_ref = ref
        if arm in ("auto", None):
            if grasp_ref is None:
                raise ApiError("arm='auto' needs a grasp reference from grasp_candidates")
            arm = grasp_ref.values["arm"]
        arm = self._arm(arm)
        if grasp_ref is not None and all(v is None for v in (yaw, pitch, roll, quat)):
            yaw, pitch = grasp_ref.values["yaw_deg"], grasp_ref.values["pitch_deg"]
        path = path or self.default_path
        if path not in ("auto", "line", "joint", "ee"):
            raise ApiError("path must be auto, line, joint or ee")
        point = self._point(target)
        if offset is not None:
            point = point + np.asarray(offset, dtype=float).reshape(3)
        rotation = self._rotation(arm, yaw, pitch, roll, quat)
        flange = self._flange_for(arm, point, rotation, frame)
        reach = self._reach(arm, flange)
        if reach.reachable is False:
            return ToolOutcome({"status": "unreachable", "arm": arm, "reach": reach.as_dict(),
                                "hint": "reach_query lists reachable pitch/height variants"})
        limits = limits or self.limits
        if speed == "slow":
            limits = dataclasses.replace(limits, translation_step_m=0.008, rotation_step_rad=0.06)
        elif translation_step is not None:
            limits = dataclasses.replace(limits, translation_step_m=float(translation_step))
        signature = (arm, tuple(np.round(flange.position, 3)), tuple(np.round(flange.quat, 2)), path, precision)
        if self._failed_moves.get(arm) == signature and self.obs.arms[arm].ee_pose.errors_to(flange)[0] > 0.01:
            return ToolOutcome({"status": "repeat_refused", "arm": arm,
                                "reason": "the same move stalled or was blocked just before; change arm, pitch, height, "
                                          "path or approach"})
        grip = self.executor.hold_command(arm).gripper if gripper is None else float(gripper)
        plan = plan_move(self.profile, self.obs.arms[arm], arm, flange, grip=grip, limits=limits, precision=precision,
                         straight=stop_on_contact, path=path, reach=reach)
        if plan.blocked:
            self._failed_moves[arm] = signature
            return ToolOutcome({"status": "path_blocked", "arm": arm, "reason": plan.why})
        refusal = self.governor.check_move(arm, flange, plan.planned_actions, final=final)
        if refusal:
            return ToolOutcome({"status": "over_budget", "arm": arm, "budget": refusal})
        result = self._run([arm], lambda ctx: follow(ctx, plan, gripper=grip, limits=limits, precision=precision,
                                                     stop_on_contact=stop_on_contact, max_ticks=max_ticks),
                           label=f"move_to {arm}", wait=wait, queue=queue)
        retry = {"line": "joint", "joint": "line"}.get(str(plan.how).split("+")[0])
        if (result.get("status") in ("stalled", "timeout") and wait and not queue and not precision
                and not stop_on_contact and max_ticks is None and retry and path in ("auto", "line", "joint")):
            # One automatic retry along the other kind of path from where the arm stopped: after a stall the
            # agent otherwise spent a whole program switching line <-> joint by hand.
            second = plan_move(self.profile, self.obs.arms[arm], arm, flange, grip=grip, limits=limits,
                               path=retry)
            if not second.blocked and not self.governor.check_move(arm, flange, second.planned_actions, final=final):
                first = {k: result.get(k) for k in ("status", "reason", "path", "ticks", "position_error_mm")}
                result = self._run([arm], lambda ctx: follow(ctx, second, gripper=grip, limits=limits),
                                   label=f"move_to {arm}", wait=True, queue=False)
                result["ticks"] = int(result.get("ticks") or 0) + int(first.get("ticks") or 0)
                result["retried_after"] = first
        if result.get("status") in ("stalled", "timeout"):
            self._failed_moves[arm] = signature
        else:
            self._failed_moves.pop(arm, None)
        if reach.reachable is None:
            result["reach_warning"] = f"reachability unknown ({reach.reason}); such moves stall more often"
        return ToolOutcome(result)

    def reset(self) -> dict:
        """Practice only: restore the same scene. Tracks, holds, budget origin and references start over."""
        if not getattr(self.backend, "supports_reset", False):
            raise ApiError("this backend cannot reset the scene")
        for track in self.executor.active():
            self.executor.cancel(track.id, reason="scene reset")
        self.backend.reset()
        self.executor.rebind()
        self.governor = BudgetGovernor(self, self.governor.mode, self.governor.reserve)
        self.checkpoints.begin()
        self.tip_offsets.clear()
        self._failed_moves.clear()
        for summary in self.ctx.references.summaries(self.tick, limit=10_000):
            self.ctx.references.invalidate(summary["ref"], "scene reset")
        self.note_start_poses()
        return {"status": "reset", "budget": self.budget()}

    def subgoal_done(self, name: str) -> ToolOutcome:
        return ToolOutcome(self.governor.subgoal_done(name))

    def checkpoint(self, name: str, kind: str, **params) -> ToolOutcome:
        return ToolOutcome(self.checkpoints.run(name, kind, **params))

    def budget_panel(self) -> dict:
        return self.governor.panel()

    def _relative_move(self, arm: str, action: str, flange: Pose, *, translation_m: float, rotation_deg: float,
                       direction=None, stop_on_contact: bool, precision: bool, wait: bool, label: str) -> ToolOutcome:
        self._authorize(action, arm, translation_m=translation_m, rotation_deg=rotation_deg,
                        new_flange_xyz=flange.position, direction=direction)
        reach = self._reach(arm, flange)
        if reach.reachable is False:
            return ToolOutcome({"status": "unreachable", "arm": arm, "reach": reach.as_dict()})
        result = self._run([arm], lambda ctx: move_flange(ctx, arm, flange, limits=self.limits, precision=precision,
                                                          stop_on_contact=stop_on_contact),
                           label=label, wait=wait, queue=False)
        return ToolOutcome(result)

    def shift(self, arm: str, camera: str, right_m: float = 0.0, down_m: float = 0.0,
              du_px: Optional[float] = None, dv_px: Optional[float] = None, depth_m: Optional[float] = None,
              wait: bool = True) -> ToolOutcome:
        """Translate in the reference camera's image plane: +right and +down as seen in that image."""
        arm = self._arm(arm)
        frame = self.obs.camera(self.camera_name(camera))
        if not frame.calibrated:
            raise ApiError(f"camera {camera} has no calibration; cannot map image directions")
        r_cam = np.asarray(frame.extrinsics)[:3, :3]
        if du_px is not None or dv_px is not None:
            if depth_m is None:
                grasp = self.grasp_point(arm)
                depth_m = float((r_cam.T @ (grasp - np.asarray(frame.extrinsics)[:3, 3]))[2])
                if depth_m <= 0.02:
                    raise ApiError("cannot infer the depth of the correction; pass depth_m")
            fx = float(np.asarray(frame.intrinsics)[0, 0])
            right_m = float(du_px or 0.0) * depth_m / fx
            down_m = float(dv_px or 0.0) * depth_m / fx
        delta = r_cam[:, 0] * float(right_m) + r_cam[:, 1] * float(down_m)
        current = self.obs.arms[arm].ee_pose
        target = current.with_position(current.position + delta)
        return self._relative_move(arm, "shift", target, translation_m=float(np.linalg.norm(delta)), rotation_deg=0.0,
                                   stop_on_contact=False, precision=True, wait=wait,
                                   label=f"shift {arm} in {camera} view")

    def _axis(self, arm: str, axis: Any, frame: str, camera: Optional[str]) -> np.ndarray:
        names = {"x": 0, "y": 1, "z": 2}
        if frame == "world":
            base = np.eye(3)
        elif frame == "tool":
            base = self.obs.arms[arm].ee_pose.rotation
        elif frame == "camera":
            cam = self.obs.camera(self.camera_name(camera or self.profile.wrist_camera[arm]))
            if not cam.calibrated:
                raise ApiError("camera frame needs a calibrated camera")
            base = np.asarray(cam.extrinsics)[:3, :3]
        else:
            raise ApiError("frame must be world, tool or camera")
        if isinstance(axis, str):
            sign = -1.0 if axis.startswith("-") else 1.0
            key = axis.lstrip("+-")
            if key not in names:
                raise ApiError("axis must be x, y, z (optionally signed) or a vector")
            return sign * base[:, names[key]]
        return unit(base @ np.asarray(axis, dtype=float).reshape(3))

    def _pivot(self, arm: str, pivot: str) -> np.ndarray:
        flange = self.obs.arms[arm].ee_pose
        if pivot == "flange":
            return flange.position
        if pivot == "grasp":
            return self.grasp_point(arm)
        if pivot == "tip":
            if arm not in self.tip_offsets:
                raise ApiError("pivot='tip' needs inspect(mode='axis_tip', arm=...) first")
            return flange.transform(self.tip_offsets[arm]["tip_in_flange"])
        raise ApiError("pivot must be grasp, flange or tip")

    def rotate(self, arm: str, degrees: float, axis: Any = "z", frame: str = "world", camera: Optional[str] = None,
               pivot: str = "grasp", wait: bool = True) -> ToolOutcome:
        arm = self._arm(arm)
        degrees = float(degrees)
        if abs(degrees) > 180:
            raise ApiError("rotate at most 180 degrees per call")
        direction = self._axis(arm, axis, frame, camera)
        r_delta = axis_angle_matrix(direction, math.radians(degrees))
        current = self.obs.arms[arm].ee_pose
        p = self._pivot(arm, pivot)
        new_pos = p + r_delta @ (current.position - p)
        target = Pose.make(new_pos, matrix_to_quat(r_delta @ current.rotation))
        return self._relative_move(arm, "rotate", target, translation_m=float(np.linalg.norm(new_pos - current.position)),
                                   rotation_deg=abs(degrees), stop_on_contact=False, precision=False, wait=wait,
                                   label=f"rotate {arm} {degrees:+.0f}deg")

    def _direction(self, arm: str, direction: Any, camera: Optional[str]) -> np.ndarray:
        if direction is None and self.contract is not None and self.contract.advance_direction is not None:
            return np.asarray(self.contract.advance_direction, dtype=float)
        if direction in (None, "tool"):
            return unit(self.obs.arms[arm].ee_pose.rotation @ np.asarray(self.profile.tool_axis, dtype=float))
        if direction == "down":
            return np.array([0.0, 0.0, -1.0])
        if direction == "up":
            return np.array([0.0, 0.0, 1.0])
        if direction == "camera":
            cam = self.obs.camera(self.camera_name(camera or self.profile.wrist_camera[arm]))
            return unit(np.asarray(cam.extrinsics)[:3, 2])
        return unit(np.asarray(direction, dtype=float).reshape(3))

    def advance(self, arm: str, distance_m: float, direction: Any = None, camera: Optional[str] = None,
                stop_on_contact: bool = True, wait: bool = True) -> ToolOutcome:
        arm = self._arm(arm)
        distance_m = float(distance_m)
        if abs(distance_m) > TASK_MAX_ADVANCE_M:
            raise ApiError(f"advance at most {TASK_MAX_ADVANCE_M} m per call")
        d = self._direction(arm, direction, camera)
        current = self.obs.arms[arm].ee_pose
        target = current.with_position(current.position + d * distance_m)
        out = self._relative_move(arm, "advance", target, translation_m=abs(distance_m), rotation_deg=0.0,
                                  direction=d * np.sign(distance_m or 1.0), stop_on_contact=stop_on_contact,
                                  precision=abs(distance_m) <= 0.03, wait=wait, label=f"advance {arm}")
        out.data["direction"] = _r(d.tolist())
        return out

    def gripper(self, arm: str, command: Any = "close", settle_ticks: Optional[int] = None, wait: bool = True,
                until_still: bool = True) -> ToolOutcome:
        arm = self._arm(arm)
        value = {"open": 1.0, "close": 0.0}.get(command) if isinstance(command, str) else float(command)
        if value is None or not 0.0 <= value <= 1.0:
            raise ApiError("gripper command must be 'open', 'close' or a number in [0, 1] (1 = open)")
        opening = value > float(self.obs.arms[arm].gripper_command) + 1e-6
        self._authorize("gripper", arm, opening_gripper=opening)
        result = self._run([arm], lambda ctx: set_gripper(ctx, arm, value, settle_ticks=settle_ticks, until_still=until_still),
                           label=f"gripper {arm} {'open' if value > 0.5 else 'close'}", wait=wait, queue=False)
        flagged = self.ctx.references.note_interaction(self.grasp_point(arm), f"{arm} gripper {'opened' if opening else 'closed'}"
                                                      f" at tick {self.tick}")
        if opening and arm in self.tip_offsets:
            self.tip_offsets.pop(arm)
        if flagged:
            result["references_flagged"] = flagged
        return ToolOutcome(result)

    def observe(self, arm: str, target: Any, view: Any = "down", distance_m: float = 0.22,
                wait: bool = True) -> ToolOutcome:
        """Move this arm's wrist camera to look at `target` along `view` from `distance_m` away."""
        arm = self._arm(arm)
        self._authorize("observe", arm)
        cam_name = self.profile.wrist_camera[arm]
        frame = self.obs.camera(cam_name)
        if not frame.calibrated:
            raise ApiError(f"{cam_name} has no calibration; cannot plan a view")
        flange = self.obs.arms[arm].ee_pose
        t_fc = np.linalg.inv(flange.matrix()) @ np.asarray(frame.extrinsics)
        axis_f, mount_f = t_fc[:3, 2], t_fc[:3, 3]
        point = self._point(target)
        d = self._view_direction(view)
        best = None
        base = _rotation_between(axis_f, d)
        for spin in range(0, 360, 30):
            rotation = axis_angle_matrix(d, math.radians(spin)) @ base
            cam_pos = point - d * float(distance_m)
            pose = Pose.make(cam_pos - rotation @ mount_f, matrix_to_quat(rotation))
            if not self.profile.in_workspace(pose.xyz):
                continue
            reach = self._reach(arm, pose)
            image_down = rotation @ t_fc[:3, 1]
            readability = -float(image_down[1])  # image 'down' toward the robot (-y) reads naturally
            key = ({True: 0, None: 1, False: 2}[reach.reachable], reach.joint_travel_deg or 999.0, -readability)
            if best is None or key < best[0]:
                best = (key, pose, reach, spin)
        if best is None or best[2].reachable is False:
            raise ApiError("no reachable pose gives that view; try a different view direction or distance")
        result = self._run([arm], lambda ctx: move_flange(ctx, arm, best[1], limits=self.limits),
                           label=f"observe {arm}", wait=wait, queue=False)
        result.update(camera=cam_name, view_direction=_r(d.tolist()), distance_m=distance_m)
        if best[2].reachable is None:
            result["reach_warning"] = best[2].reason
        images = []
        if wait:
            images = [(f"{cam_name} after observe", self.obs.camera(cam_name).rgb)]
            projection = self._project(cam_name, point)
            if projection:
                result["target_pixel"] = projection
        return ToolOutcome(result, images)

    def _project(self, camera: str, point: np.ndarray) -> Optional[list]:
        frame = self.obs.camera(camera)
        if not frame.calibrated:
            return None
        t = np.linalg.inv(np.asarray(frame.extrinsics))
        p = t[:3, :3] @ point + t[:3, 3]
        if p[2] <= 1e-6:
            return None
        uvw = np.asarray(frame.intrinsics) @ p
        return [round(float(uvw[0] / uvw[2]), 1), round(float(uvw[1] / uvw[2]), 1)]

    @staticmethod
    def _view_direction(view: Any) -> np.ndarray:
        named = {"down": [0, 0, -1], "front": [0, 1, -0.6], "back": [0, -1, -0.6],
                 "left_side": [1, 0, -0.6], "right_side": [-1, 0, -0.6], "oblique": [0, 1, -1.2]}
        if isinstance(view, str):
            if view not in named:
                raise ApiError(f"view must be one of {sorted(named)} or a direction vector")
            return unit(named[view])
        return unit(np.asarray(view, dtype=float).reshape(3))

    def wait(self, ticks: Optional[int] = None, seconds: Optional[float] = None,
             tracks: Optional[Sequence[str]] = None, max_ticks: int = 600) -> ToolOutcome:
        if tracks:
            targets = [self.executor.track(t) for t in tracks]
            before = self.tick
            self.executor.run(targets, max_ticks=max_ticks)
            return ToolOutcome({"waited_ticks": self.tick - before,
                                "tracks": [self._track_result(t) if t.finished else t.summary() for t in targets]})
        if seconds is not None:
            ticks = int(round(float(seconds) * self.profile.control_hz))
        ticks = max(0, min(int(ticks or 0), max_ticks))
        done = self.executor.advance(ticks)
        return ToolOutcome({"waited_ticks": done, "episode_over": self.executor.episode_over})

    # -- bounded geometric controllers ------------------------------------------------------------

    def orient_held(self, arm: str, target_direction: Sequence[float], pivot: str = "tip",
                    tolerance_deg: float = 3.0, max_iterations: int = 3, camera: Optional[str] = None) -> ToolOutcome:
        arm = self._arm(arm)
        if arm not in self.tip_offsets:
            raise ApiError("orient_held needs inspect(mode='axis_tip', arm=...) of the held object first")
        goal = unit(np.asarray(target_direction, dtype=float).reshape(3))
        history = []
        for iteration in range(max(1, int(max_iterations))):
            flange = self.obs.arms[arm].ee_pose
            axis = unit(flange.rotation @ np.asarray(self.tip_offsets[arm]["axis_in_flange"], dtype=float))
            error = math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ goal)))))
            history.append(round(error, 2))
            if error <= tolerance_deg:
                return ToolOutcome({"status": "aligned", "error_deg": round(error, 2), "iterations": iteration,
                                    "history_deg": history})
            rot_axis = np.cross(axis, goal)
            if np.linalg.norm(rot_axis) < 1e-9:
                rot_axis = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
            step = min(error, 45.0)
            out = self.rotate(arm, step, axis=unit(rot_axis).tolist(), frame="world", pivot=pivot)
            if out.data.get("status") not in ("reached",):
                return ToolOutcome({"status": out.data.get("status"), "error_deg": round(error, 2), "history_deg": history,
                                    "last_move": out.data})
            if camera is not None:
                label = self.tip_offsets[arm]["label"]
                self.inspect("axis_tip", label, camera=camera, arm=arm)
        flange = self.obs.arms[arm].ee_pose
        axis = unit(flange.rotation @ np.asarray(self.tip_offsets[arm]["axis_in_flange"], dtype=float))
        error = math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ goal)))))
        history.append(round(error, 2))
        return ToolOutcome({"status": "aligned" if error <= tolerance_deg else "residual", "error_deg": round(error, 2),
                            "history_deg": history,
                            "note": "residual is model-based unless a camera re-measured the axis"})

    def carry_tip_to(self, arm: str, target: Any, approach_offset_m: float = 0.0, precision: bool = True,
                     wait: bool = True) -> ToolOutcome:
        arm = self._arm(arm)
        if arm not in self.tip_offsets:
            raise ApiError("carry_tip_to needs inspect(mode='axis_tip', arm=...) of the held object first")
        point = self._point(target)
        flange = self.obs.arms[arm].ee_pose
        tip_offset = np.asarray(self.tip_offsets[arm]["tip_in_flange"], dtype=float)
        axis = unit(flange.rotation @ np.asarray(self.tip_offsets[arm]["axis_in_flange"], dtype=float))
        point = point - axis * float(approach_offset_m)
        goal = flange_for_point(point, flange.rotation, tip_offset)
        reach = self._reach(arm, goal)
        if reach.reachable is False:
            return ToolOutcome({"status": "unreachable", "arm": arm, "reach": reach.as_dict()})
        result = self._run([arm], lambda ctx: move_flange(ctx, arm, goal, limits=self.limits, precision=precision,
                                                          stop_on_contact=True),
                           label=f"carry_tip_to {arm}", wait=wait, queue=False)
        if wait:
            tip_now = self.obs.arms[arm].ee_pose.transform(tip_offset)
            result["tip_error_mm"] = round(float(np.linalg.norm(tip_now - point)) * 1000, 2)
        return ToolOutcome(result)

    def center_over(self, arm: str, target: str, camera: Optional[str] = None, height_m: Optional[float] = None,
                    tolerance_mm: float = 1.5, max_iterations: int = 4) -> ToolOutcome:
        arm = self._arm(arm)
        cam = self.camera_name(camera or self.profile.wrist_camera[arm])
        text = self._label_of(target) or target
        history = []
        for iteration in range(max(1, int(max_iterations))):
            found = self.locate(text, cam).data["objects"]
            if not found:
                return ToolOutcome({"status": "not_found", "camera": cam, "history_mm": history})
            obj = found[0]
            grasp = self.grasp_point(arm)
            error = np.asarray(obj["center_xyz"][:2]) - grasp[:2]
            err_mm = float(np.linalg.norm(error)) * 1000
            history.append(round(err_mm, 2))
            if err_mm <= tolerance_mm:
                return ToolOutcome({"status": "centered", "error_mm": round(err_mm, 2), "iterations": iteration,
                                    "object_ref": obj["ref"], "history_mm": history})
            z = grasp[2] if height_m is None else float(obj["top_z"]) + float(height_m)
            goal = np.array([obj["center_xyz"][0], obj["center_xyz"][1], z])
            move = self.move_to(arm, goal.tolist(), frame="grasp", precision=err_mm < 20)
            if move.data.get("status") not in ("reached",):
                return ToolOutcome({"status": move.data.get("status"), "history_mm": history, "last_move": move.data})
        return ToolOutcome({"status": "residual", "history_mm": history,
                            "note": "not within tolerance after the iteration limit"})

    def advance_until(self, arm: str, direction: Any = "down", max_distance_m: float = 0.08, step_m: float = 0.004,
                      camera: Optional[str] = None) -> ToolOutcome:
        arm = self._arm(arm)
        if not 0 < float(max_distance_m) <= TASK_MAX_ADVANCE_M:
            raise ApiError(f"max_distance_m must be in (0, {TASK_MAX_ADVANCE_M}]")
        d = self._direction(arm, direction, camera)
        start = self.obs.arms[arm].ee_pose
        target = start.with_position(start.position + d * float(max_distance_m))
        limits = limits_with(translation_step_m=float(step_m), precision_step_m=min(float(step_m), 0.003),
                             stall_window=6)
        result = self._run([arm], lambda ctx: move_flange(ctx, arm, target, limits=limits, stop_on_contact=True),
                           label=f"advance_until {arm}", wait=True, queue=False)
        travelled = float((self.obs.arms[arm].ee_pose.position - start.position) @ d)
        stop = {"blocked": "contact", "reached": "max_distance"}.get(result.get("status"), result.get("status"))
        result.update(travelled_m=round(travelled, 4), stopped_by=stop, direction=_r(d.tolist()))
        return ToolOutcome(result)

    # -- convenience for skills ---------------------------------------------------------------------

    def other_arm(self, arm: str) -> str:
        return other_arm(arm)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return axis_angle_matrix(perp, math.pi)
    return axis_angle_matrix(v, math.atan2(float(np.linalg.norm(v)), c))
