"""The function surface agent programs call: H4/H5 names, signatures and result fields on KnowinAgent's runtime.

Compatibility layer: get_instruction, get_observation (RoboDojo-native layout), step (one native action for both
arms), reset (practice only), move_to, set_gripper, check_reach, checkpoint, subgoal_done, budget_panel/status.
Status names follow H5 (reached -> 'tracking_reached', unreachable -> 'model_unreachable'); a stall with
on_stall='interrupt' stops the whole program, like an interrupted H5 stage.

KnowinAgent capabilities in the same namespace: locate / inspect / observe / grasp_candidates / run_skill, asynchronous
moves (move_to(..., wait=False) then wait()), both() for two concurrent arm moves, tracks(), cancel_track().

Every robot call checks the program's wall-clock deadline. A stop (motion interrupt, failed checkpoint, budget
refusal, episode end, timeout) is latched: any later robot call in the same program raises it again, even if the
program caught the first one.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from collections import Counter
from typing import Any, Callable, Optional

import numpy as np

from knowin_agent.core.geometry import Pose, as_vector, matrix_to_quat, quat_angle, quat_normalize, quat_to_matrix, wrap180
from knowin_agent.runtime.api import ApiError, RobotRuntime
from knowin_agent.runtime.backend import ARMS, ArmCommand
from knowin_agent.runtime.budget import CheckpointFailed as RuntimeCheckpointFailed
from knowin_agent.runtime.executor import EpisodeOver
from knowin_agent.runtime.motion import GRIP_EMPTY_CLOSED, GRIP_HOLD_MARGIN
from knowin_agent.program.sandbox import ProgramStop, ProgramTimeout, Sandbox

STATUS_NAMES = {"reached": "tracking_reached", "unreachable": "model_unreachable"}
FRAMES = {"ee": "flange", "flange": "flange", "grasp": "grasp", "tip": "tip", "tcp": "flange"}
MOVED = ("tracking_reached", "blocked", "stalled", "dropped", "timeout", "episode_over", "error", "cancelled")
LIFT_MIN_M, LIFT_MIN_RAD = 0.025, math.radians(10)
WAIT_MAX_TICKS = 600
RESET_MIN_EXECUTIONS = 10


class MotionInterrupted(ProgramStop):
    reason = "motion_interrupted"


class CheckpointFailed(ProgramStop):
    reason = "checkpoint_failed"


class OverBudgetRefused(ProgramStop):
    reason = "over_budget_refused"


class EpisodeEnded(ProgramStop):
    reason = "episode_over"


def native_camera_name(profile, logical: str) -> str:
    """RoboDojo-native camera key ('cam_head', 'cam_left_wrist', ...) for a runtime camera name."""
    native = str(profile.cameras.get(logical, logical))
    return native if native.startswith("cam_") else "cam_" + logical


def _r(value, digits: int = 4):
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_r(v, digits) for v in value]
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or \
            not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _steps(value, name: str, low: int = 1, high: int = 1000) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


class ProgramAPI:
    def __init__(self, runtime: RobotRuntime, *, phase: str = "practice", library=None):
        if phase not in ("practice", "test"):
            raise ValueError("phase must be 'practice' or 'test'")
        if library is None:
            from knowin_agent.skills.library import SkillLibrary

            library = SkillLibrary()
        self.rt = runtime
        self.profile = runtime.profile
        self.phase = phase
        self.library = library
        self.executions_left: Optional[int] = None  # counted programs left, set by the episode before each program
        self.sandbox: Optional[Sandbox] = None
        self.labeled_stages: dict[str, dict] = {}
        self.begin()

    # -- per-program state ----------------------------------------------------------------------

    def bind(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def begin(self) -> None:
        self.calls: Counter = Counter()
        self.latched: Optional[ProgramStop] = None
        self.interruption: Optional[dict] = None
        self.pending: dict[str, dict] = {}
        self.touched: list[str] = []
        self.images: list = []
        self.resets = 0
        self.native_spent = 0
        self.segment_start = self.rt.backend.budget().ticks_used

    def native_used(self) -> int:
        return self.native_spent + self.rt.backend.budget().ticks_used - self.segment_start

    def stop(self, exc: ProgramStop) -> ProgramStop:
        """Latch a stop and record where it happened; the caller raises the returned exception."""
        if self.latched is None:
            where = self.sandbox.current_location() if self.sandbox else {}
            self.latched = exc
            self.interruption = {"reason": exc.reason, "message": str(exc), **where,
                                 "native_steps": self.rt.backend.budget().ticks_used,
                                 "detail": _compact(exc.detail)}
        return exc

    def record_stop(self, exc: ProgramStop, where: dict) -> None:
        """A stop that reached the top of the program without passing through a robot call (deadline in a loop)."""
        if self.latched is None:
            self.latched = exc
            self.interruption = {"reason": exc.reason, "message": str(exc), "file": where.get("file"),
                                 "line": where.get("line"), "program_line": where.get("program_line"),
                                 "native_steps": self.rt.backend.budget().ticks_used, "detail": _compact(exc.detail)}

    def _touch(self, result: dict) -> None:
        track = result.get("track") if isinstance(result, dict) else None
        if track and track not in self.touched:
            self.touched.append(track)

    def _note_episode_end(self) -> None:
        if self.rt.executor.episode_over and self.latched is None:
            self.stop(EpisodeEnded(f"episode is over ({self.rt.executor.over_reason})"))

    def _robot(self, name: str, fn: Callable) -> Callable:
        def call(*args, **kwargs):
            self.calls[name] += 1
            if self.latched is not None:
                raise type(self.latched)(f"program already stopped ({self.latched.reason}): {self.latched}",
                                         self.latched.detail)
            try:
                if self.sandbox is not None:
                    self.sandbox.check()
                return fn(*args, **kwargs)
            except ProgramTimeout as exc:
                raise self.stop(exc) from None
            except EpisodeOver as exc:
                raise self.stop(EpisodeEnded(str(exc))) from None

        call.__name__ = name
        call.__doc__ = fn.__doc__
        return call

    def _counted(self, name: str, fn: Callable) -> Callable:
        def call(*args, **kwargs):
            self.calls[name] += 1
            return fn(*args, **kwargs)

        call.__name__ = name
        call.__doc__ = fn.__doc__
        return call

    def namespace(self) -> dict:
        robot = ("get_observation", "step", "reset", "move_to", "set_gripper", "check_reach", "checkpoint",
                 "subgoal_done", "locate", "inspect", "observe", "grasp_candidates", "run_skill", "wait", "both",
                 "cancel_track")
        plain = ("get_instruction", "budget_panel", "status", "arm_status", "tracks")
        helpers = ("gripper_rotation", "wrist_yaw", "rotation_quaternion")
        ns = {name: self._robot(name, getattr(self, name)) for name in robot}
        ns.update({name: self._counted(name, getattr(self, name)) for name in plain})
        ns.update({name: getattr(self, name) for name in helpers})
        ns.update(self.constants())
        return ns

    def constants(self) -> dict:
        p = self.profile
        return {"ARMS": list(ARMS), "PHASE": self.phase, "GRASP_OFFSET": [float(v) for v in p.grasp_offset],
                "FINGERTIP_BEYOND_GRASP_M": float(p.fingertip_beyond_grasp_m),
                "TABLE_Z": None if p.table_z is None else float(p.table_z),
                "TOOL_AXIS": [float(v) for v in p.tool_axis], "CLOSING_AXIS": [float(v) for v in p.closing_axis],
                "WRIST_CAMERAS": {arm: native_camera_name(p, p.wrist_camera[arm]) for arm in ARMS}}

    # -- observation and raw actions -------------------------------------------------------------

    def get_instruction(self) -> str:
        """The task instruction."""
        fn = getattr(self.rt.backend, "instruction", None)
        return str(fn()) if callable(fn) else str(self.rt.obs.instruction)

    def get_observation(self) -> dict:
        """RoboDojo-native observation: instruction, vision (read-only RGB), state, camera_calibration."""
        obs = self.rt.obs
        state = {}
        for arm in ARMS:
            st = obs.arms[arm]
            opening = st.gripper_command if st.gripper_opening is None else st.gripper_opening
            state[f"{arm}_ee_pose"] = np.asarray(st.ee_pose.array(), dtype=float)
            state[f"{arm}_arm_joint_state"] = np.array(st.joints, dtype=float).reshape(-1)
            state[f"{arm}_ee_joint_state"] = np.array([float(opening)])
        vision, calibration = {}, {}
        for logical, frame in obs.cameras.items():
            native = native_camera_name(self.profile, logical)
            rgb = np.asarray(frame.rgb).view()
            rgb.flags.writeable = False
            vision[native] = {"color": rgb}
            if frame.calibrated:
                calibration[native] = {"intrinsics": np.array(frame.intrinsics, dtype=float),
                                       "extrinsics_world": np.array(frame.extrinsics, dtype=float),
                                       "size": list(frame.size)}
        return {"instruction": self.get_instruction(), "vision": vision, "state": state,
                "camera_calibration": calibration, "tick": int(obs.tick), "obs_id": obs.obs_id}

    def step(self, action: dict, *, stage: Optional[dict] = None):
        """One native action for both arms (ee mode: *_ee_pose + *_ee_joint_state; joint mode: *_arm_joint_state
        + *_ee_joint_state), executed as exactly one tick. Returns (observation, 0.0, terminated, truncated, info)."""
        if not isinstance(action, dict):
            raise TypeError("step(action) needs a native action dict")
        if "high_level" in action:
            return self._high_level(action["high_level"])
        allowed = {f"{arm}_{key}" for arm in ARMS for key in ("ee_pose", "arm_joint_state", "ee_joint_state")}
        unknown = sorted(set(action) - allowed)
        if unknown:
            raise ValueError(f"unknown action keys {unknown}; allowed: {sorted(allowed)}")
        ex = self.rt.executor
        commands = {}
        for arm in ARMS:
            pose, joints, grip = (action.get(f"{arm}_{k}") for k in ("ee_pose", "arm_joint_state", "ee_joint_state"))
            if pose is None and joints is None:
                raise ValueError(f"the action needs {arm}_ee_pose or {arm}_arm_joint_state (both arms every step)")
            ee = None if pose is None else Pose.from_array(np.asarray(pose, dtype=float).reshape(-1))
            q = None
            if joints is not None:
                q = np.asarray(joints, dtype=float).reshape(-1)
                expected = np.asarray(self.rt.obs.arms[arm].joints).reshape(-1).shape
                if q.shape != expected or not np.isfinite(q).all():
                    raise ValueError(f"{arm}_arm_joint_state must be {expected[0]} finite joint angles")
                if ee is None and self.profile.kinematics is None:
                    raise ApiError("joint actions need a kinematics model; this robot has none (send *_ee_pose)")
            g = ex.hold_command(arm).gripper if grip is None else float(np.asarray(grip, dtype=float).reshape(-1)[0])
            commands[arm] = ArmCommand(ee, g, joints=q)
        busy = ex.busy_arms()
        if busy:
            detail = ", ".join(f"{arm} arm runs track {t.id} ({t.label})" for arm, t in busy.items())
            raise ApiError(f"step needs both arms idle: {detail}; wait(tracks=[...]) or cancel_track first")
        stage = dict(stage or {})
        label = stage.get("label")
        self._reserve(stage.get("reserve_native"), "step")
        before = self.rt.backend.budget().ticks_used

        def one_tick(ctx):
            yield commands
            return {"status": "done"}

        track = ex.submit(ARMS, one_tick, label=f"step {label}" if label else "step", owner=self.rt.track_owner)
        ex.run([track])
        after = self.rt.backend.budget().ticks_used
        reason = ex.over_reason if ex.episode_over else None
        truncated = reason in ("action_limit", "truncated", "time_limit")
        terminated = bool(ex.episode_over and not truncated)
        info = {"execution_receipt": {"native_steps_before": before, "native_steps_after": after,
                                      "native_steps_delta": after - before, "track": track.id,
                                      "status": track.status.value, **({"episode_over": reason} if reason else {})}}
        if stage:
            info["stage_receipt"] = self._stage_receipt(stage)
        self._account(label, info.get("stage_receipt", {}).get("status") == "tracking_reached", after - before)
        self._note_episode_end()
        return self.get_observation(), 0.0, terminated, truncated, info

    def _stage_receipt(self, stage: dict) -> dict:
        out = {"label": stage.get("label"), "status": "tracking"}
        goal, arm = stage.get("goal_pose"), stage.get("arm")
        if goal is not None and arm in ARMS:
            target = Pose.from_array(np.asarray(goal, dtype=float).reshape(-1))
            dist, angle = self.rt.obs.arms[arm].ee_pose.errors_to(target)
            ok = dist <= float(stage.get("position_tolerance", 0.003)) and \
                angle <= float(stage.get("orientation_tolerance", 0.08))
            out.update(status="tracking_reached" if ok else "tracking", position_error_m=round(dist, 5),
                       orientation_error_rad=round(angle, 5))
        return out

    def _high_level(self, request: dict):
        """H5's step({'high_level': {...}}) envelope, for frozen code that wraps the tools itself."""
        request = dict(request)
        op = request.pop("operation", None)
        ops = {"move_to": self.move_to, "set_gripper": self.set_gripper, "check_reach": self.check_reach,
               "checkpoint": self.checkpoint, "subgoal_done": self.subgoal_done}
        if op not in ops:
            raise ValueError(f"high_level operation must be one of {sorted(ops)}")
        result = ops[op](**request)
        terminated = bool(self.rt.executor.episode_over)
        return self.get_observation(), 0.0, terminated, False, {"high_level": result}

    def reset(self, force=False) -> dict:
        """Practice only: restart this scene from its initial state; returns the first observation. With fewer
        than RESET_MIN_EXECUTIONS counted programs left it is refused unless force=True (the scene cannot be
        completed): late resets threw away held scores that the remaining programs could not earn back."""
        if self.phase != "practice":
            raise PermissionError("reset() is only available in practice; the test episode cannot be reset")
        left = self.executions_left
        if not force and left is not None and left < RESET_MIN_EXECUTIONS:
            raise ApiError(f"reset() refused: only {left} counted programs are left, too few to earn back what a "
                           f"reset throws away; keep improving this attempt, or reset(force=True) if the scene "
                           f"can no longer be completed")
        spent = self.native_used()
        self.rt.reset()
        self.rt.governor.reset()
        self.rt.checkpoints.begin()
        self.pending.clear()
        self.native_spent = spent
        self.segment_start = self.rt.backend.budget().ticks_used
        self.resets += 1
        return self.get_observation()

    # -- budget helpers ---------------------------------------------------------------------------

    def _reserve(self, reserve_native, what: str) -> Optional[int]:
        """Native actions this stage may still use before `reserve_native` is reached (None: unlimited)."""
        if reserve_native is None:
            return None
        reserve = _steps(reserve_native, "reserve_native", 0, 100_000)
        remaining = self.rt.backend.budget().remaining
        if remaining is None:
            return None
        if remaining <= reserve:
            raise self.stop(OverBudgetRefused(
                f"{what}: native_steps_remaining {remaining} <= reserve_native {reserve}; the stage did not start",
                {"native_steps_remaining": remaining, "reserve_native": reserve}))
        return remaining - reserve

    def _account(self, label: Optional[str], reached: bool, native: int) -> None:
        if not label:
            return
        row = self.labeled_stages.setdefault(str(label), {"stages": 0, "reached": 0, "native_steps": 0})
        row["stages"] += 1
        row["reached"] += int(bool(reached))
        row["native_steps"] += int(native)

    # -- motion -----------------------------------------------------------------------------------

    def _orientation(self, arm: str, quaternion, yaw, pitch, roll) -> tuple[dict, np.ndarray]:
        angles = (yaw, pitch, roll)
        if quaternion is not None:
            if any(a is not None for a in angles):
                raise ValueError("give a quaternion or yaw/pitch/roll, not both")
            q = quat_normalize(np.asarray(quaternion, dtype=float).reshape(-1))
            return {"quat": [float(v) for v in q]}, quat_to_matrix(q)
        if all(a is None for a in angles):
            return {}, self.rt.obs.arms[arm].ee_pose.rotation
        y = 0.0 if yaw is None else _finite(yaw, "yaw")
        p = 90.0 if pitch is None else _finite(pitch, "pitch")
        r = 0.0 if roll is None else _finite(roll, "roll")
        return {"yaw": y, "pitch": p, "roll": r}, self.profile.orientation(y, p, r)

    def _frame_point(self, arm: str, frame: str, offset: Optional[np.ndarray]) -> np.ndarray:
        flange = self.rt.obs.arms[arm].ee_pose
        if frame == "grasp":
            return self.rt.grasp_point(arm)
        if frame == "tcp" and offset is not None:
            return flange.transform(offset)
        if frame == "tip" and arm in self.rt.tip_offsets:
            return flange.transform(self.rt.tip_offsets[arm]["tip_in_flange"])
        return flange.position

    def move_to(self, arm, xyz, quaternion=None, *, yaw=None, pitch=None, roll=None, grip=None, max_steps=None,
                position_tolerance=None, angle_tolerance=None, translation_step=None, rotation_step=None,
                frame="ee", tcp_offset_local=None, calibration_ref=None, label=None, reserve_native=None,
                precision=False, stop_on_contact=False, on_stall="interrupt", path=None, final=False,
                wait=True, queue=False):
        """Move the EE/flange (frame 'ee'), the grasp point ('grasp'), a calibrated TCP ('tcp') or a measured
        held tip ('tip') to xyz with a wxyz quaternion or yaw/pitch/roll degrees (pitch defaults to 90)."""
        if arm not in ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        for name, value in (("precision", precision), ("stop_on_contact", stop_on_contact), ("final", final),
                            ("wait", wait), ("queue", queue)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be True or False")
        if on_stall not in ("interrupt", "return"):
            raise ValueError("on_stall must be 'interrupt' or 'return'")
        if path is not None and path not in ("auto", "line", "joint", "ee"):
            raise ValueError("path must be auto, line, joint or ee")
        if frame not in FRAMES:
            raise ValueError("frame must be 'ee', 'grasp', 'tcp' (with tcp_offset_local and calibration_ref) or 'tip'")
        max_steps = _steps(max_steps, "max_steps")
        if grip is not None:
            grip = _finite(grip, "grip")
            if not 0.0 <= grip <= 1.0:
                raise ValueError("grip must be in [0, 1] (0 closed, 1 open)")
        if isinstance(xyz, str):
            ref = self.rt.ctx.references.get(xyz)
            if ref.kind == "grasp" and quaternion is None and all(a is None for a in (yaw, pitch, roll)):
                yaw, pitch = ref.values["yaw_deg"], ref.values["pitch_deg"]
        orient, rotation = self._orientation(arm, quaternion, yaw, pitch, roll)
        point = self.rt.ctx.references.resolve_point(xyz) if isinstance(xyz, str) else as_vector(xyz, 3, "xyz")
        offset = None
        target: Any = xyz if isinstance(xyz, str) else [float(v) for v in point]
        if frame == "tcp":
            if not calibration_ref or tcp_offset_local is None:
                raise ValueError("frame='tcp' needs tcp_offset_local=[x, y, z] and a calibration_ref")
            offset = as_vector(tcp_offset_local, 3, "tcp_offset_local")
            target = [float(v) for v in point - rotation @ offset]
        elif tcp_offset_local is not None:
            raise ValueError("tcp_offset_local is only used with frame='tcp'")
        overrides = {}
        if translation_step is not None:
            overrides["translation_step_m"] = _finite(translation_step, "translation_step")
        if rotation_step is not None:
            overrides["rotation_step_rad"] = _finite(rotation_step, "rotation_step")
        if position_tolerance is not None:
            tol = _finite(position_tolerance, "position_tolerance")
            overrides.update(position_tol_m=tol, precision_tol_m=min(tol, self.rt.limits.precision_tol_m))
        if angle_tolerance is not None:
            tol = _finite(angle_tolerance, "angle_tolerance")
            overrides.update(rotation_tol_rad=tol, precision_tol_rad=min(tol, self.rt.limits.precision_tol_rad))
        if any(v <= 0 for v in overrides.values()):
            raise ValueError("motion steps and tolerances must be positive")
        allowance = self._reserve(reserve_native, f"move_to {arm}")
        max_ticks = max_steps
        capped = allowance is not None and (max_ticks is None or allowance < max_ticks)
        if capped:
            max_ticks = allowance
        limits = dataclasses.replace(self.rt.limits, **overrides) if overrides else None
        result = self.rt.move_to(arm, target, frame=FRAMES[frame], precision=precision,
                                 stop_on_contact=stop_on_contact, gripper=grip, wait=wait, queue=queue,
                                 max_ticks=max_ticks, path=path, final=final, limits=limits, **orient).data
        context = {"kind": "move", "arm": arm, "frame": frame, "point": point, "rotation": rotation,
                   "offset": offset, "on_stall": on_stall, "label": label, "capped": capped,
                   "calibration_ref": calibration_ref}
        if result.get("status") in ("running", "queued") and result.get("track"):
            self.pending[result["track"]] = context
            self._touch(result)
            return {"status": result["status"], "track": result["track"], "arm": arm,
                    "note": "running; call wait([track]) (or wait()) for its result"}
        out, stop = self._finish_move(result, context)
        if stop is not None:
            raise self.stop(stop)
        return out

    def _finish_move(self, result: dict, ctx: dict) -> tuple[dict, Optional[ProgramStop]]:
        arm = ctx["arm"]
        raw = result.get("status")
        status = STATUS_NAMES.get(raw, raw)
        out = dict(result)
        out["status"] = status
        quat = [round(float(v), 5) for v in matrix_to_quat(ctx["rotation"])]
        reachable = False if status == "model_unreachable" else ("unknown" if "reach_warning" in result else True)
        out.update(arm=arm, frame=ctx["frame"], reachable=reachable,
                   requested_pose=_r(list(ctx["point"]) + quat, 5))
        if ctx.get("calibration_ref"):
            out["calibration_ref"] = ctx["calibration_ref"]
        if ctx.get("label"):
            out["label"] = ctx["label"]
        calls = int(result.get("ticks") or 0)
        out["calls"] = calls
        if status in MOVED:
            actual = self._frame_point(arm, ctx["frame"], ctx.get("offset"))
            now = self.rt.obs.arms[arm].ee_pose
            error = float(np.linalg.norm(actual - np.asarray(ctx["point"], dtype=float)))
            out.update(actual_xyz=_r(actual.tolist()), position_error_m=round(error, 5),
                       position_error_mm=round(error * 1000, 2),
                       orientation_error_rad=round(quat_angle(now.quat, matrix_to_quat(ctx["rotation"])), 4))
        self._touch(out)
        self._account(ctx.get("label"), status == "tracking_reached", calls)
        self._note_episode_end()
        stop = None
        if ctx["on_stall"] == "interrupt":
            if status == "timeout" and ctx.get("capped"):
                stop = OverBudgetRefused(f"move_to {arm} stopped at reserve_native", out)
            elif status == "dropped":
                stop = MotionInterrupted(f"move_to {arm} dropped: the held part slipped out "
                                         f"{out.get('position_error_mm', '?')} mm from the goal; look where it fell, "
                                         f"locate it again and regrasp (on_stall='return' handles this in code)", out)
            elif status in ("stalled", "timeout", "error"):
                stop = MotionInterrupted(f"move_to {arm} {status} {out.get('position_error_mm', '?')} mm from the "
                                         f"goal; pass on_stall='return' to handle stalls in code", out)
            elif status == "over_budget":
                stop = OverBudgetRefused(f"move_to {arm} refused: not enough native actions for the move and the "
                                         f"way back (final=True for the last scoring move)", out)
        return out, stop

    def set_gripper(self, arm, command, *, settle_steps=None, until_still=True, label=None, reserve_native=None,
                    prepare_lift_ref=None, verify_lift_ref=None, wait=True):
        """Command the gripper (0 closed .. 1 open, or 'open'/'close') until the fingers stop; returns the
        measured opening as `width` and, after a close, `holding` ('yes' / 'no')."""
        if arm not in ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        if isinstance(command, str):
            if command not in ("open", "close"):
                raise ValueError("command must be 'open', 'close' or a number in [0, 1]")
            value = 1.0 if command == "open" else 0.0
        else:
            value = _finite(command, "command")
            if not 0.0 <= value <= 1.0:
                raise ValueError("gripper command must be in [0, 1] (0 closed, 1 open)")
        settle_steps = _steps(settle_steps, "settle_steps", 1, 100)
        if not isinstance(until_still, bool) or not isinstance(wait, bool):
            raise ValueError("until_still and wait must be True or False")
        self._reserve(reserve_native, f"set_gripper {arm}")
        result = self.rt.gripper(arm, value, settle_ticks=settle_steps, until_still=until_still, wait=wait).data
        context = {"kind": "gripper", "arm": arm, "label": label,
                   "refs": {"prepare_lift": prepare_lift_ref, "verify_lift": verify_lift_ref}}
        if result.get("status") in ("running", "queued") and result.get("track"):
            self.pending[result["track"]] = context
            self._touch(result)
            return {"status": result["status"], "track": result["track"], "arm": arm}
        out, _ = self._finish_grip(result, context)
        return out

    def _finish_grip(self, result: dict, ctx: dict) -> tuple[dict, None]:
        raw = result.get("status")
        opening = result.get("opening")
        out = {"status": "command_sent" if raw == "done" else raw, "arm": ctx["arm"],
               "commanded": result.get("commanded"), "width": opening, "holding": result.get("holding", "unknown"),
               "attachment": "unknown", "calls": int(result.get("ticks") or 0), "track": result.get("track"),
               "evidence": {"status": "pending", "provided_references": ctx.get("refs"),
                            "interpretation": "holding comes from the measured finger opening, not from images"}}
        if ctx.get("label"):
            out["label"] = ctx["label"]
        if result.get("references_flagged"):
            out["references_flagged"] = result["references_flagged"]
        self._touch(out)
        self._account(ctx.get("label"), raw == "done", out["calls"])
        self._note_episode_end()
        return out, None

    def check_reach(self, xyz, quaternion=None, *, yaw=None, pitch=None, roll=None, frame="grasp",
                    arms=("left", "right")):
        """No motion: model reachability per arm, with reachable variants (lower pitch, higher point)."""
        if frame not in ("ee", "flange", "grasp"):
            raise ValueError("check_reach supports frame 'ee' or 'grasp'")
        if isinstance(arms, str):
            arms = [arms]
        point = as_vector(xyz, 3, "xyz")
        out = {}
        for arm in arms:
            if arm not in ARMS:
                raise ValueError("arms must list 'left' and/or 'right'")
            orient, _ = self._orientation(arm, quaternion, yaw, pitch, roll)
            data = self.rt.reach_query(arm, point.tolist(), frame=FRAMES[frame], **orient).data["arms"][arm]
            row = {"reachable": data["reachable"], "variants": [], "reason": data.get("reason")}
            if "joint_travel_deg" in data:
                row["joint_travel_deg"] = data["joint_travel_deg"]
            for variant in data.get("reachable_variants") or []:
                if "pitch" in variant:
                    row["variants"].append({"pitch": variant["pitch"], "yaw": orient.get("yaw"),
                                            "roll": orient.get("roll")})
                elif "higher_m" in variant:
                    dz = float(variant["higher_m"])
                    row["variants"].append({"xyz": _r((point + [0.0, 0.0, dz]).tolist()),
                                            "note": f"{dz * 100:.0f} cm higher"})
            out[arm] = row
        return out

    # -- guards and budget ------------------------------------------------------------------------

    def checkpoint(self, name, kind, **params):
        """Guard evaluated mid-program (no native action): mark, native_reserve(min_remaining), pose(arm, xyz,
        tolerance[, frame]), lift_retained(arm, before, ...), or a condition the program computed
        (checkpoint(name, ok, **evidence) with ok a bool). A failure stops the rest of the program."""
        line = self.sandbox.program_line() if self.sandbox else None
        if isinstance(kind, (bool, np.bool_)):
            # checkpoint(name, <bool>, evidence=...) was written in two tasks' skills and raised every time
            entry = {"name": str(name), "kind": "condition", "passed": bool(kind),
                     "native_steps": self.rt.backend.budget().ticks_used,
                     "detail": {k: _compact(v) for k, v in params.items()}}
            self.rt.checkpoints.log.append(entry)
        elif kind == "lift_retained":
            entry = self._lift_retained(name, **params)
        else:
            if kind == "pose" and params.get("frame") == "ee":
                params["frame"] = "flange"
            try:
                entry = self.rt.checkpoints.run(name, kind, **params)
            except RuntimeCheckpointFailed as exc:
                entry = self.rt.checkpoints.log[-1]
                entry["program_line"] = line
                raise self.stop(CheckpointFailed(str(exc), copy.deepcopy(entry))) from None
        entry["program_line"] = line
        if not entry["passed"]:
            raise self.stop(CheckpointFailed(f"checkpoint_failed: {name} ({entry['kind']}) {entry['detail']}",
                                             copy.deepcopy(entry)))
        return copy.deepcopy(entry)

    def _lift_retained(self, name, *, arm=None, before=None, source=None, support_z=None, allow_unknown=False):
        """Something stayed between the fingers while the gripper moved since mark `before` (finger opening)."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
            raise ValueError("checkpoint name must be 1..40 characters of [A-Za-z0-9_-]")
        cps = self.rt.checkpoints
        entry = {"name": name, "kind": "lift_retained", "native_steps": self.rt.backend.budget().ticks_used}
        try:
            if arm not in ARMS:
                raise ValueError("arm must be left or right")
            mark = cps.marks.get(before)
            if mark is None:
                raise ValueError(f"lift_retained needs a mark {before!r} taken earlier in this program")
            for key, value in (("support_z", support_z),):
                if value is not None:
                    _finite(value, key)
            state = self.rt.obs.arms[arm]
            moved, turned = mark["poses"][arm].errors_to(state.ee_pose)
            opening = state.gripper_opening
            if opening is None:
                evidence = "unknown"
            elif state.gripper_command >= 0.5 or opening <= GRIP_EMPTY_CLOSED + GRIP_HOLD_MARGIN:
                evidence = "source_material_remains"
            elif moved < LIFT_MIN_M and turned < LIFT_MIN_RAD:
                evidence = "unknown"
            else:
                evidence = "motion_coupled"
            passed = evidence == "motion_coupled" or (bool(allow_unknown) and evidence == "unknown")
            detail = {"evidence": evidence, "ee_translation_m": round(moved, 4), "ee_rotation_rad": round(turned, 4),
                      "finger_opening": None if opening is None else round(float(opening), 4),
                      "basis": "measured finger opening while the gripper moved (no image test)"}
        except (ValueError, KeyError, TypeError) as exc:
            passed, detail = False, {"error": str(exc)}
        entry.update(passed=bool(passed), detail=detail)
        cps.log.append(entry)
        return entry

    def subgoal_done(self, name):
        """Mark a finished sub-goal; the budget panel then reports native actions per sub-goal."""
        return self.rt.subgoal_done(name).data

    def budget_panel(self) -> dict:
        """What the remaining native actions can still buy (way back to origin and reserve included)."""
        panel = self.rt.budget_panel()
        if self.labeled_stages:
            panel["labeled_stages"] = copy.deepcopy(self.labeled_stages)
        return panel

    def status(self) -> dict:
        """Budget panel, arm states and the track lanes."""
        return {"phase": self.phase, "tick": int(self.rt.tick), "budget_panel": self.budget_panel(),
                "arms": self.rt.arm_status(), "tracks": self.rt.executor.lanes()}

    def arm_status(self) -> dict:
        return self.rt.arm_status()

    # -- KnowinAgent perception and skills ---------------------------------------------------------------

    def locate(self, text, camera="head", pixel=None, max_objects=1, label=None):
        """Ground objects by text; returns H5-style xyz (top-face centre) and object_pose per object."""
        try:
            data = self.rt.locate(text, camera=camera, hint_px=pixel, max_objects=max_objects, label=label).data
        except ApiError as exc:
            if not str(exc).startswith("locate failed"):
                raise
            return {"status": "not_found", "reason": str(exc), "objects": [], "ref": None, "xyz": None,
                    "object_pose": None}
        objects = []
        for obj in data["objects"]:
            heading = obj.get("grasp_yaw_deg")
            if heading is None:
                heading = float(obj.get("long_axis_deg") or 0.0) + 90.0
            center = [float(v) for v in obj["center_xyz"]]
            pose = {k: obj[k] for k in ("center_xyz", "top_z", "height_m", "length_m", "width_m", "long_axis_deg")
                    if k in obj}
            pose.update(grasp_yaw=round(wrap180(self.profile.yaw_for_heading(heading, 90.0)), 1),
                        closing_heading_deg=round(wrap180(heading), 1))
            objects.append({"ref": obj["ref"], "xyz": _r([center[0], center[1], float(obj["top_z"])]),
                            "object_pose": pose, "score": obj.get("score"), "warnings": obj.get("warnings", [])})
        first = objects[0] if objects else {}
        return {"status": "located" if objects else "not_found", "camera": data.get("camera"),
                "obs_id": data.get("obs_id"), "objects": objects, "ref": first.get("ref"), "xyz": first.get("xyz"),
                "object_pose": first.get("object_pose"), "warnings": first.get("warnings", [])}

    def inspect(self, mode, target=None, camera="head", arm=None, radius_m=0.08):
        """shape | axis_tip (pass arm when held) | support_plane | opening. A failed measurement returns
        status 'not_found' instead of stopping the program (as locate does)."""
        try:
            data = self.rt.inspect(mode, target, camera=camera, arm=arm, radius_m=radius_m).data
        except ApiError as exc:
            if not str(exc).startswith(f"inspect({mode}) failed"):
                raise
            return {"status": "not_found", "mode": mode, "camera": camera, "reason": str(exc), "ref": None}
        return {"status": "measured", **data}

    def observe(self, arm, target, view="down", distance_m=0.22):
        """Move this arm's wrist camera to view target from a direction at a distance (KnowinAgent active view)."""
        outcome = self.rt.observe(arm, target, view=view, distance_m=distance_m)
        self.images.extend(outcome.images)
        out = dict(outcome.data)
        out["status"] = STATUS_NAMES.get(out.get("status"), out.get("status"))
        self._touch(out)
        self._note_episode_end()
        return out

    def grasp_candidates(self, target, arm="auto", pitch=90.0, max_candidates=4):
        """Ranked top-grasp poses for an object reference; each yaw_deg is a move_to yaw."""
        return self.rt.grasp_candidates(target, arm=arm, pitch=pitch, max_candidates=max_candidates).data

    def run_skill(self, skill_id, **bindings):
        """Run a library skill to its endpoint as one track; returns its execution receipt."""
        result = self.library.run(self.rt, skill_id, bindings, owner=self.rt.track_owner)
        self._touch(result)
        self._note_episode_end()
        return result

    # -- asynchronous tracks ------------------------------------------------------------------------

    def _settle(self, track_id: str) -> tuple[dict, Optional[ProgramStop]]:
        track = self.rt.executor.track(track_id)
        if not track.finished:
            return track.summary(), None
        result = dict(track.result or {})
        result.setdefault("status", track.status.value)
        result["track"] = track_id
        ctx = self.pending.pop(track_id, None)
        if ctx is None:
            return result, None
        if ctx["kind"] == "gripper":
            return self._finish_grip(result, ctx)
        return self._finish_move(result, ctx)

    def wait(self, tracks=None, ticks=None, seconds=None):
        """Wait for tracks (default: every asynchronous move this program started) or a number of ticks;
        results carry the same fields as the synchronous calls."""
        if isinstance(tracks, str):
            tracks = [tracks]
        if tracks is None and ticks is None and seconds is None:
            tracks = [tid for tid in self.pending if not self.rt.executor.track(tid).finished]
            if not tracks:
                return {"waited_ticks": 0, "tracks": []}
        before = self.rt.tick
        if tracks:
            self.rt.wait(tracks=list(tracks), max_ticks=WAIT_MAX_TICKS)
            ids = list(tracks)
        else:
            self.rt.wait(ticks=ticks, seconds=seconds, max_ticks=WAIT_MAX_TICKS)
            ids = [tid for tid in list(self.pending) if self.rt.executor.track(tid).finished]
        rows, stops = [], []
        for tid in ids:
            row, stop = self._settle(tid)
            rows.append(row)
            if stop is not None:
                stops.append(stop)
        self._note_episode_end()
        if stops:
            raise self.stop(stops[0])
        return {"waited_ticks": int(self.rt.tick - before), "tracks": rows}

    def both(self, left=None, right=None):
        """Run one command per arm concurrently and wait for both: each is a dict of move_to keyword arguments
        (with xyz=...) or set_gripper arguments (with command=...)."""
        if left is None and right is None:
            raise ValueError("both() needs left=dict(...) and/or right=dict(...)")
        started = {}
        for arm, spec in (("left", left), ("right", right)):
            if spec is None:
                continue
            if not isinstance(spec, dict):
                raise TypeError(f"{arm} must be a dict of move_to or set_gripper keyword arguments")
            spec = {k: v for k, v in spec.items() if k != "wait"}
            if "xyz" in spec:
                self.calls["move_to"] += 1
                started[arm] = self.move_to(arm, wait=False, **spec)
            elif "command" in spec:
                self.calls["set_gripper"] += 1
                started[arm] = self.set_gripper(arm, wait=False, **spec)
            else:
                raise ValueError(f"{arm}: give move_to arguments (xyz=...) or set_gripper arguments (command=...)")
        before = self.rt.tick
        ids = [r["track"] for r in started.values() if r.get("status") in ("running", "queued") and r.get("track")]
        if ids:
            self.rt.wait(tracks=ids, max_ticks=WAIT_MAX_TICKS)
        out, stops = {}, []
        for arm, result in started.items():
            if result.get("track") in ids:
                result, stop = self._settle(result["track"])
                if stop is not None:
                    stops.append(stop)
            out[arm] = result
        out["ticks"] = int(self.rt.tick - before)
        self._note_episode_end()
        if stops:
            raise self.stop(stops[0])
        return out

    def tracks(self) -> dict:
        """Running and queued tracks per arm lane, plus recent finishes."""
        return self.rt.executor.lanes()

    def cancel_track(self, track, reason="cancelled by program"):
        self.pending.pop(track, None)
        return self.rt.executor.cancel(track, reason=reason).summary()

    def settle(self, stopped: bool) -> list[dict]:
        """At program end: wait for this program's asynchronous tracks (completed) or cancel them (stopped)."""
        rows = []
        ids = [tid for tid in self.pending if not self.rt.executor.track(tid).finished]
        if ids and not stopped and not self.rt.executor.episode_over:
            self.rt.executor.run([self.rt.executor.track(t) for t in ids], max_ticks=WAIT_MAX_TICKS)
        for tid in ids:
            track = self.rt.executor.track(tid)
            if not track.finished:
                self.rt.executor.cancel(tid, reason="program ended" if not stopped else "program stopped")
            row, _ = self._settle(tid)
            rows.append(row)
        return rows

    # -- geometry helpers (embodiment conventions) ---------------------------------------------------

    def gripper_rotation(self, yaw, pitch=90.0, roll=0.0) -> np.ndarray:
        """Flange rotation of move_to(yaw=, pitch=, roll=) on this robot."""
        return self.profile.orientation(_finite(yaw, "yaw"), _finite(pitch, "pitch"), _finite(roll, "roll"))

    def wrist_yaw(self, rotation, pitch=90.0) -> float:
        """The move_to yaw that reproduces this rotation's finger closing line."""
        return float(self.profile.yaw_of(np.asarray(rotation, dtype=float).reshape(3, 3), float(pitch)))

    def rotation_quaternion(self, rotation) -> list:
        """wxyz quaternion of a 3x3 rotation matrix."""
        return [float(v) for v in matrix_to_quat(np.asarray(rotation, dtype=float).reshape(3, 3))]


def _compact(detail: Any, depth: int = 0) -> Any:
    if isinstance(detail, dict):
        if depth >= 2:
            return detail.get("status", "...") if "status" in detail else "..."
        return {str(k): _compact(v, depth + 1) for k, v in detail.items()}
    if isinstance(detail, (list, tuple)):
        return [_compact(v, depth + 1) for v in detail[:12]]
    if isinstance(detail, np.ndarray):
        return _r(detail.tolist())
    if isinstance(detail, (np.floating, np.integer)):
        return detail.item()
    return detail
