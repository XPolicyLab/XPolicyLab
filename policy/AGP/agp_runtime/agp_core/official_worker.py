"""The migration worker on top of agp_core.official_env.OfficialEnv (official TASK_ENV interface only).

migration/worker.py (shared, unchanged) reads the benchmark's robot manager, camera prims, simulator clock, end flag
and measured finger joints. OfficialBaseWorker keeps all of its motion code (preparation, preflight, sampling,
settling, diagnostics, the bridge protocol) and replaces only what needs simulator internals:

    __init__                 arm bases, TCP, initial joints and finger limits from public config and the first
                             official observation (no articulation, no link-pose query); the instruction from
                             the observation; robot_frame_validation.json from the public URDF
    _transit_joint_velocity  open-gripper transit rate when the last gripper command is open (the measured
                             finger opening is not observed)
    _errors (gripper only)   an open is taken as reached (nothing blocks an opening); gripper_measured_m null,
                             gripper_observed false. A close, with the RGB-only observer in the class (OURS_RGB_ONLY):
                             the fingers are estimated from the arm's wrist image in the latest official observation
                             (agp_core/visual_gap.py, public constants only) and judged with the rule of migration/worker.py
                             on measured fingers: every finger within 3 mm of its closed endpoint -> target_reached,
                             motion_outcome reached (closed fully); fingers stopped apart -> held_short (something
                             between the pads); the result carries gripper_estimate_m / _pad_gap_m / _quality / _source.
                             An estimate without quality ok, or a worker without the observer, leaves the close
                             unconfirmed (target_reached false, held_short). The runtime (control_runtime/runtime.py)
                             invalidates measurements and both arms' target confirmation and lift state after a
                             held_short or not-reached outcome, not after every close.
    execute                  `estimate` is refused (it needs measured fingers; the MCP server does not offer it)
    serve                    the bridge loop, ending on the policy's finish file, agent exit, episode end plus a
                             grace period (at once with OURS_HIDE_END), or the wall-clock limit (see
                             agp_core/official_deploy.py)

build_worker() composes it with the policy-side PolicyPlanner (planner_injection) and the RGB-only observer
(rgb_observe), in the same order as agp_core/sim_main.py: RgbOnlyMixin -> PolicyPlannerMixin -> OfficialBaseWorker ->
Worker. agp_core/approx_home.py's ApproxHomeMixin goes in front (OURS_RGB_ONLY, which official conditions require): a home
that completes the return of both arms ends inside the official return tolerance instead of at the exact start joints.
agp_core/speed_profiles.py's SpeedProfilesMixin goes in front of that when any of OURS_FREE_TOPT, OURS_ARM_GUARD,
OURS_HOLD_GUARD, OURS_GRIP_OBS, OURS_PREC_ZONE or OURS_HOLD_PROFILE is on (all off: the class is unchanged and the
module is not imported); also when one of OURS_GUARD_V2, OURS_GF_OPEN, OURS_HOLD_CLEARANCE, OURS_DROP_STOP or
OURS_HOLD_WATCH2 acts (speed_profiles.effective_r2a: each with the switches it acts with), and when one of these is on,
each acting by itself: OURS_SPIN_FAST (the in-place twist), OURS_YAW_FLIP (the symmetric-gripper flip),
OURS_PRECISION_2X (the precision-segment limits scaled by speed_profiles.PRECISION_2X_SCALE) or OURS_PLAN_GUARD (a
plan-mode move whose TCP path would swing far beyond its straight distance is refused as "Planning rejected").
agp_core/grip_width.py's GripWidthWorkerMixin goes in front of that when OURS_GRIP_WIDTH is on (the width action; off: the
class is unchanged and the module is not imported).
agp_core/handover.py's HandoverMixin goes in front of that when OURS_HANDOVER and OURS_ARM_GUARD are both on (the declared
handover), then agp_core/return_reserve.py's ReturnReserveMixin in front of everything when OURS_RETURN_RESERVE is on (the
return reserve); with both off the class is unchanged and neither module is imported.
HideEndMixin goes in front of everything when OURS_HIDE_END is on (no reply carries episode_ended, nothing is answered
after the official end); off: the class is unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from migration import worker as base
from migration.kinematics import x5_tcp_transform


class OfficialBaseWorker(base.Worker):
    """migration.worker.Worker without simulator internals (see the module docstring)."""

    def __init__(self, env, session):
        self.env = env
        self.session = Path(session)
        self.bridge = self.session / "bridge"
        self.bridge.mkdir(exist_ok=True)
        self.rm = env.robot_manager
        self.arms, self.grips = {}, {}
        state = env.get_obs()["state"]
        for arm in ("left", "right"):
            robot = next(r for r in self.rm.robot_list if r.type == "target" and r.arm_name.startswith(arm))
            pose = base.transform(robot.entity_origin_pose)
            self.arms[arm] = {"robot": robot, "base": pose, "invbase": np.linalg.inv(pose),
                              "initial": self.rm.get_joint(robot)[0].copy(),
                              "tcp": x5_tcp_transform(float(robot.gripper_bias)),
                              "gripper_limits_m": np.asarray(robot.gripper_limits_m, dtype=float).copy()}
            # The worker only ever commands 1 (open) or 0 (closed); the episode starts open.
            command = state.get(f"{arm}_ee_joint_state")
            opened = command is None or float(np.asarray(command, float).ravel()[0]) >= 0.5
            self.grips[robot.arm_name] = 1.0 if opened else 0.0
        self.started = time.monotonic()
        self.tools = 0
        self.motion_sequence = 0
        self.dt = 1 / env.obs_manager.collect_freq
        goal = self.session / "goal"
        goal.mkdir(exist_ok=True)
        (goal / "INSTRUCTION.md").write_text(str(env.obs_manager.instruction[0]))
        base.atomic(self.session / "robot_frame_validation.json", self._frames_from_urdf())
        base.atomic(self.session / "control_task.json", {"release_requirement": "support"})
        base.atomic(self.session / "worker_ready.json", {
            "arms": list(self.arms), "pid": os.getpid(), "task": env.task, "action_budget": int(env.step_lim),
            "control_dt_s": self.dt, "dual_arm_exercise": False,
            "environment": "official TASK_ENV whitelist (agp_core/official_env.py)"})

    def _frames_from_urdf(self):
        from agp_core import public_camera as pc
        from agp_core.rgb_observe import link_poses_in_base
        model = pc.default_model()
        frames = {"source": "public URDF (fingers at their open limit); the simulator is not queried"}
        for arm, context in self.arms.items():
            upper = float(context["gripper_limits_m"][0][1])
            links = link_poses_in_base(model, np.zeros(6), upper)
            link6 = links[model.ee_link]
            frames[arm] = {"finger_poses_in_link6": {name: base.pose(np.linalg.inv(link6) @ links[name]).tolist()
                                                     for name in ("link7", "link8")},
                           "T_link6_tcp": context["tcp"].tolist()}
        return frames

    def _transit_joint_velocity(self, arm):
        # Calibrated for open-gripper free travel; the finger opening itself is not observed.
        return 2.5 if self.grips[self.arms[arm]["robot"].arm_name] == 1.0 else 1.5

    def _gripper_estimate(self, arm):
        """Visual finger estimate (agp_core/visual_gap.py) from this arm's wrist image in the latest official
        observation (OfficialEnv returns the observation taken after the last control step; no new render)."""
        from agp_core import public_camera as pc
        from agp_core import visual_gap
        try:
            camera = next(name for name, own in pc.default_model().wrist_arm.items() if own == arm)
            color = np.asarray(self.env.get_obs()["vision"][camera]["color"])
            return visual_gap.estimate(color[..., :3].astype(np.uint8))
        except Exception as exc:          # an unusable image leaves the close unconfirmed
            return {"quality": "unavailable", "pad_gap_m": None, "finger_joints_m": None,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _errors(self, arm, plan):
        if plan["command"] != "gripper":
            return super()._errors(arm, plan)
        robot = self.arms[arm]["robot"]
        achieved = self.tcp_base(arm)
        joints = self.rm.get_joint(robot)[0]
        opening = plan["args"]["action"] == "open"
        mapping = robot.gripper_move
        index = (1 if opening else 0) if mapping["sign"] == 1 else (0 if opening else 1)
        target = float(robot.gripper_scale[index])
        desired = np.array([target, target * mapping["mimic"][1] + mapping["mimic"][2]])
        limits = self.arms[arm]["gripper_limits_m"]
        endpoint = np.clip(desired, limits[:, 0], limits[:, 1])
        result = {"target_reached": bool(opening), "position_error_m": None, "rotation_error_rad": None,
                  "joint_error_rad": float(np.max(np.abs(joints - plan["points"][-1]))),
                  "gripper_error_m": None, "gripper_drive_error_m": None, "gripper_endpoint_m": endpoint.tolist(),
                  "gripper_measured_m": None, "gripper_drive_target_m": desired.tolist(),
                  "gripper_observed": False,
                  "gripper_outcome_basis": ("command only: the finger opening is not observed; an open is taken as "
                                            "reached, a close is never confirmed"),
                  "achieved_tcp_pose_base": base.pose(achieved).tolist(), "position_residual_m": None}
        from agp_core import rgb_observe
        if opening or not isinstance(self, rgb_observe.RgbOnlyMixin):
            return result
        # Under OURS_RGB_ONLY: a close is judged from the fingers estimated in the arm's wrist image,
        # with the rule of migration/worker.py on measured fingers (every finger within 3 mm of its closed
        # endpoint -> reached; fingers stopped apart by something between the pads -> held_short). An estimate
        # without quality ok leaves the close unconfirmed as before.
        from agp_core import visual_gap
        estimate = self._gripper_estimate(arm)
        outcome = visual_gap.close_outcome(estimate, endpoint)
        result.update(gripper_estimate_m=estimate.get("finger_joints_m"),
                      gripper_estimate_pad_gap_m=estimate.get("pad_gap_m"),
                      gripper_estimate_quality=estimate.get("quality"),
                      gripper_estimate_source=visual_gap.SOURCE)
        if outcome is not None:
            result.update(target_reached=outcome["reached"], gripper_error_m=outcome["gripper_error_m"],
                          gripper_outcome_basis=(
                              "wrist-image estimate of the fingers (agp_core/visual_gap.py, quality ok): every finger "
                              f"within {visual_gap.CLOSE_TOLERANCE_M * 1000:.0f} mm of its closed endpoint -> "
                              "reached (closed fully); otherwise held_short (the fingers stopped apart)"))
        else:
            result["gripper_outcome_basis"] = (f"wrist-image estimate quality {estimate.get('quality')}: the close "
                                               "is not confirmed (as without an estimate)")
        return result

    def execute(self, command, args):
        if command == "estimate":
            return {"ok": False, "error": "estimate is not available under official conditions",
                    "motion_started": False, "motion_executed": False, "low_level_actions": 0}
        return super().execute(command, args)

    # -- the bridge
    def _bridge_open(self):
        """Whether a new request may be claimed (HideEndMixin closes the bridge at the official end)."""
        return True

    def _deliver(self, response, rid, result):
        """Write the reply the policy side reads (HideEndMixin withholds it after the official end)."""
        base.atomic(response, dict(result, id=rid))

    def _serve_pending(self, processed):
        handled = 0
        for path in sorted(self.bridge.glob("req_*.json")):
            if path.name in processed:
                continue
            if not self._bridge_open():
                break
            req = json.loads(path.read_text())
            rid = req["id"]
            response = self.bridge / f"resp_{rid:06d}.json"
            if response.exists():
                processed.add(path.name)
                continue
            claim = self.bridge / f"started_{rid:06d}.json"
            if claim.exists():
                raise RuntimeError("Unresolved worker execution claim. Replay forbidden.")
            with claim.open("x") as stream:
                json.dump(req, stream)
                stream.flush()
                os.fsync(stream.fileno())
            directory_fd = os.open(self.bridge, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.tools += 1
            count_before = self.env.take_action_cnt[0]
            try:
                result = self.observe() if req["cmd"] == "observe" else self.execute(req["cmd"], req["args"])
            except Exception as exc:
                traceback.print_exc()
                result = {"ok": False, "error": str(exc),
                          "motion_started": self.env.take_action_cnt[0] > count_before,
                          "low_level_actions": self.env.take_action_cnt[0] - count_before}
            self._deliver(response, rid, result)
            processed.add(path.name)
            handled += 1
        return handled

    def serve(self, should_stop=None, poll_s=0.01):
        """Answer bridge requests until should_stop() returns a reason (checked only when nothing is pending)."""
        processed = set()
        while True:
            handled = self._serve_pending(processed)
            if not handled:
                reason = should_stop() if should_stop is not None else (
                    "policy finished" if (self.session / "policy_finished.json").exists() else None)
                if reason:
                    return reason
            time.sleep(poll_s)


END_KEY = "episode_ended"


def without_end_flag(value):
    """A copy of a reply with every episode_ended key removed, at any depth (OURS_HIDE_END)."""
    if isinstance(value, dict):
        return {k: without_end_flag(v) for k, v in value.items() if k != END_KEY}
    if isinstance(value, list):
        return [without_end_flag(v) for v in value]
    return value


class HideEndMixin:
    """Hidden end (OURS_HIDE_END): the official end flag never leaves the adapter. Every reply loses its episode_ended
    keys. Once is_episode_end() is true (success, an official failure or the step limit) the bridge answers nothing
    more: the reply of the request during which the episode ended goes to withheld_<id>.json in the bridge folder (a
    record for us; the policy side reads only resp_<id>.json), later requests stay unclaimed, and
    agp_core/official_deploy.py stops codex at once, as the official deploy loop stops calling its policy. The flag itself
    is still read here and by the worker's own motion code."""

    def _bridge_open(self):
        return not self.env.is_episode_end()

    def _deliver(self, response, rid, result):
        if self.env.is_episode_end():
            base.atomic(response.with_name(f"withheld_{rid:06d}.json"), dict(result, id=rid))
            return
        base.atomic(response, dict(without_end_flag(result), id=rid))


def planner_factory(task):
    """PolicyPlanner for one public arm descriptor: public cuRobo yml, the task's table, the control period."""
    def factory(robot, env):
        from agp_core.policy_planner import PolicyPlanner
        return PolicyPlanner.from_public_config(
            robot.robot_name, list(robot.entity_origin_pose), task,
            active_joints_name=list(robot.arm_joints_name), interpolation_dt=1 / env.obs_manager.collect_freq)
    return factory


def worker_class(environ, factory):
    """OfficialBaseWorker with the policy planner and the RGB-only observer (both switches must be on), and the
    approximate home (OURS_RGB_ONLY); the speed profiles and guards in front when one of their switches is on, then
    the further mixins of the module docstring when theirs are on."""
    from agp_core import approx_home, planner_injection, rgb_observe
    cls = planner_injection.worker_class(OfficialBaseWorker, environ, factory=factory)
    cls = rgb_observe.worker_class(cls, environ)
    if not (issubclass(cls, planner_injection.PolicyPlannerMixin) and issubclass(cls, rgb_observe.RgbOnlyMixin)):
        raise ValueError("official conditions need OURS_POLICY_PLANNER=1 and OURS_RGB_ONLY=1")
    cls = approx_home.worker_class(cls, environ)
    from agp_core import features
    if any(features.enabled(name, environ) for name in ("free_topt", "arm_guard", "hold_guard",
                                                         "grip_obs", "prec_zone", "hold_profile",
                                                         "guard_v2", "gf_open", "hold_clearance", "drop_stop",
                                                         "hold_watch2", "spin_fast", "yaw_flip", "precision_2x",
                                                         "plan_guard", "prec_roll")):
        from agp_core import speed_profiles
        cls = speed_profiles.worker_class(cls, environ)
    if features.enabled("grip_width", environ):                                              # the width action
        from agp_core import grip_width
        cls = grip_width.worker_class(cls, environ)
    if features.enabled("handover", environ) and features.enabled("arm_guard", environ):     # the declared handover
        from agp_core import handover
        cls = handover.worker_class(cls)
    if features.enabled("return_reserve", environ):                                          # the return reserve
        from agp_core import return_reserve
        cls = return_reserve.worker_class(cls)
    if features.enabled("hide_end", environ):                                                # the hidden end
        cls = type(f"HideEnd{cls.__name__}", (HideEndMixin, cls),
                   {"__module__": __name__, "__doc__": f"{cls.__qualname__} with the official end kept in the adapter."})
    return cls


def build_worker(env, session, environ, factory=None):
    cls = worker_class(environ, factory if factory is not None else planner_factory(env.task))
    return cls(env, session)
