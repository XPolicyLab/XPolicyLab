"""Grip width (switch OURS_GRIP_WIDTH, opt-in): a gripper action to a requested pad gap.

The official X5 gripper target is continuous (finger joints joint7 / joint8, 0 .. 0.044 m each; the policy's gripper value
c in [0, 1] is the finger target -0.01 m + c * 0.054 m, Assets/Robots/x5/robot_config.yml gripper_scale [-0.01, 0.044],
src/eval_client/eval_env.py; joint8 follows the public mimic q8 = 1.0 * q7 + 0). `open` sends c = 1 (the fingers at the
open limit, the pads 88.8 mm apart) and `close` c = 0 (the target 10 mm past closed: the drive squeezes until the
fingers stop on an object or close fully, 0.8 mm). The width action sends the c whose finger target gives the requested
pad gap (pad gap = q7 + q8 + 0.807 mm, agp_core/visual_gap.py): the fingers move to that gap and stop there, or earlier on
an object, where the drive presses only by the remaining distance to the target (a close presses by that distance plus
10 mm: a close is the firm grasp). Requested gaps WIDTH_MIN_MM .. WIDTH_MAX_MM (closed .. fully open, the public finger
range; the finger target is clipped to the joint range).

Ground truth (an offline simulator probe with partial commands 0.2 .. 0.85 that read the measured joints, which the
policy never reads): the fingers come to rest at the commanded target within 0.003 mm and the official reading
<arm>_ee_joint_state equals c. The official reading is the target clamped to the measured finger
+- 10.8 mm (env/robot_manager/control_manager.py process_gripper_val): it equals c while the fingers are within 10.8 mm
of the target and shows the finger only beyond that. So the width outcome is judged from the wrist-image estimate
(agp_core/visual_gap.py, quality ok) first, then from the reading's clamp:

    stopped_on_object  the estimated pad gap is at least the target gap + MARGIN_M (the fingers stopped wider than
                       asked: something is between the pads), or, without a quality-ok estimate, the reading shows the
                       clamp acting from above (n > c: a finger held more than 10.8 mm wider than its target)
    at_gap             the estimated pad gap is within MARGIN_M of the target gap
    short              the estimated pad gap is at least MARGIN_M below the target gap (an opening blocked from outside
                       or still under way), or, without a quality-ok estimate, the reading's clamp acts from below
    unconfirmed        no quality-ok estimate and the reading equals c (within 10.8 mm of the target: not known)

MARGIN_M = 1.0 mm (engineering value): about twice the largest error of the wrist-image estimate at a partial width
(0.56 mm at 20 mm in the probe; at most 0.46 mm in the visual-gap evaluation; the fingers themselves rest within
0.003 mm of the target). F's and D's "fully closed on nothing" bound for a close (1.5 mm = the closed gap 0.807 mm +
0.693 mm) stays as it is: there the fingers rest on their joint limit, where the estimate errs by less than 0.1 mm. An
object at most MARGIN_M wider than the requested gap is pressed by at most about 1 N per finger (stiffness 2300 N/m,
x5.py): at_gap. target_reached = at_gap, or unconfirmed while widening (an opening is taken as reached, as
OfficialBaseWorker._errors takes an open). The action is ok like a close (a stop on an object is its normal outcome);
motion_outcome is held_short unless target_reached.

The widening of a width action is judged by command (the requested gap against the gap the arm's gripper is commanded
to now: after a close the closed end, so every width; after a width that width; after an open the open limit): a wider
command lowers the squeeze, so while the arm holds something (the runtime's grasp state) it is a release that needs
open's release evidence (agp_core/runtime_ext.py GripWidthMixin). The worker side (GripWidthWorkerMixin):
  * preparation: the worker's gripper plan (SAMPLES = 10 samples of the current joints, migration/worker.py) with the
    gripper value c of the requested gap, SAMPLES_LONG = 12 when the finger target moves by more than LONG_CHANGE_M =
    35 mm from the previous command's (a change of the pad gap over 70 mm); the record ours_grip_width in the plan and the
    motion record. Why (engineering values from the open curve of agp_core/evaluation/estimator.py, the probe's per-step
    fingers of an open from closed: about 8.6 mm per step while the official clamp leads the finger, then 0.59 of the
    remaining distance per step): a finger 35 mm from its target is within 0.35 mm after 10 steps, one 44 mm away (the
    whole range) only within 0.5 mm, the outcome's margin per finger, and within 0.3 mm after 12;
  * G (OURS_GRIP_OBS) does not end a width action early: the reading cannot show the fingers' arrival within 10.8 mm of a
    partial target, so the action runs its samples (the ground truth above: the fingers arrive within them);
  * the outcome (_errors): the rule above, the result field ours_grip_width {requested_gap_mm, target_gap_mm, command,
    previous_command, widening, state, gap_mm, gap_source, reading, estimate_quality, estimate_gap_mm};
  * ok like a close (execute_actions: the arm's ok true; the stage ok unless a joint tracking fault);
  * F-g (OURS_ARM_GUARD): a widening width places the stage's fingers at the larger of the current outermost opening
    and the requested target.
The speed and guard rules of agp_core/speed_profiles.py read a partial command c (0 < c < 1, only this switch sends one)
through partial(), at_command() and target_gap_m() (F's two-signal empty rule, H, D; see there).
With the switch off nothing here is imported or composed: the runtime refuses the action (width_mm an unsupported
argument; "Use open or close ..."), the step verb `width` is unknown, the worker class and every record are as before.
"""
from __future__ import annotations

import math

ACTION = "width"                 # gripper {action: "width", width_mm: <pad gap in mm>}
ARG = "width_mm"
PAD_GAP0_M = 0.000807            # pad separation at finger joints 0 (agp_core/visual_gap.py; speed_profiles.GRIP_PAD_GAP0_M)
FINGER_LIMITS = (0.0, 0.044)     # joint7 / joint8 (Assets/Robots/x5 URDF, env/robot_manager/robot_config/x5.py)
SCALE = (-0.01, 0.044)           # gripper_scale (Assets/Robots/x5/robot_config.yml)
MIMIC = (1.0, 0.0)               # q8 = 1.0 * q7 + 0 (public mimic)
WIDTH_MIN_MM = 0.8               # closed: round(PAD_GAP0_M * 1000, 1)
WIDTH_MAX_MM = 88.8              # fully open: round((2 * 0.044 + PAD_GAP0_M) * 1000, 1)
MARGIN_M = 0.001                 # around a requested gap: at it / stopped on an object / short (see the docstring)
READING_TOL = 1e-6               # the reading equals c (the clamp does not act); as speed_profiles.CLOSED_EMPTY
CLAMP_M = 0.2 * (SCALE[1] - SCALE[0])   # 10.8 mm: the official clamp of the finger target (gripper_eps 0.2)
SAMPLES = 10                     # the worker's gripper samples (migration/worker.py _prepare_from_state)
SAMPLES_LONG = 12                # a change of the finger target over LONG_CHANGE_M (see the docstring)
LONG_CHANGE_M = 0.035
STATES = ("at_gap", "stopped_on_object", "short", "unconfirmed")
FINGERS_TEXT = {"at_gap": "at the requested gap", "stopped_on_object": "stopped on an object",
                "short": "short of the requested gap", "unconfirmed": "not confirmed (the wrist image cannot show them)"}
RANGE_TEXT = (f"width_mm is the pad gap in mm, from {WIDTH_MIN_MM} (closed) to {WIDTH_MAX_MM} (fully open)")


def width_mm(args):
    """The requested pad gap (mm) of a width action's args, or ValueError (exactly {action, width_mm}, a finite number
    in WIDTH_MIN_MM .. WIDTH_MAX_MM; extra keys are refused by the caller's own rules)."""
    value = (args or {}).get(ARG)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(RANGE_TEXT)
    if not WIDTH_MIN_MM <= float(value) <= WIDTH_MAX_MM:
        raise ValueError(RANGE_TEXT + f"; got {value}")
    return float(value)


def is_width(command, args):
    return command == "gripper" and isinstance(args, dict) and args.get("action") == ACTION


def finger_target_m(width_m, limits=FINGER_LIMITS, mimic=MIMIC):
    """The primary finger target (m) whose pad gap is width_m (pad gap = q7 + m1 * q7 + m2 + PAD_GAP0_M), clipped to the
    finger range."""
    q = (float(width_m) - PAD_GAP0_M - float(mimic[1])) / (1.0 + float(mimic[0]))
    return float(min(max(q, float(limits[0])), float(limits[1])))


def command_of_finger(q, scale=SCALE, sign=1.0):
    """The gripper value c in [0, 1] whose finger target is q (eval_env.py: target = s0 + c (s1 - s0), sign 1)."""
    span = float(scale[1]) - float(scale[0])
    c = (float(q) - float(scale[0])) / span if sign >= 0 else (float(scale[1]) - float(q)) / span
    return float(min(max(c, 0.0), 1.0))


def finger_of_command(c, scale=SCALE, limits=FINGER_LIMITS, sign=1.0):
    """The finger target of c, clipped to the finger range (where the fingers come to rest with nothing between)."""
    span = float(scale[1]) - float(scale[0])
    t = float(scale[0]) + float(c) * span if sign >= 0 else float(scale[1]) - float(c) * span
    return float(min(max(t, float(limits[0])), float(limits[1])))


def pad_gap_m(q, mimic=MIMIC):
    return float(q) + float(q) * float(mimic[0]) + float(mimic[1]) + PAD_GAP0_M


def target_gap_m(c, scale=SCALE, limits=FINGER_LIMITS, mimic=MIMIC, sign=1.0):
    """The pad gap (m) at which a gripper commanded to c comes to rest with nothing between the pads."""
    return pad_gap_m(finger_of_command(c, scale, limits, sign), mimic)


def partial(c):
    """A partial command: 0 < c < 1 (only the width action sends one; open is 1, close 0)."""
    return isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and 0.0 < float(c) < 1.0


def at_command(c, n):
    """The official reading n equals the command c: the clamp does not act (the fingers within 10.8 mm of the target)."""
    return (isinstance(n, (int, float)) and math.isfinite(n) and isinstance(c, (int, float))
            and abs(float(n) - float(c)) <= READING_TOL)


def reading_gap_m(c, n, scale=SCALE, limits=FINGER_LIMITS, mimic=MIMIC, sign=1.0):
    """The pad gap (m) the reading shows while its clamp acts (the finger 10.8 mm beyond the clamped target: wider when
    n > c, narrower when n < c), else None (the reading equals c or is missing)."""
    if not isinstance(n, (int, float)) or not math.isfinite(n) or at_command(c, n):
        return None
    span = float(scale[1]) - float(scale[0])
    t = float(scale[0]) + float(n) * span if sign >= 0 else float(scale[1]) - float(n) * span
    q = t + CLAMP_M if float(n) > float(c) else t - CLAMP_M
    return pad_gap_m(min(max(q, float(limits[0])), float(limits[1])), mimic)


def outcome(target_gap, widening, c, reading, estimate, scale=SCALE, limits=FINGER_LIMITS, mimic=MIMIC, sign=1.0):
    """The width action's outcome after its samples (see the module docstring): {state, target_reached, gap_m,
    gap_source, reading, estimate_quality, estimate_gap_m}."""
    estimate = estimate if isinstance(estimate, dict) else {}
    quality, gap = estimate.get("quality"), estimate.get("pad_gap_m")
    ok = quality == "ok" and isinstance(gap, (int, float)) and math.isfinite(gap)
    seen = reading_gap_m(c, reading, scale, limits, mimic, sign)
    if ok:
        gap = float(gap)
        state = ("stopped_on_object" if gap >= target_gap + MARGIN_M else
                 "short" if gap <= target_gap - MARGIN_M else "at_gap")
        shown, source = gap, "wrist image"
    elif seen is not None:
        state = "stopped_on_object" if float(reading) > float(c) else "short"
        shown, source = seen, "official reading"
    else:
        state, shown, source = "unconfirmed", None, None
    reached = state == "at_gap" or (state == "unconfirmed" and bool(widening))
    return {"state": state, "target_reached": reached, "gap_m": shown, "gap_source": source,
            "reading": None if not isinstance(reading, (int, float)) else float(reading),
            "estimate_quality": quality, "estimate_gap_m": float(gap) if ok else None}


def counts_as_close(state, widening):
    """The runtime's grasp rule for a width action that executed: a stop on an object, or an unconfirmed narrowing
    (the image could not show the fingers, as an unconfirmed close), starts a new grasp."""
    return state == "stopped_on_object" or (state == "unconfirmed" and not widening)


# ----------------------------------------------------------------------------- the worker (official conditions)
def _mm(value):
    return None if value is None else round(float(value) * 1000.0, 2)


class GripWidthWorkerMixin:
    """The width action in the official worker (see the module docstring); in front of the speed profiles."""

    def _gw_robot(self, arm):
        robot = self.arms[arm]["robot"]
        limits = self.arms[arm]["gripper_limits_m"]
        try:
            finger = (float(limits[0][0]), float(limits[0][1]))
        except (TypeError, IndexError):
            finger = FINGER_LIMITS
        move = dict(robot.gripper_move)
        return {"scale": tuple(float(v) for v in robot.gripper_scale), "limits": finger,
                "mimic": (float(move["mimic"][1]), float(move["mimic"][2])), "sign": float(move.get("sign", 1.0))}

    def _prepare_from_state(self, arm, command, args, start, tcp, transit_velocity):
        if not is_width(command, args):
            return super()._prepare_from_state(arm, command, args, start, tcp, transit_velocity)
        try:
            requested = width_mm(args)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "motion_started": False, "motion_executed": False}
        geometry = self._gw_robot(arm)
        q = finger_target_m(requested / 1000.0, geometry["limits"], geometry["mimic"])
        c = command_of_finger(q, geometry["scale"], geometry["sign"])
        # the worker's own gripper plan (the hooks of the classes behind see a gripper action, as for a close)
        plan = super()._prepare_from_state(arm, command, dict(args, action="close"), start, tcp, transit_velocity)
        if not plan.get("ok"):
            return plan
        previous = float(self.grips[self.arms[arm]["robot"].arm_name])
        target = pad_gap_m(q, geometry["mimic"])
        previous_q = finger_of_command(previous, geometry["scale"], geometry["limits"], geometry["sign"])
        previous_gap = pad_gap_m(previous_q, geometry["mimic"])
        change = abs(q - previous_q)
        samples = SAMPLES_LONG if change > LONG_CHANGE_M else SAMPLES
        plan.update(args=args, gripper_command=c, points=[plan["points"][0]] * samples, ours_grip_width={
            "requested_gap_mm": requested, "target_gap_mm": _mm(target), "finger_target_m": round(q, 6),
            "command": c, "previous_command": previous, "previous_target_gap_mm": _mm(previous_gap),
            "finger_change_m": round(change, 6), "widening": bool(c > previous), "samples": samples,
            "early_end": "none: G does not end a width action early (the reading cannot show the fingers' arrival "
                         "within 10.8 mm of a partial target)"})
        return plan

    def _sp_grip_rule(self, arm, plan):
        """G (OURS_GRIP_OBS): a width action runs its samples (a GripRule of source "fixed")."""
        if plan.get("command") == "gripper" and (plan.get("args") or {}).get("action") == ACTION:
            from agp_core import speed_profiles as sp
            g = self._gw_robot(arm)
            return sp.GripRule(ACTION, "fixed", None, None, g["scale"], g["limits"], g["mimic"], g["sign"],
                               len(plan["points"]))
        return super()._sp_grip_rule(arm, plan)

    def _sp_stage_state(self, plans):
        """F-g: a widening width places the fingers at the larger of the current outermost opening and its target."""
        fingers, held = super()._sp_stage_state(plans)
        for arm, plan in plans.items():
            record = plan.get("ours_grip_width") if plan.get("command") == "gripper" else None
            if record and arm in fingers:
                fingers[arm] = max(float(fingers[arm]), float(record["finger_target_m"]))
        return fingers, held

    def _errors(self, arm, plan):
        record = plan.get("ours_grip_width") if plan.get("command") == "gripper" else None
        if record is None:
            return super()._errors(arm, plan)
        from migration import worker as base
        geometry = self._gw_robot(arm)
        joints = self.rm.get_joint(self.arms[arm]["robot"])[0]
        try:
            from agp_core import speed_profiles as sp
            reading = sp.gripper_observation(self.env.get_obs()["state"], arm)
        except Exception:                                     # noqa: BLE001 - no reading: judged from the image only
            reading = None
        estimate_of = getattr(self, "_gripper_estimate", None)
        estimate = estimate_of(arm) if estimate_of is not None else None
        target = record["target_gap_mm"] / 1000.0
        verdict = outcome(target, record["widening"], record["command"], reading, estimate, geometry["scale"],
                          geometry["limits"], geometry["mimic"], geometry["sign"])
        q = record["finger_target_m"]
        desired = [q, q * geometry["mimic"][0] + geometry["mimic"][1]]
        achieved = self.tcp_base(arm)
        estimate = estimate if isinstance(estimate, dict) else {}
        info = dict(record, state=verdict["state"], fingers=FINGERS_TEXT[verdict["state"]],
                    gap_mm=_mm(verdict["gap_m"]), gap_source=verdict["gap_source"], reading=verdict["reading"],
                    estimate_quality=verdict["estimate_quality"], estimate_gap_mm=_mm(verdict["estimate_gap_m"]),
                    margin_mm=round(MARGIN_M * 1000.0, 3))
        return {"target_reached": bool(verdict["target_reached"]), "position_error_m": None,
                "rotation_error_rad": None,
                "joint_error_rad": float(max(abs(float(a) - float(b)) for a, b in zip(joints, plan["points"][-1]))),
                "gripper_error_m": None, "gripper_drive_error_m": None, "gripper_endpoint_m": desired,
                "gripper_measured_m": None, "gripper_drive_target_m": desired, "gripper_observed": False,
                "gripper_estimate_m": estimate.get("finger_joints_m"),
                "gripper_estimate_pad_gap_m": estimate.get("pad_gap_m"),
                "gripper_estimate_quality": estimate.get("quality"),
                "gripper_outcome_basis": ("width action (agp_core/grip_width.py): the wrist-image estimate of the pad gap "
                                          "(quality ok) against the requested gap, else the official reading's clamp"),
                "achieved_tcp_pose_base": base.pose(achieved).tolist(), "position_residual_m": None,
                "ours_grip_width": info}

    def execute_actions(self, actions):
        out = super().execute_actions(actions)
        widths = [arm for arm, action in (actions or {}).items()
                  if isinstance(action, dict) and is_width(action.get("command"), action.get("args"))]
        arms = (out or {}).get("arms") or {}
        if not widths or not arms:
            return out
        changed = False
        for arm in widths:
            result = arms.get(arm)
            if not isinstance(result, dict) or result.get("ok") is True or "ours_grip_width" not in result:
                continue
            result["ok"] = True                               # like a close: a stop on an object is its outcome
            self._gw_rewrite_record(arm, result)
            changed = True
        if changed:                                           # the stage's ok as the loop computes it: no joint
            faulted = any(str((r or {}).get("stop_reason") or "").startswith(f"{name} joint tracking error exceeded")
                          for name, r in arms.items())        # tracking fault and every arm ok
            out["ok"] = bool(not faulted and all(isinstance(r, dict) and r.get("ok") for r in arms.values()))
        return out

    def _gw_rewrite_record(self, arm, result):
        """The arm's motion record of this stage gets the corrected ok (the record was written inside the loop)."""
        from migration.worker import atomic
        import json
        try:
            paths = sorted((self.session / "motion_diagnostics").glob(f"*_{self.motion_sequence:04d}_{arm}_gripper.json"))
            if paths:
                record = json.loads(paths[-1].read_text())
                record["ok"] = True
                atomic(paths[-1], record)
        except Exception:                                     # noqa: BLE001 - a record never fails the stage
            pass


def worker_class(base_cls, environ=None):
    """`base_cls` with GripWidthWorkerMixin in front when OURS_GRIP_WIDTH is on, else `base_cls` unchanged."""
    from agp_core import features
    if not features.enabled("grip_width", environ):
        return base_cls
    return type(f"GripWidth{base_cls.__name__}", (GripWidthWorkerMixin, base_cls),
                {"__module__": __name__, "__doc__": f"{base_cls.__qualname__} with the grip width action."})
