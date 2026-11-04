"""P (switch OURS_PREC_ZONE, opt-in): precision speed only for the last part of a contact or place
descent and the first part of a lift. This module is the run's half (the contact and place splits, agp_core/runner.py)
and holds P's constants; the lift is re-timed by the worker (agp_core/speed_profiles.py). With the switch off nothing of
this module is imported by the run.

Design: P = precision only in the last part of contact, lift and place; the run fills the runtime's evidence for the
fast part (the clearance declaration is a formality once the split is automatic), but a descent is split only when its
target comes from a measurement whose height source is match or plane (never fallback, uncertain or none) and the
descent is nearly vertical; the contact tail is 1.5 cm, the place tail 2 cm and the lift head 1 cm.

Contact and place (one `run` step; the `step` tool is never split). An arm action of the step is split when all of
these hold, otherwise the reason is recorded and the step runs as written:
  1. intent contact or place, command move_ee to a measurement-relative target (`label+dx,dy,dz`: position_offset with
     a measurement_id), linear, precision speed, no max_samples;
  2. the measurement is valid, belongs to that arm, is an RGB-only measurement, and its height source is match or plane
     (SOURCES: a fallback, uncertain or none height is never split);
  3. nearly vertical: the arm's TCP in the observation before the step is within VERTICAL_XY_M = 1 cm horizontally of
     the target (target = the measured point + the offset), and the move turns the wrist by at most MAX_TURN_RAD =
     0.05 rad;
  4. the descent (TCP height before the step minus the target height) is at least the tail + MIN_FAST_M (contact:
     1.5 + 0.5 cm; place: 2 + 0.5 cm), so the fast part is at least 5 mm long;
  5. the gripper state fits the kind: contact with an open gripper (the observation's reading n = 1, the open has
     converged) and no grasp attempt active in the runtime (so the approach, intent observe, changes no grasp state),
     place while holding (0 < n < 1 and the runtime's grasp attempt active);
  6. the action carries target_unchanged evidence (the runtime requires it for both parts).
A step is split only when every arm action in it is split (contact and/or place; one arm or both arms), so the arms
keep moving on a shared clock: first both fast parts, then both original actions. A step with one qualifying arm and
another action (or a non-qualifying descent) of the other arm runs as written.

The split: the fast part is a runtime step of its own with action id "<run id>.<k>.p": the same move_ee with the offset
raised by the tail (CONTACT_TAIL_M = 1.5 cm or PLACE_TAIL_M = 2 cm above the target), speed_profile transit, the intent
the runtime requires for transit (FAST_INTENT: observe for a contact approach with an empty gripper, as a staging move;
transfer for a place approach while holding, carrying toward the destination), the same measurement_id, and the
action's evidence with clearance replaced by a text that states the automatic split, the measurement and the reason
(plus `automatic_split`, the same facts as data). Its speed is then the worker's: F re-times an empty arm's transit
move (OURS_FREE_TOPT), H a holding arm's (OURS_HOLD_PROFILE), otherwise the transit rate. When it reaches its target,
the original action runs unchanged from there (action id "<run id>.<k>", precision for the last 1.5 or 2 cm). When the
runtime refuses the fast part before dispatch (its evidence or input), the step runs as written; when the fast part ran
and did not reach its target, the run stops at step k with its result.

Records: prec_zone.jsonl in the session, one line per run step that has a contact or place descent (decision, the
reasons per arm, the fast part's and the original action's ids and outcomes, the TCP distance left for the precision
part); the run's full result carries the same under `prec_zone`, and the step's view a short `split` entry.

The place-only split (switch OURS_PLACE_SPLIT, opt-in): P's place half only, with OURS_PREC_ZONE off (with it on,
P's own split acts and this switch changes nothing). No contact split, no lift head, no prompt text, and nothing of it
in a model-visible result (plan_place_step, place_fast_action, place_tail_actions, as_written; the run's half is in
agp_core/runner.py). An arm action of a run step qualifies when all of these hold:
  1. intent place, command move_ee to a measurement-relative target (P's rules: linear, precision speed, no
     max_samples, the measurement valid, of that arm, RGB-only, height source match or plane, target_unchanged
     evidence) or move_delta (a relative displacement: precision speed, no max_samples); an absolute move_ee target
     is not split;
  2. nearly vertical: the TCP before the step within VERTICAL_XY_M = 1 cm horizontally of the target, the wrist turning
     by at most MAX_TURN_RAD = 0.05 rad (move_delta: its drot_deg);
  3. the arm holds something: the runtime's grasp attempt is active and the two signals do not confirm the gripper
     empty (agp_core/speed_profiles.py empty_state: the reading n = 0 AND the wrist-image pad gap < 1.5 mm, quality ok, in
     the observation before the step); its lift status is set (post_grasp_motion_seen), so that the fast part, a
     transit move with intent observe, changes no runtime state (it would set the lift status; a measurement-relative
     place passes the runtime only with that status set anyway);
  4. the fast part is at least MIN_FAST_M = 5 mm (descent >= PLACE_TAIL_M + 5 mm), and the split saves steps:
     estimate_steps (the worker's own sampling of straight moves: precision 1.2 mm and 0.0043 rad per sample, 3.0 mm
     and 0.0108 rad with OURS_PRECISION_2X (x 2.5; agp_core/runner.py passes the switch); the
     fast part at H's s-curve when OURS_HOLD_PROFILE is on, else 4.8 mm and 0.0168 rad per sample; the tail the
     precision 2 cm; plus SPLIT_EXTRA_SETTLE for the extra action's settle) is smaller split than unsplit.
A step is split only when every arm action in it qualifies (P's rule): a two-arm step whose arms both place and both
qualify runs both fast parts on one shared clock (one runtime step), then both precision tails on one shared clock;
a step with any other action, or a place that does not qualify, runs as written (the qualifying arm's reason says the
other arm's action is not split). The fast part is the runtime step "<run id>.<k>.p": the same move_ee with the offset
raised by PLACE_TAIL_M (or the move_delta's displacement shortened by it, its wrist turn included), speed_profile
transit, intent observe, the action's evidence plus an automatic clearance and `automatic_split` (OURS_PLACE_SPLIT
...): the worker times it as a holding transit move (H when on), D and F-g watch it as usual. After it reached its
target the tail runs: the measurement-relative action unchanged, or the move_delta of the remaining displacement to
the original target (from the fast part's achieved TCP, no further wrist turn). The fast part's result is not in the
run's result; when it does not reach its target, the run stops at that step and shows its result restated against the
model's own targets (as_written: every split arm not at its target, the distance to that target; restate_guidance:
the runtime's recovery guidance for the model's own actions). Records in
prec_zone.jsonl only (mode "place_split": decision, reasons, the estimate, both parts' outcomes and steps).

The support-dependent tail (part of OURS_PLACE_SPLIT; it replaces the fixed 2 cm tail above). On top of rules 1-4 of
the place-only split, the referenced support decides:
  * the referenced support: a measurement-relative place's own measurement; a move_delta place's arm's
    target_confirmation in the runtime (the measurement its last transfer or contact cited). Its height source fallback,
    uncertain or none (or no referenced measurement at all) -> no split (BRANCH_UNCERTAIN);
  * the held object's lowest point at the end of the place = the target TCP height - the hang, the hang being the arm's
    TCP height above the table at the close that started the grasp (the runtime journal's achieved pose of that close;
    counted from the table, as the hold clearance of OURS_HOLD_CLEARANCE does). It ends within ONTO_M = 1 cm above the
    referenced surface (the support measurement's height; below it counts as onto) -> the precision tail is
    ONTO_TAIL_M = 0.5 cm (BRANCH_ONTO); higher -> a hanging release (for example into a container whose floor was not
    measured): the 2 cm tail PLACE_TAIL_M (BRANCH_HANGING); a hang that cannot be read (no executed close of this grasp
    in the journal) counts as hanging;
  * the estimate (rule 4) uses the chosen tail; the fast part's law is the verified fast hold's (0.3 m/s, 2 m/s^2) for
    an arm whose grasp the worker verified (session file hold_verified.json, OURS_HOLD_VERIFIED_FAST on), else H's. The
    worker times the fast part itself by the same rule (a held transit move).
The decision records name the branch (`branch`) with the support's measurement, source and height, the hang and the
lowest point; the fast part's evidence names the tail. Everything else as in the place-only split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

# ---- design values
CONTACT_TAIL_M = 0.015          # a contact descent keeps its last 1.5 cm at precision
SOURCES = ("match", "plane")    # only these height sources are split
# ---- place tail and lift head
PLACE_TAIL_M = 0.02             # a place descent keeps its last 2 cm at precision
LIFT_HEAD_M = 0.01              # a lift keeps its first 1 cm at precision (agp_core/speed_profiles.py)
# ---- engineering values
VERTICAL_XY_M = 0.01            # nearly vertical: the start TCP within 1 cm horizontally of the target (in
                                # simulator runs 192 of 216 contact moves started within 1 cm, the p90 was 1.14 cm)
MAX_TURN_RAD = 0.05             # nearly vertical also means no wrist turn on the way down (the staging pose's residual
                                # is at most 0.02 rad, the worker's rotation tolerance)
MIN_FAST_M = 0.005              # a fast part shorter than 5 mm saves at most about 2 steps: not split
FAST_INTENT = {"contact": "observe", "place": "transfer"}
TAILS = {"contact": CONTACT_TAIL_M, "place": PLACE_TAIL_M}
OPEN_CONVERGED = 1.0 - 1e-6     # as agp_core/speed_profiles.py
CLOSED_EMPTY = 1e-6
RECORD_FILE = "prec_zone.jsonl"


def descent_geometry(start_xyz, target_xyz, turn_rad):
    """Horizontal distance, descent (start height minus target height) and wrist turn of a move."""
    dx, dy = float(target_xyz[0]) - float(start_xyz[0]), float(target_xyz[1]) - float(start_xyz[1])
    return {"horizontal_m": math.hypot(dx, dy), "descent_m": float(start_xyz[2]) - float(target_xyz[2]),
            "turn_rad": float(turn_rad)}


def quat_angle(a, b):
    """Rotation angle (rad) between two quaternions (w, x, y, z)."""
    na = math.sqrt(sum(v * v for v in a)) or 1.0
    nb = math.sqrt(sum(v * v for v in b)) or 1.0
    dot = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return 2.0 * math.acos(min(1.0, dot))


def measurement(runtime, measurement_id):
    """(entry, valid) of a journaled measurement, read only; (None, False) when unknown."""
    try:
        row = runtime.db.execute("SELECT value, valid FROM measurements WHERE id=?",
                                 (int(str(measurement_id)[1:]),)).fetchone()
    except Exception:                       # noqa: BLE001 - no journal: nothing to split on
        return None, False
    if not row:
        return None, False
    return json.loads(row[0]), bool(row[1])


def _arm_state(runtime, arm):
    try:
        return (runtime.state.get("arms") or {}).get(arm) or {}
    except Exception:                       # noqa: BLE001
        return {}


def arm_plan(runtime, arm, action, obs_state):
    """(plan, reason) for one runtime arm action: plan when it qualifies (see the module docstring), else None and the
    reason. kind None means the action is not a contact or place descent at all (not a candidate)."""
    intent = action.get("intent")
    if intent not in TAILS or action.get("command") not in ("move_ee", "move_delta"):
        return {"kind": None}, f"{action.get('command')} with intent {intent}: not a contact or place descent"
    kind = intent
    tail = TAILS[kind]
    args = action.get("args") or {}
    if action.get("command") != "move_ee" or "position_offset" not in args or not action.get("measurement_id"):
        return {"kind": kind}, f"{kind}: not a move to a measurement-relative target (label+dx,dy,dz)"
    if args.get("mode", "linear") != "linear":
        return {"kind": kind}, f"{kind}: plan mode"
    if args.get("speed_profile", "precision") != "precision":
        return {"kind": kind}, f"{kind}: not at precision speed"
    if "max_samples" in args:
        return {"kind": kind}, f"{kind}: a checkpoint (max_samples)"
    mid = action["measurement_id"]
    entry, valid = measurement(runtime, mid)
    if entry is None or not valid or not entry.get("ok"):
        return {"kind": kind}, f"{kind}: measurement {mid} absent or invalid"
    if entry.get("arm") != arm:
        return {"kind": kind}, f"{kind}: measurement {mid} belongs to the other arm"
    result = entry.get("result") or {}
    height = result.get("height") or {}
    source = height.get("source")
    if not result.get("rgb_only") or source not in SOURCES:
        return {"kind": kind, "source": source}, (f"{kind}: measurement {mid} height source {source} (only "
                                                   f"{' or '.join(SOURCES)} is split)")
    point = result.get("point_base") or {}
    offset = args["position_offset"]
    try:
        target = [float(point[k]) + float(offset[k]) for k in "xyz"]
    except (KeyError, TypeError, ValueError):
        return {"kind": kind}, f"{kind}: measurement {mid} has no point"
    state = (obs_state.get("arms") or {}).get(arm) or {}
    pose = state.get("tcp_pose_base")
    if not pose or len(pose) != 7:
        return {"kind": kind}, f"{kind}: no TCP pose in the observation before the step"
    rotation = args.get("rotation") or {}
    try:
        turn = quat_angle([float(rotation[k]) for k in "wxyz"], [float(v) for v in pose[3:]])
    except (KeyError, TypeError, ValueError):
        return {"kind": kind}, f"{kind}: no rotation"
    geometry = descent_geometry(pose[:3], target, turn)
    facts = {"kind": kind, "measurement_id": mid, "source": source, "target": [round(v, 6) for v in target],
             "start": [round(float(v), 6) for v in pose[:3]], **{k: round(v, 6) for k, v in geometry.items()},
             "tail_m": tail}
    if geometry["horizontal_m"] > VERTICAL_XY_M:
        return facts, (f"{kind}: not nearly vertical, the TCP starts {geometry['horizontal_m'] * 1000:.1f} mm from the "
                       f"target horizontally (at most {VERTICAL_XY_M * 1000:.0f} mm)")
    if geometry["turn_rad"] > MAX_TURN_RAD:
        return facts, f"{kind}: turns the wrist by {geometry['turn_rad']:.3f} rad (at most {MAX_TURN_RAD} rad)"
    if geometry["descent_m"] < tail + MIN_FAST_M:
        return facts, (f"{kind}: descends {geometry['descent_m'] * 1000:.1f} mm, less than the "
                       f"{tail * 1000:.0f} mm tail + {MIN_FAST_M * 1000:.0f} mm")
    n = state.get("gripper_command")
    try:
        n = float(n)
    except (TypeError, ValueError):
        n = None
    facts["gripper_reading"] = n
    if n is None or not math.isfinite(n):
        return facts, f"{kind}: no gripper reading in the observation"
    if kind == "contact" and not (n >= OPEN_CONVERGED and not _arm_state(runtime, arm).get("grasp_active")):
        return facts, (f"contact: the gripper is not open and settled with no grasp attempt active (n = {n:.3f}, "
                       f"grasp attempt {bool(_arm_state(runtime, arm).get('grasp_active'))})")
    if kind == "place" and not (CLOSED_EMPTY < n < OPEN_CONVERGED and _arm_state(runtime, arm).get("grasp_active")):
        return facts, f"place: the arm does not hold anything (n = {n:.3f})"
    evidence = action.get("evidence") or {}
    if not isinstance(evidence.get("target_unchanged"), str) or not evidence["target_unchanged"].strip():
        return facts, f"{kind}: no target_unchanged evidence"
    return dict(facts, ok=True), None


def fast_action(arm, action, plan):
    """The fast part of one split arm action (see the module docstring)."""
    tail = plan["tail_m"]
    args = dict(action["args"])
    offset = dict(args["position_offset"])
    offset["z"] = float(offset["z"]) + tail
    args.update(position_offset=offset, speed_profile="transit")
    reason = (f"the target height is a {plan['source']} measurement and the descent is nearly vertical (the TCP starts "
              f"{plan['horizontal_m'] * 1000:.1f} mm from the target horizontally, the wrist turns "
              f"{plan['turn_rad']:.3f} rad)")
    clearance = (f"automatic split by the run (OURS_PREC_ZONE): the {arm} arm's {plan['kind']} move first goes straight "
                 f"down at transit speed to {tail * 1000:.0f} mm above its target from measurement "
                 f"{plan['measurement_id']}, along the same line as the {plan['kind']} move itself, because {reason}; "
                 f"the last {tail * 1000:.0f} mm follow at precision speed as the {plan['kind']} move")
    evidence = dict(action.get("evidence") or {})
    evidence.update(clearance=clearance, automatic_split=f"OURS_PREC_ZONE {plan['kind']} split: {reason}")
    return {"command": "move_ee", "intent": FAST_INTENT[plan["kind"]], "args": args,
            "measurement_id": plan["measurement_id"], "evidence": evidence}


def plan_step(runtime, actions, obs_full):
    """The P decision for one run step: None when no arm action is a contact or place descent (nothing to record);
    else {"split": bool, "arms": {arm: plan facts}, "reasons": {arm: reason}, "fast": {arm: action} (when split)}."""
    obs_state = (obs_full or {}).get("state") or {}
    plans, reasons, candidates = {}, {}, False
    for arm, action in actions.items():
        plan, reason = arm_plan(runtime, arm, action, obs_state)
        plans[arm] = plan
        if plan.get("kind") is not None:
            candidates = True
        if reason:
            reasons[arm] = reason
    if not candidates:
        return None
    split = not reasons
    decision = {"split": split, "arms": plans, "reasons": reasons}
    if split:
        decision["fast"] = {arm: fast_action(arm, actions[arm], plans[arm]) for arm in actions}
    elif any(p.get("ok") for p in plans.values()):
        for arm, plan in plans.items():
            if plan.get("ok"):
                reasons[arm] = "qualifies, but the other arm's action in this step is not split"
    return decision


def approach_verdict(result):
    """'reached' (every arm of the fast part reached its target), 'refused' (the runtime refused it before dispatch:
    nothing moved, the step runs as written) or 'stopped' (it ran, or was refused by the worker, and not every arm
    reached)."""
    if result.get("status") != "done":
        if result.get("error_code") in ("INVALID_EVIDENCE", "INVALID_INPUT") and "action" not in result:
            return "refused"
        return "stopped"
    arms = ((result.get("action") or {}).get("arms") or {})
    if result.get("ok") and arms and all(a.get("target_reached") is True for a in arms.values()):
        return "reached"
    return "stopped"


def outcome(result):
    """A short record of one runtime step result (ids, samples, per-arm outcome and achieved TCP)."""
    action = result.get("action") or {}
    out = {"action_id": result.get("action_id"), "ok": result.get("ok"), "status": result.get("status"),
           "samples": action.get("low_level_actions"), "arms": {}}
    if result.get("error"):
        out["error"] = str(result["error"])[:300]
    for arm, a in (action.get("arms") or {}).items():
        out["arms"][arm] = {"outcome": a.get("motion_outcome"), "reached": a.get("target_reached"),
                            "trajectory_samples": a.get("trajectory_samples"), "settle_samples": a.get("settle_samples"),
                            "achieved_tcp_pose_base": a.get("achieved_tcp_pose_base"),
                            "stop_reason": a.get("stop_reason")}
    return out


def view(decision, approach, tail):
    """The step view's `split` entry (compact, neutral)."""
    arms = decision.get("arms") or {}
    entry = {"auto": "precision only for the last part: a transit approach ran first",
             "tail_mm": {arm: round(p["tail_m"] * 1000) for arm, p in arms.items() if p.get("ok")},
             "approach_id": approach.get("action_id"),
             "approach_samples": (approach.get("action") or {}).get("low_level_actions")}
    return entry


def record(session, entry):
    """Append one line to the session's prec_zone.jsonl (never raises: a failed record never stops the run)."""
    try:
        path = Path(session) / RECORD_FILE
        with path.open("a") as stream:
            stream.write(json.dumps(dict(entry, wall=time.time()), default=str) + "\n")
    except Exception:                       # noqa: BLE001
        pass


def precision_left(decision, approach):
    """TCP distance from each arm's achieved pose after the fast part to its target (the precision part's length)."""
    out = {}
    arms = ((approach.get("action") or {}).get("arms") or {})
    for arm, plan in (decision.get("arms") or {}).items():
        pose = (arms.get(arm) or {}).get("achieved_tcp_pose_base")
        if pose and plan.get("target"):
            out[arm] = round(math.dist([float(v) for v in pose[:3]], plan["target"]), 6)
    return out



# ----------------------------------------------------------------------------- the place-only split
PLACE_SPLIT = "place_split"         # the records' mode (prec_zone.jsonl) and the decision's marker
PLACE_FAST_INTENT = "observe"       # the fast part's intent: a transit move that needs no measurement and, with the lift
                                    # status already set (rule 3), changes no runtime state (P's transfer would set the
                                    # target confirmation)
PLACE_SPLIT_MARK = "OURS_PLACE_SPLIT"   # the start of the fast part's automatic_split evidence (the ledger's marker)
# the worker's own sampling of a straight move (migration/worker.py _prepare_from_state: rate x dt x 0.6 per sample)
PRECISION_SAMPLE_M = 0.05 * 0.04 * 0.6      # 1.2 mm
PRECISION_SAMPLE_RAD = 0.18 * 0.04 * 0.6    # 0.00432 rad
TRANSIT_SAMPLE_M = 0.20 * 0.04 * 0.6        # 4.8 mm (a closed gripper keeps the transit rate when H is off)
TRANSIT_SAMPLE_RAD = 0.7 * 0.04 * 0.6       # 0.0168 rad
SPLIT_EXTRA_SETTLE = 1              # engineering value: the extra action's own settle samples in the estimate (an
                                    # s-curve ends at rest; 0 to 1 settle sample is the usual)
# ---- the support-dependent tail of the place split
ONTO_M = 0.01                       # the held object's lowest point ends within 1 cm above the referenced surface
ONTO_TAIL_M = 0.005                 # ... then the precision tail is 0.5 cm (a support of certain height)
BRANCH_UNCERTAIN, BRANCH_ONTO, BRANCH_HANGING = "uncertain support", "onto the referenced surface", "hanging release"
PLACE_RULE = "R3I"                  # the place_split records' rule marker (prec_zone.jsonl)
TABLE_Z_M = 0.0                     # the table top in the arm's base frame (agp_core/rgb_measure.py, public camera model)


def _turn_of_delta(drot_deg):
    """Rotation angle (rad) of a move_delta's drot_deg (the worker: Rotation.from_euler("xyz", drot_deg, degrees))."""
    if not drot_deg:
        return 0.0
    rx, ry, rz = (math.radians(float(v)) for v in drot_deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    # extrinsic xyz: R = Rz @ Ry @ Rx; angle from the trace
    trace = cy * cz + (sx * sy * sz + cx * cz) + cx * cy
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def holding_state(runtime, arm, state):
    """(holding, reason) of an arm for the place split: the runtime's grasp attempt is active and the observation's two
    signals do not confirm the gripper empty (agp_core/speed_profiles.py empty_state with the gripper commanded closed: the
    reading n = 0 and the wrist-image pad gap, quality ok, below 1.5 mm)."""
    from agp_core import speed_profiles as sp
    if not _arm_state(runtime, arm).get("grasp_active"):
        return False, "no grasp attempt active in the runtime"
    n = state.get("gripper_command")
    try:
        n = float(n)
    except (TypeError, ValueError):
        n = None
    if n is None or not math.isfinite(n):
        return False, "no gripper reading in the observation"
    empty, why = sp.empty_state(0.0, n, state.get("gripper_gap_estimate"))
    return (not empty), why


def estimate_steps(geometry, tail, hold_profile, verified=False, precision_2x=False):
    """{unsplit, fast, tail, split} control steps of one arm's descent (see the module docstring, rule 4): the worker's
    own sampling of straight moves; the fast part at H's s-curve (agp_core/speed_profiles.py scurve_positions with H's caps,
    as retime_parts times a straight path) when OURS_HOLD_PROFILE is on -- with the verified fast hold's TCP cap and
    acceleration for a verified grasp (verified True) -- else the closed-gripper transit sampling. precision_2x
    (OURS_PRECISION_2X): the unsplit move and the tail at the raised precision rate (3.0 mm and 0.0108 rad per
    sample at agp_core/speed_profiles.py PRECISION_2X_SCALE = 2.5), as the worker then prepares them; the record names the scale."""
    from agp_core import speed_profiles as sp
    whole = math.hypot(geometry["horizontal_m"], geometry["descent_m"])
    fast_len = math.hypot(geometry["horizontal_m"], max(0.0, geometry["descent_m"] - tail))
    turn = geometry["turn_rad"]
    scale = sp.PRECISION_2X_SCALE if precision_2x else 1.0
    sample_m, sample_rad = ((PRECISION_SAMPLE_M, PRECISION_SAMPLE_RAD) if scale == 1.0 else
                            (PRECISION_SAMPLE_M * scale, PRECISION_SAMPLE_RAD * scale))
    unsplit = max(1, math.ceil(whole / sample_m - 1e-9), math.ceil(turn / sample_rad - 1e-9))
    if hold_profile:
        speed, peak = ((sp.FAST_HOLD_TCP_SPEED, sp.FAST_HOLD_TCP_ACCEL) if verified else
                       (sp.HOLD_TCP_SPEED, sp.HOLD_TCP_ACCEL))
        U = max(fast_len / (speed * sp.CAP_SAFETY), turn / (sp.HOLD_ROTATION * sp.CAP_SAFETY))
        accel = peak / (speed * sp.CAP_SAFETY)
        fast = len(sp.scurve_positions(U, accel)[0])
        law = ("verified hold s-curve (OURS_HOLD_VERIFIED_FAST)" if verified else
               "hold s-curve (OURS_HOLD_PROFILE)")
    else:
        fast = max(1, math.ceil(fast_len / TRANSIT_SAMPLE_M - 1e-9), math.ceil(turn / TRANSIT_SAMPLE_RAD - 1e-9))
        law = "closed-gripper transit sampling"
    tail_steps = max(1, math.ceil(tail / sample_m - 1e-9))
    out = {"unsplit": unsplit, "fast": fast, "tail": tail_steps, "split": fast + SPLIT_EXTRA_SETTLE + tail_steps,
           "fast_law": law}
    if scale != 1.0:                                    # OURS_PRECISION_2X (no key with the switch off)
        out["precision_scale"] = scale
    return out


def close_hang(runtime, arm):
    """The arm's TCP height above the table at the close that started its current grasp (the achieved TCP pose of
    its latest executed close in the runtime journal), or None when the journal has none (or an executed open of the
    arm came after it, or the close has no pose)."""
    try:
        rows = runtime.db.execute("SELECT payload, result FROM actions WHERE status='done' ORDER BY rowid DESC")
    except Exception:                       # noqa: BLE001 - no journal: unknown
        return None
    for payload, result in rows:
        try:
            spec = (json.loads(payload).get("actions") or {}).get(arm) or {}
        except (TypeError, ValueError):
            continue
        if spec.get("command") != "gripper":
            continue
        try:
            outcome = (((json.loads(result) or {}).get("action") or {}).get("arms") or {}).get(arm) or {}
        except (TypeError, ValueError):
            outcome = {}
        if outcome.get("motion_executed") is False:
            continue                        # refused before any motion: it changed nothing
        if (spec.get("args") or {}).get("action") != "close":
            return None                     # an open came after the last close: no grasp of this arm
        pose = outcome.get("achieved_tcp_pose_base")
        try:
            return float(pose[2]) - TABLE_Z_M
        except (TypeError, ValueError, IndexError):
            return None
    return None


def support_of(runtime, arm, action):
    """(measurement id, entry) of the place's referenced support: a move_ee's own measurement, a move_delta's arm's
    target_confirmation (the measurement its last transfer or contact cited); (None, None) without one."""
    mid = action.get("measurement_id") if action.get("command") == "move_ee" else \
        _arm_state(runtime, arm).get("target_confirmation")
    if not mid:
        return None, None
    entry, _ = measurement(runtime, mid)
    return mid, entry


def verified_arms(session):
    """{arm: True} for the arms whose grasp the worker verified (session file hold_verified.json written by
    agp_core/speed_profiles.py with OURS_HOLD_VERIFIED_FAST on); {} without the file."""
    try:
        from agp_core import speed_profiles as sp
        data = json.loads((Path(session) / sp.HOLD_VERIFIED_FILE).read_text())
        return {arm: bool((entry or {}).get("verified")) for arm, entry in data.items()}
    except Exception:                       # noqa: BLE001 - no file (switch off or no grasp yet): not verified
        return {}


def place_arm_plan(runtime, arm, action, obs_state, hold_profile=False, verified=False, precision_2x=False):
    """(plan, reason) of one runtime arm action for the place-only split (module docstring): plan with ok True when it
    qualifies (the step-level saving is decided by plan_place_step), else the reason; the plan names the branch once
    the support is read. kind None: not a place descent at all (a move with another intent, a gripper action, a home,
    a hold). verified: the arm's grasp is verified by the verified fast hold (the fast part's law).
    precision_2x: OURS_PRECISION_2X (the estimate at the doubled precision rate)."""
    intent, command = action.get("intent"), action.get("command")
    if intent != "place" or command not in ("move_ee", "move_delta"):
        return {"kind": None}, f"{command} with intent {intent}: not a place descent"
    args = action.get("args") or {}
    if args.get("speed_profile", "precision") != "precision":
        return {"kind": "place"}, "place: not at precision speed"
    if "max_samples" in args:
        return {"kind": "place"}, "place: a checkpoint (max_samples)"
    state = (obs_state.get("arms") or {}).get(arm) or {}
    pose = state.get("tcp_pose_base")
    if not pose or len(pose) != 7:
        return {"kind": "place"}, "place: no TCP pose in the observation before the step"
    start = [float(v) for v in pose[:3]]
    facts = {"kind": "place"}
    if command == "move_ee":
        if "position_offset" not in args or not action.get("measurement_id"):
            return facts, "place: an absolute target (neither measurement-relative nor a relative displacement)"
        if args.get("mode", "linear") != "linear":
            return facts, "place: plan mode"
        mid = action["measurement_id"]
        entry, valid = measurement(runtime, mid)
        if entry is None or not valid or not entry.get("ok"):
            return facts, f"place: measurement {mid} absent or invalid"
        if entry.get("arm") != arm:
            return facts, f"place: measurement {mid} belongs to the other arm"
        result = entry.get("result") or {}
        point, offset = result.get("point_base") or {}, args["position_offset"]
        facts.update(measurement_id=mid, relative=False)
        try:
            target = [float(point[k]) + float(offset[k]) for k in "xyz"]
        except (KeyError, TypeError, ValueError):
            return facts, f"place: measurement {mid} has no point"
        rotation = args.get("rotation") or {}
        try:
            turn = quat_angle([float(rotation[k]) for k in "wxyz"], [float(v) for v in pose[3:]])
        except (KeyError, TypeError, ValueError):
            return facts, "place: no rotation"
        evidence = action.get("evidence") or {}
        if not isinstance(evidence.get("target_unchanged"), str) or not evidence["target_unchanged"].strip():
            return facts, "place: no target_unchanged evidence"
    else:
        try:
            dpos = [float(v) for v in args.get("dpos") or (0.0, 0.0, 0.0)]
        except (TypeError, ValueError):
            return facts, "place: no displacement"
        if len(dpos) != 3:
            return facts, "place: no displacement"
        target = [start[i] + dpos[i] for i in range(3)]
        turn = _turn_of_delta(args.get("drot_deg"))
        facts.update(relative=True, dpos=[round(v, 6) for v in dpos])
    geometry = descent_geometry(start, target, turn)
    facts.update(target=[round(v, 6) for v in target], start=[round(v, 6) for v in start],
                 **{k: round(v, 6) for k, v in geometry.items()})
    if geometry["horizontal_m"] > VERTICAL_XY_M:
        return facts, (f"place: not nearly vertical, the TCP starts {geometry['horizontal_m'] * 1000:.1f} mm from the "
                       f"target horizontally (at most {VERTICAL_XY_M * 1000:.0f} mm)")
    if geometry["turn_rad"] > MAX_TURN_RAD:
        return facts, f"place: turns the wrist by {geometry['turn_rad']:.3f} rad (at most {MAX_TURN_RAD} rad)"
    holding, why = holding_state(runtime, arm, state)
    facts["holding"] = why
    if not holding:
        return facts, f"place: the arm is not holding anything ({why})"
    if not _arm_state(runtime, arm).get("post_grasp_motion_seen"):
        return facts, "place: the arm's lift status is not set (the fast part would set it)"
    # the referenced support and its height source
    mid, entry = support_of(runtime, arm, action)
    result = (entry or {}).get("result") or {}
    source = (result.get("height") or {}).get("source") if entry else None
    facts.update(support_measurement=mid, source=source)
    if not mid:
        return dict(facts, branch=BRANCH_UNCERTAIN), ("place: support height uncertain: no referenced measurement (no "
                                                      "target confirmation of the arm)")
    if entry is None or not entry.get("ok") or entry.get("arm") != arm:
        return dict(facts, branch=BRANCH_UNCERTAIN), (f"place: support height uncertain: the referenced measurement "
                                                      f"{mid} is absent, not ok or of the other arm")
    if not result.get("rgb_only") or source not in SOURCES:
        return dict(facts, branch=BRANCH_UNCERTAIN), (f"place: support height uncertain: measurement {mid} height "
                                                      f"source {source} (only {' or '.join(SOURCES)} is split)")
    try:
        surface = float((result.get("point_base") or {})["z"])
    except (KeyError, TypeError, ValueError):
        return dict(facts, branch=BRANCH_UNCERTAIN), f"place: support height uncertain: measurement {mid} has no point"
    facts["support_z_m"] = round(surface, 6)
    # onto the referenced surface (0.5 cm tail) or a hanging release (2 cm)
    hang = close_hang(runtime, arm)
    if hang is None:
        branch, tail = BRANCH_HANGING, PLACE_TAIL_M
        facts.update(hang_m=None, lowest_end_m=None, lowest_above_support_m=None,
                     branch_reason="the hang is unknown (no executed close of this grasp in the journal)")
    else:
        lowest = target[2] - hang
        above = lowest - surface
        branch, tail = (BRANCH_ONTO, ONTO_TAIL_M) if above <= ONTO_M + 1e-9 else (BRANCH_HANGING, PLACE_TAIL_M)
        facts.update(hang_m=round(hang, 6), lowest_end_m=round(lowest, 6), lowest_above_support_m=round(above, 6))
    facts.update(branch=branch, tail_m=tail)
    if geometry["descent_m"] < tail + MIN_FAST_M:
        return facts, (f"place: descends {geometry['descent_m'] * 1000:.1f} mm, less than the "
                       f"{tail * 1000:.0f} mm tail + {MIN_FAST_M * 1000:.0f} mm")
    facts["estimate"] = estimate_steps(geometry, tail, hold_profile, verified, precision_2x)
    if verified:
        facts["verified_hold"] = True
    return dict(facts, ok=True), None


def plan_place_step(runtime, actions, obs_full, hold_profile=False, hold_verified=False, precision_2x=False):
    """The place-only split decision for one run step (module docstring): None when no arm action is a
    place descent (nothing to record); else {"mode": PLACE_SPLIT, "split", "arms", "reasons", "estimate", "fast" (when
    split)}. Every arm action must qualify, and the step's estimate (the larger arm of each part) must save steps.
    hold_verified (OURS_HOLD_VERIFIED_FAST on): the worker's verified arms (hold_verified.json) estimate their fast
    part at the verified fast hold's law. precision_2x (OURS_PRECISION_2X on): the unsplit move and the tail at the
    doubled precision rate."""
    obs_state = (obs_full or {}).get("state") or {}
    verified = verified_arms(getattr(runtime, "session", None)) if hold_verified else {}
    plans, reasons, candidates = {}, {}, False
    for arm, action in actions.items():
        plan, reason = place_arm_plan(runtime, arm, action, obs_state, hold_profile, bool(verified.get(arm)),
                                      precision_2x)
        plans[arm] = plan
        if plan.get("kind") is not None:
            candidates = True
        if reason:
            reasons[arm] = reason
    if not candidates:
        return None
    decision = {"mode": PLACE_SPLIT, "split": False, "arms": plans, "reasons": reasons}
    if not reasons:
        est = [plans[arm]["estimate"] for arm in actions]
        unsplit = max(e["unsplit"] for e in est)
        split = max(e["fast"] for e in est) + SPLIT_EXTRA_SETTLE + max(e["tail"] for e in est)
        decision["estimate"] = {"unsplit": unsplit, "split": split, "saves": unsplit - split}
        if split < unsplit:
            decision["split"] = True
            decision["fast"] = {arm: place_fast_action(arm, actions[arm], plans[arm]) for arm in actions}
        else:
            for arm in actions:
                reasons[arm] = (f"place: the split would not save steps (estimated {split} against {unsplit} as "
                                f"written)")
    elif any(p.get("ok") for p in plans.values()):
        for arm, plan in plans.items():
            if plan.get("ok"):
                reasons[arm] = "qualifies, but the other arm's action in this step is not split"
    return decision


def place_fast_action(arm, action, plan):
    """The fast part of one qualifying place (module docstring): it ends the plan's tail (ONTO_TAIL_M or
    PLACE_TAIL_M) above the target."""
    tail = plan["tail_m"]
    args = dict(action["args"])
    args["speed_profile"] = "transit"
    support = (f"; the support is a {plan.get('source')} measurement ({plan.get('support_measurement')}), "
               f"{plan.get('branch')}")
    if plan.get("relative"):
        dpos = list(plan["dpos"])
        dpos[2] = dpos[2] + tail
        args["dpos"] = dpos
        why = (f"the place is a displacement that goes nearly straight down (the target {plan['horizontal_m'] * 1000:.1f}"
               f" mm away horizontally, a wrist turn of {plan['turn_rad']:.3f} rad) while the arm holds something"
               f"{support}")
    else:
        offset = dict(args["position_offset"])
        offset["z"] = float(offset["z"]) + tail
        args["position_offset"] = offset
        why = (f"the target height is a {plan['source']} measurement ({plan['measurement_id']}) and the descent is "
               f"nearly vertical (the TCP starts {plan['horizontal_m'] * 1000:.1f} mm from the target horizontally, the "
               f"wrist turns {plan['turn_rad']:.3f} rad) while the arm holds something{support}")
    clearance = (f"automatic split by the run ({PLACE_SPLIT_MARK}): the {arm} arm's place move first goes down at "
                 f"transit speed to {tail * 1000:g} mm above its target, along the same line as the place move "
                 f"itself, because {why}; the last {tail * 1000:g} mm follow at precision speed as the place move")
    evidence = dict(action.get("evidence") or {})
    evidence.update(clearance=clearance,
                    automatic_split=f"{PLACE_SPLIT_MARK} place split (tail {tail * 1000:g} mm, {plan.get('branch')}): "
                                    f"{why}")
    out = {"command": action["command"], "intent": PLACE_FAST_INTENT, "args": args, "evidence": evidence}
    if action.get("measurement_id") and action["command"] == "move_ee":
        out["measurement_id"] = action["measurement_id"]
    return out


def _achieved(result, arm):
    """An arm's achieved TCP position after a runtime step: the worker's achieved_tcp_pose_base, else the new
    observation's tcp_pose_base."""
    pose = (((result.get("action") or {}).get("arms") or {}).get(arm) or {}).get("achieved_tcp_pose_base")
    if not pose:
        pose = ((((result.get("observation") or {}).get("state") or {}).get("arms") or {}).get(arm) or {}).get(
            "tcp_pose_base")
    return [float(v) for v in pose[:3]] if pose and len(pose) >= 3 else None


def place_tail_actions(decision, actions, approach):
    """The step's actions after a fast part that reached its targets: a measurement-relative place unchanged; a
    move_delta place as the remaining displacement from the fast part's achieved TCP to the original target, without a
    further wrist turn (the fast part made it). None when an achieved pose is missing (the run then stops)."""
    out = {}
    for arm, action in actions.items():
        plan = decision["arms"][arm]
        if not plan.get("relative"):
            out[arm] = action
            continue
        achieved = _achieved(approach, arm)
        if achieved is None:
            return None
        args = {k: v for k, v in (action.get("args") or {}).items() if k != "drot_deg"}
        args["dpos"] = [round(plan["target"][i] - achieved[i], 6) for i in range(3)]
        out[arm] = dict(action, args=args)
    return out


def as_written(result, decision):
    """A fast part that did not reach its targets, restated against the model's own step (a deep copy): every split
    arm that is not at its target (no tail ran) is not reached, with its distance to the model's target as the
    position error and, where the fast part itself reached its own target, the outcome held_short; nothing else
    changes (the ids stay the runtime's, for action_status)."""
    shown = json.loads(json.dumps(result, default=str))
    arms = ((shown.get("action") or {}).get("arms") or {})
    for arm, outcome in arms.items():
        plan = (decision.get("arms") or {}).get(arm) or {}
        achieved = outcome.get("achieved_tcp_pose_base")
        if not plan.get("target") or not achieved:
            continue
        outcome["position_error_m"] = round(math.dist([float(v) for v in achieved[:3]], plan["target"]), 6)
        outcome["position_residual_m"] = [round(plan["target"][i] - float(achieved[i]), 6) for i in range(3)]
        if outcome.get("target_reached") is not False:
            outcome["target_reached"] = False
            if outcome.get("motion_outcome") == "reached":
                outcome["motion_outcome"] = "held_short"
    return shown


def restate_guidance(shown, actions, guidance_of):
    """The recovery guidance of a fast part shown as the model's own step (as_written): the runtime's rule
    (ControlRuntime._recovery_guidance, passed as guidance_of) for each arm's own action and its restated outcome,
    as the runtime gives it after a step (_complete_dual), in place of the fast part's (whose intent observe would ask to
    verify that the held object follows). Left as it is when the result has no action or the rule cannot be applied."""
    response = shown.get("action")
    if not isinstance(response, dict):
        return shown
    observation = shown.get("observation") or {}
    guidance = {}
    try:
        for arm, spec in actions.items():
            outcome = dict((response.get("arms") or {}).get(arm, {"ok": False}),
                           episode_ended=response.get("episode_ended", False))
            outcome.setdefault("motion_started", response.get("motion_started"))
            item = guidance_of(spec, outcome, observation, arm)
            if item is not None:
                guidance[arm] = item
    except Exception:                       # noqa: BLE001 - keep the runtime's own guidance
        return shown
    if guidance:
        shown["recovery_guidance"] = {"arms": guidance}
    else:
        shown.pop("recovery_guidance", None)
    return shown
