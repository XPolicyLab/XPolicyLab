"""F-g v2's runtime half (OURS_GUARD_V2 with OURS_ARM_GUARD) and the runtime-side switches built on it.

The control runtime (agp.experiments.control_runtime.runtime.ControlRuntime, W/source, read only) treats a step whose
worker response is not ok as uncertain: _complete_dual then invalidates every measurement and both arms' target
confirmation and lift state (_invalidate_dual_evidence); before dispatch, step() has already invalidated the
measurements when the step held a gripper action or a contact, and opened a new grasp for a close
(_change_dual_revision). A step in which neither arm started changed nothing physical. KeepUnstartedRuntime lets the
runtime do all of that unchanged and then restores, for such a step only:
  * every measurement that was valid before the step (the ids valid when _change_dual_revision ran);
  * each arm's target confirmation, grasp (grasp_epoch, grasp_active) and lift state (post_grasp_motion_seen), from the
    state the runtime saved with the action before dispatch (actions.prior_state).
The observation the runtime takes after the step, the result returned and the action's journal rows are the runtime's
own; one journal event KEPT_EVENT records what was restored (the step ledger counts it, agp_core/evaluation/ledger.py).

"Neither arm started" (neither_arm_moved): the response has motion_started false and motion_executed not true, each of
its counts (low_level_actions, trajectory_samples, settle_samples) is 0 or absent (F-g's layer-1 refusal has no
trajectory_samples: None), and the same holds for every arm outcome it lists. The worker (agp_core/speed_profiles.py over
migration/worker.py) answers so for: F-g's layer-1 refusal (two-arm clearance), a preparation refusal of either arm
(IK or planning rejected, a linear-IK branch jump, the Cartesian rate preparation, an approximate home that does not
fit or cannot be planned, the held-clearance refusal), the action budget refusal, the joint-trajectory preflight
refusal, an episode that already ended, a layer-2 stop or an official episode end before the first sample, and a worker
exception before any sample. Not covered: a transport refusal before dispatch (NOT_DISPATCHED: the runtime's step()
handles it itself and another request may have been pending) and an unresolved action (no response).

runtime_class(environ) is ControlRuntime itself unless both switches are on: with either off nothing here is used.

Intent manipulate: OURS_INTENT_MANIPULATE adds a seventh, generic intent "manipulate" (ManipulateMixin). With the
switch off the runtime class is the one above and "manipulate" stays an unknown intent
(ControlRuntime._validate_action: "Unknown action intent"). With it on, an action with intent manipulate:
  * runs at precision speed only: speed_profile transit is refused (MANIPULATE_PRECISION), and so is `home`, which the
    worker always runs at its transit profile (migration/worker.py _prepare_from_state); move_ee, move_delta,
    move_joints (always precision), gripper and hold are accepted;
  * needs no measurement: ControlRuntime._check_evidence asks one only of transfer, contact and a move_ee to a
    measurement-relative target (position_offset), so a manipulate move to such a target still needs a valid
    measurement of that arm (with its grasp-epoch and lift rules) and evidence.target_unchanged; an open still needs
    its release evidence whatever the intent;
  * is not the lift after a close (post_grasp_motion_seen is set only by observe, retreat and grasp motions), so it
    unlocks no transfer, and it never sets the arm's target confirmation (only contact and transfer do; and a relative
    contact continuation must follow a contact);
  * invalidates every measurement, as a contact does: before dispatch (_change_dual_revision, _change_revision), with
    contact's treatment of the other arm (its target confirmation is cleared) and, after the step, of the other arm's
    contact or transfer in the same step (it changes the shared support: no confirmation is set). Both dual-arm hooks
    hand the base class the action map with each manipulate arm written as a contact without a measurement_id
    (as_contact): a contact without a measurement keeps the arm's confirmation as it is, sets no lift, and counts as
    support-changing for the other arm. The journal keeps the action as sent (intent manipulate).
Composed with KeepUnstartedRuntime when OURS_GUARD_V2 and OURS_ARM_GUARD are on as well (runtime_class).

Keep unstarted: OURS_KEEP_UNSTARTED turns on KeepUnstartedRuntime alone, the runtime half of F-g v2, without
OURS_GUARD_V2's worker changes (its 15 mm layer 1, the delays and the pair-only refusal stay off; the worker answers a
step that no arm started with the same zero counts whether or not F-g v2 is on). runtime_class: keep-unstarted when
OURS_GUARD_V2 and OURS_ARM_GUARD are both on or when OURS_KEEP_UNSTARTED is on; with OURS_KEEP_UNSTARTED off the class
is exactly the one above, and with OURS_GUARD_V2 and OURS_ARM_GUARD on the switch changes nothing. It composes with the
intent manipulate (ManipulateKeepUnstartedRuntime).

Declarations (return reserve and handover): with OURS_RETURN_RESERVE or OURS_HANDOVER on, DeclareMixin goes in front
of the class above (Declare<that class>). The worker sees no evidence, so the declarations reach it as keys in the
arm's dispatched args, added after the runtime's own checks of that arm (ControlRuntime.step calls _check_evidence and
then _resolve_args for each arm in turn; the validated payload, the journal's action row and every rule of the runtime
are unchanged; the action_prepared event shows the dispatched args with the keys):
  * OURS_HANDOVER: evidence `handover` must be a text that names the other gripper (agp_core/handover.py
    names_other_gripper), else the step is refused before dispatch (INVALID_EVIDENCE, handover.HANDOVER_TEXT); then
    handover.ARGS_KEY: true (agp_core/handover.py relaxes F-g only when both arms of the stage carry it);
  * OURS_RETURN_RESERVE: evidence `final` or `no_return` must be a non-empty text, then return_reserve.FINAL_KEY: the
    key's name (the reserve lets that arm's action run, agp_core/return_reserve.py); an open whose evidence is a release at
    the goal (release_purpose goal, the default, and release_requirement not "empty") gets return_reserve.RELEASE_KEY:
    true (the worker counts it once it executed).
With both off, runtime_class() returns exactly the classes above and nothing here is used.

Grip width (agp_core/grip_width.py): with OURS_GRIP_WIDTH on, GripWidthMixin goes in front of the class chosen above (and
DeclareMixin, when on, in front of that). It accepts a third gripper action {action "width", width_mm: the requested
pad gap in mm, grip_width.WIDTH_MIN_MM .. WIDTH_MAX_MM} and keeps every rule of the runtime as for open and close:
  * release evidence: the arm holds something (its grasp_active: an attempted grasp since its last close, or since a
    width that stopped on an object, and no release since) and the requested gap is wider than the gap its gripper is
    commanded to now (the commanded gap of its last executed gripper action in the journal: a close's target is past
    closed, so every width; a width its own gap; an open the open limit; none: open, the episode starts open) -> the
    action is a release and ControlRuntime._check_evidence checks it exactly as an open (release_requirement, support,
    landing, empty, recovery). An empty arm, or a width that does not widen, needs no release evidence;
  * before dispatch: every measurement invalidated and the other arm's target confirmation cleared, as for any gripper
    action (ControlRuntime._change_dual_revision); no new grasp yet;
  * after the response (the worker's outcome ours_grip_width.state, agp_core/grip_width.py): a width that executed and
    stopped on an object, or an unconfirmed narrowing, counts as a close: a new grasp epoch, grasp_active, no lift
    yet, no target confirmation (as _change_dual_revision does for a close); a release that reached its gap (or an
    unconfirmed widening, taken as reached like an open) ends the grasp as an open does (the base class completes it
    as an open); otherwise the grasp state stays. A held_short or unreached outcome invalidates both arms' evidence in
    the base class, as after a close on an object. The journal keeps the action as sent; one event GRIP_WIDTH_EVENT
    records each width arm's decision. Recovery guidance: a close's for a width that counts as a close, else an open's;
  * DeclareMixin (OURS_RETURN_RESERVE): a width that is a release with goal release evidence carries
    return_reserve.RELEASE_KEY like an open (the reserve counts it once it reached its gap).
With the switch off, runtime_class() returns exactly the classes above and none of this is used.
"""
from __future__ import annotations

import json

from agp.experiments.control_runtime.runtime import ControlRuntime

from agp_core import features

KEPT_EVENT = "ours_unstarted_kept"
KEPT_KEYS = ("target_confirmation", "grasp_epoch", "grasp_active", "post_grasp_motion_seen")
COUNTS = ("low_level_actions", "trajectory_samples", "settle_samples")


def _zero(value):
    return value is None or (type(value) in (int, float) and value == 0)


def neither_arm_moved(response):
    """True when the worker response shows that no sample was sent on either arm (see the module docstring)."""
    if not isinstance(response, dict) or response.get("motion_started") is not False:
        return False
    if response.get("motion_executed") is True or not all(_zero(response.get(key)) for key in COUNTS):
        return False
    for outcome in (response.get("arms") or {}).values():
        if not isinstance(outcome, dict):
            return False
        if outcome.get("motion_started") is True or outcome.get("motion_executed") is True:
            return False
        if not all(_zero(outcome.get(key)) for key in COUNTS):
            return False
    return True


class KeepUnstartedRuntime(ControlRuntime):
    """ControlRuntime that keeps the evidence of a step no arm started (see the module docstring)."""

    def _change_dual_revision(self, actions):
        row = self.db.execute("SELECT id FROM actions WHERE status='pending' ORDER BY rowid DESC LIMIT 1").fetchone()
        valid = sorted(int(r[0]) for r in self.db.execute("SELECT id FROM measurements WHERE valid=1"))
        self._ours_valid_before = {row[0]: valid} if row is not None else {}
        return super()._change_dual_revision(actions)

    def _complete_dual(self, action_id, payload, response):
        valid = getattr(self, "_ours_valid_before", {}).pop(action_id, None)
        if valid is None or not neither_arm_moved(response):
            return super()._complete_dual(action_id, payload, response)
        row = self.db.execute("SELECT prior_state FROM actions WHERE id=?", (action_id,)).fetchone()
        prior = json.loads(row[0]) if row and row[0] else {}
        result = super()._complete_dual(action_id, payload, response)
        restored = {}
        with self.db:
            again = [i for i in valid if not self.db.execute("SELECT valid FROM measurements WHERE id=?",
                                                             (i,)).fetchone()[0]]
            self.db.executemany("UPDATE measurements SET valid=1 WHERE id=?", [(i,) for i in valid])
            for arm, state in (self.state.get("arms") or {}).items():
                before = ((prior.get("arms") or {}).get(arm)) or {}
                kept = {key: before[key] for key in KEPT_KEYS if key in before}
                restored[arm] = {key: value for key, value in kept.items() if state.get(key) != value}
                state.update(kept)
            self._save_state()
            self._event(KEPT_EVENT, {"action_id": action_id, "measurements_kept": [f"m{i}" for i in valid],
                                     "measurements_restored": [f"m{i}" for i in again], "arms_restored": restored,
                                     "refusal": str(response.get("error") or response.get("stop_reason") or "")[:400]})
        return result


MANIPULATE = "manipulate"
MANIPULATE_PRECISION = "Manipulate runs at precision speed only: transit speed and home are refused"


def as_contact(actions):
    """The action map with every manipulate arm written as a contact without a measurement_id (see the module
    docstring); the other arms and the map itself unchanged. Returns the same object when no arm manipulates."""
    if not any(isinstance(spec, dict) and spec.get("intent") == MANIPULATE for spec in actions.values()):
        return actions
    return {arm: (dict(spec, intent="contact", measurement_id=None)
                  if isinstance(spec, dict) and spec.get("intent") == MANIPULATE else spec)
            for arm, spec in actions.items()}


class ManipulateMixin:
    """OURS_INTENT_MANIPULATE: the generic intent manipulate over a ControlRuntime class (module docstring)."""

    def _validate_action(self, command, args, intent):
        if intent != MANIPULATE:
            return super()._validate_action(command, args, intent)
        self._validate_command(command, args)
        if command == "home" or args.get("speed_profile") == "transit":
            raise ValueError(MANIPULATE_PRECISION)

    def _change_revision(self, command, args, intent):          # the single-arm runtime: as after a contact
        super()._change_revision(command, args, "contact" if intent == MANIPULATE else intent)

    def _change_dual_revision(self, actions):
        return super()._change_dual_revision(as_contact(actions))

    def _complete_dual(self, action_id, payload, response):
        mapped = as_contact(payload["actions"])
        if mapped is not payload["actions"]:
            payload = dict(payload, actions=mapped)
        return super()._complete_dual(action_id, payload, response)


class ManipulateRuntime(ManipulateMixin, ControlRuntime):
    """ControlRuntime with the intent manipulate (OURS_INTENT_MANIPULATE on, F-g v2's runtime half off)."""


class ManipulateKeepUnstartedRuntime(ManipulateMixin, KeepUnstartedRuntime):
    """KeepUnstartedRuntime with the intent manipulate (OURS_INTENT_MANIPULATE, OURS_GUARD_V2 and OURS_ARM_GUARD on)."""


def declaration_markers(arm, command, args, evidence, reserve, handover, width_release=False):
    """The declaration keys to add to one arm's dispatched args (module docstring); ValueError for a declaration the
    runtime refuses. width_release (grip width): this arm's width action is a release (GripWidthMixin), which counts
    here as an open."""
    from agp_core import handover as ho, return_reserve as rr
    evidence = evidence if isinstance(evidence, dict) else {}
    markers = {}
    if handover and ho.EVIDENCE_KEY in evidence:
        if not ho.names_other_gripper(arm, evidence[ho.EVIDENCE_KEY]):
            raise ValueError(ho.HANDOVER_TEXT.format(other=ho.other_arm(arm) if arm in ("left", "right") else "other"))
        markers[ho.ARGS_KEY] = True
    if reserve:
        for key in rr.FINAL_EVIDENCE:
            if key in evidence:
                value = evidence[key]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} needs concrete visible evidence")
                markers.setdefault(rr.FINAL_KEY, key)
        if (command == "gripper" and isinstance(args, dict)
                and (args.get("action") == "open" or (width_release and args.get("action") == "width"))
                and evidence.get("release_requirement") != "empty"
                and evidence.get("release_purpose", "goal") == "goal"):
            markers[rr.RELEASE_KEY] = True
    return markers


class DeclareMixin:
    """OURS_RETURN_RESERVE, OURS_HANDOVER: declarations as keys of the dispatched args (module docstring)."""

    def __init__(self, session, adapter):
        super().__init__(session, adapter)
        self._ours_declare = {"reserve": features.enabled("return_reserve"), "handover": features.enabled("handover")}
        self._ours_markers = None

    def _check_evidence(self, command, args, intent, measurement_id, evidence, arm=None):
        self._ours_markers = None
        super()._check_evidence(command, args, intent, measurement_id, evidence, arm)
        if getattr(self, "_gw_releases", None):              # grip width: a width that is a release (GripWidthMixin)
            self._ours_markers = declaration_markers(arm, command, args, evidence, self._ours_declare["reserve"],
                                                     self._ours_declare["handover"],
                                                     width_release=bool(self._gw_releases.pop(arm, False)))
            return
        self._ours_markers = declaration_markers(arm, command, args, evidence, self._ours_declare["reserve"],
                                                 self._ours_declare["handover"])

    def _resolve_args(self, command, args, measurement_id):
        resolved = super()._resolve_args(command, args, measurement_id)
        markers, self._ours_markers = self._ours_markers, None
        return dict(resolved, **markers) if markers else resolved


class DeclareRuntime(DeclareMixin, ControlRuntime):
    """ControlRuntime with the return-reserve and handover declarations."""


class DeclareKeepUnstartedRuntime(DeclareMixin, KeepUnstartedRuntime):
    """KeepUnstartedRuntime with the return-reserve and handover declarations."""


class DeclareManipulateRuntime(DeclareMixin, ManipulateRuntime):
    """ManipulateRuntime with the return-reserve and handover declarations."""


class DeclareManipulateKeepUnstartedRuntime(DeclareMixin, ManipulateKeepUnstartedRuntime):
    """ManipulateKeepUnstartedRuntime with the return-reserve and handover declarations."""


DECLARE = {ControlRuntime: DeclareRuntime, KeepUnstartedRuntime: DeclareKeepUnstartedRuntime,
           ManipulateRuntime: DeclareManipulateRuntime,
           ManipulateKeepUnstartedRuntime: DeclareManipulateKeepUnstartedRuntime}


# ----------------------------------------------------------------------------- grip width (OURS_GRIP_WIDTH)
GRIP_WIDTH_EVENT = "ours_grip_width"
CLOSE_COMMAND_GAP_MM = -19.193    # the pad gap of a close's finger target (-10 mm per finger + 0.807 mm): below every
                                  # requested gap, so every width widens after a close


class GripWidthMixin:
    """OURS_GRIP_WIDTH: the gripper action width (module docstring)."""

    def _validate_command(self, command, args):
        from agp_core import grip_width as gw
        if not gw.is_width(command, args):
            return super()._validate_command(command, args)
        unsupported = set(args) - {"action", gw.ARG}
        if unsupported:
            raise ValueError(f"Unsupported arguments for gripper: {', '.join(sorted(unsupported))}. "
                             f"Allowed arguments: action, {gw.ARG}")
        gw.width_mm(args)
        return super()._validate_command(command, {"action": "close"})     # the generic checks of a gripper action

    def _gw_commanded_gap_mm(self, arm, before_rowid=None):
        """The gap (mm) the arm's gripper is commanded to: that of its last executed gripper action in the journal
        (before the action of `before_rowid` when given); the open limit when there is none (the episode starts open)."""
        from agp_core import grip_width as gw
        full = gw.target_gap_m(1.0) * 1000.0
        for row in self.db.execute("SELECT rowid, payload, response, status FROM actions ORDER BY rowid DESC"):
            if (before_rowid is not None and row[0] >= before_rowid) or row["status"] != "done" or not row["response"]:
                continue
            payload, response = json.loads(row["payload"]), json.loads(row["response"])
            if "actions" in payload:
                spec = (payload.get("actions") or {}).get(arm)
                outcome = ((response.get("arms") or {}).get(arm)) or {}
            else:
                spec, outcome = (payload if arm is None else None), response
            if not isinstance(spec, dict) or spec.get("command") != "gripper":
                continue
            if outcome.get("motion_started", response.get("motion_started")) is False:
                continue                                       # not executed: the command did not change
            action = (spec.get("args") or {}).get("action")
            if action == "close":
                return CLOSE_COMMAND_GAP_MM
            if action == gw.ACTION:
                return gw.target_gap_m(gw.command_of_finger(gw.finger_target_m(float(spec["args"][gw.ARG]) / 1000.0)))\
                    * 1000.0
            return full
        return full

    def _gw_requested_gap_mm(self, args):
        from agp_core import grip_width as gw
        return gw.target_gap_m(gw.command_of_finger(gw.finger_target_m(gw.width_mm(args) / 1000.0))) * 1000.0

    def _gw_widening(self, arm, args, before_rowid=None):
        return self._gw_requested_gap_mm(args) > self._gw_commanded_gap_mm(arm, before_rowid) + 1e-6

    def _check_evidence(self, command, args, intent, measurement_id, evidence, arm=None):
        from agp_core import grip_width as gw
        if not gw.is_width(command, args):
            return super()._check_evidence(command, args, intent, measurement_id, evidence, arm)
        release = bool(self._arm_state(arm).get("grasp_active")) and self._gw_widening(arm, args)
        releases = getattr(self, "_gw_releases", None)
        if releases is None:
            releases = self._gw_releases = {}
        releases[arm] = release
        # a release is checked exactly as an open; any other width as a gripper action without release evidence
        return super()._check_evidence(command, dict(args, action="open" if release else gw.ACTION), intent,
                                       measurement_id, evidence, arm)

    def _gw_decide(self, arm, spec, outcome, response, prior_arm, before_rowid):
        from agp_core import grip_width as gw
        info = outcome.get("ours_grip_width") or {}
        started = outcome.get("motion_started", response.get("motion_started")) is not False
        executed = bool(outcome.get("ok")) and started and bool(info)
        holding = bool((prior_arm or {}).get("grasp_active"))
        widening = self._gw_widening(arm, spec["args"], before_rowid)
        state = info.get("state")
        close_like = executed and gw.counts_as_close(state, widening)
        released = (executed and holding and widening and not close_like and outcome.get("target_reached") is True
                    and outcome.get("motion_outcome") not in {"held_short", "checkpoint"})
        return {"requested_gap_mm": float(spec["args"][gw.ARG]), "state": state, "executed": executed,
                "holding_before": holding, "widening": widening, "release": holding and widening,
                "counts_as_close": close_like, "released": released}

    def _complete_dual(self, action_id, payload, response):
        from agp_core import grip_width as gw
        widths = {arm: spec for arm, spec in payload["actions"].items()
                  if isinstance(spec, dict) and gw.is_width(spec.get("command"), spec.get("args"))}
        if not widths:
            return super()._complete_dual(action_id, payload, response)
        row = self.db.execute("SELECT rowid, prior_state FROM actions WHERE id=?", (action_id,)).fetchone()
        prior = json.loads(row[1]) if row and row[1] else {}
        decisions, mapped = {}, dict(payload["actions"])
        for arm, spec in widths.items():
            outcome = ((response.get("arms") or {}).get(arm)) or {}
            decisions[arm] = self._gw_decide(arm, spec, outcome, response, (prior.get("arms") or {}).get(arm),
                                             row[0] if row else None)
            if decisions[arm]["released"]:
                mapped[arm] = dict(spec, args={"action": "open"})     # the base class ends the grasp as for an open
        result = super()._complete_dual(action_id, dict(payload, actions=mapped), response)
        with self.db:
            for arm, decision in decisions.items():
                if decision["counts_as_close"]:                       # as _change_dual_revision does for a close
                    state = self.state["arms"][arm]
                    state["grasp_epoch"] += 1
                    state.update(grasp_active=True, post_grasp_motion_seen=False, target_confirmation=None)
            self._save_state()
            self._event(GRIP_WIDTH_EVENT, {"action_id": action_id, "arms": decisions})
        return result

    def _complete(self, action_id, payload, response):
        from agp_core import grip_width as gw
        if "actions" in payload or not gw.is_width(payload.get("command"), payload.get("args")):
            return super()._complete(action_id, payload, response)
        row = self.db.execute("SELECT rowid, prior_state FROM actions WHERE id=?", (action_id,)).fetchone()
        prior = json.loads(row[1]) if row and row[1] else {}
        decision = self._gw_decide(None, payload, response, response, prior, row[0] if row else None)
        mapped = dict(payload, args={"action": "open"}) if decision["released"] else payload
        result = super()._complete(action_id, mapped, response)
        with self.db:
            if decision["counts_as_close"]:
                self.state["grasp_epoch"] += 1
                self.state.update(grasp_active=True, post_grasp_motion_seen=False, target_confirmation=None)
            self._save_state()
            self._event(GRIP_WIDTH_EVENT, {"action_id": action_id, "arms": {"single": decision}})
        return result

    def _recovery_guidance(self, payload, response, observation, arm=None):
        from agp_core import grip_width as gw
        if payload.get("command") == "gripper" and gw.is_width(payload.get("command"), payload.get("args")):
            info = (response or {}).get("ours_grip_width") or {}
            close_like = gw.counts_as_close(info.get("state"), bool(info.get("widening")))
            payload = dict(payload, args={"action": "close" if close_like else "open"})
        return super()._recovery_guidance(payload, response, observation, arm)


_GRIP_WIDTH_CLASSES = {}


def grip_width_class(chosen, declare):
    """GripWidthMixin in front of `chosen` (and DeclareMixin in front of that when `declare`), one class per pair."""
    key = (chosen, bool(declare))
    if key not in _GRIP_WIDTH_CLASSES:
        cls = type(f"GripWidth{chosen.__name__}", (GripWidthMixin, chosen),
                   {"__module__": __name__, "__doc__": f"{chosen.__name__} with the grip width action."})
        if declare:
            cls = type(f"Declare{cls.__name__}", (DeclareMixin, cls),
                       {"__module__": __name__, "__doc__": f"{cls.__name__} with the GB declarations."})
        _GRIP_WIDTH_CLASSES[key] = cls
    return _GRIP_WIDTH_CLASSES[key]


def runtime_class(environ=None):
    """KeepUnstartedRuntime when OURS_GUARD_V2 and OURS_ARM_GUARD are both on, or when OURS_KEEP_UNSTARTED is on,
    else ControlRuntime itself; with OURS_INTENT_MANIPULATE on, the same class with ManipulateMixin in front; with
    OURS_RETURN_RESERVE or OURS_HANDOVER on, that class with DeclareMixin in front. With OURS_GRIP_WIDTH on,
    GripWidthMixin in front of the class before DeclareMixin (grip_width_class)."""
    keep = ((features.enabled("guard_v2", environ) and features.enabled("arm_guard", environ))
            or features.enabled("keep_unstarted", environ))
    if features.enabled("intent_manipulate", environ):
        chosen = ManipulateKeepUnstartedRuntime if keep else ManipulateRuntime
    elif keep:
        chosen = KeepUnstartedRuntime
    else:
        chosen = ControlRuntime
    if features.enabled("grip_width", environ):                    # grip width
        return grip_width_class(chosen, features.enabled("return_reserve", environ)
                                or features.enabled("handover", environ))
    if features.enabled("return_reserve", environ) or features.enabled("handover", environ):
        return DECLARE[chosen]
    return chosen
