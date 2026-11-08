"""Observation-derived complements, optional and episode-local; no simulator imports.

Assertions in evidence are caller interpretations of CURRENT RGB, not ground truth.
Receipts attest motions, never object placement or task reward. Fail closed when
identity, count, contact or a handover cannot be measured. All calls end in an event.
"""

import ast
import copy
from fractions import Fraction
import hashlib
import json
import math
import re

from harness_policy.runtime import events as ev
from harness_policy.runtime.events import EpisodeEnded, Escalation, Event, ProgramFinished
from harness_policy.geometry import down_with_yaw
from harness_policy.runtime.session import ARMS
from skills.vla import _OWN_TOOLS, _WALL_BASKET_BINDING, _WALL_COLOUR_POUR


PLANAR = re.compile(r"T-shaped (?:block|object)[\s\S]*?(?:pad|align)|set square[\s\S]*?(?:align|row|push)", re.I)
TOOLS = re.compile(r"(?:xylophone|sweep[\s\S]*?(?:broom|dustpan)|(?:broom|dustpan)[\s\S]*?sweep)", re.I)
SWAP = re.compile(r"(?:swap|exchange)[\s\S]*?(?:positions?[\s\S]*?blocks?|blocks?[\s\S]*?(?:positions?|empty mat))", re.I)
MEMORY = re.compile(r"(?:remember|memorize|imitate|cover[\s\S]*?blocks?|conveyor|discard|kong|"
                    r"(?:swap|exchange)[\s\S]*?T-shaped)", re.I)
PRECISION = re.compile(r"(?:screw|nut\b|fasten|tubes?|charger|keyhole|peg|stacking toy|tower|"
                       r"(?:insert|plug)[\s\S]*?(?:slot|socket|hole|\bkey\b)|\bkey\b[\s\S]*?\block\b)", re.I)
TAIL = re.compile(r"(?:pen holder|(?:pack|place all (?:the )?objects)[\s\S]*?box|"
                  r"laptop[\s\S]*?headphones|headphones[\s\S]*?laptop)", re.I)


def instruction_route(instruction):
    """Capability routing from native instruction ONLY (no benchmark identifiers).

    Named 10 cm pickup is A/WALL, not a hard semantic read; it has no own-tool gate.
    """
    text = str(instruction or "")
    precision = bool(PRECISION.search(text))
    hard = not precision and bool(_OWN_TOOLS.search(text) or _WALL_BASKET_BINDING.search(text) or
                                  _WALL_COLOUR_POUR.search(text) or
                                  re.search(r"\bequation\b|\blargest\s+(?:possible\s+)?number\b", text, re.I))
    # Number-card reads are B-hard, not KFMem ownership. A visible block swap
    # protects its identity ledger separately; generic "swap" is not memory.
    # Explicit remembered work (especially swapping T-shaped BLOCKS) wins over
    # the generic block-swap complement, not just the T-shaped-object wording.
    memory = not hard and bool(MEMORY.search(text))
    identity_sensitive = not precision and not memory and bool(SWAP.search(text))
    own = hard or (not precision and not memory and bool(PLANAR.search(text) or TOOLS.search(text) or identity_sensitive))
    tail = not own and not memory and not precision and bool(TAIL.search(text))
    return {"owner": "B" if own else "C" if tail else "A", "hard": hard,
            "memory": memory, "identity_sensitive": identity_sensitive, "precision": precision, "tail": tail,
            "rule": "semantic_read_binding" if hard else "measured_complement" if own else
                    "static_tail" if tail else "wall_preserve"}


def _angle(degrees):
    return (float(degrees) + 180.0) % 360.0 - 180.0


def _finite(values, length=None):
    try:
        return (length is None or len(values) == length) and all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _arithmetic(text):
    """Small exact arithmetic grammar, never eval/calls/names/powers or OCR guesses."""
    tree = ast.parse(text.strip(), mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("expression too large")
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 1000000:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return a / b
        raise ValueError("unsupported arithmetic")
    return visit(tree.body)


def turn_report(session):
    state = session.__dict__.get("_complement")
    if not state:
        return None
    return {"text": "Complementary ownership/ledger (observations, NOT reward): " + json.dumps({
        "ownership": state["route"], "failure_counts": state["failures"],
        "last_event": state.get("last"), "ledger": state["ledger"], "cards": state["cards"],
        "swap": {k: v for k, v in state.get("swap", {}).items() if k != "receipts"},
        "motion_uncertain": state["uncertain"],
        "completion": state.get("completion", "unknown")}, default=str)}


class Complementary:
    def _complement_enabled(self):
        return ("complementary" in getattr(self, "skills", ()) and
                getattr(self, "_policy_span", "episode") == "cell" and set(self.policies) == {"wall-oss"})

    def _comp(self):
        if not self._complement_enabled():
            raise ValueError("complementary requires the explicit WALL cell skill configuration")
        if "_complement" not in self.__dict__:
            instruction = self._task_instruction()
            route = instruction_route(instruction)
            route.update(instruction=instruction, phase="own" if route["owner"] == "B" else "policy",
                         correction_count=0, wall_returns=0, transitions=[])
            if not instruction:
                route.update(phase="blocked", rule="native_instruction_unavailable")
            self.__dict__["_complement"] = {"route": route, "ledger": {}, "cards": [], "failures": {},
                                           "attempts": {}, "completion": "unknown", "uncertain": False}
        return self._complement

    def _evidence(self, evidence, *, fresh=True):
        return (isinstance(evidence, dict) and bool(evidence.get("source")) and
                isinstance(evidence.get("observed_step"), int) and
                (not fresh or evidence["observed_step"] == self.state.get("step")) and
                evidence.get("episode_id", self.episode_id) == self.episode_id)

    def _comp_result(self, skill, ok, reason=None, *, motion=False, next_hop=None, receipt=None, **extra):
        state = self._comp()
        detail = {"ok": bool(ok), "failure_reason": reason, "episode_id": self.episode_id,
                  "observed_step": self.state.get("step"), "motion_executed": motion,
                  "next_hop": next_hop or ("observe_postcondition" if ok else "remeasure_no_motion"),
                  "ownership": copy.deepcopy(state["route"]), "receipt": receipt, **extra}
        event = Event(ev.COMPLETED if ok else ev.SKILL_ERROR, skill, detail, self.state.get("step"))
        result = {"ok": bool(ok), "reason": reason, **detail, "event": event.to_dict()}
        state["last"] = {k: result[k] for k in ("ok", "failure_reason", "next_hop", "motion_executed", "observed_step")}
        self._events.append({"skill": skill, "event": event.to_dict(), "ticks": 0})
        return result

    def ownership_status(self):
        """Return frozen instruction, owner/phase, safety flags and transition receipts."""
        state = self._comp()
        route = state["route"]
        if not route["instruction"] and not state["uncertain"]:
            # An unavailable FIRST observation never grants WALL ownership. Re-read
            # explicitly before any motion; a known original instruction is immutable.
            instruction = self._task_instruction()
            if instruction:
                new = instruction_route(instruction)
                route.update(new, instruction=instruction, phase="own" if new["owner"] == "B" else "policy")
                if self.__dict__.get("_strategy"):
                    self._strategy.update(strategy=new["owner"], policy=None if new["owner"] == "B" else "wall-oss")
        return self._comp_result("ownership_status", bool(route["instruction"]),
                                 None if route["instruction"] else "native_instruction_unavailable",
                                 status=copy.deepcopy(route))

    def _complement_start(self, strategy, kind, policy=None):
        if str(strategy).upper() not in {"A", "B", "C"}:
            raise ValueError("strategy must be A, B or C")
        route = self._comp()["route"]
        if self.__dict__.get("_strategy"):
            return None
        self.__dict__["_strategy"] = {"strategy": route["owner"], "kind": str(kind),
                                      "policy": None if route["owner"] == "B" else "wall-oss",
                                      "step": self.state.get("step"), "rule": route["rule"]}
        self._comp_result("start", True, requested=str(strategy), selected=route["owner"])
        return None

    def _ownership_guard(self, skill):
        if not self._complement_enabled():
            return None
        state = self._comp()
        route = state["route"]
        if self.state.get("ended"):
            return "episode_ended"
        if state["completion"] == "failed":
            return "terminal_honest_failure"
        if not route["instruction"]:
            return "native_instruction_unavailable"
        if skill == "run_policy":
            if route["owner"] == "B" or route["phase"] != "policy":
                return "own_owner_requires_explicit_failure_handover"
            return None
        if state["uncertain"]:
            return "motion_uncertain_no_retry"
        if skill in {"home", "clear_view"} and state["completion"] != "observed":
            return "forbidden_home_recovery"
        if skill in self.still_executions:
            if route["memory"] and not self.__dict__.get("_comp_active"):
                return "memory_owner_forbids_time_advancing_watch"
            return None
        if state["completion"] == "observed" and self.__dict__.get("_wrapping_up"):
            return None
        if route["phase"] == "policy":
            return "policy_owner_no_own_motion"
        if route["owner"] == "C" and not self.__dict__.get("_comp_active"):
            return "C_requires_one_tail_correct_call"
        return None

    def _complement_policy_arguments(self, instruction, arms, reset, region, until, from_anywhere):
        route = self._comp()["route"]
        reason = self._ownership_guard("run_policy")
        if reason:
            self._comp_result("run_policy", False, reason, next_hop="failure_route")
            raise ValueError(reason)
        if instruction is not None or set(arms) != set(ARMS) or reset or region is not None or until is not None:
            self._comp_result("run_policy", False, "wall_requires_original_instruction_both_arms_no_reset")
            raise ValueError("WALL ownership requires original instruction, both arms and no reset/early predicate")
        if from_anywhere and not route["wall_returns"]:
            raise ValueError("from_anywhere requires an explicit safe failure handover")

    def _complement_policy_feedback(self, trace):
        """Keep actual WALL events/restarts observable, never invent restart counts."""
        route = self._comp()["route"]
        restarts = trace.event.detail.get("memory_restarts", [])
        if route["transitions"] and route["transitions"][-1].get("memory_restart_pending"):
            route["transitions"][-1].update(memory_restart_pending=False, memory_restarts=copy.deepcopy(restarts))
        self._comp_result("wall_cell", trace.ok, None if trace.ok else trace.event.kind,
                          motion=trace.ticks > 0, receipt=trace.to_dict(), memory_restarts=copy.deepcopy(restarts),
                          next_hop="host_ended" if trace.event.kind == ev.EPISODE_ENDED else
                          "continue_same_wall" if trace.ok else "inspect_receipt_no_retry")

    def safe_boundary(self, evidence):
        """Open C-own ONCE after fresh RGB proves empty hands, stable support,
        exposed reachable static remainder, measured target and no precision/memory.
        Required evidence booleans: empty_hands, stable, exposed, reachable, measured.
        Evidence includes source, observed_step and optional episode_id. It is an
        auditable visual assertion, not an automatic detector. No movement here."""
        state = self._comp()
        route = state["route"]
        checks = {key: isinstance(evidence, dict) and evidence.get(key) is True
                  for key in ("empty_hands", "stable", "exposed", "reachable", "measured")}
        allowed = (self._evidence(evidence) and all(checks.values()) and route["tail"] and
                   route["owner"] == "C" and route["phase"] == "policy" and
                   route["correction_count"] == 0 and not state["uncertain"] and
                   not self._eef_grips() and not route["memory"] and not route["precision"])
        if not allowed:
            return self._comp_result("safe_boundary", False, "boundary_rejected", checks=checks,
                                     next_hop="continue_same_wall")
        route["phase"] = "own"
        route["transitions"].append({"from": "C-policy", "to": "C-own", "evidence": copy.deepcopy(evidence)})
        return self._comp_result("safe_boundary", True, boundary=copy.deepcopy(evidence))

    def failure_route(self, reason, *, motion_executed=False, receipt=None, evidence=None, request=None):
        """Record a failure and return its next hop; NEVER execute/retry a motion.
        request='return_wall' is one explicit safe handover for non-hard, non-memory,
        non-identity-sensitive, non-precision B/C. request='finish' records honest
        failure unless a recoverable measurement or remaining direct work forbids it.
        Unknown motion needs inspect_receipt; a timeout is not proof nothing moved.
        Clearing uncertainty requires a fresh resolved_motion visual/receipt statement;
        the OLD action signature remains consumed, even after resolution."""
        state = self._comp()
        route = state["route"]
        reason = str(reason)
        counted = state.get("last_counted_press")
        if (counted and state["completion"] != "failed" and counted["observed_step"] == self.state.get("step") and
                receipt is not None and (receipt == counted or receipt == counted["receipt"]) and
                (("timeout" in reason.lower() and any(word in reason.lower() for word in
                    ("press", "confirm", "count", "downstroke"))) or
                 reason in {"count_uncertain", "confirm_uncertain"})):
            # The caller must supply the ORIGINAL successful receipt. Do not turn
            # its nested bounded contact TIMEOUT into a new terminal count failure.
            return self._comp_result("failure_route", True, "counted_press_already_complete",
                                     rejected_reason=reason, receipt=receipt, press_retry_allowed=False,
                                     next_hop="inspect_receipt_no_retry" if state["uncertain"] else
                                     "continue_after_counted_press")
        count = state["failures"][reason] = state["failures"].get(reason, 0) + 1
        before = route["owner"] + "-" + route["phase"]
        if state["uncertain"] and self._evidence(evidence) and evidence.get("resolved_motion") is True:
            state["uncertain"] = False
        irreversible = any(word in reason for word in ("irreversible", "spill", "count_uncertain", "confirm_uncertain",
                                                       "memory_identity_lost", "support_destroyed"))
        uncertain = motion_executed == "unknown" or (state["uncertain"] and not irreversible)
        staging_failure = reason in {"staging_required", "unreachable_staging", "staging_measurement_missing",
                                     "staging_overlaps_protected_object", "staging_measurement_unknown"}
        transfer = state.get("transfer_plan", {})
        direct_remaining = (transfer.get("observed_step") == self.state.get("step") and
                            any(state["ledger"].get(key, {}).get("status", "pending") == "pending"
                                for key in transfer.get("direct_keys", [])))
        # Do not translate a partial no-motion plan into an episode-wide gap.
        staging_context = staging_failure or (reason == "capability_missing" and transfer.get("blocked") and
                                               transfer.get("observed_step") == self.state.get("step"))
        measured_failure = (route["owner"] == "B" and not route["hard"] and reason in {
            "planar_pose_unknown", "planar_contact_unknown", "planar_contact_height_unsafe", "unreachable",
            "rotation_requires_visible_lever_contact", "directed_orientation_unknown",
            "stale_or_missing_measurement", "tool_contact_measurement_missing",
            "tool_contact_capability_missing", "invalid_contact_stroke", "invalid_contact_direction",
            "tool_contact_effector_below_table", "staging_measurement_unknown"})
        recoverable = staging_context or measured_failure
        budget_count = count
        if staging_context:
            # Renaming staging_required to unreachable_staging/capability_missing
            # cannot buy another two attempts. Ready directs still remain usable.
            budget_count = state["staging_failure_count"] = state.get("staging_failure_count", 0) + 1
        if irreversible:
            state["uncertain"] = state["uncertain"] or motion_executed == "unknown"
            hop = "finish_honest_failure"
        elif uncertain:
            state["uncertain"] = True
            hop = "inspect_receipt_no_retry"
        elif reason in {"policy_slice_end", "policy_timeout"} and route["phase"] == "policy":
            hop = "continue_same_wall"
        elif staging_context and direct_remaining and not motion_executed:
            hop = "continue_direct_transfers"
        elif recoverable and budget_count <= 2:
            hop = "change_action_no_silent_retry" if motion_executed else "remeasure_no_motion"
        elif (reason in {"placement_unverified", "swap_placement_unverified", "directed_orientation_unknown"} and
              self._swap_facing_recovery(evidence) is not None):
            # A newly measured small facing error permits ONE different corrective
            # action, even after two placement checks. This is not a motion retry.
            hop = "pick_observe_orient_reseat_then_confirm"
        elif request == "finish" or budget_count > 2:
            hop = "finish_honest_failure"
        elif "capability" in reason or "precision_pose" in reason or "pour_stream" in reason:
            hop = "finish_honest_failure" if route["hard"] or route["memory"] else "capability_preserve"
        elif motion_executed:
            hop = "change_action_no_silent_retry"
        else:
            hop = "remeasure_no_motion"
        if request == "return_wall" and hop != "finish_honest_failure":
            safe = (self._evidence(evidence) and all(evidence.get(k) is True for k in
                    ("empty_hands", "stable", "exposed", "reachable", "measured")) and
                    not self._eef_grips() and not state["uncertain"] and
                    route["owner"] in {"B", "C"} and not route["hard"] and not route["memory"] and
                    not route["identity_sensitive"] and not route["precision"] and route["wall_returns"] == 0)
            if safe:
                route.update(owner="A", phase="policy", wall_returns=1)
                route["transitions"].append({"from": before, "to": "A-policy", "reason": reason,
                                             "evidence": copy.deepcopy(evidence), "memory_restart_pending": True})
                # Discard actions computed before an own interval. Keep the KFMem client:
                # it will record a REAL frame-gap restart on next infer, not a fake count.
                self.__dict__.setdefault("_policy_rests", {}).pop("wall-oss", None)
                self.__dict__["_strategy"] = {"strategy": "A", "kind": reason, "policy": "wall-oss",
                                               "step": self.state.get("step"), "after": before}
                hop = "return_wall_original_instruction_both_arms"
            else:
                hop = "wall_fallback_denied"
        if route["owner"] == "C" and route["correction_count"] and request != "return_wall":
            hop = "inspect_receipt_no_retry" if uncertain else "finish_honest_failure"
        if hop == "finish_honest_failure":
            state["completion"] = "failed"
        result = self._comp_result("failure_route", False, reason, motion=motion_executed,
                                   next_hop=hop, receipt=receipt, failure_count=count, ownership_before=before,
                                   recovery_budget_count=budget_count, recovery_budget_exhausted=budget_count > 2,
                                   evidence=copy.deepcopy(evidence))
        if request == "finish" and hop == "finish_honest_failure":
            raise ProgramFinished("FAILED: " + reason)
        return result

    def complete_task(self, evidence):
        """Record completion only with fresh RGB: complete=True and safe_to_finish=True.
        All identity ledger items must have observed postconditions, all cards must be
        confirmed, and no motion may be uncertain. Allows final benchmark wrap-up;
        evidence may retain a held object for a named-pick task. No reward access."""
        state = self._comp()
        ready = (self._evidence(evidence) and evidence.get("complete") is True and
                 state["completion"] != "failed" and
                 evidence.get("safe_to_finish") is True and not state["uncertain"] and
                 all(item["status"] == "done" for item in state["ledger"].values()) and
                 all(card["status"] == "confirmed" for card in state["cards"]) and
                 (not state.get("swap") or state["swap"]["confirmed_legs"] == 3))
        if not ready:
            return self._comp_result("complete_task", False, "completion_unverified")
        state["completion"] = "observed"
        return self._comp_result("complete_task", True, evidence=copy.deepcopy(evidence))

    def _comp_motion(self, skill, signature, action, evidence):
        state = self._comp()
        route = state["route"]
        if not self._evidence(evidence):
            return self._comp_result(skill, False, "stale_or_missing_measurement")
        if state["uncertain"] or route["phase"] != "own" or state["completion"] == "failed":
            return self._comp_result(skill, False, "unsafe_owner_or_uncertain_motion", next_hop="failure_route")
        if route["owner"] == "C" and skill != "tail_correct":
            return self._comp_result(skill, False, "C_requires_one_tail_correct_call")
        # Even with a new observation, repeating an identical failed action is refused.
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
        if state["attempts"].get(key) in {"failed", "unknown", "pending"}:
            return self._comp_result(skill, False, "same_failed_action_no_retry", next_hop="change_action_no_silent_retry")
        state["attempts"][key] = "pending"
        start = self.state.get("step")
        previous_active = self.__dict__.get("_comp_active", False)
        self.__dict__["_comp_active"] = True
        try:
            raw = action()
        except (EpisodeEnded, ProgramFinished):
            state["attempts"][key] = "unknown"
            self._comp_result(skill, False, "episode_ended_during_motion", motion="unknown", next_hop="host_ended")
            raise
        except Exception as error:
            receipt = error.trace.to_dict() if isinstance(error, Escalation) else {
                "exception_type": type(error).__name__, "exception": str(error)}
            state["attempts"][key] = "unknown"
            state["uncertain"] = True
            return self._comp_result(skill, False, "motion_uncertain", motion="unknown", receipt=receipt,
                                     next_hop="inspect_receipt_no_retry")
        finally:
            self.__dict__["_comp_active"] = previous_active
        receipt = raw.to_dict() if hasattr(raw, "to_dict") else raw
        ok = bool(raw.get("ok")) if isinstance(raw, dict) else bool(getattr(raw, "ok", False))
        reason = None if ok else (raw.get("reason") if isinstance(raw, dict) else raw.event.kind)
        def uncertain_receipt(value):
            if isinstance(value, dict):
                return (value.get("kind") in {ev.TIMEOUT, ev.BLOCKED} or
                        value.get("motion_executed") == "unknown" or
                        value.get("reason") == "motion_uncertain" or
                        any(uncertain_receipt(v) for v in value.values()))
            if isinstance(value, list):
                return any(uncertain_receipt(v) for v in value)
            return False
        clearance_uncertain = (skill in {"press_card", "confirm_card", "swap_confirm"} and ok and
                               isinstance(receipt, dict) and receipt.get("arm_clear") is False)
        uncertain = clearance_uncertain or (not ok and uncertain_receipt(receipt))
        state["attempts"][key] = "completed" if ok else "unknown" if uncertain else "failed"
        state["uncertain"] = uncertain
        return self._comp_result(skill, ok, reason or (None if ok else "motion_failed"),
                                 motion="unknown" if uncertain else self.state.get("step") != start,
                                 receipt=receipt, next_hop="inspect_receipt_no_retry" if uncertain else
                                 "observe_postcondition" if ok else "change_action_no_silent_retry")

    def _comp_counted_motion(self, skill, signature, target, count, arm, evidence):
        """Exact contact count and arm clearance are separate; never auto-retry either."""
        self.__dict__["_counted_press"] = True
        try:
            result = self._comp_motion(skill, signature,
                                       lambda: self.press(target, times=count, arm=arm), evidence)
        finally:
            self.__dict__["_counted_press"] = False
        raw = result.get("receipt") or {}
        exact = (result["ok"] and type(raw.get("presses")) is int and raw["presses"] == count and
                 raw.get("count_complete", True) is True)
        if exact:
            result = self._comp_result(skill, True, motion=result["motion_executed"], receipt=raw,
                                       count_complete=True, press_retry_allowed=False,
                                       next_hop=result["next_hop"])
            self._comp()["last_counted_press"] = copy.deepcopy(result)
        return result, exact

    def _swap_facing_recovery(self, evidence):
        """Read-only eligibility: same visible identity, frozen leg, small measured yaw."""
        swap = self._comp().get("swap")
        if not swap or swap["confirmed_legs"] >= 3 or not self._evidence(evidence):
            return None
        index = swap["confirmed_legs"]
        leg = swap["legs"][index]
        item = self._comp()["ledger"].get(leg["key"], {})
        if (index in swap["attempted"] or index in swap.get("facing_corrections", []) or
                item.get("observed_step") != self.state.get("step") or
                item.get("measure_serial") != self.__dict__.get("_eef_measure_serial", 0) or
                self._comp()["uncertain"] or self._comp()["completion"] == "failed" or
                not all(evidence.get(k) is True for k in
                    ("identity_unique", "at_destination", "released", "directed_axis", "orientation_measured", "path_clear")) or
                evidence.get("facing_correct") is not False or evidence.get("key") != leg["key"] or
                not _finite([evidence.get("yaw_deg")]) or
                not 0 < abs(_angle(leg["yaw_deg"] - evidence["yaw_deg"])) <= 30):
            return None
        return leg

    def identity_ledger(self, entries=None, evidence=None):
        """Bind stable keys (not oN) to visible identities and directed yaw.
        entries=[{key, object: current oN, identity: RGB description, destination,
        yaw_deg: directed facing or None}]. Evidence identity_unique=True required.
        Rebinding retains identity/destination; ambiguous aliases cannot auto-match.
        A ledger survives cells, not episodes. Calling without entries is read-only."""
        ledger = self._comp()["ledger"]
        if entries is None:
            return self._comp_result("identity_ledger", True, ledger=copy.deepcopy(ledger))
        if not self._evidence(evidence) or evidence.get("identity_unique") is not True:
            return self._comp_result("identity_ledger", False, "identity_ambiguous")
        if self.__dict__.get("_eef_measured", {}).get("step") != self.state.get("step"):
            return self._comp_result("identity_ledger", False, "fresh_locate_required")
        updated = copy.deepcopy(ledger)
        for entry in entries:
            key, identity = str(entry["key"]), str(entry["identity"])
            obj = self._eef_object(entry["object"])
            if key in updated and (updated[key]["identity"] != identity or
                                   updated[key]["destination"] != entry.get("destination")):
                return self._comp_result("identity_ledger", False, "identity_or_destination_changed")
            if obj.get("aliases") and evidence.get("aliases_resolved") is not True:
                return self._comp_result("identity_ledger", False, "identity_ambiguous")
            if any(other != key and item["object"] == obj["id"] and item.get("status") != "done" and
                   item.get("measure_serial") == self.__dict__.get("_eef_measure_serial", 0)
                   for other, item in updated.items()):
                return self._comp_result("identity_ledger", False, "identity_ambiguous")
            old = updated.get(key, {})
            yaw = entry.get("yaw_deg")
            if yaw is not None and (not _finite([yaw]) or evidence.get("directed_axis") is not True):
                return self._comp_result("identity_ledger", False, "directed_orientation_unknown")
            updated[key] = {**old, "key": key, "identity": identity, "object": obj["id"],
                            "destination": entry.get("destination"), "yaw_deg": yaw,
                            "destination_object": entry.get("destination_object", entry.get("destination")),
                            "position": list(obj["position"]), "status": old.get("status", "pending"),
                            "observed_step": evidence["observed_step"], "evidence": copy.deepcopy(evidence),
                            "measure_serial": self.__dict__.get("_eef_measure_serial", 0),
                            "last_transfer_serial": old.get("last_transfer_serial", -1),
                            "last_receipt": old.get("last_receipt")}
        self._comp()["ledger"] = updated
        return self._comp_result("identity_ledger", True, ledger=copy.deepcopy(updated))

    def transfer_plan(self, entries, staging=None, evidence=None):
        """Plan direct same-arm work FIRST, grouped by destination/category.
        entries=[{key, source:oN, destination:oN, category}]. Arms must reach both
        straight-down (unknown reach is not permission). staging is a measured
        empty table point {x,y}; only with staging_clear=True. Explicit two-leg
        staging, never a blind gripper handoff. Partial plans retain ready legs and
        structured blocked rows: staging failure is recoverable, not capability_missing.
        Execute ready direct work before remeasuring staging for the remainder.
        No movement in this planner."""
        if not self._evidence(evidence):
            return self._comp_result("transfer_plan", False, "stale_or_missing_measurement")
        plan, blocked = [], []
        staging_reach = None
        for entry in entries:
            source, goal = self._eef_object(entry["source"]), self._eef_object(entry["destination"])
            a = {arm for arm in ARMS if source.get("reach", {}).get(arm) == "down"}
            b = {arm for arm in ARMS if goal.get("reach", {}).get(arm) == "down"}
            shared = sorted(a & b)
            row = {**entry, "legs": []}
            if shared:
                row["legs"] = [{"arm": shared[0], "source": source["id"], "into": goal["id"]}]
            elif staging and self._evidence(evidence) and evidence.get("staging_clear") is True and _finite(
                    [staging.get("x"), staging.get("y")], 2):
                if staging_reach is None:
                    staging_reach = {arm for arm, reach in self._eef_reach(staging["x"], staging["y"]).items()
                                     if reach == "down"}
                c = staging_reach
                if not a & c or not b & c:
                    blocked.append({**entry, "reason": "unreachable_staging",
                                    "source_arms": sorted(a), "destination_arms": sorted(b),
                                    "staging_arms": sorted(c)})
                    continue
                row["legs"] = [{"arm": sorted(a & c)[0], "source": source["id"], "staging": True, **staging},
                               {"arm": sorted(b & c)[0], "source": "RELOCATE_SAME_IDENTITY", "into": goal["id"]}]
            else:
                blocked.append({**entry, "reason": "staging_required",
                                "source_arms": sorted(a), "destination_arms": sorted(b)})
                continue
            plan.append(row)
        plan.sort(key=lambda row: (len(row["legs"]) != 1, str(row.get("category", "")), str(row["destination"])))
        direct_keys = [row["key"] for row in plan if len(row["legs"]) == 1]
        self._comp()["transfer_plan"] = {"observed_step": self.state.get("step"),
                                         "direct_keys": direct_keys, "blocked": copy.deepcopy(blocked)}
        return self._comp_result("transfer_plan", not blocked, blocked[0]["reason"] if blocked else None,
                                 plan=plan, blocked=blocked, recoverable=bool(blocked),
                                 next_hop="continue_direct_transfers" if direct_keys else
                                 "remeasure_no_motion" if blocked else "execute_planned_transfers",
                                 completion="partial_plan_only_not_motion" if blocked else "plan_only_not_motion")

    def equation_solve(self, expression, candidates, evidence):
        """No motion. expression='? + 2 = 5'; candidates=[{key, token, value}].
        value is an integer or one of +,-,*,/; token is the caller's visible answer
        label. Fresh read_confident/identity_unique evidence must repeat expression
        and candidates exactly. One '?' and exactly one satisfying candidate only.
        Exact Python Fraction check is a reasoning receipt, NOT OCR verification."""
        if (not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("read_confident", "identity_unique")) or evidence.get("ambiguous") or
                evidence.get("expression") != expression or evidence.get("candidates") != candidates):
            return self._comp_result("equation_solve", False, "equation_read_ambiguous")
        try:
            if not isinstance(expression, str) or len(expression) > 160 or expression.count("?") != 1 or expression.count("=") != 1:
                raise ValueError("one missing token and equality required")
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 20:
                raise ValueError("bounded candidates required")
            keys, matches, checks = set(), [], []
            normalized = expression.translate(str.maketrans({"×": "*", "÷": "/", "−": "-"}))
            for candidate in candidates:
                key, token, value = candidate["key"], candidate["token"], candidate["value"]
                if not isinstance(key, str) or not key or key in keys or not isinstance(token, str) or not token:
                    raise ValueError("unique visible keys/tokens required")
                keys.add(key)
                if not ((type(value) is int and 0 <= value <= 9) or
                        (isinstance(value, str) and value in ("+", "-", "*", "/"))):
                    raise ValueError("candidate must be digit or operator")
                if token.strip().translate(str.maketrans({"×": "*", "÷": "/", "−": "-"})) != str(value):
                    raise ValueError("visible token/value disagree")
                replaced = normalized.replace("?", str(value))
                left, right = replaced.split("=")
                try:
                    a, b = _arithmetic(left), _arithmetic(right)
                    check = {"key": key, "left": str(a), "right": str(b), "equal": a == b}
                except ZeroDivisionError:
                    check = {"key": key, "equal": False, "reason": "division_by_zero"}
                checks.append(check)
                if check["equal"]:
                    matches.append(candidate)
        except (ValueError, SyntaxError, KeyError, TypeError, RecursionError) as error:
            return self._comp_result("equation_solve", False, "equation_invalid", receipt={"error": str(error)})
        receipt = {"python_check": "restricted AST + exact fractions.Fraction", "checks": checks,
                   "evidence": copy.deepcopy(evidence)}
        if len(matches) != 1:
            return self._comp_result("equation_solve", False, "equation_no_unique_answer", receipt=receipt)
        answer = copy.deepcopy(matches[0])
        return self._comp_result("equation_solve", True, answer_token=answer["token"], answer=answer,
                                 receipt=receipt, next_hop="bind_answer_identity_then_place_on_pad")

    def largest_order(self, digits, evidence):
        """No motion. digits=[{key, value:0..9}]; preserve duplicate-digit identities.
        Fresh read_confident/identity_unique evidence repeats digits exactly. Returns
        ordered keys and values; directed upright facing still needs RGB measurement."""
        if (not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("read_confident", "identity_unique")) or evidence.get("ambiguous") or
                evidence.get("digits") != digits):
            return self._comp_result("largest_order", False, "digit_read_ambiguous")
        try:
            if not isinstance(digits, list) or not 1 <= len(digits) <= 20:
                raise ValueError("bounded digit list required")
            keys = [row["key"] for row in digits]
            if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
                raise ValueError("unique identity keys required")
            if any(type(row["value"]) is not int or not 0 <= row["value"] <= 9 for row in digits):
                raise ValueError("integer digits required")
            ordered = sorted(digits, key=lambda row: -row["value"])
        except (ValueError, KeyError, TypeError) as error:
            return self._comp_result("largest_order", False, "digits_invalid", receipt={"error": str(error)})
        return self._comp_result("largest_order", True, ordered_keys=[row["key"] for row in ordered],
                                 ordered_values=[row["value"] for row in ordered],
                                 receipt={"python_check": "stable descending integer sort", "evidence": copy.deepcopy(evidence)},
                                 next_hop="bind_digit_identities_then_place_on_pad")

    def read_cards(self, evidence):
        """No motion. evidence.cards=[{side:'left'/'middle',value:0..20,target:oN}],
        blue_confirm=oN, read_confident=True,target_unique=True. Freeze counts/order
        episode-locally; fresh readings may update IDs but never counts after a press.
        Returns press_card arguments in left/middle order, blue confirm separately."""
        if (not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("read_confident", "target_unique")) or evidence.get("ambiguous")):
            return self._comp_result("read_cards", False, "card_read_uncertain")
        try:
            rows, blue = evidence["cards"], evidence["blue_confirm"]
            if not isinstance(rows, list) or len(rows) != 2 or {r["side"] for r in rows} != {"left", "middle"}:
                raise ValueError("left and middle readings required")
            ordered = sorted(rows, key=lambda row: row["side"] != "left")
            targets = [row["target"] for row in ordered] + [blue]
            if any(not isinstance(t, str) or not t for t in targets) or len(set(targets)) != 3:
                raise ValueError("distinct red/red/blue targets required")
            if any(type(row["value"]) is not int or not 0 <= row["value"] <= 20 for row in ordered):
                raise ValueError("bounded integer counts required")
            readings = {"cards": [{"index": i, "side": r["side"], "count": r["value"], "target": r["target"]}
                                   for i, r in enumerate(ordered)], "blue_confirm": blue}
        except (KeyError, TypeError, ValueError) as error:
            return self._comp_result("read_cards", False, "card_read_uncertain", receipt={"error": str(error)})
        state = self._comp()
        previous = state.get("card_readings")
        if previous and state["cards"] and [r["count"] for r in previous["cards"]] != [r["count"] for r in readings["cards"]]:
            return self._comp_result("read_cards", False, "card_read_changed_after_motion")
        state["card_readings"] = copy.deepcopy(readings)
        return self._comp_result("read_cards", True, **readings, receipt={"evidence": copy.deepcopy(evidence)},
                                 next_hop="press_card_left_then_confirm_then_middle")

    def place_on_pad(self, key, arm, pad, *, yaw_deg=None, evidence=None):
        """One ledger-bound pick/release onto a measured pad (not a container/hole).
        Fresh identity_unique/pad_measured/path_clear evidence and same-arm down
        reach required. For yaw, FIRST pick, observe/rebind held identity; supply
        held_identity=True, orientation_measured=True and current yaw_deg. Calls
        orient_held for one <=30deg correction preserving tilt. Never auto-observe
        or infer orientation after a grasp. Release is placed_unverified; re-locate,
        rebind then placement_check needs inside=True,seated=True and facing if set."""
        item = self._comp()["ledger"].get(str(key))
        if not item or item["status"] == "done":
            return self._comp_result("place_on_pad", False, "identity_missing_or_already_done")
        if item["status"] in {"placed_unverified", "staged_unverified"}:
            return self._comp_result("place_on_pad", False, "placement_unverified_no_replay")
        if (not self._evidence(evidence) or item["observed_step"] != evidence.get("observed_step") or
                item["measure_serial"] != self.__dict__.get("_eef_measure_serial", 0) or
                not all(evidence.get(k) is True for k in ("identity_unique", "pad_measured", "path_clear"))):
            return self._comp_result("place_on_pad", False, "pad_measurement_missing")
        if arm not in ARMS or pad != item["destination_object"] or self._comp()["route"]["precision"]:
            return self._comp_result("place_on_pad", False, "pad_destination_or_precision_rejected")
        source, goal = self._eef_object(item["object"]), self._eef_object(pad)
        held = source.get("held_by") == arm and arm in self._eef_grips() and evidence.get("held_identity") is True
        if (goal.get("container") or goal.get("held_by") or goal.get("reach", {}).get(arm) != "down" or
                (not held and (source.get("held_by") or self._eef_grips() or source.get("reach", {}).get(arm) != "down"))):
            return self._comp_result("place_on_pad", False, "pad_reach_or_holding_unknown")
        if yaw_deg is not None and (not held or item["yaw_deg"] is None or
                not _finite([yaw_deg, evidence.get("yaw_deg")]) or evidence.get("orientation_measured") is not True or
                abs(_angle(float(yaw_deg) - float(item["yaw_deg"]))) > 1e-6 or
                abs(_angle(float(yaw_deg) - float(evidence["yaw_deg"]))) > 30):
            return self._comp_result("place_on_pad", False, "observe_held_orientation_before_place",
                                     next_hop="pick_then_reobserve_rebind_held_identity")
        def action():
            pick = {"ok": True, "already_held": True} if held else self.eef_pick(arm, item["object"])
            if not pick["ok"]:
                return {"ok": False, "reason": "grasp_failed", "pick": pick}
            orient = self.orient_held(key, arm, yaw_deg, evidence) if yaw_deg is not None else None
            if orient is not None and not orient["ok"]:
                return {"ok": False, "reason": orient["reason"], "pick": pick, "orient": orient}
            release = self.eef_release(arm, onto=pad, withdraw=False)
            return {"ok": release["ok"], "reason": release.get("reason"), "pick": pick, "orient": orient, "release": release}
        result = self._comp_motion("place_on_pad", ["pad", key, arm, pad, yaw_deg, item["position"]], action, evidence)
        if result["receipt"] is not None:
            item.update(status="placed_unverified" if result["ok"] else "failed", placement_kind="pad",
                        staging=False, last_receipt=result["receipt"], last_transfer_serial=item["measure_serial"])
        return result

    def transfer_item(self, key, arm, *, into=None, x=None, y=None, staging=False, grasp="auto", evidence=None):
        """One ledger-bound pick/release; confirms grasp before release. Compact
        withdrawal saves a redundant park; remeasure before next item. Doesn't mark
        done: placement_check requires fresh RGB inside/below-rim/facing evidence.
        A staging leg uses x,y and must be visually re-bound before the next leg."""
        item = self._comp()["ledger"].get(str(key))
        if not item or item["status"] == "done":
            return self._comp_result("transfer_item", False, "identity_missing_or_already_done")
        if item["status"] in {"placed_unverified", "staged_unverified"}:
            return self._comp_result("transfer_item", False, "placement_unverified_no_replay",
                                     next_hop="remeasure_no_motion")
        if not self._evidence(evidence) or item["observed_step"] != evidence.get("observed_step"):
            return self._comp_result("transfer_item", False, "rebind_identity_from_fresh_locate")
        if into is not None and into != item["destination_object"]:
            return self._comp_result("transfer_item", False, "wrong_destination")
        if staging and (into is not None or not _finite([x,y]) or evidence.get("staging_clear") is not True):
            return self._comp_result("transfer_item", False, "staging_measurement_missing")
        signature = ["transfer", key, arm, into, x, y, staging, grasp, item["position"]]
        def action():
            pick = self.eef_pick(arm, item["object"], grasp=grasp)
            if not pick["ok"]:
                return {"ok": False, "reason": "grasp_failed", "pick": pick}
            release = self.eef_release(arm, into=into, x=x, y=y, withdraw=False)
            return {"ok": release["ok"], "reason": release.get("reason"), "pick": pick, "release": release}
        result = self._comp_motion("transfer_item", signature, action, evidence)
        item.update(status=("staged_unverified" if staging else "placed_unverified") if result["ok"] else "failed",
                    last_receipt=result["receipt"],
                    last_transfer_serial=item["measure_serial"])
        if result["ok"]:
            item["placement_kind"] = "container" if into is not None else "surface"
            item["staging"] = bool(staging)
        return result

    def placement_check(self, key, evidence):
        """Commit ledger ONLY after fresh identity_unique=True, destination_correct=True,
        inside=True, below_rim=True (container) / seated=True (surface) and holding=False.
        Directed facing entries also need facing_correct=True. Caller must rebind the
        identity from new locate before checking; no scene position fabricated by release
        or its ok flag can verify a placement."""
        item = self._comp()["ledger"].get(str(key))
        if not item:
            return self._comp_result("placement_check", False, "identity_missing")
        if not isinstance(evidence, dict):
            return self._comp_result("placement_check", False, "placement_unverified")
        placement = ((evidence.get("inside") is True and evidence.get("seated") is True)
                     if item.get("placement_kind") == "pad" else
                     (evidence.get("inside") is True and evidence.get("below_rim") is True)
                     if item.get("placement_kind") == "container" else evidence.get("seated") is True or
                     (evidence.get("inside") is True and evidence.get("below_rim") is True))
        valid = (self._evidence(evidence) and item["observed_step"] == evidence.get("observed_step") and
                 item["measure_serial"] > item["last_transfer_serial"] and
                 evidence.get("identity_unique") is True and
                 (evidence.get("staging_matches") is True if item.get("staging") else
                  evidence.get("destination_correct") is True) and
                 evidence.get("holding") is False and placement and
                 (item["yaw_deg"] is None or evidence.get("facing_correct") is True))
        if not valid:
            return self._comp_result("placement_check", False, "placement_unverified")
        item.update(status="staged" if item.get("staging") else "done", placement_evidence=copy.deepcopy(evidence))
        return self._comp_result("placement_check", True, key=key, completion="observed_not_reward")

    def press_card(self, index, count, target, *, arm=None, evidence=None):
        """Press one read card's red button count times; no automatic confirm/retry.
        index is 0 then 1, count is nonnegative integer <=20 (bounded safety cap, not
        hidden card labels). Fresh evidence read_confident=True and target_unique=True.
        Physical down/up receipts are NOT evaluator counts; confirm_card requires
        visible count_matches=True. Failed/uncertain press poisons this card forever."""
        cards = self._comp()["cards"]
        if type(index) is not int or index != len(cards) or index not in (0, 1) or (cards and cards[-1]["status"] != "confirmed"):
            return self._comp_result("press_card", False, "card_order_or_already_attempted", next_hop="finish_honest_failure")
        if self._comp()["uncertain"]:
            return self._comp_result("press_card", False, "motion_uncertain_no_retry",
                                     next_hop="inspect_receipt_no_retry")
        if type(count) is not int or not 0 <= count <= 20 or not self._evidence(evidence) or not all(
                evidence.get(k) is True for k in ("read_confident", "target_unique")):
            return self._comp_result("press_card", False, "card_read_uncertain")
        readings = self._comp().get("card_readings")
        if readings and (count != readings["cards"][index]["count"] or target != readings["cards"][index]["target"]):
            return self._comp_result("press_card", False, "card_read_binding_mismatch")
        card = {"index": index, "count": count, "target": target, "status": "attempted", "read": copy.deepcopy(evidence)}
        cards.append(card)  # BEFORE motion: replaying the cell cannot press twice.
        if count == 0:
            card["status"] = "pressed_unconfirmed"
            return self._comp_result("press_card", True, receipt={"ok": True, "presses": 0}, motion=False)
        result, exact = self._comp_counted_motion("press_card", ["card", index, count, target, arm],
                                                  target, count, arm, evidence)
        raw = result.get("receipt") or {}
        card.update(status="pressed_unconfirmed" if exact else "uncertain", receipt=raw)
        if not exact:
            self._comp()["uncertain"] = True
            return self.failure_route("count_uncertain", motion_executed="unknown", receipt=raw)
        return result

    def confirm_card(self, index, target, *, arm=None, evidence=None):
        """After that card's count_matches=True and target_unique=True in fresh RGB,
        press blue exactly ONCE. No retry, even if receipt or visible confirmation is
        uncertain. Before second card, call again with fresh confirmed=True to commit
        the first confirmation without moving. Failed confirm -> honest failure."""
        cards = self._comp()["cards"]
        if type(index) is not int or not 0 <= index < len(cards):
            return self._comp_result("confirm_card", False, "card_missing")
        card = cards[index]
        if not self._evidence(evidence):
            return self._comp_result("confirm_card", False, "stale_or_missing_measurement")
        readings = self._comp().get("card_readings")
        if readings and target != readings["blue_confirm"]:
            return self._comp_result("confirm_card", False, "blue_confirm_binding_mismatch")
        if card["status"] == "confirm_sent":
            if evidence.get("confirmed") is True:
                card["status"] = "confirmed"
                return self._comp_result("confirm_card", True, confirmation="observed", motion=False)
            return self.failure_route("confirm_uncertain", receipt=card.get("confirm_receipt"))
        if card["status"] != "pressed_unconfirmed" or not all(evidence.get(k) is True for k in ("count_matches", "target_unique")):
            return self._comp_result("confirm_card", False, "count_unverified_no_confirm")
        if self._comp()["uncertain"]:
            return self._comp_result("confirm_card", False, "motion_uncertain_no_retry",
                                     next_hop="inspect_receipt_no_retry")
        card["status"] = "confirm_attempted"
        result, exact = self._comp_counted_motion("confirm_card", ["confirm", index, target, arm],
                                                  target, 1, arm, evidence)
        card["confirm_receipt"] = result["receipt"]
        card["status"] = "confirm_sent" if exact else "uncertain"
        if not exact:
            return self.failure_route("confirm_uncertain", motion_executed="unknown", receipt=result["receipt"])
        return result

    def planar_push(self, arm, object, goal, *, evidence=None):
        """ONE measured <=3cm planar translation or tangential rotation contact.
        goal={x,y,yaw_deg}; evidence directed_axis=True, yaw_deg (source's directed
        T axis), contact=[x,y,z] (visible boundary contact), approach_clear=True.
        Contact stays in the table plane; never pick/lift the T. Use fresh locate and
        image error after EACH segment. No guessed bounding-circle contact. For align
        with set square, use tool_contact, not naked jaws on the blocks."""
        obj = self._eef_object(object)
        if re.search(r"set square", self._comp()["route"]["instruction"], re.I):
            return self._comp_result("planar_push", False, "align_requires_tool_contact")
        if not self._evidence(evidence) or not all(evidence.get(k) is True for k in ("directed_axis", "approach_clear")):
            return self._comp_result("planar_push", False, "planar_pose_unknown")
        contact = evidence.get("contact")
        if not _finite(contact, 3) or not _finite([goal.get("x"), goal.get("y"), goal.get("yaw_deg"), evidence.get("yaw_deg")]):
            return self._comp_result("planar_push", False, "planar_contact_unknown")
        if not self.table_z + 0.005 <= contact[2] <= self.table_z + 0.04:
            return self._comp_result("planar_push", False, "planar_contact_height_unsafe")
        if obj.get("reach", {}).get(arm) != "down":
            return self._comp_result("planar_push", False, "unreachable")
        ox, oy = obj["position"][:2]
        dx, dy = goal["x"] - ox, goal["y"] - oy
        yaw_error = _angle(goal["yaw_deg"] - evidence["yaw_deg"])
        errors = self._comp().setdefault("planar_errors", {})
        previous = errors.get(obj["name"])
        xy_error = math.hypot(dx, dy)
        stagnant = (previous["stagnant"] + 1 if previous and
                    xy_error >= previous["xy"] - 0.002 and abs(yaw_error) >= previous["yaw"] - 2 else 0)
        errors[obj["name"]] = {"xy": xy_error, "yaw": abs(yaw_error), "stagnant": stagnant}
        if stagnant >= 2:
            return self._comp_result("planar_push", False, "planar_no_progress", next_hop="failure_route")
        if abs(yaw_error) > 8.0:
            rx, ry = contact[0] - ox, contact[1] - oy
            radius = math.hypot(rx, ry)
            if radius < 0.015:
                return self._comp_result("planar_push", False, "rotation_requires_visible_lever_contact")
            sign = 1.0 if yaw_error > 0 else -1.0
            ux, uy = -ry / radius * sign, rx / radius * sign
            distance = min(0.025, radius * math.radians(abs(yaw_error)))
            mode = "tangential_rotation"
        else:
            distance = math.hypot(dx, dy)
            if distance < 0.004:
                return self._comp_result("planar_push", True, completion="within_local_deadband_not_task_success",
                                         error={"xy_m": distance, "yaw_deg": yaw_error})
            ux, uy = dx / distance, dy / distance
            distance = min(0.03, distance)
            mode = "translation"
        orientation = down_with_yaw(math.atan2(uy, ux) + math.pi / 2)
        start = [contact[0] - ux * 0.02, contact[1] - uy * 0.02, contact[2]]
        end = [contact[0] + ux * distance, contact[1] + uy * distance, contact[2]]
        def action():
            for method, args, kwargs in [
                (self.close_gripper, (arm,), {}),
                (self.move_eef, (arm, self.link6_for_tip(start[:2] + [contact[2] + 0.06], orientation)), {"speed": 0.4}),
                (self.move_eef, (arm, self.link6_for_tip(start, orientation)), {"speed": 0.08}),
                (self.move_eef, (arm, self.link6_for_tip(end, orientation)), {"speed": 0.04, "max_seconds": 2.0})]:
                trace = method(*args, **kwargs)
                if not trace.ok:
                    return trace
            # Retract the effector, not the object; no invented scene update.
            return self.move_eef(arm, self.link6_for_tip(end[:2] + [end[2] + 0.06], orientation), speed=0.3)
        result = self._comp_motion("planar_push", [object, arm, contact, end, mode], action, evidence)
        result.update(mode=mode, error_before={"xy_m": math.hypot(dx, dy), "yaw_deg": yaw_error},
                      commanded_contact_end=end, completion="motion_only_remeasure_SE2")
        return result

    def tool_contact(self, arm, tool, target, *, tip_offset, direction, distance=0.02, evidence=None):
        """ONE bounded held-tool contact stroke. target/offset/direction are xyz in
        world frame, measured NOW. tip_offset is actual tool head/contact edge minus
        jaws, NOT an arbitrary pivot. Requires tool_held, contact_visible, offset_measured,
        path_clear=True. Use for set-square alignment, broom or xylophone. Tool must
        be in eef grip ledger; no guessed 3D pose, millimetre assembly or blind tilt.
        Stroke <=3cm, normalised direction, max_seconds=2; observe after every stroke."""
        if not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("tool_held", "contact_visible", "offset_measured", "path_clear")):
            return self._comp_result("tool_contact", False, "tool_contact_measurement_missing")
        obj = self._eef_object(tool)
        if arm not in self._eef_grips() or obj.get("held_by") != arm:
            return self._comp_result("tool_contact", False, "tool_contact_lost")
        if not all(_finite(v, 3) for v in (target, tip_offset, direction)) or not _finite([distance]) or not 0 < distance <= 0.03:
            return self._comp_result("tool_contact", False, "invalid_contact_stroke")
        norm = math.sqrt(sum(v*v for v in direction))
        if norm < 1e-6:
            return self._comp_result("tool_contact", False, "invalid_contact_direction")
        tip = [target[i] - tip_offset[i] for i in range(3)]
        end = [tip[i] + direction[i] / norm * distance for i in range(3)]
        if min(tip[2], end[2]) < self.table_z + 0.01:
            return self._comp_result("tool_contact", False, "tool_contact_effector_below_table")
        orientation = self.state[f"{arm.split('_')[0]}_ee_pose"][3:]
        def action():
            approach = self.move_eef(arm, self.link6_for_tip(tip, orientation), speed=0.08, max_seconds=2)
            if not approach.ok:
                return approach
            return self.move_eef(arm, self.link6_for_tip(end, orientation), speed=0.04, max_seconds=2)
        return self._comp_motion("tool_contact", [arm, tool, target, tip_offset, direction, distance], action, evidence)

    def orient_held(self, key, arm, yaw_deg, evidence=None):
        """One <=30deg yaw correction of a held identity, based on directed observed
        facing. Not an insertion/tilt primitive. Requires orientation_measured/path_clear,
        updates no object truth; fresh facing confirmation required after release.
        For a swap: same freshly rebound held_identity/identity_unique/directed_axis,
        frozen current-leg yaw, and ONE correction consumed before motion (even failure)."""
        item = self._comp()["ledger"].get(str(key))
        if not item or not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("orientation_measured", "path_clear")) or not _finite([yaw_deg, evidence.get("yaw_deg")]):
            return self._comp_result("orient_held", False, "directed_orientation_unknown")
        if self._comp()["route"]["precision"] or arm not in self._eef_grips():
            return self._comp_result("orient_held", False, "precision_capability_missing")
        delta = _angle(yaw_deg - evidence["yaw_deg"])
        if abs(delta) > 30:
            return self._comp_result("orient_held", False, "orientation_requires_smaller_observed_segment")
        swap = self._comp().get("swap")
        if swap:
            index = swap["confirmed_legs"]
            if index >= 3 or index in swap["attempted"] or index in swap.get("facing_corrections", []):
                return self._comp_result("orient_held", False, "swap_facing_correction_exhausted")
            leg = swap["legs"][index]
            if (str(key) != leg["key"] or abs(_angle(yaw_deg - leg["yaw_deg"])) > 1e-6 or
                    item["observed_step"] != self.state.get("step") or
                    item["measure_serial"] != self.__dict__.get("_eef_measure_serial", 0) or
                    not all(evidence.get(k) is True for k in ("identity_unique", "held_identity", "directed_axis")) or
                    self._eef_object(item["object"]).get("held_by") != arm):
                return self._comp_result("orient_held", False, "swap_held_identity_or_facing_unverified")
            if self._comp()["uncertain"] or self._comp()["completion"] == "failed":
                return self._comp_result("orient_held", False, "unsafe_owner_or_uncertain_motion")
            swap.setdefault("facing_corrections", []).append(index)  # BEFORE motion; no retry on error
        # Rotate the CURRENT EE orientation about world z, preserving its tilt.
        from harness_policy.geometry import pose_matrix, matrix_quaternion
        import numpy as np
        old = self.state[f"{arm.split('_')[0]}_ee_pose"][3:]
        a = math.radians(delta)
        rotation = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0,0,1]])
        orientation = matrix_quaternion(rotation @ pose_matrix([0, 0, 0, *old])[:3, :3])
        tip = self._eef_tip(arm)
        return self._comp_motion("orient_held", [key, arm, yaw_deg, evidence["yaw_deg"]],
                                 lambda: self.move_eef(arm, self.link6_for_tip(tip, orientation),
                                                       speed=0.08, max_seconds=2), evidence)

    def swap_plan(self, first, second, staging, evidence=None):
        """Freeze two identity+directed-yaw destinations from the pre-swap observation;
        return three table legs via measured empty staging. Does NOT track through
        occlusion or execute: rebind SAME identity before every leg. Missing identity
        continuity/facing fails, never nearest-position relabeling or WALL restart.
        first/second are ledger keys or identity_ledger entry dicts bound together
        under the same evidence rules. staging is [x,y], {position/xy:[x,y,...]},
        or a current locate object id/{object:id}; explicit coordinates take priority."""
        ledger = self._comp()["ledger"]
        if self._comp().get("swap"):
            return self._comp_result("swap_plan", True, swap=copy.deepcopy(self._comp()["swap"]))
        if not self._evidence(evidence):
            return self._comp_result("swap_plan", False, "stale_or_missing_measurement")
        if evidence.get("identity_unique") is not True:
            return self._comp_result("swap_plan", False, "memory_identity_lost", next_hop="finish_honest_failure")

        keys, entries = [], []
        for ref in (first, second):
            if isinstance(ref, str) and ref in ledger:
                keys.append(ref)
            elif isinstance(ref, dict) and all(isinstance(ref.get(field), str) and ref[field].strip()
                                               for field in ("key", "object", "identity")):
                keys.append(ref["key"])
                entries.append(ref)
            else:
                return self._comp_result("swap_plan", False, "memory_identity_lost",
                                         next_hop="finish_honest_failure")
        if keys[0] == keys[1]:
            return self._comp_result("swap_plan", False, "identity_ambiguous" if entries else "memory_identity_lost",
                                     next_hop="remeasure_no_motion" if entries else "finish_honest_failure")
        if entries:
            # One atomic ledger binding, not a dict's repr used as a key. Reuse
            # aliases, directed-axis, freshness and immutable identity checks.
            try:
                binding = self.identity_ledger(entries, evidence)
            except ValueError as error:  # a supplied object id is absent from current locate
                return self._comp_result("swap_plan", False, "identity_ambiguous",
                                         receipt={"exception_type": type(error).__name__, "exception": str(error)})
            if not binding["ok"]:
                return self._comp_result("swap_plan", False, binding["reason"])
            ledger = self._comp()["ledger"]
        first, second = keys
        a, b = ledger[first], ledger[second]
        if a["yaw_deg"] is None or b["yaw_deg"] is None:
            return self._comp_result("swap_plan", False, "directed_orientation_unknown")
        if evidence.get("staging_clear") is not True or evidence.get("mat_observed") is not True:
            return self._comp_result("swap_plan", False, "staging_measurement_unknown")
        if isinstance(staging, dict) and ("position" in staging or "xy" in staging):
            staging = staging["position"] if "position" in staging else staging["xy"]
            staging = staging[:2] if isinstance(staging, (list, tuple)) and len(staging) >= 2 else None
        elif isinstance(staging, str) or (isinstance(staging, dict) and "object" in staging):
            if self.__dict__.get("_eef_measured", {}).get("step") != self.state.get("step"):
                return self._comp_result("swap_plan", False, "staging_measurement_unknown")
            try:
                mat = self._eef_object(staging["object"] if isinstance(staging, dict) else staging)
            except ValueError as error:
                return self._comp_result("swap_plan", False, "staging_measurement_unknown",
                                         receipt={"exception_type": type(error).__name__, "exception": str(error)})
            if mat.get("aliases") and evidence.get("aliases_resolved") is not True:
                return self._comp_result("swap_plan", False, "staging_measurement_unknown")
            staging = mat["position"][:2]
        if not isinstance(staging, (list, tuple)) or not _finite(staging, 2):
            return self._comp_result("swap_plan", False, "staging_measurement_unknown")
        staging = [float(value) for value in staging]
        if math.dist(staging, a["position"][:2]) < 0.08 or math.dist(staging, b["position"][:2]) < 0.08:
            return self._comp_result("swap_plan", False, "staging_overlaps_protected_object")
        legs = [{"key": first, "xy": list(staging), "yaw_deg": a["yaw_deg"]},
                {"key": second, "xy": a["position"][:2], "yaw_deg": b["yaw_deg"]},
                {"key": first, "xy": b["position"][:2], "yaw_deg": a["yaw_deg"]}]
        for leg in legs:
            if not any(v == "down" for v in self._eef_reach(*leg["xy"]).values()):
                return self._comp_result("swap_plan", False, "unreachable")
        self._comp()["swap"] = {"legs": copy.deepcopy(legs), "confirmed_legs": 0, "attempted": [], "receipts": []}
        return self._comp_result("swap_plan", True, legs=legs, completion="plan_only")

    def swap_confirm(self, index, target, *, arm=None, evidence=None):
        """Press the swap's button ONCE after each visually completed move to its
        frozen destination (including the actual empty mat). Requires fresh identity_unique,
        at_destination, facing_correct, released=True. No retry/replan after uncertainty;
        final placement commitment is per stable identity, not moving oN labels.
        A complete contact count confirms the leg despite bounded downstroke TIMEOUT;
        failed final clearance preserves confirmation but blocks motion until inspected."""
        swap = self._comp().get("swap")
        if not swap or type(index) is not int or index != swap["confirmed_legs"] or index in swap["attempted"] or index >= 3:
            return self._comp_result("swap_confirm", False, "swap_order_or_already_attempted")
        if self._comp()["uncertain"]:
            # No new press has run: do not poison/consume the NEXT leg just because
            # the previous, already counted press still needs clearance inspection.
            return self._comp_result("swap_confirm", False, "motion_uncertain_no_retry",
                                     next_hop="inspect_receipt_no_retry")
        if not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("identity_unique", "at_destination", "facing_correct", "released", "target_unique")):
            return self._comp_result("swap_confirm", False, "swap_placement_unverified")
        swap["attempted"].append(index)
        result, exact = self._comp_counted_motion("swap_confirm", ["swap_confirm", index, target, arm],
                                                  target, 1, arm, evidence)
        swap["receipts"].append(result["receipt"])
        if not exact:
            return self.failure_route("confirm_uncertain", motion_executed="unknown", receipt=result["receipt"])
        swap["confirmed_legs"] += 1
        if index > 0:
            self._comp()["ledger"][swap["legs"][index]["key"]].update(status="done", placement_evidence=copy.deepcopy(evidence))
        result = self._comp_result("swap_confirm", True, receipt=result["receipt"],
                                   motion=result["motion_executed"], count_complete=True,
                                   confirmation="counted_press_complete", confirmed_legs=swap["confirmed_legs"],
                                   press_retry_allowed=False, next_hop="inspect_receipt_no_retry" if
                                   self._comp()["uncertain"] else "rebind_next_swap_leg" if index < 2 else
                                   "observe_swap_completion")
        self._comp()["last_counted_press"] = copy.deepcopy(result)
        return result

    def tail_correct(self, arm, object, *, into=None, x=None, y=None, yaw_deg=None, evidence=None):
        """C-own's ONE high-level visible/exposed transfer, never inside blind packing,
        contact assembly, occupied support removal or dynamic/remembered tasks.
        Fresh evidence exposed, stable, reachable, measured, empty_hands=True; optional
        face correction must be measured before calling, not a second correction."""
        route = self._comp()["route"]
        if route["owner"] != "C" or route["phase"] != "own" or route["correction_count"]:
            return self._comp_result("tail_correct", False, "correction_exhausted", next_hop="finish_honest_failure")
        if not self._evidence(evidence) or not all(evidence.get(k) is True for k in
                ("empty_hands", "exposed", "stable", "reachable", "measured")):
            return self._comp_result("tail_correct", False, "boundary_rejected")
        # Cooperative pen-holder manipulation is not generic empty-hand transfer.
        if re.search(r"hold[\s\S]*?pen holder[\s\S]*?one hand", route["instruction"], re.I):
            return self._comp_result("tail_correct", False, "cooperative_holder_capability_missing",
                                     next_hop="capability_preserve")
        if yaw_deg is not None and (not _finite([yaw_deg, evidence.get("yaw_deg")]) or
                                   evidence.get("orientation_measured") is not True or
                                   abs(_angle(yaw_deg - evidence["yaw_deg"])) > 30):
            return self._comp_result("tail_correct", False, "directed_orientation_unknown")
        route["correction_count"] = 1  # consume BEFORE any motion, even if it errors
        def action():
            if yaw_deg is None:
                return self.eef_transfer(arm, object, into=into, x=x, y=y, withdraw=False)
            picked = self.eef_pick(arm, object)
            if not picked["ok"]:
                return {"ok": False, "reason": "grasp_failed", "pick": picked}
            from harness_policy.geometry import pose_matrix, matrix_quaternion
            import numpy as np
            a = math.radians(_angle(yaw_deg - evidence["yaw_deg"]))
            rotation = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0,0,1]])
            old = self.state[f"{arm.split('_')[0]}_ee_pose"][3:]
            orientation = matrix_quaternion(rotation @ pose_matrix([0,0,0,*old])[:3,:3])
            turned = self.move_eef(arm, self.link6_for_tip(self._eef_tip(arm), orientation), speed=.08, max_seconds=2)
            if not turned.ok:
                return turned
            released = self.eef_release(arm, into=into, x=x, y=y, withdraw=False)
            return {"ok": released["ok"], "reason": released.get("reason"), "pick": picked,
                    "orientation": turned.to_dict(), "release": released}
        return self._comp_motion("tail_correct", [arm, object, into, x, y, yaw_deg], action, evidence)
