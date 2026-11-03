"""Review guard: the teacher-side v2 switches of review mode, under either schedule.

``review.guard`` holds one key per switch. ``enforce`` is required; every other key may be left out and is then off.
Without the block nothing here runs: the validated review block, every packet, prompt, tool and row are as before.

Refusal rules (checked on every validated teacher decision before admission, in ``TeacherBrain._decide`` or, on the
brain channel, ``TeacherBridge._validate``; they replay over finished attempts):

``handback_failed`` / ``handback_still`` / ``handback_home`` / ``handback_after`` (hand-back on no progress)
    Counted since OpenWAM last executed a whole chunk: once ``handback_failed`` teacher decisions in a row were made
    on an execution the teacher itself judged ``failed``, or ``handback_still`` takeovers (``edit``, ``eef``,
    ``home``) in a row left both arms where they were (measured from the packets: under 5 mm and 0.05 rad, no gripper
    change of ``gripper_delta``), or the teacher's ``home`` failed (it left an arm farther than 0.15 m or 20 degrees
    from its start pose, RoboDojo's own return check, without bringing it 5 mm or 0.05 rad closer: the packets'
    ``home_offset``), or ``handback_after`` takeovers ran (a backstop count, off by default), OpenWAM gets one whole
    chunk (a ``student`` decision with the whole proposal) before the teacher may take over again. Until then
    takeovers and shorter ``student`` prefixes are refused, except ``home`` in the ``final_home_steps`` window (the
    final return comes first there). A no-progress trigger, not a plain count: caps of 16 takeovers or fewer would
    have cut streaks in wins and successes of both formal rounds.
``after_home`` (misjudged completion)
    After the teacher's own ``home`` has executed, while its assessment only inspects or confirms (its
    ``current_subgoal`` names no manipulation work, and either names an inspection or leaves nothing but completion
    items in ``task_progress.remaining``), ``edit``/``eef`` are refused and a ``student`` decision takes the whole
    proposal: an episode that goes on is not complete for the benchmark, and OpenWAM may still finish it. A
    correction that names its manipulation work in ``current_subgoal`` stays allowed (a motion of the arms alone, such
    as a return or an approach, names none).
    The rule binds only while the proposal has admissible rows (without them ``student`` is unavailable and an
    ``eef`` is the way on). (A text rule on the teacher's own words; the patterns are below.)
``release_check`` (look before a release)
    A ``student`` decision stops before a planned release more than ``release_check - 1`` rows ahead: the first
    proposal row where an arm that grips (an opening of 0.6 or less) opens out of the grip band, above 0.6 and by
    ``gripper_delta`` or more from its lowest opening so far (a rise within OpenWAM's carry band is no release). The
    teacher sees where the object is before OpenWAM lets go, then approves the release or corrects its position: the
    packet after such a stop is not limited again. Yields to the whole chunk the hand-back or ``after_home`` require.
``wrist_turn_deg``
    An ``eef`` target may not turn an arm more than this many degrees from its orientation when the current stretch
    of takeovers began (it begins anew after each whole OpenWAM chunk): a limit on long in-place wrist rolls, which
    sweep the arm and the held object through what is around them.
``full_chunk``, ``min_student_steps`` (whole student chunks; not in the v2 template)
    When the teacher's own assessment says the last chunk progressed (or the robot recovered) and OpenWAM's next intent
    is aligned, a ``student`` decision takes the whole proposal (the rule SKILL.md states; under on_stall this is the
    hand-back). ``min_student_steps``: any ``student`` decision takes at least that many rows. Both allow a shorter
    prefix that ends at a gripper change of at least ``gripper_delta`` in the proposal, and both yield to a
    ``release_check`` stop.

Switches that change what the teacher sees or what executes (they need ``enforce: true`` and do not replay):

``show_bounds``
    Every packet shows the frozen workspace bounds (the z floor is ``bounds_xyz[2][0]``) and each arm's margin to
    them, and an ``eef`` target outside them is refused here with its axis and limit, before admission would refuse
    it. (The rows of an arm measured outside start from its clip into the bounds: oeh_policy.review, start_clipped.)
``grip_note``
    Every packet states how OpenWAM's commanded gripper opening maps to holding: it grasps and carries objects at about
    0.3-0.6, often just above 0.5, so the 0.5 cut of ``executed_gripper_closed`` and the proposal diagnostics is no
    release point for it (oeh_policy.escalation documents the same ranges). Per arm: the last command, the proposal's
    opening range and its release row (as ``release_check`` finds it). The harness measures no finger gap (the
    state's opening is the previous command), so the note is all the packet can add.
``reason_note``
    Every packet states the takeover reasons the formal rounds support: an observed failure or a wrong subgoal; not
    the release position or a gripper opening alone; near standing objects, approach from above and back off at the
    first contact.
``hold_grip``, ``even_eef``, ``home_lift`` (how a correction's rows are built, oeh_policy.review.ReviewMethod)
    ``hold_grip``: a "closed" command keeps the grip an arm already holds (its last command, the ledger, is 0.6 or
    less) instead of commanding 0.0: an ``eef`` keeps that grip, and an ``edit`` may tighten it with OpenWAM's own row
    but never loosens it. From an open gripper (the ledger above 0.6) "closed" still closes fully. It needs
    ``grip_note``: a kept grip of 0.5-0.6 reads as open in ``executed_gripper_closed`` (cut at 0.5), and the note
    explains that cut.
    ``even_eef``: an ``eef``'s rows move evenly from where they start (the measured pose, or its clip into the
    bounds) to the target (distance / steps per row, at most the row limits) instead of at the row limit until the
    target (2, 2, 1, 0, 0 cm for 5 cm in 5 rows).
    ``home_lift``: ``home`` first raises an arm that is below its start height straight up to it, then moves toward
    the start pose.
``final_home_steps``
    In the last that many native steps (the packet's ``remaining_steps``), ``home`` needs no failed or misaligned
    assessment (oeh_policy.review.validate_teacher_response) and the hand-back does not refuse it; its trigger is
    recorded as ``final_return``. Both grippers must still be open (``home``'s own precondition).

With ``enforce: false`` the refusal rules run in shadow: what they would refuse is recorded (``review_guard`` events
and the teacher summary's ``guard``) and nothing the teacher sees or does changes. With ``enforce: true`` a refused
decision executes nothing and goes back to the teacher like an invalid one (it answers the same packet again within
``review.teacher.max_invalid``), and every packet carries a ``review_guard`` status: the numbers behind each rule in
every packet; all the rule sentences (``rules``) in the first packet and again every ``RESEND_EVERY`` packets (the
Codex app-server compacts a long episode's history); in between, a rule's sentence again whenever that rule starts to
bind (the hand-back, ``after_home``, a ``release_check`` stop, the final-home window).

The refusal rules read only the packets the teacher is shown (measured poses, the gripper command ledger, the
proposal table, the start-pose offsets, the previous result and its assessment), so they also replay over a finished
attempt's ``teacher/packets``: ``python -m oeh_policy.review_guard ATTEMPT_DIR...``. Pure Python, no numpy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping

ARMS = ("left", "right")
TAKEOVERS = ("edit", "eef", "home")
CORRECTIONS = ("edit", "eef")
RULES = ("handback", "after_home", "wrist_turn", "bounds", "release_check", "full_chunk", "min_student_steps")
# The assessment that obliges a whole chunk; equal to takeover.HANDBACK_EXECUTION / HANDBACK_INTENT (a test checks).
FULL_EXECUTION = ("progressing", "recovered")
FULL_INTENT = ("aligned",)
GUARD_KEYS = ("enforce", "handback_failed", "handback_still", "handback_home", "handback_after", "after_home",
              "release_check", "wrist_turn_deg", "final_home_steps", "show_bounds", "grip_note", "reason_note",
              "hold_grip", "even_eef", "home_lift", "full_chunk", "min_student_steps", "gripper_delta")
DEFAULTS = {"handback_failed": 0, "handback_still": 0, "handback_home": False, "handback_after": 0,
            "after_home": False, "release_check": 0, "wrist_turn_deg": 0, "final_home_steps": 0, "show_bounds": False,
            "grip_note": False, "reason_note": False, "hold_grip": False, "even_eef": False, "home_lift": False,
            "full_chunk": False, "min_student_steps": 0, "gripper_delta": 0.25}
BOOLEANS = ("enforce", "handback_home", "after_home", "show_bounds", "grip_note", "reason_note", "hold_grip",
            "even_eef", "home_lift", "full_chunk")
RANGES = {"handback_failed": (0, 100), "handback_still": (0, 100), "handback_after": (0, 1000),
          "release_check": (0, 32), "wrist_turn_deg": (0, 360), "final_home_steps": (0, 1000),
          "min_student_steps": (0, 256), "gripper_delta": (0.05, 1.0)}
INTEGERS = ("handback_failed", "handback_still", "handback_after", "release_check", "wrist_turn_deg",
            "final_home_steps", "min_student_steps")
# Switches that change what the teacher sees or what executes, with their off values: they need enforce true and
# do not replay.
ACTING = {"show_bounds": False, "grip_note": False, "reason_note": False, "hold_grip": False, "even_eef": False,
          "home_lift": False, "final_home_steps": 0}
# examples/openwam-review-v2.template.json's block (the v2 development arm); its refusal rules are the replay's
# default.
EXAMPLE = {"enforce": True, "handback_failed": 3, "handback_still": 2, "handback_home": True, "handback_after": 0,
           "after_home": True, "release_check": 2, "wrist_turn_deg": 120, "final_home_steps": 40, "show_bounds": True,
           "grip_note": True, "reason_note": True, "hold_grip": True, "even_eef": True, "home_lift": True,
           "full_chunk": False, "min_student_steps": 0, "gripper_delta": 0.25}
# handback_still: a takeover that moved neither arm this far or turned it this much (and changed no gripper) is still.
STILL_M, STILL_RAD = 0.005, 0.05
# handback_home: RoboDojo's return check, "roughly within 0.15 m and 20 degrees" (teacher/SKILL.md). A home that
# leaves an arm outside it has failed when it brought that arm less than STILL_M / STILL_RAD closer.
HOME_M, HOME_DEG = 0.15, 20.0
# An opening at or below this is a grip OpenWAM holds; equal to oeh_policy.review.HOLD_MAX (a test checks).
HOLD_MAX = 0.6
# All rule sentences go out again every RESEND_EVERY packets: the Codex app-server compacts a long episode's history.
RESEND_EVERY = 8

# after_home: the teacher's own words (both formal rounds) for "only inspecting or confirming". A subgoal naming an
# inspection, or a remaining list of nothing but completion items (native success or its confirmation, an
# unspecified discrepancy, the arms at or back to their start poses, a stable result); neither the subgoal nor an
# item may name manipulation work ("place the remaining pieces and return both arms home" is work, and so is a
# correction such as "re-seat the green block"). A motion of the arms alone ("lower the left arm toward its start
# pose", "approach the green piece from above") names no work: after a home, those are the returns and inspection
# approaches the rule is for.
INSPECTION = re.compile(r"\b(inspect|verif|confirm|check|view|observ|look)\w*", re.I)
COMPLETION_ITEM = re.compile(
    r"(native (task )?(success|completion)|success (confirmation|signal)|completion (confirmation|signal)|"
    r"confirm\w* (task |native )?(completion|success)|unconfirmed|resolve (any|the) .*discrepanc|"
    r"(identify|find) (and \w+ )?(any|the) .*(discrepanc|requirement)|"
    r"\barms? ((remain|stay)s? |back )?((at|near|to) (their |its )?)?(home|start)|"
    r"(remain|stay)s? (stable|standing|undisturbed)|hold the final|leave .* undisturbed)", re.I)
WORK = re.compile(r"\b(place|places|placing|placement|placements|put|puts|putting|pick|picks|picking|"
                  r"(pre|re)?grasp(s|ing)?|insert\w*|stack|stacks|stacking|push\w*|pull\w*|lift|lifts|lifting|"
                  r"transfer\w*|carry|carries|carrying|sort|sorts|sorting|pack|packs|packing|hang|hangs|hanging|"
                  r"pour\w*|wipe\w*|fold\w*|press\w*|re-?orient\w*|orient|orients|orienting|rotate|rotates|"
                  r"rotating|"
                  # corrections, as verbs ("the centered block" or "is aligned" describes a state, not work)
                  r"(re-?)?seat|(re-?)?seats|(re-?)?seating|(re-?)?cent(er|re)|(re-?)?cent(er|re)s|"
                  r"(re-?)?cent(er|re)ing|(re-?)?align|(re-?)?aligns|(re-?)?aligning|shift|shifts|shifting|slide|"
                  r"slides|sliding|nudge|nudges|nudging|adjust|adjusts|adjusting|straighten|"
                  r"straightens|straightening|re-?position|re-?positions|re-?positioning|flatten|flattens|"
                  r"flattening|tighten|tightens|tightening|fix|fixes|fixing)\b", re.I)
REASON_NOTE = ("reason_note: take over for an observed failure (a drop, a missed grasp, contact, an arm that stalls, "
               "deflects or moves away from its target) or a wrong subgoal (the wrong object, destination or order; "
               "idling or returning with work left). The release position alone or a gripper opening alone is no "
               "reason: OpenWAM stages and releases in its own way. Near standing or stacked objects approach from "
               "above with clearance and back off at the first contact or stop.")


def validate_guard(value) -> dict:
    """``review.guard``: known keys, ``enforce`` given, every effective value frozen (left-out keys get their off
    value), at least one switch on (without one, leave the block out), the acting switches only when enforced, and
    ``hold_grip`` only with ``grip_note``."""
    if not isinstance(value, Mapping):
        raise ValueError("review.guard must be a mapping")
    unknown = set(value) - set(GUARD_KEYS)
    if unknown:
        raise ValueError(f"unknown review.guard settings: {sorted(unknown)}")
    if "enforce" not in value:
        raise ValueError("review.guard.enforce is required")
    guard = {key: value[key] if key in value else DEFAULTS[key] for key in GUARD_KEYS}
    for key in BOOLEANS:
        if type(guard[key]) is not bool:
            raise ValueError(f"review.guard.{key} must be a boolean")
    for key, (low, high) in RANGES.items():
        number = guard[key]
        integer = key in INTEGERS
        if (isinstance(number, bool) or not isinstance(number, int if integer else (int, float))
                or not math.isfinite(number) or not low <= number <= high):
            raise ValueError(f"review.guard.{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")
    if not any(guard[key] != DEFAULTS[key] for key in GUARD_KEYS if key not in ("enforce", "gripper_delta")):
        raise ValueError("review.guard switches nothing on; without a switch, leave review.guard out")
    acting = [key for key, off in ACTING.items() if guard[key] != off]
    if acting and not guard["enforce"]:
        raise ValueError(f"review.guard.{acting[0]} changes what the teacher sees or what executes: it needs enforce "
                         "true (shadow mode records the refusal rules only)")
    if guard["hold_grip"] and not guard["grip_note"]:
        raise ValueError("review.guard.hold_grip needs grip_note: a kept grip of 0.5-0.6 reads as open in "
                         "executed_gripper_closed (cut at 0.5), and grip_note tells the teacher so")
    return guard


# --------------------------------------------------------------------------------------------------------------
# What a packet shows.

def measure(packet: Mapping) -> dict:
    """Each arm's measured position and orientation and its gripper command (the ledger) as the packet reports them."""
    eef = packet["current_eef"]
    return {arm: {"position": tuple(float(v) for v in eef[arm]["position"]),
                  "quaternion": tuple(float(v) for v in eef[arm]["quaternion_wxyz"]),
                  "gripper": float(eef[arm]["gripper_command"])} for arm in ARMS}


def turned(a, b) -> float:
    """The rotation angle (rad) between two wxyz quaternions."""
    na, nb = math.sqrt(sum(v * v for v in a)), math.sqrt(sum(v * v for v in b))
    dot = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return 2 * math.acos(min(1.0, dot))


def moved(before: Mapping, after: Mapping, delta: float) -> bool:
    """Whether either arm moved, turned or changed its gripper command between two measures."""
    return any(math.dist(before[arm]["position"], after[arm]["position"]) >= STILL_M
               or turned(before[arm]["quaternion"], after[arm]["quaternion"]) >= STILL_RAD
               or abs(before[arm]["gripper"] - after[arm]["gripper"]) >= delta - 1e-9 for arm in ARMS)


def proposal_openings(packet: Mapping) -> list[dict]:
    """The proposal's commanded gripper opening per row and arm, read from the table the teacher sees."""
    view = packet["student_proposal"]
    if not view.get("rows"):
        return []
    columns = list(view["columns"])
    index = {arm: columns.index(arm + "_opening") for arm in ARMS}
    return [{arm: float(line[index[arm]]) for arm in ARMS} for line in view["table"]]


def first_change(openings, ledger: Mapping, delta: float) -> int | None:
    """The first row whose commanded opening is ``delta`` or more from the ledger on either arm, or None."""
    for row, values in enumerate(openings):
        if any(abs(values[arm] - ledger[arm]) >= delta - 1e-9 for arm in ARMS):
            return row
    return None


def release_rows(openings, ledger: Mapping, delta: float) -> dict:
    """Per arm, the first proposal row where an arm that grips (its lowest opening so far, the last command included,
    is ``HOLD_MAX`` or less) opens out of the grip band: above ``HOLD_MAX`` and by ``delta`` or more from that low.
    A rise within the band (OpenWAM closing at 0.3 and carrying at 0.55, say) is no release. None when there is none."""
    rows = {}
    for arm in ARMS:
        low, found = ledger[arm], None
        for row, values in enumerate(openings):
            if low <= HOLD_MAX and values[arm] > HOLD_MAX and values[arm] >= low + delta - 1e-9:
                found = row
                break
            low = min(low, values[arm])
        rows[arm] = found
    return rows


def inspection_only(assessment) -> bool:
    """Whether an assessment only inspects or confirms: its subgoal names no manipulation work, and either names an
    inspection or leaves nothing but completion items naming no work remaining (an empty remaining list included)."""
    if not isinstance(assessment, Mapping):
        return False
    subgoal = str(assessment.get("current_subgoal") or "")
    if WORK.search(subgoal):
        return False
    if INSPECTION.search(subgoal):
        return True
    progress = assessment.get("task_progress")
    remaining = progress.get("remaining") if isinstance(progress, Mapping) else None
    if not isinstance(remaining, list):
        return False
    return all(COMPLETION_ITEM.search(item) and not WORK.search(item)
               for item in remaining if isinstance(item, str) and item.strip())


def home_missed(packet: Mapping, before: Mapping | None = None) -> dict:
    """The arms farther than RoboDojo's return check from their start poses, with the packet's offsets; with the
    offsets ``before`` a home, only those it brought less than STILL_M and STILL_RAD closer (a failed home)."""
    offsets = packet.get("home_offset") or {}
    missed = {}
    for arm in ARMS:
        if arm not in offsets or not (offsets[arm]["distance_m"] > HOME_M or offsets[arm]["rotation_deg"] > HOME_DEG):
            continue
        if before and arm in before and (
                before[arm]["distance_m"] - offsets[arm]["distance_m"] >= STILL_M
                or before[arm]["rotation_deg"] - offsets[arm]["rotation_deg"] >= math.degrees(STILL_RAD)):
            continue  # it is not there yet, but the home brought it closer
        missed[arm] = dict(offsets[arm])
    return missed


def workspace_view(now: Mapping, bounds: Mapping) -> dict:
    """Per arm: the frozen bounds, the measured position's distance to the nearest face (negative outside) and the
    axes it is outside on."""
    view = {}
    for arm in ARMS:
        margins, outside = [], []
        for axis, value, (low, high) in zip("xyz", now[arm]["position"], bounds[arm]):
            margins.append(min(value - low, high - value))
            if value < low:
                outside.append(f"{axis} {value:.3f} below {low}")
            elif value > high:
                outside.append(f"{axis} {value:.3f} above {high}")
        view[arm] = {"bounds_xyz": [list(axis) for axis in bounds[arm]], "margin_m": round(min(margins), 4),
                     "outside": outside}
    return view


def outside(position, bounds) -> list[str]:
    """The axes of a position outside the bounds, with the limit."""
    found = []
    for axis, value, (low, high) in zip("xyz", position, bounds):
        if value < low:
            found.append(f"{axis} {value:.3f} is below {low}")
        elif value > high:
            found.append(f"{axis} {value:.3f} is above {high}")
    return found


# --------------------------------------------------------------------------------------------------------------
# The guard.

class ReviewGuard:
    """One episode's guard. The runtime shows it every teacher packet in order (``at_packet``) and asks it about each
    validated decision before admission (``check``)."""

    def __init__(self, cfg: Mapping, student_max_steps: int, bounds: Mapping | None = None):
        self.cfg = validate_guard(cfg)
        self.enforce = self.cfg["enforce"]
        self.cap = int(student_max_steps)
        self.bounds = ({arm: [[float(v) for v in axis] for axis in bounds[arm]] for arm in ARMS}
                       if bounds is not None else None)
        cfg = self.cfg
        self.handback_on = bool(cfg["handback_failed"] or cfg["handback_still"] or cfg["handback_home"]
                                or cfg["handback_after"])
        # Since OpenWAM last executed a whole chunk:
        self.stretch = 0  # executed teacher takeovers
        self.failed_run = 0  # executed teacher decisions in a row made on an execution judged failed
        self.still_run = 0  # executed takeovers in a row that left both arms where they were
        self.missed = {}  # arms the teacher's last home failed to bring back toward their start poses
        self.reference = None  # each arm's orientation when the stretch began (wrist_turn_deg)
        self.looked = False  # the last teacher decision was a student prefix that stopped at a release (release_check)
        self.homed = False  # the teacher's home has executed in this episode
        self.counted = None  # the decision index of the last executed record taken into account
        self.at = {}  # decision index -> (whole chunk rows, measure, release limit, home offsets) at that packet
        self.current = None  # (request_id, what check needs about that packet)
        self.required = False  # whether the current packet requires the hand-back
        self.packets = self.gripper_changes = self.handback_packets = self.after_home_packets = 0
        self.release_packets = self.longest_stretch = 0
        self.handbacks = []  # [native step, why] of each packet where a hand-back became required
        self.refused = dict.fromkeys(RULES, 0)
        self.sentences = self._sentences()  # rule name -> the sentence the teacher reads
        self.binding = set()  # the rules that bound at the previous packet (their sentences were sent then)
        self.rules_packets = 0  # packets whose shown status carried rule sentences (none in shadow mode)

    def _account(self, packet: Mapping, now: Mapping) -> None:
        """What executed since the previous teacher packet (its previous_result, once per decision)."""
        previous = packet.get("previous_result") or {}
        source, mode, decision = previous.get("source"), previous.get("mode"), previous.get("decision")
        if not previous or (source != "auto" and decision == self.counted):
            return
        if source != "auto":
            self.counted = decision
        full, before, limit, offsets = self.at.get(decision, (1, None, None, None))
        if source in ("auto", "teacher"):
            self.looked = (source == "teacher" and mode == "student" and limit is not None
                           and previous.get("executed_steps", 0) <= limit)
        if mode == "student" and (source == "auto" or previous.get("executed_steps", 0) >= full):
            self.stretch = self.failed_run = self.still_run = 0  # OpenWAM executed a whole chunk
            self.missed = {}
            return
        if source != "teacher":
            return
        assessment = previous.get("assessment") or {}
        self.failed_run = self.failed_run + 1 if assessment.get("execution_status") == "failed" else 0
        went = before is None or moved(before, now, self.cfg["gripper_delta"])
        if mode in TAKEOVERS:
            self.stretch += 1
            self.still_run = 0 if went else self.still_run + 1
        elif went:
            self.still_run = 0
        if mode == "home":
            self.homed = True
            self.missed = home_missed(packet, offsets)

    def _why(self) -> list[str]:
        cfg, why = self.cfg, []
        if cfg["handback_failed"] and self.failed_run >= cfg["handback_failed"]:
            why.append("failed")
        if cfg["handback_still"] and self.still_run >= cfg["handback_still"]:
            why.append("still")
        if cfg["handback_home"] and self.missed:
            why.append("home")
        if cfg["handback_after"] and self.stretch >= cfg["handback_after"]:
            why.append("count")
        return why

    def _sentences(self) -> dict:
        """One sentence per switch that is on, by rule name, in the order the teacher reads them. They depend only on
        the configuration, so they are built once."""
        cfg, out = self.cfg, {}
        if self.handback_on:
            triggers = []
            if cfg["handback_failed"]:
                triggers.append(f"{cfg['handback_failed']} decisions in a row were made on an execution you judged "
                                "failed (failed_run)")
            if cfg["handback_still"]:
                triggers.append(f"{cfg['handback_still']} takeovers in a row left both arms where they were "
                                f"(still_run: under {STILL_M * 1000:.0f} mm and {STILL_RAD} rad, no gripper change)")
            if cfg["handback_home"]:
                triggers.append(f"your home left an arm more than {HOME_M} m or {HOME_DEG:.0f} degrees from its start "
                                f"pose without bringing it {STILL_M * 1000:.0f} mm or {STILL_RAD} rad closer "
                                "(home_offset)")
            if cfg["handback_after"]:
                triggers.append(f"{cfg['handback_after']} takeovers have run (takeovers_since_whole_chunk)")
            out["handback"] = ("handback: since OpenWAM last executed a whole proposal, once "
                               + "; or ".join(triggers) + ", OpenWAM gets one whole chunk before you may take over "
                               "again: the next decision must be student with the whole proposal (edit, eef, home and "
                               "shorter prefixes are refused"
                               + (", except home in the final_home window" if cfg["final_home_steps"] else "") + ").")
        if cfg["after_home"]:
            out["after_home"] = ("after_home: once your home has executed and the episode goes on, the task is not "
                                 "complete for the benchmark. While your assessment only inspects or confirms (your "
                                 "current_subgoal names no manipulation work, and it names an inspection or nothing "
                                 "but completion items remain) and the proposal has rows, edit and eef are refused and "
                                 "a student decision takes the whole proposal. A correction for concrete remaining "
                                 "work stays allowed: name that work in your current_subgoal (a return or an approach "
                                 "of the arms alone names none).")
        if cfg["release_check"]:
            out["release_check"] = (
                f"release_check: release_rows is the first proposal row where an arm that grips (opening {HOLD_MAX} "
                f"or less) opens out of the grip band: above {HOLD_MAX} and {cfg['gripper_delta']} or more above its "
                f"lowest opening so far. A student decision stops before a release {cfg['release_check']} or more "
                "rows ahead (at most release_check_steps rows): look where the object is, then let OpenWAM release "
                "or correct the release position first. The packet after that stop (release_looked) is not limited "
                "again; a whole chunk the handback or after_home rule requires runs through it.")
        if cfg["wrist_turn_deg"]:
            out["wrist_turn"] = ("wrist_turn: since OpenWAM last executed a whole proposal, an eef target may not "
                                 f"turn an arm more than {cfg['wrist_turn_deg']} degrees from its orientation when "
                                 "your takeovers began (wrist_turned_deg is where it is now). To turn an object "
                                 "further, set it down and regrasp it.")
        if cfg["final_home_steps"]:
            out["final_home"] = (f"final_home: in the last {cfg['final_home_steps']} native steps (remaining_steps; "
                                 "final_home_window), home needs no failed or misaligned assessment"
                                 + (" and the handback rule does not refuse it" if self.handback_on else "")
                                 + "; both grippers must still be open.")
        if cfg["show_bounds"] and self.bounds is not None:
            out["bounds"] = ("bounds: every native row stays inside workspace.<arm>.bounds_xyz (m, [low, high] per "
                             "axis; bounds_xyz[2][0] is the z floor); margin_m is the measured position's distance to "
                             "the nearest face, negative outside. An eef target outside the bounds is refused; keep "
                             "targets inside, with margin above the floor.")
        if cfg["grip_note"]:
            out["grip_note"] = (
                "grip_note: OpenWAM grasps and carries objects at commanded openings of about 0.3-0.6, often just "
                "above 0.5, and releases by opening well above its grip (usually toward 1). An opening above 0.5 is "
                "therefore no release: executed_gripper_closed, closes_at_row and opens_at_row (cut at 0.5) do not "
                "separate OpenWAM's grip from a release. gripper.<arm>.release_row is the first proposal row where an "
                f"arm that grips ({HOLD_MAX} or less) opens above {HOLD_MAX} and {cfg['gripper_delta']} or more above "
                "its lowest opening so far, null when it keeps its grip (a rise within 0.3-0.6 is its carry). "
                "Whether an object is held shows in the views (it moves with the fingers); while it is, a takeover "
                "only to keep a gripper closed is not needed.")
        if cfg["reason_note"]:
            out["reason_note"] = REASON_NOTE
        if cfg["hold_grip"]:
            out["hold_grip"] = (
                f"hold_grip: when an arm already grips (its last command, current_eef.<arm>.gripper_command, is "
                f"{HOLD_MAX} or less), gripper_closed=true (eef) keeps that grip instead of commanding 0.0, and "
                "gripper closed (edit) keeps it too, tightening with OpenWAM's own row but never loosening. From an "
                "open gripper they close fully. To close fully from a grip, open first. A kept grip above 0.5 reads "
                "as open in executed_gripper_closed (see grip_note).")
        if cfg["even_eef"]:
            out["even_eef"] = ("even_eef: an eef's rows move evenly from the measured pose (for an arm measured "
                               "outside the bounds, its nearest pose inside them) to your target over the steps you "
                               "give (distance / steps per row, at most 2 cm and 0.1 rad per row): 5 rows for 5 cm "
                               "move 1 cm each, and one row moves at most 2 cm.")
        if cfg["home_lift"]:
            out["home_lift"] = ("home_lift: home first raises an arm that is below its start height straight up to it, "
                                "keeping its orientation, then moves it toward its start pose.")
        if cfg["full_chunk"]:
            out["full_chunk"] = ("full_chunk: a student decision whose assessment is progressing or recovered and "
                                 "aligned takes full_chunk_steps rows, unless the proposal opens or closes a gripper "
                                 "within them (gripper_change_row is not null) or a release_check stop comes first.")
        if cfg["min_student_steps"]:
            out["min_student_steps"] = ("min_student_steps: any student decision takes at least min_student_steps rows "
                                        "(this proposal's shorter length, through gripper_change_row, or a "
                                        "release_check stop may lower it).")
        return out

    def at_packet(self, packet: Mapping) -> dict:
        """Account for what executed since the previous teacher packet; return this packet's status."""
        now = measure(packet)
        self._account(packet, now)
        if self.stretch == 0:
            self.reference = {arm: now[arm]["quaternion"] for arm in ARMS}
        self.longest_stretch = max(self.longest_stretch, self.stretch)
        first = self.packets == 0
        self.packets += 1
        full = min(int(packet["student_proposal"]["rows"]), self.cap)
        cfg = self.cfg
        why = self._why() if full > 0 else []
        required = bool(why)
        if required:
            self.handback_packets += 1
            if not self.required:
                self.handbacks.append([packet.get("native_step"), why])
        self.required = required
        # after_home binds only while the proposal has admissible rows: without them student and edit are
        # unavailable, and an eef (back into the bounds, say) is the teacher's way on.
        after_home = bool(cfg["after_home"] and self.homed and full > 0)
        self.after_home_packets += after_home
        ledger = {arm: now[arm]["gripper"] for arm in ARMS}
        openings = proposal_openings(packet)[:full]
        change = first_change(openings, ledger, cfg["gripper_delta"])
        self.gripper_changes += change is not None
        releases = release_rows(openings, ledger, cfg["gripper_delta"])
        ahead = [row for row in releases.values() if row is not None and row >= cfg["release_check"]]
        release_limit = min(ahead) if cfg["release_check"] and ahead and not self.looked else None
        self.release_packets += release_limit is not None
        self.at = {packet.get("decision"): (full, now, release_limit, packet.get("home_offset"))}
        # A release_check stop lowers the floor too: the two rules never ask for opposite prefixes.
        floor = min(cfg["min_student_steps"], full, full if change is None else change + 1,
                    full if release_limit is None else release_limit)
        final = bool(cfg["final_home_steps"]) and packet.get("remaining_steps", math.inf) <= cfg["final_home_steps"]
        status = {}
        if self.handback_on:
            status.update(takeovers_since_whole_chunk=self.stretch, failed_run=self.failed_run,
                          still_run=self.still_run, handback_required=required, handback_why=why)
        if cfg["after_home"]:
            status["after_home"] = after_home
        if cfg["release_check"]:
            status.update(release_rows=releases, release_check_steps=release_limit, release_looked=self.looked)
        if cfg["wrist_turn_deg"]:
            status["wrist_turned_deg"] = {arm: round(math.degrees(turned(self.reference[arm], now[arm]["quaternion"])),
                                                     1) for arm in ARMS}
        if cfg["final_home_steps"]:
            status["final_home_window"] = final
        if cfg["show_bounds"] and self.bounds is not None:
            status["workspace"] = workspace_view(now, self.bounds)
        if cfg["grip_note"]:
            status["gripper"] = {arm: {"last_command": round(now[arm]["gripper"], 3),
                                       "proposal_opening_range": [round(min(row[arm] for row in openings), 3),
                                                                  round(max(row[arm] for row in openings), 3)]
                                       if openings else None, "release_row": releases[arm]} for arm in ARMS}
        if cfg["full_chunk"] or cfg["min_student_steps"]:
            status["gripper_change_row"] = change
        if cfg["full_chunk"]:
            status["full_chunk_steps"] = full
        if cfg["min_student_steps"]:
            status["min_student_steps"] = floor
        # The sentences: all of them in the first packet and every RESEND_EVERY packets after it (packets 1, 9, 17,
        # ...), and in between a rule's sentence when it starts to bind. The numbers above come with every packet.
        binding = {name for name, on in (("handback", required), ("after_home", after_home),
                                         ("release_check", release_limit is not None), ("final_home", final)) if on}
        if first or (self.packets - 1) % RESEND_EVERY == 0:
            status["rules"] = list(self.sentences.values())
        elif binding - self.binding:
            status["rules"] = [text for name, text in self.sentences.items() if name in binding - self.binding]
        if "rules" in status and self.enforce:  # only an enforced guard's status is shown to the teacher
            self.rules_packets += 1
        self.binding = binding
        self.current = (packet["request_id"], {"full": full, "floor": floor, "change": change, "why": why,
                                               "after_home": after_home, "release_limit": release_limit,
                                               "final": final, "now": now})
        return json.loads(json.dumps(status))

    def _handback_reason(self, why, full) -> str:
        parts = {"failed": f"{self.failed_run} decisions in a row were made on an execution you judged failed",
                 "still": f"{self.still_run} takeovers in a row left both arms where they were",
                 "home": "your home left " + ", ".join(
                     f"the {arm} arm {item['distance_m']} m and {item['rotation_deg']} degrees" for arm, item in
                     self.missed.items()) + " from the start pose and brought it no closer",
                 "count": f"{self.stretch} takeovers have run"}
        return (f"review guard (handback): since OpenWAM last executed a whole proposal, "
                + "; ".join(parts[item] for item in why) + ". Hand OpenWAM one whole chunk first: student with "
                f"steps={full}. After it you may take over again.")

    def check(self, request_id, decision: Mapping, assessment) -> tuple[str, str] | None:
        """The rule a validated decision for the current packet breaks and the reason for the teacher, or None."""
        current = self.current
        if current is None or current[0] != request_id:
            return None  # not the current packet: the runtime refuses it as stale anyway
        info, cfg = current[1], self.cfg
        mode, steps, full = decision["mode"], decision["steps"], info["full"]
        assessment = assessment if isinstance(assessment, Mapping) else {}
        cut = mode == "student" and steps < full
        if info["why"] and (mode in TAKEOVERS or cut) and not (mode == "home" and info["final"]):
            return self._refuse("handback", self._handback_reason(info["why"], full))
        whole = bool(info["why"]) or (info["after_home"] and inspection_only(assessment))
        if info["after_home"] and (mode in CORRECTIONS or cut) and inspection_only(assessment):
            # info["after_home"] holds only while the proposal has rows (full > 0).
            return self._refuse("after_home", (
                "review guard (after_home): your home has executed and the episode goes on, so the benchmark does not "
                "count the task as complete; your assessment only inspects or confirms. Let OpenWAM run its whole "
                f"proposal (student, steps={full}): it may still finish the task. A correction for concrete "
                "remaining work stays allowed: name that work in your current_subgoal (a return or an approach of "
                "the arms alone names none)."))
        target = decision.get("target") if mode == "eef" else None
        if cfg["wrist_turn_deg"] and target and self.reference:
            for arm in ARMS:
                angle = math.degrees(turned(self.reference[arm], target[arm]["quaternion_wxyz"]))
                if angle > cfg["wrist_turn_deg"] + 1e-6:
                    return self._refuse("wrist_turn", (
                        f"review guard (wrist_turn): this eef would turn the {arm} wrist {angle:.0f} degrees from its "
                        f"orientation when your takeovers began (limit {cfg['wrist_turn_deg']}). A long in-place "
                        "wrist roll swings the arm through what is around it: set the object down and regrasp it "
                        "from the side you need, or let OpenWAM run a whole chunk."))
        if cfg["show_bounds"] and target and self.bounds is not None:
            for arm in ARMS:
                found = outside(target[arm]["position"], self.bounds[arm])
                if found:
                    return self._refuse("bounds", (
                        f"review guard (bounds): target.{arm} lies outside the workspace bounds ({'; '.join(found)}); "
                        "every row must stay inside workspace.<arm>.bounds_xyz. Choose a target inside, with "
                        "margin above the z floor."))
        limit = info["release_limit"]
        if limit is not None and mode == "student" and steps > limit and not whole:
            return self._refuse("release_check", (
                f"review guard (release_check): the proposal releases at row {limit}; take at most "
                f"steps={limit} and look where the object is before OpenWAM lets go (then approve the release, or "
                "correct its position first)."))
        execution, intent = assessment.get("execution_status"), assessment.get("intent_status")
        # full_chunk yields to a release_check stop (steps == limit), as the min_student_steps floor does.
        if (cfg["full_chunk"] and cut and info["change"] is None and execution in FULL_EXECUTION
                and intent in FULL_INTENT and steps != limit):
            take = (f"accept the whole proposal: student with steps={full}. A shorter student prefix is for observing "
                    f"a decisive moment sooner, and this proposal opens or closes no gripper in its {full} rows."
                    if limit is None else
                    f"accept the proposal up to its release: student with steps={limit} (release_check). A shorter "
                    "student prefix is for observing a decisive moment sooner, and the release is this proposal's.")
            return self._refuse("full_chunk", (
                f"review guard (full_chunk): your assessment says the last chunk is {execution} and the next intent "
                f"aligned, so {take}"))
        if cfg["min_student_steps"] and mode == "student" and steps < info["floor"]:
            through = ("" if info["change"] is None or info["change"] + 1 != info["floor"] else
                       f", through the gripper change at row {info['change']}")
            return self._refuse("min_student_steps", (
                f"review guard (min_student_steps): a student decision takes at least {info['floor']} rows of this "
                f"proposal{through}; OpenWAM's chunks are not cut shorter. Take steps={info['floor']} or more."))
        return None

    def _refuse(self, rule, message):
        self.refused[rule] += 1
        return rule, message

    def summary(self) -> dict:
        return {"enforce": self.enforce, "packets": self.packets, "handback_packets": self.handback_packets,
                "handbacks": [list(item) for item in self.handbacks], "longest_stretch": self.longest_stretch,
                "after_home_packets": self.after_home_packets, "release_packets": self.release_packets,
                "gripper_change_packets": self.gripper_changes, "rules_packets": self.rules_packets,
                ("refused" if self.enforce else "would_refuse"): dict(self.refused)}


# --------------------------------------------------------------------------------------------------------------
# Offline replay over a finished attempt's packets.

def replay(packets: list[Mapping], cfg: Mapping = EXAMPLE, student_max_steps: int = 32) -> dict:
    """What the guard's refusal rules would have refused in one attempt, from its teacher packets in order. The
    decision taken on a packet and its assessment are read from the next packet's ``previous_result`` (the last
    packet's are unknown; so is an eef target, so ``wrist_turn`` does not replay). Decisions are not changed: after a
    refusal the teacher would have answered differently, so only the first refusal of a stretch is exact. The acting
    switches do not replay (they refuse nothing)."""
    guard = ReviewGuard({**dict(cfg), "enforce": False, **ACTING}, student_max_steps)
    findings = []
    for packet, following in zip(packets, list(packets[1:]) + [None]):
        guard.at_packet(packet)
        record = (following or {}).get("previous_result") or {}
        if record.get("source") != "teacher" or record.get("decision") != packet.get("decision"):
            continue  # unknown, the harness fallback, or OpenWAM alone
        decision = {"mode": record["mode"], "steps": record["requested_steps"]}
        found = guard.check(packet["request_id"], decision, record.get("assessment"))
        if found is not None:
            findings.append({"decision": packet.get("decision"), "native_step": packet.get("native_step"),
                             "rule": found[0], "mode": record["mode"], "steps": record["requested_steps"],
                             "full_chunk_steps": guard.current[1]["full"]})
    return {"summary": guard.summary(), "findings": findings}


def _packet_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("[0-9][0-9][0-9][0-9].json")) if directory.is_dir() else []


def load_packets(path: Path) -> list[dict]:
    """One attempt's teacher packets in decision order: from a runtime directory (its ``teacher/packets``), a packets
    directory, or any directory with exactly one ``teacher/packets`` below it (an attempt directory, say)."""
    path = Path(path)
    found = next((files for files in map(_packet_files, (path / "teacher" / "packets", path / "packets", path))
                  if files), None)
    if found is None and path.is_dir():
        below = sorted(item for item in path.rglob("packets") if item.parent.name == "teacher" and _packet_files(item))
        if len(below) > 1:
            raise ValueError(f"{len(below)} teacher/packets directories under {path}, one per run: name one of "
                             f"{[str(item) for item in below[:4]]}")
        found = _packet_files(below[0]) if below else None
    if not found:
        raise FileNotFoundError(f"no teacher packets under {path}")
    return [json.loads(item.read_text(encoding="utf-8")) for item in found]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay the review guard over finished attempts' teacher packets.")
    parser.add_argument("attempts", nargs="+", type=Path)
    parser.add_argument("--guard", type=Path, help="a JSON file holding a review.guard mapping (default: the v2 one)")
    parser.add_argument("--student-max-steps", type=int, default=32)
    args = parser.parse_args(argv)
    cfg = json.loads(args.guard.read_text(encoding="utf-8")) if args.guard else EXAMPLE
    try:
        validate_guard({**cfg, "enforce": False, **ACTING} if isinstance(cfg, Mapping) else cfg)
    except ValueError as exc:
        parser.error(f"--guard: {exc} (only the refusal rules replay)")
    failed = 0
    for attempt in args.attempts:
        try:
            line = {"attempt": str(attempt), **replay(load_packets(attempt), cfg, args.student_max_steps)}
        except (OSError, ValueError, KeyError, TypeError) as exc:  # one unreadable attempt does not stop the others
            failed += 1
            line = {"attempt": str(attempt), "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(line, sort_keys=True) + "\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
