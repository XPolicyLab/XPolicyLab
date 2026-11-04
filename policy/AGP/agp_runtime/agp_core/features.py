"""Switches for the individual speed-ups, so each one can be turned off for ablation.

The fixed system is SYSTEM below, the value of every switch: the adapter (agp_core/official_deploy.py load_config) gives
the policy processes and the worker exactly these values as environment variables, and the freeze manifest
(agp_core/freeze.py) records the flags they give. Only development runs change single values (an A/B arm's overlay,
agp_core/freeze.py validate_overlay); a frozen measurement runs SYSTEM unchanged.

Every flag is read from the environment (value "0" turns it off) by the MCP server,
the launcher and the prompt renderer, so one setting changes the tools and the
prompt text together. A name missing from an environment falls back to the default of
enabled() (OPT_IN), which only the development tree's older pipelines and unit tests
rely on. "direct" follows OURS_TOOL_MODE (hybrid: direct tool calls plus exec; code:
exec only).
"""
import os
import re

FLAGS = {
    # Tools and returns.
    "run": "OURS_RUN",                  # server-side multi-step `run` tool
    "compact": "OURS_COMPACT",          # compact JSON views instead of full runtime results
    # Shorthands.
    "ev_defaults": "OURS_EV_DEFAULTS",  # default clearance for retreat moves and home
    "preload": "OURS_PRELOAD",          # task text and first observation attached to the launch message
    "candidates": "OURS_CANDIDATES",    # several candidate pixels per measurement query
    "dsl": "OURS_DSL",                  # short command strings for steps and queries
    # Prompt sections with general guidance.
    "p_time": "OURS_P_TIME",            # where the time goes: plan once, chain, short text
    "p_images": "OURS_P_IMAGES",        # which images to look at
    "p_reach": "OURS_P_REACH",          # measured reach table and how to use it
    "p_motion": "OURS_P_MOTION",        # general motion and manipulation experience
    "p_dual": "OURS_P_DUAL",            # required dual-arm parallelism
    # Speed variants (ported from the real-robot version). All five off reproduce the default pipeline exactly;
    # see OPT_IN below for the defaults.
    "p_occlusion": "OURS_P_OCCLUSION",        # prompt: arms and held objects hide parts of the top image
    "end_on_assess": "OURS_END_ON_ASSESS",    # launcher ends codex once assess is accepted (no closing turn)
    "fast_precision": "OURS_FAST_PRECISION",  # simulator: precision linear 3 -> 8 cm/s, joint x1.75 (worker_ext)
    "trial_transit": "OURS_TRIAL_TRANSIT",    # prompt: the verification lift may run at transit (intent retreat)
    "grip_converge": "OURS_GRIP_CONVERGE",    # simulator: a gripper action ends when the fingers arrive (worker_ext)
    # Grasp stability. Both off reproduce the default.
    "p_grasp": "OURS_P_GRASP",                # prompt: grasp on faces parallel to the pads, near the centre of mass
    "hold_smooth": "OURS_HOLD_SMOOTH",        # simulator: smooth start and stop ramps while an object is held (worker_ext)
    # Official leaderboard conditions. Off reproduces the pipeline above unchanged.
    "policy_planner": "OURS_POLICY_PLANNER",  # simulator: the worker plans with agp_core/policy_planner.py (planner_injection)
    "rgb_only": "OURS_RGB_ONLY",              # observe from get_obs only (rgb_observe), measure without depth
                                              # (rgb_measure), official step limit kept, prompt with a step budget
    # Speed profiles (agp_core/speed_profiles.py, composed by agp_core/official_worker.py under official conditions only).
    # All three off reproduce the pipeline above unchanged (the mixin is not composed).
    "free_topt": "OURS_FREE_TOPT",            # F: empty-gripper transit moves and home re-timed along their path
                                              # (joint trapezoid 4 rad/s, 20 rad/s^2; TCP 1 m/s, 3 rad/s; zones)
    "arm_guard": "OURS_ARM_GUARD",            # F-g: two-arm clearance before a stage and before every control step
    "hold_guard": "OURS_HOLD_GUARD",          # D: stop a motion when the official observation shows the held
                                              # object lost (fingers fully closed)
    # Gripper, hold and precision-zone profiles (G and H in agp_core/speed_profiles.py, P there (the lift) and in
    # agp_core/prec_zone.py + agp_core/runner.py (contact and place splits of a run)). All three off: unchanged.
    "grip_obs": "OURS_GRIP_OBS",              # G: a gripper action ends when the official observation shows the
                                              # fingers stopped (close) or open enough (open), else 10 samples
    "prec_zone": "OURS_PREC_ZONE",            # P: precision only for the last 1.5 cm of a contact (match / plane
                                              # height, nearly vertical), the last 2 cm of a place, the first 1 cm
                                              # of a lift
    "hold_profile": "OURS_HOLD_PROFILE",      # H: transit moves while holding as a Cartesian s-curve (0.2 m/s,
                                              # 1 m/s^2, 1 rad/s, joints 2 rad/s)
    # Guard and hold refinements (agp_core/speed_profiles.py, agp_core/runtime_ext.py). All five off reproduce the pipeline
    # above unchanged; each acts only together with the speed-profile switches named here.
    "guard_v2": "OURS_GUARD_V2",              # F-g v2 (needs OURS_ARM_GUARD): layer 1 at 15 mm, a delay of up to the
                                              # other arm's whole stage, a refusal that names only this pair; the
                                              # runtime keeps measurements and arm states after a step no arm started
    "gf_open": "OURS_GF_OPEN",                # G x F (needs OURS_GRIP_OBS and OURS_FREE_TOPT): an open that G ended
                                              # as wide enough counts as empty for F
    "hold_clearance": "OURS_HOLD_CLEARANCE",  # a held transit move keeps the held object's lowest point 3 cm above
                                              # the table along its lateral part (lift first, descend last)
    "drop_stop": "OURS_DROP_STOP",            # (needs OURS_HOLD_GUARD): a watched lift or transport stops when the
                                              # official reading falls to 0 while holding
    "hold_watch2": "OURS_HOLD_WATCH2",        # (needs OURS_HOLD_GUARD): D starts watching unless both signals show
                                              # the closed gripper empty (F's rule)
    # Place split and geometry prompts (agp_core/prec_zone.py + agp_core/runner.py, agp_core/PROMPT.md). All three off reproduce
    # the pipeline above unchanged (no prompt text, no runner path).
    "place_split": "OURS_PLACE_SPLIT",        # the run splits a long, nearly vertical place descent of a holding arm
                                              # (measurement-relative as P, or a relative displacement) into a transit
                                              # part and a precision tail, when that saves steps; invisible to the model
                                              # (no prompt text, no split entry in a result); official conditions only
                                              # (OURS_RGB_ONLY); with OURS_PREC_ZONE on, P's split acts. Not split when
                                              # the referenced support's height source is fallback, uncertain or none;
                                              # a 0.5 cm tail when the held object's lowest point ends within 1 cm
                                              # above the referenced surface; the 2 cm tail for a hanging release
                                              # above it
    "p_geometry": "OURS_P_GEOMETRY",          # prompt: the gripper body's geometry and the side-by-side TCP
                                              # separations instead of "keep the TCPs at least 12 cm apart" (acts with
                                              # OURS_ARM_GUARD under OURS_RGB_ONLY; the margin follows OURS_GUARD_V2)
    "p_hold_cost": "OURS_P_HOLD_COST",        # prompt: a held transit move only by its cost per centimetre, the
                                              # stop-accuracy sentence, no s-curve or rate words (acts with
                                              # OURS_HOLD_PROFILE under OURS_RGB_ONLY)
    # Verified fast hold and grasp depth (agp_core/speed_profiles.py, agp_core/grasp_depth.py + agp_core/runner.py). Both off
    # reproduce the pipeline above unchanged; both are invisible to the model (no prompt text, no result text).
    "hold_verified_fast": "OURS_HOLD_VERIFIED_FAST",  # (needs OURS_HOLD_PROFILE): a grasp whose first lift after its
                                              # close kept the pad gap from the official reading within 0.5 mm and
                                              # ended with the reading > 0 carries its later held transit moves at
                                              # 0.3 m/s and 2 m/s^2 (H's other limits); every other grasp keeps H
    "grasp_depth": "OURS_GRASP_DEPTH",        # (official conditions, OURS_RGB_ONLY): a contact descent within 45 deg
                                              # of vertical onto a target measured <= 25 mm above the table goes down
                                              # until the fingers' lowest point is 4 mm above the support measured under
                                              # the fingertips (at most 3 mm lower when that support has no match/plane
                                              # result, never more than 10 mm lower, never up); only when the agent's
                                              # own target puts the fingers' lowest point <= 25 mm above the table
    # Upright symbols (agp_core/PROMPT.md). Off reproduces the pipeline above unchanged (no prompt text); no runtime code
    # reads it.
    "p_upright": "OURS_P_UPRIGHT",            # prompt: an object in the form of (or marked with) a character or
                                              # symbol ends upright in the top image; the turn a wrist rotation or a
                                              # tilted release gives the held object; check it after the release
    # Intent manipulate (agp_core/runtime_ext.py, agp_core/syntax.py, agp_core/runner.py, agp_core/server.py, agp_core/PROMPT.md). Off
    # reproduces the pipeline above unchanged (the runtime refuses the word as an unknown intent).
    "intent_manipulate": "OURS_INTENT_MANIPULATE",  # a seventh, generic runtime intent "manipulate": precision speed
                                              # only; no measurement required (a measurement-relative target still
                                              # needs a valid one); not the lift after a close, no transfer unlock, no
                                              # target confirmation; every measurement invalidated, as after contact
    # Waiting, counting, checklist and carry guidance (agp_core/PROMPT.md, agp_core/syntax.py + agp_core/runner.py,
    # agp_core/runtime_ext.py). Each off reproduces the pipeline above unchanged (no prompt text, no syntax, no runtime
    # class change).
    "wait_still": "OURS_WAIT_STILL",          # prompt: while another robot or a demonstration moves, both arms stay
                                              # at their start poses until it is at rest; when taking turns, home and
                                              # wait after each own move; holds cost steps. run: the hold option
                                              # `still` (`B hold N still`, JSON "still": true) holds up to N steps in
                                              # chunks and ends once the top image is unchanged over two chunks
    "p_counted": "OURS_P_COUNTED",            # (acts with OURS_P_MOTION): prompt: counted contacts are separate
                                              # events, lift at least 3 cm clear after each, no extra one when unsure
    "p_checklist": "OURS_P_CHECKLIST",        # prompt: the instruction as a checklist of objects and subgoals;
                                              # before assess every item checked in the latest top image, fixed while
                                              # the remaining steps allow, listed in a success summary
    "p_turn_carry": "OURS_P_TURN_CARRY",      # (acts with OURS_P_MOTION): prompt: a large turn of a held object
                                              # in the transit carry (intent transfer), the slow last part at precision
                                              # (intent manipulate when OURS_INTENT_MANIPULATE is on)
    "keep_unstarted": "OURS_KEEP_UNSTARTED",  # the runtime keeps measurements and arm states after a step no arm
                                              # started (runtime_ext.KeepUnstartedRuntime, OURS_GUARD_V2's runtime half)
                                              # without F-g v2's worker changes; with OURS_GUARD_V2 and OURS_ARM_GUARD
                                              # on it changes nothing
    # Return reserve and handover (agp_core/return_reserve.py, agp_core/handover.py, agp_core/runtime_ext.py,
    # agp_core/official_worker.py, agp_core/compact.py, agp_core/PROMPT.md). Each off reproduces the pipeline above unchanged (no
    # prompt text, no runtime class, no worker mixin, no result field).
    "return_reserve": "OURS_RETURN_RESERVE",  # (official conditions, OURS_RGB_ONLY): every observation reports the
                                              # step accounting (goal releases, steps per release, the steps a home of
                                              # both arms needs now, the spare steps after it and a margin, the
                                              # releases they allow); a step that is not a home and needs more steps
                                              # than the spare ones is refused before any motion unless its evidence
                                              # declares it final or that no return is needed (no_return)
    "handover": "OURS_HANDOVER",              # (acts with OURS_ARM_GUARD): in a step whose two arms both declare a
                                              # handover (evidence handover naming the other gripper) F-g keeps 5 mm
                                              # between the two grippers' finger and wrist-camera links instead of its
                                              # clearance; every other link pair and every undeclared step unchanged
    # Task notes (agp_core/task_notes.py, agp_core/task_notes.json, agp_core/run_agent.py): the official scoring requirements of
    # each task. Off reproduces the pipeline above unchanged (the launcher appends nothing).
    "task_notes": "OURS_TASK_NOTES",          # the launcher appends, right after the rendered main prompt, the note of
                                              # the episode's official task (TASK_ENV.task_name as the adapter recorded
                                              # it in episode.json; a _random configuration gets its base task's note)
    # Task hints (agp_core/task_hints.json, agp_core/task_notes.py, agp_core/run_agent.py): a hint for 18 tasks, a layer
    # separate from the task notes, behind its own switch. Off reproduces the pipeline above unchanged (the launcher
    # appends nothing).
    "task_hints": "OURS_TASK_HINTS",          # the launcher appends, right after the task-notes section (right after the
                                              # main prompt when OURS_TASK_NOTES is off), the hint for the episode's
                                              # official task (a _random configuration gets its base task's hint; a
                                              # task without one gets no section)
    # In-place twist (agp_core/speed_profiles.py, composed by agp_core/official_worker.py under official conditions, and the
    # speeds line of agp_core/PROMPT.md). Off reproduces the pipeline above unchanged (no prompt text, the worker class and
    # its records as before).
    "spin_fast": "OURS_SPIN_FAST",            # a precision move_ee / move_delta that only turns the gripper about its own
                                              # axis (the TCP within 5 mm of its start and the tool axis within 2 deg of
                                              # its start direction along the whole path) runs as an s-curve at up to
                                              # 1 rad/s at the TCP and 2 rad/s per joint (H's values), translation
                                              # at the precision cap; any intent, holding or not
    # Yaw flip (agp_core/speed_profiles.py, composed by agp_core/official_worker.py under official conditions). Off reproduces
    # the pipeline above unchanged (the worker class binds the same switch set, no record is written). Invisible to the
    # model: no prompt text, no result text.
    "yaw_flip": "OURS_YAW_FLIP",              # a move_ee (linear or plan, any intent) of an EMPTY gripper (F's two-signal
                                              # rule) executes the requested orientation or the same one turned 180 deg
                                              # about the gripper's own axis (TCP z), whichever goal is nearer the current
                                              # joints (L-infinity, the IK's own measure; only one reachable: that one);
                                              # a holding arm makes no choice and keeps its frame; poses are reported in
                                              # the frame the agent asked for
    # Grip width (agp_core/grip_width.py, agp_core/runtime_ext.py, agp_core/syntax.py + agp_core/runner.py, agp_core/server.py,
    # agp_core/compact.py, agp_core/speed_profiles.py, agp_core/rgb_observe.py + agp_core/rgb_measure.py, agp_core/PROMPT.md). Off
    # reproduces the pipeline above unchanged (the runtime refuses the action, the step verb is unknown, no prompt text,
    # the worker class and its records as before).
    "grip_width": "OURS_GRIP_WIDTH",          # a gripper action that moves the fingers to a requested pad gap (step verb
                                              # `width <mm>`, 0.8 .. 88.8 mm) and stops there or earlier on an object;
                                              # widening while holding is a release (open's evidence), stopping on an
                                              # object counts as a close (grasp state); the result reports the gap
    # Pose query (agp_core/rgb_pose.py, agp_core/syntax.py + agp_core/runner.py, agp_core/server.py, agp_core/PROMPT.md). Off reproduces
    # the pipeline above unchanged (the token and the field are refused, no prompt text, the tool schema as before,
    # agp_core/rgb_pose.py is not imported).
    "pose_query": "OURS_POSE_QUERY",          # a measure query with `pose` also returns the footprint of the colour
                                              # region at its pixel back-projected to the measured height: centroid,
                                              # long-axis heading and its tail, straight-edge direction, extents
    # Faster precision segments (agp_core/speed_profiles.py, composed by agp_core/official_worker.py under official conditions;
    # agp_core/prec_zone.py + agp_core/runner.py; agp_core/PROMPT.md; agp_core/evaluation/estimator.py). Off reproduces the pipeline
    # above unchanged (the worker prepares, re-times and records exactly as before, no prompt text changes, the place
    # split's estimate as before).
    "precision_2x": "OURS_PRECISION_2X",      # official conditions (OURS_RGB_ONLY): a move_ee / move_delta / move_joints
                                              # at speed_profile precision (grasp, contact, place, manipulate, a slow
                                              # move; a place split's precision tail) is prepared at 2.5 times the
                                              # worker's precision limits (0.03 m/s, 0.108 rad/s, 0.3 rad/s per
                                              # joint): 0.075 m/s and 0.27 rad/s at the TCP, 0.75 rad/s per joint;
                                              # P's lift head and the in-place twist's translation cap
                                              # (OURS_SPIN_FAST) likewise; F's landing and take-off zones 0.30 m/s and
                                              # the landing zone's rotation 2.5 rad/s (from 0.12 m/s, 1 rad/s); the
                                              # place split's step estimate and the prompt's precision numbers follow
    # Plan guard (agp_core/speed_profiles.py, composed by agp_core/official_worker.py under official conditions;
    # agp_core/compact.py, agp_core/PROMPT.md). Off reproduces the pipeline above unchanged (the worker prepares and records
    # exactly as before, no result text, no prompt text).
    "plan_guard": "OURS_PLAN_GUARD",          # a move_ee / move_delta in plan mode (holding or empty, any intent, each
                                              # candidate of OURS_YAW_FLIP) whose prepared plan moves the TCP along more
                                              # than 3 times the straight distance between its ends and more than 0.6 m
                                              # (an IK goal on another arm branch: the arm would swing around) is not
                                              # executed: "Planning rejected" with a reason naming the guard and the
                                              # numbers, as when cuRobo rejects the plan; linear moves, home and
                                              # move_joints are not checked
    # Hide end (agp_core/official_worker.py, agp_core/official_deploy.py, agp_core/compact.py, agp_core/PROMPT.md). Off reproduces the
    # pipeline above unchanged (every reply as before, the end grace, the compact views and the prompt as before).
    "hide_end": "OURS_HIDE_END",              # official conditions: the official end flag never leaves the adapter; no
                                              # reply carries episode_ended, no request is answered after the official
                                              # end and the adapter stops codex at once (no end grace); the prompt says
                                              # that a policy still running after the arms are home has not finished
    # Hold frames (a single still hold through a whole demonstration shows only its end, so the order of the shown
    # moves is lost; agp_core/hold_frames.py, agp_core/runner.py, agp_core/PROMPT.md). Off reproduces the pipeline above unchanged
    # (no result field, no image, no prompt text; agp_core/hold_frames.py is not imported).
    "hold_frames": "OURS_HOLD_FRAMES",        # a `still` hold step of the run (OURS_WAIT_STILL) that held more than 50
                                              # steps also returns the top images its chunks' observations carry at
                                              # hold steps 50, 100, ..., each as its own 640x480 image labelled with its
                                              # step in the hold and the episode's steps_used (fixed spacing, no
                                              # scene-change detection); the step's view lists the steps
    "prec_roll": "OURS_PREC_ROLL",
}


# Opt-in switches (off unless set to "1"). For the speed variants, simulator A/B runs found the motion changes
# together raised digit slips (clean 0/5 against 3/5) and the occlusion rule without measurable effect; a 10-pair
# check of trial_transit + grip_converge on the default saved ~90 samples per episode but lost more digits in transport
# and was not faster (paired wall +15 s, turns +5), so they stay opt-in.
OPT_IN = {"p_occlusion", "fast_precision", "trial_transit", "grip_converge", "p_grasp", "hold_smooth",
          "policy_planner", "free_topt", "arm_guard", "hold_guard", "grip_obs", "prec_zone",
          "hold_profile", "guard_v2", "gf_open", "hold_clearance", "drop_stop", "hold_watch2", "place_split",
          "p_geometry", "p_hold_cost", "hold_verified_fast", "grasp_depth", "p_upright", "intent_manipulate",
          "wait_still", "p_counted", "p_checklist", "p_turn_carry", "keep_unstarted",
          "return_reserve", "handover", "task_notes", "task_hints", "spin_fast", "yaw_flip", "grip_width",
          "pose_query", "precision_2x", "plan_guard", "hide_end", "hold_frames", "prec_roll"}


# The fixed system: the value of every switch, as the environment strings the adapter gives the policy processes and
# the worker (in the order of FLAGS, OURS_TOOL_MODE first). Every flag is on except those in SYSTEM_OFF. Three of these
# (fast_precision, grip_converge, hold_smooth) act on measured finger joints, which the official observation does not
# contain, so the adapter refuses them on (agp_core/official_deploy.py REQUIRED_OFF); it also refuses policy_planner or
# rgb_only off (REQUIRED_ON).
SYSTEM_OFF = ("p_occlusion", "fast_precision", "trial_transit", "grip_converge", "p_grasp", "hold_smooth", "prec_zone",
              "guard_v2", "hold_clearance", "drop_stop", "hold_watch2", "p_geometry", "p_hold_cost",
              "hold_verified_fast", "grasp_depth")
SYSTEM = {"OURS_TOOL_MODE": "hybrid", **{env: "0" if name in SYSTEM_OFF else "1" for name, env in FLAGS.items()}}
if set(SYSTEM_OFF) - set(FLAGS):
    raise RuntimeError(f"SYSTEM_OFF names unknown flags {sorted(set(SYSTEM_OFF) - set(FLAGS))}")


def system_switches():
    """A copy of SYSTEM (environment strings)."""
    return dict(SYSTEM)


def enabled(name, environ=None):
    environ = os.environ if environ is None else environ
    if name == "direct":
        return environ.get("OURS_TOOL_MODE", "hybrid") == "hybrid"
    value = environ.get(FLAGS[name], "0" if name in OPT_IN else "1")
    if name == "rgb_only" and value == "0":
        raise RuntimeError("OURS_RGB_ONLY=0 is refused: the policy observes and measures from RGB images only")
    return value != "0"


def current(environ=None):
    flags = {name: enabled(name, environ) for name in FLAGS}
    flags["direct"] = enabled("direct", environ)
    return flags


_BLOCK = re.compile(r"<!--if ([!\w,]+)-->\n(.*?)<!--endif-->\n", re.S)


def render(text, flags):
    """Keep `<!--if a,!b-->` ... `<!--endif-->` line blocks whose terms all hold (`!name`: flag off)."""
    def keep(match):
        terms, body = match.groups()
        for term in terms.split(","):
            name = term.lstrip("!")
            if name not in flags:
                raise KeyError(f"unknown prompt flag {name}")
            if flags[name] == term.startswith("!"):
                return ""
        return body
    rendered = _BLOCK.sub(keep, text)
    if "<!--if" in rendered or "<!--endif" in rendered:
        raise ValueError("unbalanced prompt condition")
    return rendered
