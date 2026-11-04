"""Scripted policy: a short fixed sequence through the same MCP tools as the agent, without any LLM.

    .venv-agent/bin/python -B agp_core/official_scripted.py SESSION --name smoke|observe|idle
        [--fault mcp_start|mcp_crash|agent_crash]

Started by agp_core/official_deploy.py (AGP_SCRIPTED=<name>) in place of agp_core/run_agent.py, with the same clean
environment and switch set, and the MCP server agp_core/server.py over stdio (the server code the agent uses).

  smoke    goal; observe (official observation: no depth files or calibration depth entries, own step count
           equals steps_used, steps_left = official limit - steps_used); measure 4 top pixels on the table plane
           for both arms (z_from plane, the two arm frames 0.6 m apart along y) and one pixel without plane_z
           (recorded only); one paired top-down move_ee plan step in free air (both reached, joint-trajectory
           preflight ok in motion_diagnostics); a paired close (official conditions: the outcome follows
           the wrist-image finger estimate recorded in the full step result -- reached when its quality is ok and
           every finger is within 3 mm of the closed endpoint, as in free air; held_short otherwise -- and no
           finger joints); a paired open with empty-release evidence (reached); a `run` of a short
           precision retreat lift and home for both arms (a run stops at a motion that ends short of its target,
           a physical outcome that is recorded; home then goes as a plain step); a final observe; assess failure
           (the task is not attempted).
           Approximate home (agp_core/approx_home.py; policy-side checks only):
             * that home: both arms homed approximately (reached inside the END margin), in fewer control steps
               than a full return (the larger of the two arms' cuRobo plans to the exact start joints, recorded as
               full_return_samples, which a paired step to the exact start joints takes);
             * a 25-sample hold of both arms right after;
             * budget, at the edge (with the public URDF as planning model, i.e. in the simulator): the right arm
               moved to FREE_TEST_JOINTS (move_joints; the start of a final home that a recorded codex episode had
               refused for budget), holds until steps_left is FREE_TEST_STEPS_LEFT (as in that episode), then a home
               of both arms: it starts although neither a full return nor the right arm's restricted plans fit, the
               right arm takes a free plan (any joint values) and the step stays within steps_left (with
               OURS_FREE_TOPT on the re-timed restricted plan fits too; the check is then that a full return does
               not fit and the home stays within steps_left). With another planning model (the unit tests' toy
               kinematics) the budget test is the paired top-down move again, holds until steps_left is the first
               home's samples + 6, then a home of both arms that starts below a full return;
           Every observation check also compares the finger gap of the compact view with the wrist-image estimate
           in the runtime's full observation (shown when its quality is ok, else null).
           Speed and guard test: on the spare layouts listed in M3S1A_LAYOUTS the smoke runs it instead (m3s1a): free
           transit moves of both arms, low moves of the layout's low arm over an area free of objects (low-altitude,
           landing and take-off zones), a rotation, a full home of one arm and an approximate home, a transit move with
           a gripper closed on nothing; with OURS_ARM_GUARD=1 also a paired stage that passes, one on a collision course
           and a crossing; then the held-object test: the right gripper closes on the left gripper's finger (the left
           arm points forward, fingers open) and lifts off it (OURS_HOLD_GUARD=1 stops that lift when the observation
           shows the fingers fully closed). Each stage's compact result is recorded (report m3s1a); the worker's records
           (motion_diagnostics, arm_guard.jsonl) hold the rest. Every other layout runs the smoke above unchanged.
           G / P / H test: on the spare layouts listed in M3S1B_LAYOUTS the smoke runs it instead (m3s1b), through `run`
           as the agent does: real blocks picked from their first top image (measured with a raised plane_z, staged 6 cm
           above, the label re-measured in the wrist view), a contact descent to them (P's split when the height is a
           match), the closes (G), the verification lift (P's lift: first 1 cm at precision), the carry while holding
           (H: a linear and a plan move), a place on the table at a measured spot (P's place split), the opens (G), an
           empty close and open of both grippers (G), a home; the contact descents are one step per arm (a step is split
           only when all its descents qualify); on layout 45 the free arm also makes an oblique contact descent and one
           to a fallback height after the pick and place (P's refusals, live; a contact move invalidates every
           measurement). The capture check accepts the matcher's review images (match_*.png) next to the capture files.
           Each run's compact result is recorded (report m3s1b); the decisions and the worker's records
           (prec_zone.jsonl, motion_diagnostics) hold the rest. Nothing here asserts a physical outcome (a missed grasp
           is recorded, not a failure).
           Pick-and-place sequence: on the spare layouts listed in M3E0_LAYOUTS (one per tested task configuration)
           the smoke runs it instead (m3e0): a pick and place of each picking arm's object through `run` (rise,
           measure, stage, the contact descents one arm per step, close, verification lift, carry linear and plan,
           the table spot at the pick pixel, place, open, retreat, home of both arms; general_pickup: the lift by
           11 cm is its task, the sequence ends there). Records only, like m3s1b (report m3e0).
           Manipulate test (OURS_INTENT_MANIPULATE): on the spare layouts listed in M4MI_LAYOUTS the smoke runs it
           instead (m4mi), in free air above the table: the paired top-down move (step tool), a table point measured for
           the left arm (plane), a paired precision manipulate move through `run` (accepted, both reached, steps used),
           a move to the measured point (refused: the manipulate step has invalidated every measurement), a transit
           manipulate move through `run` and through the step tool (both refused before any motion, no step used), a
           precision manipulate move through the step tool (accepted), a retreat and home of both arms, the preflight as
           in the other smokes (report m4mi). With the switch off the first manipulate move is refused as an unknown
           intent and the smoke fails there.
           Handover and return-reserve test (OURS_HANDOVER, OURS_RETURN_RESERVE): on the spare layouts listed in
           M4GB_LAYOUTS the smoke runs it instead (m4gb; report m4gb), in free air above the table, through the step
           tool and `run` as the agent calls them:
             * the declared handover (OURS_HANDOVER): the left gripper points at the right arm and the right gripper at
               the left arm (TCPs face to face, opening axes parallel), 7 cm between the TCPs; the right arm's precision
               move of 3 cm toward the left gripper (the left arm holds) brings the two grippers' fingers within about
               15 mm of each other (every other pair of links stays about 9 cm apart): undeclared it is refused by F-g
               before any motion (no step used); declared by the right arm only, refused as well (with the one-arm
               sentence); a `handover` evidence that does not name the other gripper is refused by the runtime before
               dispatch (INVALID_EVIDENCE); declared by both arms it runs and both reach; the right arm moves back
               undeclared (moving apart is allowed) and both arms home. Whether the declared motion stayed clear of
               contact is measured afterwards from the worker's records;
             * the return reserve (OURS_RETURN_RESERVE): a paired top-down move, a close and a release at a goal
               (counted: releases 1) and a close and an empty open (not counted) in free air; the observation's budget
               (return of both arms, spare = steps left - return - margin, steps per release, releases left) checked
               against its own numbers; holds down to a spare of M4GB_SPARE steps; then a `run` move that needs more
               than the spare steps and an undeclared hold of spare + 1 (both refused before any motion, no step used),
               the same hold with evidence no_return (runs), an undeclared hold of 1 (refused) and with evidence final
               (runs), and a home of both arms, which is never refused by the reserve: both reach the return region
               (inside the END margin), and the budget shows return 0 afterwards.
           In-place twist probe (OURS_SPIN_FAST): on the spare layout listed in M4SPIN_LAYOUTS the smoke runs it instead
           (m4spin; report m4spin), in free air 22 cm above the table, through the step tool as the agent calls it, the
           same sequence with the switch on and off (an A/B pair): the paired top-down move; a paired precision
           manipulate twist of the empty open grippers about their own axes by +90 deg (left) and -90 deg (right)
           (move_delta drot_deg); the twist back (move_ee to the start rotation at the same position); a control move of
           the left arm that turns 30 deg and moves 1 cm (not in place: precision rate in both arms); a home of both
           arms; the preflight as in the other smokes. Each stage's steps are recorded; the worker's records
           (motion_diagnostics) hold the tracking error, the TCP path and the re-timing. The check of the steps follows
           the switch: with it on each twist stage needs at most M4SPIN_ON_MAX steps, with it off at least the precision
           samples of a 90 deg turn.
           Grip width probe (OURS_GRIP_WIDTH): on the spare layout listed in M4GW_LAYOUTS the smoke runs it instead
           (m4gw; report m4gw), through `run` as the agent calls it: both arms lift, measure their blocks and stage 6 cm
           above them (fingers down, as m3s1b); widths 20, 40 and 60 mm of the empty grippers and the 55 mm pre-shape
           (each at the requested gap by the wrist image within M4GW_TOLERANCE_MM, reached, the official reading equal
           to the command, the observation's width_mm); a new measurement (the widths invalidated the old ones) and the
           contact descent around the blocks; a 25 mm width (on a block it stops on the object: held_short, ok, the
           close's check verify_grasp_attempt, not a stop of the run); a close on the blocks and a 3 cm lift; a widening
           width of the holding right arm without release evidence (refused by the runtime before any motion); the place
           descent and the release by a 60 mm width with support evidence (widening, at the requested gap, reached); a
           retreat and two widths of the empty grippers (40 mm, then 70 mm: accepted without evidence); a home of both
           arms; the preflight as in the other smokes. Per step and arm the report keeps the worker's width record, the
           official reading after the step (equal to the command, or the gap its clamp shows), the wrist-image estimate
           and the observation's width_mm; a missed block is recorded, not a failure.
  observe  goal, observe, the capture checks, assess failure (no motion).
  idle     no tool call (the episode ends with no action).

Writes scripted_report.json and policy_finished.json in the session. Exit 0 only when every check passed; any
other code is an infrastructure failure for the adapter. --fault mcp_start starts the server without the `agp_core`
package on its path (test of the adapter's infrastructure-failure path: the MCP server cannot start).

Faults (our own code dying mid-episode is a counted policy failure, kind mcp_crash):
  --fault mcp_crash    the MCP server runs over streamable HTTP as agp_core/run_agent.py starts it for codex, as
                       agp_core/mcp_crash_server.py (agp_core/server.py that crashes by itself, os._exit(70), while answering
                       its CRASH_AT_CALL-th tool call). Sequence: goal, observe, measure (table plane), the paired
                       top-down move (steps used), the paired close (the server executes it, then dies before
                       answering). The driver watches the server process as run_agent.py does (wait_official) and
                       records its exit in policy_finished.json the same way (stopped_by mcp_server_exited, mcp_server
                       {returncode, exited_during_episode}); exit 0 when the crash happened as injected. Development
                       tree only: the packaged runtime leaves agp_core/mcp_crash_server.py out (scripts/package_policy.py),
                       so the packaged driver refuses this fault (exit 2) before anything starts.
  --fault agent_crash  the driver stands in for agp_core/run_agent.py dying by itself: once its MCP session is up it writes
                       agent.pid (as run_agent.py does when codex starts), runs goal, observe and the paired top-down
                       move, saves scripted_report.json and ends with os._exit(70): no policy_finished.json, no
                       clean-up, no signal.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def _load_paths():
    """agp_core/paths.py by its path (this script runs with ROOT off sys.path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("agp_scripted_paths", ROOT / "agp_core" / "paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the agent interpreter is in AGP_HOME (agp_core/paths.py; the adapter passes AGP_HOME)
PATHS = _load_paths()
AGENT_PYTHON = PATHS.AGP_HOME / ".venv-agent" / "bin" / "python"
# test-only fault whose server exists in the development tree only (agp_core/official_deploy.py DEVELOPMENT_ONLY_FAULTS)
DEVELOPMENT_ONLY_FAULTS = {"mcp_crash": "agp_core/mcp_crash_server.py"}
EXPECTED_TOOLS = {"goal", "observe", "measure", "step", "run", "action_status", "assess"}
CAPTURE_FILES = {"top.png", "left_wrist.png", "right_wrist.png", "calibration.json", "robot_state.json"}
HEAD_FX = 288.13254096884566
MOTIONS = {"move_ee", "move_delta", "move_joints", "home"}
CRASH_EXIT = 70           # --fault mcp_crash / agent_crash: the crashing process's own exit status (not a signal)
# Last-resort budget test: the start of a final home that a recorded codex episode (stack_bowls seed 0) had refused
# for budget (the right arm's joints in the observation at control step 771, 29 of 800 steps left, the left arm back
# from an earlier approximate home), here with the left arm back from the smoke's first home. From there the right
# arm's restricted plans need 31 (standard) and 30 (last chance) steps and the free plan 22 with the STANDARD margins
# (offline, the real planner), so with FREE_TEST_STEPS_LEFT steps left only a free plan fits, as at step 771 of that
# episode. The pose is in free air (TCP 7.7 cm above the table); the right arm's motions of this test stay within
# x 0.09..0.40 m, y -0.24..-0.09 m (env frame) with the TCP at least 7.7 cm above the table, clear of the blocks of
# stack_blocks seed 0 layout 0 (the default smoke layout) and of the spare layout 29 (all blocks at x < -0.18 m).
FREE_TEST_JOINTS = {"right": [0.534021020, 2.13417673, 1.89012897, -1.31876922, 0.00191399152, 2.10260653]}
FREE_TEST_STEPS_LEFT = 29
CRASH_AT_CALL = 5         # --fault mcp_crash: goal, observe, measure, paired move, paired close (the crash)
# Speed and guard test layouts (spare layouts of seeds 0 and 2, chosen from their first top image in a scripted
# observe run): the arm that makes the low moves and where (its own base frame, metres; the area around these points,
# from TCP height 0.04 m, holds no object in that layout).
M3S1A_LAYOUTS = {
    ("arrange_largest_number", 0, 29): {"low_arm": "right", "low_xy": (0.30, -0.18), "low_z": 0.04,
                                        "low_shift": (0.0, 0.08, 0.0)},    # nearest digit ("5") 33 cm away
    ("arrange_largest_number", 2, 26): {"low_arm": "left", "low_xy": (0.30, 0.10), "low_z": 0.04,
                                        "low_shift": (0.0, -0.08, 0.0)},   # nearest digit ("7") 16 cm away
}
M3S1A_D = {   # the held-object test (base frames, metres): the left arm points forward (its start orientation) at
              # left_pose, fingers open; the right arm, top-down with its opening axis along the finger's thickness,
              # goes to grip_side (plan), then straight to grip_above over the left's inner finger (its pad region),
              # descends by grip_descent (precision) so that its fingertips are 10 mm below that finger's top edge,
              # closes on it and lifts by lift (precision). Poses from the public URDF, the finger profile of
              # the public finger meshes and the right arm's reach (offline).
    "left_pose": (0.25, -0.25, 0.15), "left_quat": (0.5, 0.5, 0.5, 0.5),
    "grip_side": (0.255, 0.20, 0.2095), "grip_above": (0.255, 0.3025, 0.2095), "grip_yaw": 90.0,
    "grip_descent": 0.05, "lift": 0.04}
# G / P / H test layouts (spare layouts of seed 0, stack_blocks, first top images of a scripted observe run): per arm
# the block it picks (top pixel), the free table spot it places it on (top pixel; the base-frame points from
# agp_core/public_camera.py: L44 left (0.405, 0.004) -> (0.32, -0.04), right (0.233, 0.094) -> (0.32, 0.00); L45 right
# (0.258, 0.078) -> (0.33, -0.02)), and the plan carry's target (base frame, 10 cm above the table, near the spot).
# Layout 45's left arm has no block in reach: it makes P's two live refusals over free table (an oblique contact, 2 cm
# off the measured spot; a contact to a raised plane_z guess the match disagrees with: fallback).
M3S1B_LAYOUTS = {
    ("stack_blocks", 0, 44): {"picks": {"left": (180, 212), "right": (429, 287)},
                              "places": {"left": (197, 253), "right": (461, 253)},
                              "carry": {"left": (0.33, -0.03, 0.10), "right": (0.31, 0.01, 0.10)}},
    ("stack_blocks", 0, 45): {"picks": {"right": (435, 275)}, "places": {"right": (469, 249)},
                              "carry": {"right": (0.32, -0.01, 0.10)},
                              "refusals": {"arm": "left", "oblique": (175, 262), "fallback": (135, 241)}},
}
# Pick-and-place smoke layouts (one per tested task configuration; spare layouts of seeds 0 and 2 that earlier codex
# runs already used, so no unused spare layout is spent): per picking arm the object's top pixel (from the layout's
# first top image and the codex agents' own measurements there), the raised plane_z guess of its top (the staged
# wrist view re-measures it), the rotation, the TCP offset below the measured top for the contact (absent: the M3S1B
# rule), and what it is (the query text). The object is put back down where it was picked (the table spot at the same
# top pixel, z = 0) after a carry (a linear move, then a plan move to M3E0_CARRY; "carry": False skips the plan move
# to stay in a short step limit). general_pickup's task is to lift the tape by 10 cm: its run ends with the lift and
# a further 8 cm transit lift (the official episode ends there on success; "carry_place": False).
M3E0_LAYOUTS = {
    ("stack_blocks", 2, 35): {"picks": {"left": (203, 244), "right": (462, 258)},
                              "z": {"left": 0.03, "right": 0.03}, "what": {"left": "pale block", "right": "striped block"}},
    ("stack_blocks_random", 2, 28): {"picks": {"left": (185, 255), "right": (451, 201)},
                                     "z": {"left": 0.025, "right": 0.025},
                                     "what": {"left": "red block", "right": "gray block"}},
    ("stack_bowls", 0, 36): {"picks": {"left": (280, 216), "right": (428, 300)},
                             "z": {"left": 0.045, "right": 0.045}, "rot": {"left": "down0", "right": "down90"},
                             "down": {"left": 0.008, "right": 0.008},
                             "what": {"left": "left bowl near rim", "right": "front bowl right rim"}},
    ("arrange_largest_number", 2, 29): {"picks": {"left": (184, 261), "right": (475, 229)},
                                        "z": {"left": 0.018, "right": 0.018},
                                        "what": {"left": "digit 3", "right": "digit 7"}},
    ("general_pickup", 0, 54): {"picks": {"right": (367, 253)}, "z": {"right": 0.025}, "down": {"right": 0.003},
                                "what": {"right": "white packing tape top rim"}, "lift_more": 0.08,
                                "carry_place": False},
    ("solve_equation", 0, 51): {"picks": {"left": (233, 272), "right": (448, 307)},
                                "z": {"left": 0.018, "right": 0.018}, "what": {"left": "yellow 4", "right": "blue six"},
                                "carry": False},
    ("classify_objects", 2, 53): {"picks": {"left": (143, 240), "right": (354, 290)},
                                  "z": {"left": 0.012, "right": 0.009}, "rot": {"left": "down15", "right": "down35"},
                                  "what": {"left": "left pen", "right": "right pen"}},
}
# The manipulate smoke's layouts (a spare layout of seed 0 that only a scripted observe run has used, so no unused spare
# layout is spent; every motion stays in free air above the table) and the top pixel of the table point measured for the
# left arm (plane_z 0: a ray-plane point, whatever lies there).
M4MI_LAYOUTS = {("arrange_largest_number", 0, 28): {"table_pixel": (320, 300)}}
MANIPULATE_REFUSAL = "Manipulate runs at precision speed only: transit speed and home are refused"   # runtime_ext.py
# The handover and return-reserve smoke's layout (a spare layout of seed 2 that earlier scripted and codex runs used:
# no unused spare layout is spent) and its poses, in each arm's base frame (metres; quaternions w, x, y, z): the left
# TCP 26.5 cm toward the right arm, its approach axis toward it (base -y), the right TCP the same toward the left arm
# (base +y), both 25 cm above the table, the opening axes along base x (the right one turned by 180 degrees about its
# approach axis, so that its IK has joint6 near 0 instead of pi, at the joint limit: with the other orientation the
# planner refuses the move). Offline (public URDF, hull geometry): at 7 cm between the TCPs the closest links are the
# two grippers' fingers, about 45 mm apart; after the right arm's 3 cm move about 15 mm, every other pair of links
# about 9 cm.
M4GB_LAYOUTS = {("arrange_largest_number", 2, 27): {
    "left": ((0.30, -0.265, 0.25), (0.70710678, 0.70710678, 0.0, 0.0)),
    "right": ((0.30, 0.265, 0.25), (0.0, 0.0, 0.70710678, 0.70710678)),
    "right_stage": (0.30, 0.185, 0.25), "approach": (0.0, 0.03, 0.0)}}
# The in-place twist probe's layout (a spare layout of seed 0 that earlier codex runs used and no scripted smoke: no
# unused spare layout is spent; every motion in free air 22 cm above the table, the poses of topdown_actions) and its
# moves
M4SPIN_LAYOUTS = {("arrange_largest_number", 0, 27): {"turn_deg": 90.0, "control": {"dpos": (0.0, 0.01, 0.0),
                                                                                    "turn_deg": 30.0}}}
# Grip width (OURS_GRIP_WIDTH): the width probe's layout (a spare layout of seed 0 that earlier codex runs used and no
# scripted smoke, so no unused spare layout is spent: its first top image shows two blocks of about 35 mm on the left
# arm's side and one on the right arm's; the pixels are those the codex agents measured there) and its widths: empty
# widths in the air above the blocks, a pre-shape wider than a block, a width narrower than a block (it stops on the
# block), the release width, two empty widths after the release (narrowing, widening)
M4GW_LAYOUTS = {("stack_blocks", 0, 47): {"picks": {"left": (193, 224), "right": (462, 207)},
                                          "empty_widths": (20.0, 40.0, 60.0), "preshape": 55.0, "on_object": 25.0,
                                          "release": 60.0, "after": (40.0, 70.0)}}
M4GW_TOLERANCE_MM = 1.0        # an empty width is at its gap within grip_width.MARGIN_M (1.0 mm)
M4SPIN_ON_MAX = 70             # with OURS_SPIN_FAST on a 90 deg twist stage needs at most this many steps (48 samples
                               # and the settle; with it off 364 precision samples)
M4SPIN_OFF_MIN = 364           # ceil((pi / 2) / 0.00432): the precision samples of a 90 deg turn
M4SPIN_OFF_MIN_2X = 146        # ceil((pi / 2) / 0.0108): the same with OURS_PRECISION_2X (the precision rate x 2.5)
M4GB_SPARE = 12                # the reserve test holds down to this many spare steps
M4GB_MARGIN = 10               # agp_core/return_reserve.py RESERVE_MARGIN_STEPS
HANDOVER_HINT = "only when both arms are in the step and each declares it"     # agp_core/handover.py ONE_ARM_HINT
RESERVE_REFUSAL = "Return reserve: this step needs"                            # agp_core/return_reserve.py REFUSAL
M3E0_CARRY = (0.28, 0.0, 0.12)  # the plan carry's target in each arm's base frame (the two arms' targets 0.6 m apart)
M3S1B_Z_GUESS = 0.03          # the blocks' top: the raised plane_z guess (the staged wrist view re-measures it)
M3S1B_STAGE = 0.06            # staging height above the block top
M3S1B_GRASP_DOWN = 0.012      # TCP below the block top for the grasp (pads 15 mm long at the fingertips) ...
M3S1B_GRASP_MIN_Z = 0.017     # ... but at least this high above the table (fingertips 12.6 mm below the TCP)
M3S1B_RELEASE_UP = 0.004      # the place target: the grasp height above the table plus this
FAULT_SIGNALS = {int(s) for s in (signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE, signal.SIGILL, signal.SIGABRT,
                                  signal.SIGTRAP, signal.SIGSYS)}


class CheckFailed(AssertionError):
    pass


def topdown(yaw_deg):
    half = math.radians(yaw_deg) / 2
    return {"w": 0.0, "x": math.cos(half), "y": math.sin(half), "z": 0.0}


def topdown_actions():
    """The smoke's paired top-down plan move in free air (left 0.30/0.10/0.22 m yaw 0, right 0.30/-0.10/0.22 m yaw 90).

    0.22 m, not 0.25 m: from 0.25 m the 3 cm precision lift of the smoke needs joint4 past the wrist range (link5
    against link3, agp_core/policy_planner.py WRIST_TABLE), so it would end about 11 mm short at the self-collision stop,
    and the planner refuses it at planning time. From 0.22 m the lift stays inside the range, so the run executes both
    of its steps."""
    targets = {"left": ((0.30, 0.10, 0.22), 0.0), "right": ((0.30, -0.10, 0.22), 90.0)}
    return {arm: {"command": "move_ee", "intent": "observe",
                  "args": {"position": dict(zip("xyz", p)), "rotation": topdown(yaw),
                           "mode": "plan", "speed_profile": "transit"},
                  "evidence": {"clearance": "free air above the table, away from objects and the other arm"}}
            for arm, (p, yaw) in targets.items()}


def m4gw_at_command(command, reading):
    """Grip width: the official reading equals the command (agp_core/grip_width.py at_command, READING_TOL 1e-6)."""
    return (isinstance(reading, (int, float)) and isinstance(command, (int, float))
            and abs(float(reading) - float(command)) <= 1e-6)


def m4gw_reading_gap_m(command, reading):
    """Grip width: the pad gap the official reading shows while its clamp acts (agp_core/grip_width.py reading_gap_m with
    the public X5 constants: target -0.01 + n * 0.054 m, clamp 10.8 mm, fingers 0 .. 0.044 m, pads 0.807 mm apart at 0),
    else None (the reading equals the command: the fingers within 10.8 mm of the target)."""
    if not isinstance(reading, (int, float)) or m4gw_at_command(command, reading):
        return None
    target = -0.01 + float(reading) * 0.054
    finger = target + 0.0108 if float(reading) > float(command) else target - 0.0108
    return 2.0 * min(max(finger, 0.0), 0.044) + 0.000807


def signal_of(code):
    """Signal behind a return code: -N or 128+N (as agp_core/run_agent.py), else None."""
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    if code < 0:
        return -code
    if 128 < code <= 128 + 64:
        return code - 128
    return None


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Script:
    def __init__(self, session, name, fault):
        self.session = session
        self.name = name
        self.fault = fault
        self.report = {"kind": f"scripted:{name}", "official_success_claim": False, "checks": {}, "calls": []}
        self.attribution = {}      # --fault mcp_crash: policy_finished.json entries as agp_core/run_agent.py writes them
        self.mcp_process = None
        ready = json.loads((session / "worker_ready.json").read_text())
        episode = json.loads((session / "episode.json").read_text())
        self.limit = int(ready["action_budget"])
        self.report.update(action_budget=self.limit, official_step_lim=episode["official_step_lim"])
        self.m3s1a_plan = M3S1A_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.m3s1b_plan = M3S1B_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.m3e0_plan = M3E0_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.m4mi_plan = M4MI_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.m4gb_plan = M4GB_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.m4spin_plan = M4SPIN_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.m4gw_plan = M4GW_LAYOUTS.get((episode.get("task"), episode.get("seed"), episode.get("layout_id")))
        self.check("worker budget is the official step limit", self.limit == episode["official_step_lim"],
                   [self.limit, episode["official_step_lim"]])

    def check(self, name, condition, detail=None):
        self.report["checks"][name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            raise CheckFailed(f"{name}: {detail}")

    async def call(self, client, tool, arguments):
        started = time.monotonic()
        result = await asyncio.wait_for(client.call_tool(tool, arguments), timeout=900)
        view = json.loads(result.content[0].text)
        labels = [b.text.split()[1] for b in result.content[1:] if b.type == "text" and b.text.startswith("Image ")]
        self.report["calls"].append({"tool": tool, "ok": view.get("ok"), "images": labels,
                                     "wall_s": round(time.monotonic() - started, 3)})
        return view, labels

    def server(self):
        from mcp import StdioServerParameters
        env = dict(os.environ, PYTHONPATH=f"{ROOT}/source/agent:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
        cwd = ROOT
        if self.fault == "mcp_start":
            # `agp_core` is neither on PYTHONPATH nor in the working directory: `-m agp_core.server` cannot start.
            env["PYTHONPATH"] = f"{ROOT}/source/agent"
            cwd = self.session
        return StdioServerParameters(command=str(AGENT_PYTHON),
                                     args=["-B", "-m", "agp_core.server", "--session", str(self.session)],
                                     env=env, cwd=str(cwd))

    async def run(self):
        if self.name == "idle":
            return
        if self.fault == "mcp_crash":
            return await self.run_mcp_crash()
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        async with stdio_client(self.server()) as (read, write):
            async with ClientSession(read, write) as client:
                await asyncio.wait_for(client.initialize(), timeout=120)
                names = {tool.name for tool in (await client.list_tools()).tools}
                self.check("MCP tool list", names == EXPECTED_TOOLS, sorted(names))
                if self.fault == "agent_crash":
                    await self.agent_crash(client)          # does not return: the process ends by itself
                goal, _ = await self.call(client, "goal", {})
                self.report["instruction"] = goal.get("instruction")
                self.check("goal returns the instruction", bool(goal.get("instruction")), goal)
                obs, labels = await self.call(client, "observe", {})
                self.check_observation("first observe", obs, labels, expect_used=0)
                if self.name == "smoke" and self.m4gw_plan is not None:
                    obs = await self.m4gw(client, obs)
                elif self.name == "smoke" and self.m4spin_plan is not None:
                    obs = await self.m4spin(client, obs)
                elif self.name == "smoke" and self.m4gb_plan is not None:
                    obs = await self.m4gb(client, obs)
                elif self.name == "smoke" and self.m4mi_plan is not None:
                    obs = await self.m4mi(client, obs)
                elif self.name == "smoke" and self.m3e0_plan is not None:
                    obs = await self.m3e0(client, obs)
                elif self.name == "smoke" and self.m3s1b_plan is not None:
                    obs = await self.m3s1b(client, obs)
                elif self.name == "smoke" and self.m3s1a_plan is not None:
                    obs = await self.m3s1a(client, obs)
                elif self.name == "smoke":
                    obs = await self.smoke(client, obs)
                assessed, _ = await self.call(client, "assess", {
                    "observation_id": obs["observation_id"], "outcome": "failure",
                    "evidence": {"summary": f"scripted {self.name}: the task was not attempted"}})
                self.check("assess accepted", assessed.get("ok") is True, assessed)
        self.check_captures()
        if self.name == "smoke" and self.m3s1a_plan is None and self.m3s1b_plan is None and self.m3e0_plan is None:
            self.check_preflight()

    # ------------------------------------------------------------------ faults (policy crash mid-episode)
    async def opening(self, client, call):
        """goal, observe and the paired top-down move: the episode is under way (steps used)."""
        goal, _ = await call(client, "goal", {})
        self.report["instruction"] = goal.get("instruction")
        obs, labels = await call(client, "observe", {})
        self.check_observation("first observe", obs, labels, expect_used=0)
        return obs

    async def paired_move(self, client, call, oid):
        moved, labels = await call(client, "step", {"action_id": "scripted-topdown", "observation_id": oid,
                                                    "actions": topdown_actions()})
        self.check("paired top-down plan move: ok, both reached",
                   moved.get("ok") is True and all(a.get("reached") is True for a in moved.get("arms", {}).values()),
                   moved)
        self.check_observation("after the move", moved["obs"], labels)
        self.check("a paired motion ran before the crash (steps used)", moved["obs"]["steps_used"] > 0,
                   moved["obs"]["steps_used"])
        return moved

    async def agent_crash(self, client):
        """--fault agent_crash: stand in for agp_core/run_agent.py dying by itself mid-episode (never returns)."""
        (self.session / "agent.pid").write_text(str(os.getpid()))      # as run_agent.py does once codex runs
        obs = await self.opening(client, self.call)
        moved = await self.paired_move(client, self.call, obs["observation_id"])
        self.report.update(verified=True, fault="agent_crash", crash={
            "process": "agp_core/official_scripted.py (stand-in for agp_core/run_agent.py)", "exit": CRASH_EXIT,
            "after_calls": len(self.report["calls"]), "steps_used": moved["obs"]["steps_used"],
            "at_wall": time.time(), "policy_finished_json": False})
        (self.session / "scripted_report.json").write_text(json.dumps(self.report, indent=1, default=str))
        print(f"[AGP fault] agent launcher stand-in dies by itself mid-episode after "
              f"{len(self.report['calls'])} tool calls and {moved['obs']['steps_used']} steps: os._exit({CRASH_EXIT}), "
              f"no policy_finished.json", flush=True)
        os._exit(CRASH_EXIT)

    async def call_watching(self, client, tool, arguments):
        """call(), or None when the MCP server process ended while the call was pending: the streamable-HTTP client
        then never receives an answer, so the call is cancelled 3 s after the process ended."""
        task = asyncio.ensure_future(self.call(client, tool, arguments))
        ended = None
        while not task.done():
            await asyncio.sleep(0.2)
            if self.mcp_process.poll() is not None:
                ended = ended or time.monotonic()
                if time.monotonic() - ended > 3.0:
                    task.cancel()
                    break
        try:
            return await task
        except (asyncio.CancelledError, Exception) as exc:
            if self.mcp_process.poll() is None:
                raise
            self.report["calls"].append({"tool": tool, "ok": None, "images": [],
                                         "error": f"no answer: the MCP server ended with code "
                                                  f"{self.mcp_process.returncode} while the call was pending "
                                                  f"({type(exc).__name__})"})
            return None

    def record_server_exit(self):
        """policy_finished.json entries for an MCP server that ended while the policy used it, as agp_core/run_agent.py's
        wait_official records them (a fault signal is the server's own crash, any other signal an external one)."""
        code = self.mcp_process.returncode
        server = {"returncode": code, "exited_during_episode": True, "at_wall": time.time()}
        number = signal_of(code)
        entry = {"process": "mcp_server", "returncode": code, "signal": number,
                 "signal_name": signal.Signals(number).name if number in {int(x) for x in signal.Signals} else None,
                 "at_wall": time.time()}
        self.attribution = {"stopped_by": "mcp_server_exited", "mcp_server": server, "external_signal": None}
        if number in FAULT_SIGNALS:
            server["crash_signal"] = entry
        elif number is not None:
            self.attribution["external_signal"] = entry

    async def run_mcp_crash(self):
        """--fault mcp_crash: the MCP server over HTTP (as agp_core/run_agent.py) crashes by itself mid-episode."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        port = free_port()
        env = dict(os.environ, PYTHONPATH=f"{ROOT}/source/agent:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
        with (self.session / "mcp_server.log").open("w") as log:
            self.mcp_process = subprocess.Popen(
                [str(AGENT_PYTHON), "-B", "-m", "agp_core.mcp_crash_server", "--session",
                 str(self.session), "--http", str(port), "--crash-at", str(CRASH_AT_CALL)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.report["mcp_server"] = {"transport": "streamable-http", "port": port, "pid": self.mcp_process.pid,
                                     "crash_at_call": CRASH_AT_CALL}
        deadline = time.monotonic() + 120
        while True:                                        # as agp_core/run_agent.py waits for its MCP server
            if self.mcp_process.poll() is not None:
                raise RuntimeError(f"MCP server exited early with code {self.mcp_process.returncode} (start-up), "
                                   f"see mcp_server.log")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("MCP server did not start")
                await asyncio.sleep(0.2)
        moved = closed = None
        transport_error = None
        try:
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (read, write, _):
                async with ClientSession(read, write) as client:
                    await asyncio.wait_for(client.initialize(), timeout=120)
                    names = {tool.name for tool in (await client.list_tools()).tools}
                    self.check("MCP tool list", names == EXPECTED_TOOLS, sorted(names))
                    obs = await self.opening(client, self.call_watching)
                    measured, _ = await self.call_watching(client, "measure", {
                        "observation_id": obs["observation_id"], "queries": [
                            {"arm": "left", "cam": "top", "u": 320, "v": 300, "plane_z": 0.0,
                             "height_basis": "table top", "target": "table surface at 320,300"}]})
                    self.check("measure on the table plane", measured.get("ok") is True, measured)
                    moved = await self.paired_move(client, self.call_watching, obs["observation_id"])
                    closed = await self.call_watching(client, "step", {
                        "action_id": "scripted-close", "observation_id": moved["obs"]["observation_id"],
                        "actions": {arm: {"command": "gripper", "intent": "grasp", "args": {"action": "close"}}
                                    for arm in ("left", "right")}})
        except BaseException as exc:                      # the transport after the server died
            detail = exc
            while isinstance(detail, BaseExceptionGroup) and detail.exceptions:   # anyio task groups wrap errors
                detail = detail.exceptions[0]
            if isinstance(detail, (CheckFailed, KeyboardInterrupt, SystemExit)) or self.mcp_process.poll() is None:
                raise
            transport_error = f"{type(detail).__name__}: {detail}"
        try:
            self.mcp_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        code = self.mcp_process.poll()
        if code is not None:
            self.record_server_exit()
        self.report["crash"] = {"server_returncode": code, "close_answered": closed is not None,
                                "steps_used_before": (moved or {}).get("obs", {}).get("steps_used"),
                                "transport_error": transport_error, "calls": len(self.report["calls"])}
        self.check(f"the MCP server crashed by itself while answering tool call {CRASH_AT_CALL} (the paired close, "
                   f"after a paired motion): exit {CRASH_EXIT}, no signal, no answer",
                   code == CRASH_EXIT and closed is None and moved is not None
                   and len(self.report["calls"]) == CRASH_AT_CALL, self.report["crash"])
        self.check_captures()

    def check_observation(self, name, obs, labels, expect_used=None):
        # a step result embeds the compact observation without its own "ok"
        self.check(f"{name}: ok with three images", obs.get("ok", True) is True and bool(obs.get("observation_id"))
                   and set(labels) == {"top", "left_wrist", "right_wrist"}, [obs.get("ok"), labels])
        used, left = obs.get("steps_used"), obs.get("steps_left")
        self.check(f"{name}: official step budget (steps_left = limit - steps_used)",
                   isinstance(used, int) and left == self.limit - used, [used, left, self.limit])
        if expect_used is not None:
            self.check(f"{name}: steps_used {expect_used}", used == expect_used, used)
        # the view shows the finger gap estimated from each wrist image (agp_core/visual_gap.py) when its
        # quality is ok, else null; never a measured finger joint (the full state has no gripper_geometry).
        full = self.full_observation(obs.get("observation_id")) or {}
        state = full.get("state") or {}
        expected = {}
        for arm, a in state.get("arms", {}).items():
            estimate = a.get("gripper_gap_estimate") or {}
            ok = estimate.get("quality") == "ok" and estimate.get("pad_gap_m") is not None
            expected[arm] = round(float(estimate["pad_gap_m"]) * 1000, 1) if ok else None
        shown = {arm: a.get("gap_mm", "absent") for arm, a in obs.get("arms", {}).items()}
        self.check(f"{name}: no simulator time; the finger gap shown is the wrist-image estimate",
                   "sim_time_s" not in obs and bool(state) and "gripper_geometry" not in json.dumps(state)
                   and shown == expected and set(expected) == {"left", "right"}, [shown, expected])
        self.report.setdefault("observations", []).append({"name": name, "steps_used": used, "steps_left": left})

    def full_observation(self, observation_id):
        """The runtime's full observation with this id, from the archived tool results (newest first)."""
        for path in sorted((self.session / "native_tool_results").glob("*.json"), reverse=True):
            result = json.loads(path.read_text())
            candidates = [result, result.get("observation") or {}]
            candidates += [(step or {}).get("observation") or {} for step in result.get("steps") or []
                           if isinstance(step, dict)]
            for candidate in candidates:
                if candidate.get("observation_id") == observation_id and "state" in candidate:
                    return candidate
        return None

    async def smoke(self, client, obs):
        oid = obs["observation_id"]
        pixels = [(320, 300), (200, 330), (440, 330), (320, 380)]
        queries = [{"arm": arm, "cam": "top", "u": u, "v": v, "plane_z": 0.0,
                    "height_basis": "table top: the arm bases stand on the table plane",
                    "target": f"table surface at {u},{v}"} for (u, v) in pixels for arm in ("left", "right")]
        measured, _ = await self.call(client, "measure", {"observation_id": oid, "queries": queries})
        points = measured.get("points", [])
        self.check("measure on the table plane: 8 points", measured.get("ok") is True and len(points) == 8, measured)
        offsets = [[round(b - a, 4) for a, b in zip(points[i]["xyz"], points[i + 1]["xyz"])]
                   for i in range(0, len(points), 2)]
        self.check("arm frames 0.6 m apart along y", all(abs(o[0]) < 1e-3 and abs(o[1] - 0.6) < 1e-3
                                                         and abs(o[2]) < 1e-3 for o in offsets), offsets)
        self.check("plane points: z_from plane, z = 0",
                   all(p.get("z_from") == "plane" and abs(p["xyz"][2]) < 1e-6 for p in points), points[:2])
        self.report["table_points_left_frame"] = {f"{u},{v}": points[2 * i]["xyz"] for i, (u, v) in enumerate(pixels)}
        free, _ = await self.call(client, "measure", {"observation_id": oid, "queries": [
            {"arm": "left", "cam": "top", "u": 320, "v": 300, "target": "table surface (height from matching)"}]})
        self.report["measure_without_plane"] = free     # recorded only: the wrist view may not contain the pixel

        # the paired top-down move at 0.22 m (see topdown_actions for why not 0.25 m)
        moved, labels = await self.call(client, "step", {"action_id": "scripted-topdown", "observation_id": oid,
                                                         "actions": topdown_actions()})
        self.check("paired top-down plan move: ok, both reached",
                   moved.get("ok") is True and all(a.get("reached") is True for a in moved.get("arms", {}).values()),
                   moved)
        self.check_observation("after the move", moved["obs"], labels)
        oid = moved["obs"]["observation_id"]

        closed, labels = await self.call(client, "step", {"action_id": "scripted-close", "observation_id": oid,
                                                          "actions": {arm: {"command": "gripper", "intent": "grasp",
                                                                            "args": {"action": "close"}}
                                                                      for arm in ("left", "right")}})
        arms = closed.get("arms", {})
        # the close outcome follows the wrist-image finger estimate (agp_core/visual_gap.py) that the worker
        # recorded in the full result (native_tool_results); in free air an ok estimate means closed fully.
        full = [json.loads(p.read_text()) for p in sorted((self.session / "native_tool_results").glob("*.json"))]
        full = [r for r in full if r.get("action_id") == "scripted-close" and r.get("action")]
        worker_arms = full[-1]["action"].get("arms", {}) if full else {}
        expected = {}
        for arm, a in worker_arms.items():
            ok_estimate = a.get("gripper_estimate_quality") == "ok" and a.get("gripper_error_m") is not None
            expected[arm] = ("reached", True) if ok_estimate and a["gripper_error_m"] < 0.003 else ("held_short", False)
        self.report["close_estimate"] = {arm: {k: a.get(k) for k in ("gripper_estimate_m", "gripper_estimate_quality",
                                                                    "gripper_error_m", "motion_outcome")}
                                         for arm, a in worker_arms.items()}
        self.check("close: ok; outcome follows the wrist-image finger estimate (no finger joints in the view)",
                   closed.get("ok") is True and set(arms) == {"left", "right"} == set(expected)
                   and all((a.get("outcome"), a.get("reached")) == expected[arm] and "finger_joints_mm" not in a
                           for arm, a in arms.items()), [closed, self.report["close_estimate"]])
        oid = closed["obs"]["observation_id"]
        evidence = {"release_requirement": "empty", "empty": "nothing between the fingers in free air",
                    "object_location": "all objects remain on the table", "opening_clearance": "free air"}
        opened, labels = await self.call(client, "step", {"action_id": "scripted-open", "observation_id": oid,
                                                          "actions": {arm: {"command": "gripper", "intent": "retreat",
                                                                            "args": {"action": "open"},
                                                                            "evidence": evidence}
                                                                      for arm in ("left", "right")}})
        self.check("open: ok, both reached", opened.get("ok") is True
                   and all(a.get("reached") is True for a in opened.get("arms", {}).values()), opened)
        oid = opened["obs"]["observation_id"]
        # A run stops at the first surprise, which includes a motion that ends short of its target; that is a
        # physical outcome, not a fault of the plumbing, so it is recorded and home is then sent as a plain step.
        steps = [{"left": {"do": "delta", "d": [0, 0, 0.03], "intent": "retreat", "speed": "precision"},
                  "right": {"do": "delta", "d": [0, 0, 0.03], "intent": "retreat", "speed": "precision"}},
                 {"left": {"do": "home"}, "right": {"do": "home"}}]
        ran, labels = await self.call(client, "run", {"observation_id": oid, "steps": steps})
        self.report["run"] = ran
        executed = [st for st in ran.get("steps", []) if "status" in st]
        reason = (ran.get("stopped") or {}).get("reason") or ""
        self.check("run: steps executed by the runtime (stopped only by a motion outcome)",
                   executed and all(st.get("status") == "done" and st.get("ok") is True for st in executed)
                   and (ran.get("ok") is True or reason.endswith("did not reach its target")), ran)
        if ran.get("done") != len(steps):
            oid = ran["obs"]["observation_id"]
            home, labels = await self.call(client, "step", {
                "action_id": "scripted-home", "observation_id": oid,
                "actions": {arm: {"command": "home", "intent": "retreat", "args": {},
                                  "evidence": {"clearance": "free air back to the initial pose"}}
                            for arm in ("left", "right")}})
            self.report["home_step"] = home
            self.check("home: ok", home.get("ok") is True, home)
        await self.approx_home_checks(client)
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        self.check("steps were used", final["steps_used"] > 0, final["steps_used"])
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ speed and guard test (m3s1a)
    async def m3s1a_stage(self, client, oid, name, actions, expect_ok=True):
        """One step; records the compact result (with the action id) and returns (view, observation id)."""
        self.m3s1a_count += 1
        action_id = f"m3s1a-{self.m3s1a_count:02d}-{name}"
        view, _ = await self.call(client, "step", {"action_id": action_id, "observation_id": oid, "actions": actions})
        obs = view.get("obs") or {}
        entry = {"name": name, "action_id": action_id, "ok": view.get("ok"), "samples": view.get("samples"),
                 "stop_reason": view.get("stop_reason"), "error": view.get("error") or view.get("action_error"),
                 "arms": view.get("arms"), "steps_used": obs.get("steps_used"), "steps_left": obs.get("steps_left")}
        self.report.setdefault("m3s1a", {}).setdefault("stages", []).append(entry)
        if expect_ok:
            self.check(f"m3s1a {name}: ok", view.get("ok") is True, entry)
        if obs.get("observation_id"):
            return view, obs["observation_id"]
        latest, _ = await self.call(client, "observe", {})
        return view, latest["observation_id"]

    @staticmethod
    def m3s1a_move(xyz, rotation, mode="plan", speed="transit", intent="observe"):
        action = {"command": "move_ee", "intent": intent,
                  "args": {"position": dict(zip("xyz", xyz)), "rotation": rotation, "mode": mode,
                           "speed_profile": speed}}
        if speed == "transit":
            action["evidence"] = {"clearance": "free air chosen from the layout's first top image"}
        return action

    @staticmethod
    def m3s1a_delta(dpos, speed="transit", intent="observe", drot=None):
        action = {"command": "move_delta", "intent": intent, "args": {"dpos": list(dpos), "speed_profile": speed}}
        if drot is not None:
            action["args"]["drot_deg"] = list(drot)
        if speed == "transit":
            action["evidence"] = {"clearance": "free air chosen from the layout's first top image"}
        return action

    @staticmethod
    def m3s1a_home():
        return {"command": "home", "intent": "retreat", "args": {},
                "evidence": {"clearance": "free air back toward the initial pose"}}

    @staticmethod
    def m3s1a_gripper(action, holding=False):
        if action == "close":
            return {"command": "gripper", "intent": "grasp", "args": {"action": "close"}}
        evidence = {"release_requirement": "empty", "empty": "nothing held between the pads"}
        if holding:
            evidence.update(object_location="no object was held", opening_clearance="free air around the fingers")
        return {"command": "gripper", "intent": "retreat", "args": {"action": "open"}, "evidence": evidence}

    async def m3s1a(self, client, obs):
        """The speed and guard test sequence (see the module docstring); returns the final observation."""
        plan = self.m3s1a_plan
        self.m3s1a_count = 0
        switches = {name: os.environ.get(name) for name in ("OURS_FREE_TOPT", "OURS_ARM_GUARD", "OURS_HOLD_GUARD")}
        self.report["m3s1a"] = {"plan": plan, "switches": switches, "stages": []}
        low = plan["low_arm"]
        other = "right" if low == "left" else "left"
        sign = {"left": 1.0, "right": -1.0}
        oid = obs["observation_id"]
        # F: free transit moves, low moves of the low arm, a rotation, curved plans, homes, a closed empty gripper
        _, oid = await self.m3s1a_stage(client, oid, "paired-observe", topdown_actions())
        down = topdown(0.0 if low == "left" else 90.0)
        x, y = plan["low_xy"]
        _, oid = await self.m3s1a_stage(client, oid, "low-descend", {low: self.m3s1a_move(
            (x, y, plan["low_z"]), down, mode="linear")})
        _, oid = await self.m3s1a_stage(client, oid, "low-sideways", {low: self.m3s1a_delta(plan["low_shift"])})
        _, oid = await self.m3s1a_stage(client, oid, "low-takeoff", {low: self.m3s1a_delta((0.0, 0.0, 0.15),
                                                                                       intent="retreat")})
        _, oid = await self.m3s1a_stage(client, oid, "rotate", {other: self.m3s1a_delta((0.0, 0.0, 0.0),
                                                                                    drot=(0.0, 0.0, 45.0))})
        _, oid = await self.m3s1a_stage(client, oid, "paired-curved", {     # yaw 45 / 135: the rotated arm keeps its
            arm: self.m3s1a_move((0.25, 0.15 * sign[arm], 0.20), topdown(45.0 if arm == "left" else 135.0))
            for arm in ("left", "right")})
        _, oid = await self.m3s1a_stage(client, oid, "full-home-one-arm", {low: self.m3s1a_home()})
        _, oid = await self.m3s1a_stage(client, oid, "approx-home-completing", {other: self.m3s1a_home()})
        _, oid = await self.m3s1a_stage(client, oid, "close-empty", {low: self.m3s1a_gripper("close")}, expect_ok=False)
        _, oid = await self.m3s1a_stage(client, oid, "closed-empty-transit", {low: self.m3s1a_delta(
            (0.0, 0.0, 0.06), intent="retreat")})
        _, oid = await self.m3s1a_stage(client, oid, "open-empty", {low: self.m3s1a_gripper("open", True)})
        _, oid = await self.m3s1a_stage(client, oid, "approx-home-low", {low: self.m3s1a_home()})
        # F-g: only with the guard on (without it the collision course would collide)
        if switches["OURS_ARM_GUARD"] == "1":
            yaw0 = topdown(0.0)
            _, oid = await self.m3s1a_stage(client, oid, "guard-pass", {
                arm: self.m3s1a_move((0.25, -0.12 * sign[arm], 0.20), yaw0) for arm in ("left", "right")})
            view, oid = await self.m3s1a_stage(client, oid, "guard-collision-course", {
                arm: self.m3s1a_move((0.25, -0.27 * sign[arm], 0.20), yaw0) for arm in ("left", "right")},
                expect_ok=False)
            self.check("m3s1a guard-collision-course: refused before any motion with the links, the step and the "
                       "distance", view.get("ok") is False and "Two-arm clearance" in json.dumps(view)
                       and not view.get("samples"), view)
            _, oid = await self.m3s1a_stage(client, oid, "guard-right-to-centre", {
                "right": self.m3s1a_move((0.25, 0.30, 0.20), yaw0)}, expect_ok=False)
            _, oid = await self.m3s1a_stage(client, oid, "guard-crossing", {
                "left": self.m3s1a_move((0.25, -0.30, 0.20), yaw0),
                "right": self.m3s1a_move((0.25, 0.05, 0.20), yaw0)}, expect_ok=False)
            _, oid = await self.m3s1a_stage(client, oid, "guard-home", {
                arm: self.m3s1a_home() for arm in ("left", "right")})
        # D: the right gripper closes on the left gripper's finger and lifts off it
        d = M3S1A_D
        _, oid = await self.m3s1a_stage(client, oid, "d-left-forward", {"left": self.m3s1a_move(
            d["left_pose"], dict(zip("wxyz", d["left_quat"])), mode="linear")})
        _, oid = await self.m3s1a_stage(client, oid, "d-right-beside", {"right": self.m3s1a_move(
            d["grip_side"], topdown(d["grip_yaw"]))})
        _, oid = await self.m3s1a_stage(client, oid, "d-right-above-finger", {"right": self.m3s1a_move(
            d["grip_above"], topdown(d["grip_yaw"]), mode="linear")})
        _, oid = await self.m3s1a_stage(client, oid, "d-right-descend", {"right": self.m3s1a_delta(
            (0.0, 0.0, -d["grip_descent"]), speed="precision")})
        _, oid = await self.m3s1a_stage(client, oid, "d-right-close-on-finger", {"right": self.m3s1a_gripper("close")},
                                        expect_ok=False)
        lift, oid = await self.m3s1a_stage(client, oid, "d-right-lift", {"right": self.m3s1a_delta(
            (0.0, 0.0, d["lift"]), speed="precision", intent="grasp")}, expect_ok=False)
        _, oid = await self.m3s1a_stage(client, oid, "d-right-open", {"right": self.m3s1a_gripper("open", True)},
                                        expect_ok=False)
        _, oid = await self.m3s1a_stage(client, oid, "d-right-retreat", {"right": self.m3s1a_delta(
            (0.0, -0.08, 0.0), intent="retreat")}, expect_ok=False)     # sideways: higher is out of the right's reach
        _, oid = await self.m3s1a_stage(client, oid, "d-home", {arm: self.m3s1a_home() for arm in ("left", "right")},
                                        expect_ok=False)
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ G / P / H test (m3s1b)
    async def m3s1b_run(self, client, oid, name, steps, evidence=None, show=None):
        """One `run` call; records its compact result; returns (view, latest observation id)."""
        args = {"observation_id": oid, "steps": steps,
                "evidence": evidence or {"clearance": "free air chosen from the layout's first top image",
                                         "target_unchanged": "the object has not moved since it was measured"}}
        if show:
            args["show"] = show
        view, _ = await self.call(client, "run", args)
        obs = view.get("obs") or {}
        entry = {"name": name, "steps_sent": steps, "ok": view.get("ok"), "done": view.get("done"),
                 "stopped": view.get("stopped"), "labels": view.get("labels"), "steps": view.get("steps"),
                 "steps_used": obs.get("steps_used"), "steps_left": obs.get("steps_left"),
                 "gaps_mm": {arm: a.get("gap_mm") for arm, a in (obs.get("arms") or {}).items()}}
        self.report["m3s1b"]["runs"].append(entry)
        oid = obs.get("observation_id")
        if not oid:
            latest, _ = await self.call(client, "observe", {})
            oid = latest["observation_id"]
        return view, oid

    @staticmethod
    def m3s1b_points(view):
        """{label: point} of a run view: the measured points, replaced by a later re-measure of the same label."""
        points = {}
        for step in view.get("steps") or []:
            for point in (step.get("measure") or []) + (step.get("remeasured") or []):
                if point.get("as") and point.get("ok", True) is not False:
                    points[point["as"]] = point
        return points

    async def m3s1b(self, client, obs):
        """The G / P / H test sequence (see the module docstring); returns the final observation."""
        plan = self.m3s1b_plan
        switches = {name: os.environ.get(name) for name in ("OURS_FREE_TOPT", "OURS_ARM_GUARD", "OURS_HOLD_GUARD",
                                                             "OURS_GRIP_OBS", "OURS_PREC_ZONE", "OURS_HOLD_PROFILE")}
        self.report["m3s1b"] = {"plan": {k: v for k, v in plan.items()}, "switches": switches, "runs": [],
                                "grasp_heights": {}}
        letter = {"left": "L", "right": "R"}
        arms = [arm for arm in ("left", "right") if arm in plan["picks"]]
        both = lambda make: " & ".join(make(arm) for arm in arms)      # noqa: E731 - one step, the picking arms
        oid = obs["observation_id"]
        # 1. lift both arms from the start pose, measure the blocks, stage above them (the label is re-measured)
        view, oid = await self.m3s1b_run(client, oid, "approach", [
            "B d 0,0,.05 retreat",
            "? " + " & ".join(f"p{letter[a].lower()}: {letter[a]} top {plan['picks'][a][0]},{plan['picks'][a][1]} "
                              f"z={M3S1B_Z_GUESS} block top" for a in arms),
            both(lambda a: f"{letter[a]} ee p{letter[a].lower()}+0,0,{M3S1B_STAGE} down90 plan obs")])
        points = self.m3s1b_points(view)
        grasp = {}
        for a in arms:
            point = points.get(f"p{letter[a].lower()}") or {}
            top = (point.get("xyz") or [None, None, M3S1B_Z_GUESS])[2]
            down = max(-M3S1B_GRASP_DOWN, M3S1B_GRASP_MIN_Z - top)
            grasp[a] = {"top_z": top, "z_from": point.get("z_from"), "offset_z": round(down, 4),
                        "tcp_z": round(top + down, 4)}
        self.report["m3s1b"]["grasp_heights"] = grasp
        # 2. the contact descents, one arm per step (P: split when the label's height is a match or plane and the
        #    descent is vertical; a step is split only when all its descents qualify, so each arm has its own step).
        #    A contact move invalidates every measurement: the second arm measures its block again from its staging
        #    pose before its own descent (the same top pixel and plane_z guess).
        contact = []
        for i, a in enumerate(arms):
            label = f"p{letter[a].lower()}"
            if i:
                contact.append(f"? {label}: {letter[a]} top {plan['picks'][a][0]},{plan['picks'][a][1]} "
                               f"z={M3S1B_Z_GUESS} block top")
            contact.append(f"{letter[a]} ee {label}+0,0,{grasp[a]['offset_z']} contact")
        view, oid = await self.m3s1b_run(client, oid, "contact", contact)
        # 3. close (G) and the verification lift (P's lift; H for its rest)
        view, oid = await self.m3s1b_run(client, oid, "grasp", [
            both(lambda a: f"{letter[a]} close"), both(lambda a: f"{letter[a]} d 0,0,.03 grasp")])
        # 4. carry while holding (H): a linear move, then a plan move to above the place spot
        sideways = {"left": "0,-.04,.02", "right": "0,-.04,.02"}
        view, oid = await self.m3s1b_run(client, oid, "carry", [
            both(lambda a: f"{letter[a]} d {sideways[a]} retreat"),
            both(lambda a: f"{letter[a]} ee {plan['carry'][a][0]},{plan['carry'][a][1]},{plan['carry'][a][2]} "
                           f"down90 plan obs")])
        # 5. measure the place spots (table plane), carry above them, place (P: split, the last 2 cm at precision)
        view, oid = await self.m3s1b_run(client, oid, "place", [
            "? " + " & ".join(f"q{letter[a].lower()}: {letter[a]} top {plan['places'][a][0]},{plan['places'][a][1]} "
                              f"z=0 table spot" for a in arms),
            both(lambda a: f"{letter[a]} ee q{letter[a].lower()}+0,0,.08 transfer"),
            both(lambda a: f"{letter[a]} ee q{letter[a].lower()}+0,0,{grasp[a]['tcp_z'] + M3S1B_RELEASE_UP:.4f} "
                           f"place")])
        # 6. open (G: the fingers keep opening during the retreat) and retreat
        view, oid = await self.m3s1b_run(client, oid, "release", [
            both(lambda a: f"{letter[a]} open support=the table surface directly below the block"),
            both(lambda a: f"{letter[a]} d 0,0,.05 retreat")])
        refusals = plan.get("refusals")
        if refusals:                         # P's live refusals with the free arm (single-arm steps; a contact move
            r = refusals["arm"]              # invalidates every measurement, so they come after the pick and place)
            L = letter[r]
            u, v = refusals["oblique"]
            view, oid = await self.m3s1b_run(client, oid, "refusal-oblique", [
                f"? t: {L} top {u},{v} z=0 table spot", f"{L} ee t+.02,0,.06 down90 plan obs",
                f"{L} ee t+0,0,.03 contact"])
            u, v = refusals["fallback"]
            view, oid = await self.m3s1b_run(client, oid, "refusal-fallback-measure", [
                f"{L} d 0,0,.06 retreat", f"? f: {L} top {u},{v} z=.04 table spot, raised guess"])
            point = self.m3s1b_points(view).get("f") or {}
            xyz = point.get("xyz")
            if xyz:
                view, oid = await self.m3s1b_run(client, oid, "refusal-fallback", [
                    f"{L} ee {xyz[0]:.4f},{xyz[1]:.4f},.14 down90 plan obs", f"{L} ee f+0,0,.03 contact",
                    f"{L} d 0,0,.05 retreat"])
        # 7. an empty close and open of both grippers (G), then home
        view, oid = await self.m3s1b_run(client, oid, "empty-grippers", [
            "B close", "B open empty=nothing between the pads object_location=no object was held "
                       "opening_clearance=free air around the fingers"])
        view, oid = await self.m3s1b_run(client, oid, "home", ["B home"])
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ pick-and-place smoke (m3e0)
    async def m3e0_run(self, client, oid, name, steps):
        """One `run` call of the pick-and-place smoke; records its compact result; returns (view, latest observation id,
        whether the official episode has ended)."""
        view, _ = await self.call(client, "run", {
            "observation_id": oid, "steps": steps,
            "evidence": {"clearance": "free air chosen from the layout's first top image",
                         "target_unchanged": "the object has not moved since it was measured"}})
        obs = view.get("obs") or {}
        self.report["m3e0"]["runs"].append({
            "name": name, "steps_sent": steps, "ok": view.get("ok"), "done": view.get("done"),
            "stopped": view.get("stopped"), "labels": view.get("labels"), "steps": view.get("steps"),
            "steps_used": obs.get("steps_used"), "steps_left": obs.get("steps_left"),
            "episode_ended": obs.get("episode_ended"),
            "gaps_mm": {arm: a.get("gap_mm") for arm, a in (obs.get("arms") or {}).items()}})
        oid = obs.get("observation_id")
        if not oid:
            obs, _ = await self.call(client, "observe", {})
            oid = obs["observation_id"]
        return view, oid, bool(obs.get("episode_ended"))

    async def m3e0(self, client, obs):
        """The pick-and-place smoke (M3E0_LAYOUTS): through `run` as the agent does, with whatever switches the run has
        (an A/B pair runs it without and with the six switches it records): the picking arms rise, measure their objects
        from the top image (raised plane_z guess), stage 6 cm above them (plan; the label is re-measured in the wrist
        view), descend to the grasp height one arm per step (contact; P splits a step when its height is match or plane
        and the descent is vertical; the second arm measures and re-stages first, since a contact invalidates every
        measurement), close (G), lift 3 cm (P's lift; H for its rest), carry (H: a linear move, then a plan move to
        M3E0_CARRY unless the layout says not), measure the table spot at the pick pixel (plane), carry above it
        (transfer), place (P's place split), open (G), retreat, home (the approximate home). Records only: a missed
        grasp is recorded, not a failure. The sequence ends early when the official episode ends (general_pickup: the
        lift is its task)."""
        plan = self.m3e0_plan
        switches = {name: os.environ.get(name) for name in ("OURS_FREE_TOPT", "OURS_ARM_GUARD", "OURS_HOLD_GUARD",
                                                             "OURS_GRIP_OBS", "OURS_PREC_ZONE", "OURS_HOLD_PROFILE")}
        self.report["m3e0"] = {"plan": {k: v for k, v in plan.items()}, "switches": switches, "runs": [],
                               "grasp_heights": {}, "ended_after": None}
        letter = {"left": "L", "right": "R"}
        arms = [arm for arm in ("left", "right") if arm in plan["picks"]]
        both = lambda make: " & ".join(make(arm) for arm in arms)       # noqa: E731 - one step, the picking arms
        rot = lambda a: (plan.get("rot") or {}).get(a, "down90")        # noqa: E731
        guess = lambda a: (plan.get("z") or {}).get(a, M3S1B_Z_GUESS)   # noqa: E731
        what = lambda a: (plan.get("what") or {}).get(a, "object top")  # noqa: E731
        pick = lambda a: f"p{letter[a].lower()}"                         # noqa: E731
        spot = lambda a: f"q{letter[a].lower()}"                         # noqa: E731
        uv = lambda a: f"{plan['picks'][a][0]},{plan['picks'][a][1]}"    # noqa: E731
        oid = obs["observation_id"]

        async def finish(ended_after=None):
            self.report["m3e0"]["ended_after"] = ended_after
            final, labels = await self.call(client, "observe", {})
            self.check_observation("final observe", final, labels)
            self.report["steps_used"] = final["steps_used"]
            return final

        # 1. rise, measure the objects, stage above them
        view, oid, ended = await self.m3e0_run(client, oid, "approach", [
            "B d 0,0,.05 retreat",
            "? " + " & ".join(f"{pick(a)}: {letter[a]} top {uv(a)} z={guess(a)} {what(a)}" for a in arms),
            both(lambda a: f"{letter[a]} ee {pick(a)}+0,0,{M3S1B_STAGE} {rot(a)} plan obs")])
        if ended:
            return await finish("approach")
        points = self.m3s1b_points(view)
        grasp = {}
        for a in arms:
            point = points.get(pick(a)) or {}
            top = (point.get("xyz") or [None, None, guess(a)])[2]
            down = (-float(plan["down"][a]) if a in (plan.get("down") or {})
                    else max(-M3S1B_GRASP_DOWN, M3S1B_GRASP_MIN_Z - top))
            grasp[a] = {"top_z": top, "z_from": point.get("z_from"), "offset_z": round(down, 4),
                        "tcp_z": round(top + down, 4)}
        self.report["m3e0"]["grasp_heights"] = grasp
        # 2. the contact descents, one arm per step (P splits a step only when all its descents qualify, so each arm
        #    has its own step). A contact move invalidates every measurement: the second arm measures its object again
        #    from the top image and re-stages (a short move to the same staging point, so the run re-measures the
        #    label in its wrist view, as after the first staging) before its own descent.
        steps = []
        for i, a in enumerate(arms):
            if i:
                steps += [f"? {pick(a)}: {letter[a]} top {uv(a)} z={guess(a)} {what(a)}",
                          f"{letter[a]} ee {pick(a)}+0,0,{M3S1B_STAGE} obs"]
            steps.append(f"{letter[a]} ee {pick(a)}+0,0,{grasp[a]['offset_z']} contact")
        view, oid, ended = await self.m3e0_run(client, oid, "contact", steps)
        if ended:
            return await finish("contact")
        # 3. close (G), the verification lift (P's lift: the first 1 cm at precision, H for the rest)
        steps = [both(lambda a: f"{letter[a]} close"), both(lambda a: f"{letter[a]} d 0,0,.03 grasp")]
        if plan.get("lift_more"):
            steps.append(both(lambda a: f"{letter[a]} d 0,0,{plan['lift_more']} retreat"))
        view, oid, ended = await self.m3e0_run(client, oid, "grasp", steps)
        if ended or plan.get("carry_place") is False:
            return await finish("grasp")
        # 4. carry while holding (H): a linear move, then a plan move (unless the layout's step limit is short)
        steps = [both(lambda a: f"{letter[a]} d 0,-.04,.02 retreat")]
        if plan.get("carry", True):
            x, y, z = M3E0_CARRY
            steps.append(both(lambda a: f"{letter[a]} ee {x},{y},{z} {rot(a)} plan obs"))
        view, oid, ended = await self.m3e0_run(client, oid, "carry", steps)
        if ended:
            return await finish("carry")
        # 5. put it back where it was picked (the table spot at the pick pixel: a plane measurement), the place descent
        #    split by P (the last 2 cm at precision)
        view, oid, ended = await self.m3e0_run(client, oid, "place", [
            "? " + " & ".join(f"{spot(a)}: {letter[a]} top {uv(a)} z=0 table spot where the {what(a)} stood"
                              for a in arms),
            both(lambda a: f"{letter[a]} ee {spot(a)}+0,0,.08 transfer"),
            both(lambda a: f"{letter[a]} ee {spot(a)}+0,0,{grasp[a]['tcp_z'] + M3S1B_RELEASE_UP:.4f} place")])
        if ended:
            return await finish("place")
        # 6. open (G), retreat
        view, oid, ended = await self.m3e0_run(client, oid, "release", [
            both(lambda a: f"{letter[a]} open support=the table surface directly below the object"),
            both(lambda a: f"{letter[a]} d 0,0,.05 retreat")])
        if ended:
            return await finish("release")
        # 7. home of both arms (the approximate home)
        view, oid, ended = await self.m3e0_run(client, oid, "home", ["B home"])
        return await finish("home" if ended else None)

    # ------------------------------------------------------------------ the intent manipulate (m4mi)
    async def m4mi_run(self, client, oid, name, steps):
        """One `run` call of the manipulate smoke; records its compact result; returns (view, latest observation id)."""
        view, _ = await self.call(client, "run", {
            "observation_id": oid, "steps": steps,
            "evidence": {"clearance": "free air above the table, away from objects and the other arm",
                         "target_unchanged": "the table point has not moved since it was measured"}})
        obs = view.get("obs") or {}
        self.report["m4mi"]["runs"].append({
            "name": name, "steps_sent": steps, "ok": view.get("ok"), "done": view.get("done"),
            "stopped": view.get("stopped"), "steps": view.get("steps"), "steps_used": obs.get("steps_used")})
        oid = obs.get("observation_id")
        if not oid:
            obs, _ = await self.call(client, "observe", {})
            oid = obs["observation_id"]
        return view, oid

    async def m4mi(self, client, obs):
        """The manipulate smoke (M4MI_LAYOUTS; see the module docstring): manipulate accepted at precision, refused at
        transit (through `run` and the step tool), every measurement invalidated by it; free air only."""
        self.report["m4mi"] = {"plan": dict(self.m4mi_plan), "switch": os.environ.get("OURS_INTENT_MANIPULATE"),
                               "runs": [], "steps": []}
        moved, labels = await self.call(client, "step", {"action_id": "scripted-topdown",
                                                         "observation_id": obs["observation_id"],
                                                         "actions": topdown_actions()})
        self.check("paired top-down plan move: ok, both reached",
                   moved.get("ok") is True and all(a.get("reached") is True for a in moved.get("arms", {}).values()),
                   moved)
        self.check_observation("after the move", moved["obs"], labels)
        oid = moved["obs"]["observation_id"]
        u, v = self.m4mi_plan["table_pixel"]
        measured, _ = await self.call(client, "measure", {"observation_id": oid, "queries": [
            {"arm": "left", "cam": "top", "u": u, "v": v, "plane_z": 0.0, "label": "tm",
             "height_basis": "table top: the arm bases stand on the table plane",
             "target": f"table surface at {u},{v}"}]})
        self.check("measure a table point (plane) for the left arm, label tm",
                   measured.get("ok") is True and (measured.get("labels") or {}).get("tm")
                   and measured["points"][0].get("z_from") == "plane", measured)
        used = moved["obs"]["steps_used"]

        # 1. a paired precision manipulate move (the step-string word, default precision): accepted, both reached
        view, oid = await self.m4mi_run(client, oid, "manipulate",
                                        ["L d 0,0,-.02 manipulate & R d 0,0,-.02 manipulate"])
        arms = (view.get("steps") or [{}])[0].get("arms") or {}
        self.check("run: a paired precision manipulate move is accepted, both arms reach, steps are used",
                   view.get("ok") is True and view.get("done") == 1 and set(arms) == {"left", "right"}
                   and all(a.get("reached") is True for a in arms.values())
                   and view["obs"]["steps_used"] > used, view)
        used = view["obs"]["steps_used"]
        # 2. the measurement taken before it is invalid now (a manipulate step invalidates every measurement)
        view, oid = await self.m4mi_run(client, oid, "stale measurement", ["L ee tm+0,0,.20 obs"])
        reason = (view.get("stopped") or {}).get("reason") or ""
        self.check("run: a move to the label measured before the manipulate step is refused (measurement invalidated), "
                   "no step used", view.get("done") == 0 and "Measurement is absent or invalidated" in reason
                   and view["obs"]["steps_used"] == used, view)
        # 3. a transit manipulate move through `run`: refused before any motion
        view, oid = await self.m4mi_run(client, oid, "transit manipulate (run)", ["L d 0,0,.02 manipulate fast"])
        self.check("run: a transit manipulate move is refused before any motion, no step used",
                   view.get("done") == 0 and (view.get("stopped") or {}).get("reason") == MANIPULATE_REFUSAL
                   and view["obs"]["steps_used"] == used, view)
        # 4. the same through the step tool (the tool's schema accepts the intent; the runtime refuses transit)
        transit = {"command": "move_delta", "intent": "manipulate", "evidence": {"clearance": "free air above the arm"},
                   "args": {"dpos": [0.0, 0.0, 0.02], "speed_profile": "transit"}}
        refused, _ = await self.call(client, "step", {"action_id": "scripted-manipulate-transit", "observation_id": oid,
                                                      "actions": {"right": transit}})
        self.report["m4mi"]["steps"].append({"name": "transit manipulate (step tool)", "view": refused})
        self.check("step tool: a transit manipulate move is refused before any motion",
                   refused.get("ok") is False and refused.get("error_code") == "INVALID_INPUT"
                   and refused.get("error") == MANIPULATE_REFUSAL and "obs" not in refused, refused)
        # 5. a precision manipulate move through the step tool: accepted
        moved, labels = await self.call(client, "step", {"action_id": "scripted-manipulate", "observation_id": oid,
                                                         "actions": {"right": {
                                                             "command": "move_delta", "intent": "manipulate",
                                                             "args": {"dpos": [0.0, 0.0, 0.02],
                                                                      "speed_profile": "precision"}}}})
        self.report["m4mi"]["steps"].append({"name": "precision manipulate (step tool)", "view": moved})
        self.check("step tool: a precision manipulate move is accepted and reached, steps are used",
                   moved.get("ok") is True and (moved.get("arms") or {}).get("right", {}).get("reached") is True
                   and moved["obs"]["steps_used"] > used, moved)
        self.check_observation("after the step-tool manipulate move", moved["obs"], labels)
        oid = moved["obs"]["observation_id"]
        # 6. retreat and home of both arms (a run stops at a motion that ends short of its target: home then as a step)
        view, oid = await self.m4mi_run(client, oid, "home", ["B d 0,0,.03 retreat", "B home"])
        if view.get("done") != 2:
            home, _ = await self.call(client, "step", {
                "action_id": "scripted-home", "observation_id": oid,
                "actions": {arm: {"command": "home", "intent": "retreat", "args": {},
                                  "evidence": {"clearance": "free air back to the initial pose"}}
                            for arm in ("left", "right")}})
            self.report["m4mi"]["steps"].append({"name": "home step", "view": home})
            self.check("home: ok", home.get("ok") is True, home)
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ the in-place twist (m4spin)
    async def m4spin_step(self, client, name, oid, actions, used):
        """One step-tool call of the twist probe: ok and every arm reached; records the steps it used; returns
        (view, the new observation id, steps used now)."""
        view, labels = await self.call(client, "step", {"action_id": "scripted-" + name.replace(" ", "-"),
                                                        "observation_id": oid, "actions": actions})
        arms = view.get("arms") or {}
        after = (view.get("obs") or {}).get("steps_used")
        self.report["m4spin"]["stages"].append({"name": name, "ok": view.get("ok"), "steps_used_before": used,
                                                "steps_used_after": after,
                                                "steps": None if after is None else after - used,
                                                "arms": {arm: {k: a.get(k) for k in ("reached", "outcome")}
                                                         for arm, a in arms.items()}})
        self.check(f"{name}: ok, every arm reached", view.get("ok") is True and set(arms) == set(actions)
                   and all(a.get("reached") is True for a in arms.values()), view)
        self.check_observation(f"after {name}", view["obs"], labels)
        return view, view["obs"]["observation_id"], after

    async def m4spin(self, client, obs):
        """The in-place twist probe (M4SPIN_LAYOUTS; see the module docstring): in-place twists of the empty grippers in
        free air, a control move that is not in place, a home of both arms."""
        plan = self.m4spin_plan
        switch = os.environ.get("OURS_SPIN_FAST")
        self.report["m4spin"] = {"plan": json.loads(json.dumps(plan)), "switch": switch, "stages": []}
        oid, used = obs["observation_id"], obs["steps_used"]
        view, oid, used = await self.m4spin_step(client, "topdown", oid, topdown_actions(), used)
        turn = float(plan["turn_deg"])

        def manipulate(command, args):
            return {"command": command, "intent": "manipulate", "args": dict(args, speed_profile="precision")}

        # 1. the twist: both empty grippers about their own axes (fingers straight down: a turn about base z)
        actions = {"left": manipulate("move_delta", {"dpos": [0.0, 0.0, 0.0], "drot_deg": [0.0, 0.0, turn]}),
                   "right": manipulate("move_delta", {"dpos": [0.0, 0.0, 0.0], "drot_deg": [0.0, 0.0, -turn]})}
        before = used
        view, oid, used = await self.m4spin_step(client, "twist", oid, actions, used)
        steps = {"twist": used - before}
        # 2. the twist back: move_ee to the top-down rotation of the first move at the same position
        back = {}
        for arm, action in topdown_actions().items():
            back[arm] = manipulate("move_ee", {"position": action["args"]["position"],
                                               "rotation": action["args"]["rotation"], "mode": "linear"})
        before = used
        view, oid, used = await self.m4spin_step(client, "twist back", oid, back, used)
        steps["twist back"] = used - before
        # 3. the control move: a turn and 1 cm of translation (not in place: the precision rate with the switch on too)
        control = plan["control"]
        before = used
        view, oid, used = await self.m4spin_step(client, "control", oid, {"left": manipulate("move_delta", {
            "dpos": list(control["dpos"]), "drot_deg": [0.0, 0.0, float(control["turn_deg"])]})}, used)
        steps["control"] = used - before
        self.report["m4spin"]["steps"] = steps
        if switch == "1":
            self.check(f"switch on: each 90 deg twist stage needs at most {M4SPIN_ON_MAX} steps",
                       steps["twist"] <= M4SPIN_ON_MAX and steps["twist back"] <= M4SPIN_ON_MAX, steps)
        else:
            # OURS_PRECISION_2X on: the precision rate is x 2.5, so a 90 deg turn takes 0.4 of the samples
            off_min = M4SPIN_OFF_MIN_2X if os.environ.get("OURS_PRECISION_2X", "0") != "0" else M4SPIN_OFF_MIN
            self.check(f"switch off: each 90 deg twist stage needs at least {off_min} steps (the precision rate)",
                       steps["twist"] >= off_min and steps["twist back"] >= off_min - 10, steps)
        # 4. home of both arms
        home = {arm: {"command": "home", "intent": "retreat", "args": {},
                      "evidence": {"clearance": "free air back to the initial pose"}} for arm in ("left", "right")}
        view, oid, used = await self.m4spin_step(client, "home", oid, home, used)
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ grip width (OURS_GRIP_WIDTH, m4gw)
    def run_full(self, run_oid):
        """The runtime's full result of the `run` call that started from run_oid (newest archived tool result)."""
        for path in sorted((self.session / "native_tool_results").glob("*.json"), reverse=True):
            result = json.loads(path.read_text())
            if result.get("run_observation_id") == run_oid:
                return result
        return None

    async def m4gw_run(self, client, oid, name, steps, evidence=None, expect_stop=False):
        """One `run` call of the width probe: records, per step and arm, the compact outcome, the worker's width record
        (ours_grip_width) and the observation after the step (the official reading, the wrist-image estimate, the
        commanded width); returns (view, the latest observation id)."""
        args = {"observation_id": oid, "steps": steps,
                "evidence": evidence or {"clearance": "free air chosen from the layout's first top image",
                                         "target_unchanged": "the block has not moved since it was measured"}}
        view, _ = await self.call(client, "run", args)
        full = self.run_full(oid) or {}
        entry = {"name": name, "steps_sent": steps, "ok": view.get("ok"), "done": view.get("done"),
                 "stopped": view.get("stopped"), "labels": view.get("labels"),
                 "steps_used": (view.get("obs") or {}).get("steps_used"), "steps": []}
        for k, (shown, step_full) in enumerate(zip(view.get("steps") or [], full.get("steps") or [])):
            record = {"k": k, "view": shown, "samples": (step_full.get("action") or {}).get("low_level_actions"),
                      "error": step_full.get("error"), "error_code": step_full.get("error_code"), "arms": {}}
            state = ((step_full.get("observation") or {}).get("state") or {}).get("arms") or {}
            for arm, outcome in ((step_full.get("action") or {}).get("arms") or {}).items():
                width = outcome.get("ours_grip_width")
                if not isinstance(width, dict):
                    continue
                n = (state.get(arm) or {}).get("gripper_command")
                estimate = (state.get(arm) or {}).get("gripper_gap_estimate") or {}
                seen = m4gw_reading_gap_m(width["command"], n)
                record["arms"][arm] = {
                    "requested_gap_mm": width.get("requested_gap_mm"), "target_gap_mm": width.get("target_gap_mm"),
                    "command": width.get("command"), "widening": width.get("widening"), "state": width.get("state"),
                    "reached": outcome.get("target_reached"), "outcome": outcome.get("motion_outcome"),
                    "ok": outcome.get("ok"), "samples": outcome.get("trajectory_samples"),
                    "result_gap_mm": width.get("gap_mm"), "result_gap_source": width.get("gap_source"),
                    "official_reading": n, "reading_at_command": m4gw_at_command(width["command"], n),
                    "reading_gap_mm": None if seen is None else round(seen * 1000.0, 2),
                    "estimate_gap_mm": (None if estimate.get("pad_gap_m") is None
                                        else round(float(estimate["pad_gap_m"]) * 1000.0, 2)),
                    "estimate_quality": estimate.get("quality"),
                    "state_width_mm": (state.get(arm) or {}).get("gripper_width_mm"),
                    "check": ((step_full.get("recovery_guidance") or {}).get("arms") or {}).get(arm, {}).get("kind")}
            entry["steps"].append(record)
        self.report["m4gw"]["runs"].append(entry)
        if not expect_stop:
            self.check(f"{name}: the run completed", view.get("ok") is True and not view.get("stopped"), view)
        oid = (view.get("obs") or {}).get("observation_id")
        if not oid:
            latest, _ = await self.call(client, "observe", {})
            oid = latest["observation_id"]
        return view, oid, entry

    def m4gw_width_arms(self, entry):
        return [(record["k"], arm, item) for record in entry["steps"] for arm, item in record["arms"].items()]

    async def m4gw(self, client, obs):
        """The grip width probe (M4GW_LAYOUTS; see the module docstring): widths of the empty grippers in the air, a
        pre-shape, a width that stops on a block, a close on it and a lift, a release refused without evidence, the
        release by width with evidence, two empty widths after it (no evidence), a home of both arms."""
        plan = self.m4gw_plan
        self.report["m4gw"] = {"plan": json.loads(json.dumps(plan)), "switch": os.environ.get("OURS_GRIP_WIDTH"),
                               "runs": []}
        letter = {"left": "L", "right": "R"}
        arms = ("left", "right")
        both = lambda make: " & ".join(make(arm) for arm in arms)      # noqa: E731
        oid = obs["observation_id"]

        def measure_step():
            return "? " + " & ".join(f"p{letter[a].lower()}: {letter[a]} top {plan['picks'][a][0]},"
                                     f"{plan['picks'][a][1]} z={M3S1B_Z_GUESS} block top" for a in arms)

        # 1. lift from the start pose, measure the blocks, stage 6 cm above them (fingers down; the label re-measured)
        view, oid, _ = await self.m4gw_run(client, oid, "approach", [
            "B d 0,0,.05 retreat", measure_step(),
            both(lambda a: f"{letter[a]} ee p{letter[a].lower()}+0,0,{M3S1B_STAGE} down90 plan obs")])
        # 2. the empty widths in the air (no evidence needed), then the pre-shape wider than a block
        widths = list(plan["empty_widths"]) + [plan["preshape"]]
        view, oid, entry = await self.m4gw_run(client, oid, "empty widths", [f"B width {w:g}" for w in widths])
        for k, arm, item in self.m4gw_width_arms(entry):
            requested = widths[k]
            self.check(f"empty width {requested:g} mm, {arm}: at the requested gap (wrist image within "
                       f"{M4GW_TOLERANCE_MM} mm), reached, ok, the reading equal to the command, the state's width",
                       item["state"] == "at_gap" and item["reached"] is True and item["ok"] is True
                       and item["result_gap_source"] == "wrist image"
                       and abs(float(item["result_gap_mm"]) - requested) <= M4GW_TOLERANCE_MM
                       and item["reading_at_command"] is True and item["state_width_mm"] == requested, item)
        self.check("empty widths: every step of both arms recorded",
                   len(self.m4gw_width_arms(entry)) == 2 * len(widths), len(self.m4gw_width_arms(entry)))
        # 3. measure again (every gripper action invalidated the measurements) and descend around the blocks
        view, oid, entry = await self.m4gw_run(client, oid, "measure", [measure_step()])
        points = self.m3s1b_points(view)
        grasp = {}
        for a in arms:
            point = points.get(f"p{letter[a].lower()}") or {}
            top = (point.get("xyz") or [None, None, M3S1B_Z_GUESS])[2]
            grasp[a] = round(max(-M3S1B_GRASP_DOWN, M3S1B_GRASP_MIN_Z - top), 4)
        self.report["m4gw"]["grasp_offsets"] = grasp
        view, oid, _ = await self.m4gw_run(client, oid, "contact", [
            both(lambda a: f"{letter[a]} ee p{letter[a].lower()}+0,0,{grasp[a]} contact")])
        # 4. a width narrower than the blocks: it stops on them (counts as a close), not a stop of the run
        view, oid, entry = await self.m4gw_run(client, oid, "width on the blocks", [f"B width {plan['on_object']:g}"])
        for _, arm, item in self.m4gw_width_arms(entry):
            if item["state"] == "stopped_on_object":
                self.check(f"width on the block, {arm}: stopped on an object is held_short, ok, with the close's check",
                           item["outcome"] == "held_short" and item["reached"] is False and item["ok"] is True
                           and item["check"] == "verify_grasp_attempt"
                           and float(item["result_gap_mm"]) >= plan["on_object"] + M4GW_TOLERANCE_MM, item)
        # 5. the close on the blocks (the firm grasp) and a short lift
        view, oid, _ = await self.m4gw_run(client, oid, "close and lift", ["B close", "B d 0,0,.03 grasp"])
        # 6. widening while holding is a release: refused without release evidence, before any motion
        before = (view.get("obs") or {}).get("steps_used")
        view, oid, entry = await self.m4gw_run(client, oid, "release refused", [f"R width {plan['release']:g}"],
                                               expect_stop=True)
        reason = str((view.get("stopped") or {}).get("reason") or "")
        self.check("a widening width while holding, without release evidence: refused before any motion",
                   view.get("stopped") is not None and "release requires" in reason
                   and (view.get("obs") or {}).get("steps_used") == before, [view.get("stopped"), before])
        # 7. set the blocks down and release them with a width and the release evidence
        view, oid, _ = await self.m4gw_run(client, oid, "place", ["B d 0,0,-.028 place"])
        view, oid, entry = await self.m4gw_run(client, oid, "release by width", [
            f"B width {plan['release']:g} support=the table surface directly below the block"])
        for _, arm, item in self.m4gw_width_arms(entry):
            self.check(f"release by width, {arm}: widening, at the requested gap, reached",
                       item["widening"] is True and item["state"] == "at_gap" and item["reached"] is True, item)
        # 8. retreat, then two widths of the empty grippers (narrowing, widening): no release evidence needed
        view, oid, entry = await self.m4gw_run(client, oid, "empty after release", [
            "B d 0,0,.05 retreat"] + [f"B width {w:g}" for w in plan["after"]])
        self.check("widths of the empty grippers after the release: accepted without release evidence",
                   len(self.m4gw_width_arms(entry)) == 2 * len(plan["after"]), self.m4gw_width_arms(entry))
        # 9. home of both arms
        view, oid, _ = await self.m4gw_run(client, oid, "home", ["B home"])
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ declared handover and return reserve (m4gb)
    async def m4gb_step(self, client, name, oid, actions):
        """One step-tool call of the handover and return-reserve smoke; records its compact result; returns the view."""
        view, labels = await self.call(client, "step", {"action_id": f"scripted-{name}", "observation_id": oid,
                                                        "actions": actions})
        self.report["m4gb"]["steps"].append({"name": name, "view": view})
        return view

    async def m4gb_latest(self, client, view):
        """The observation id after a step result (a fresh observe when the result carries none)."""
        oid = (view.get("obs") or {}).get("observation_id")
        if oid:
            return oid, view["obs"]
        obs, _ = await self.call(client, "observe", {})
        return obs["observation_id"], obs

    def m4gb_budget(self, name, obs):
        """The observation's budget against its own numbers (agp_core/return_reserve.py reserve_state)."""
        budget = obs.get("budget") or {}
        left, used = obs.get("steps_left"), obs.get("steps_used")
        ret, spare, releases = budget.get("return"), budget.get("spare"), budget.get("releases")
        per = budget.get("per_release")
        expected_per = round(used / releases, 1) if releases else None
        expected_more = (None if not releases or spare is None else max(0, int(spare // (used / releases)))
                         if spare > 0 else 0)
        self.check(f"{name}: budget shown and consistent (spare = steps_left - return - {M4GB_MARGIN}, per_release = "
                   "steps_used / releases, more_releases = spare // per_release)",
                   isinstance(ret, int) and spare == left - ret - M4GB_MARGIN and per == expected_per
                   and budget.get("more_releases") == expected_more,
                   {"budget": budget, "steps_left": left, "steps_used": used})
        self.report["m4gb"]["budgets"].append({"name": name, "steps_used": used, "steps_left": left, **budget})
        return budget

    async def m4gb(self, client, obs):
        """The handover and return-reserve smoke (M4GB_LAYOUTS; see the module docstring): the declared handover
        (OURS_HANDOVER), the return reserve (OURS_RETURN_RESERVE)."""
        plan = self.m4gb_plan
        self.report["m4gb"] = {"plan": {k: v for k, v in plan.items()}, "steps": [], "runs": [], "budgets": [],
                               "switches": {k: os.environ.get(k) for k in ("OURS_HANDOVER", "OURS_RETURN_RESERVE",
                                                                             "OURS_ARM_GUARD", "OURS_GUARD_V2")}}
        clear = {"clearance": "free air above the table, away from objects and the other arm"}

        def move(arm, position, quat, mode, speed, intent="observe", evidence=None):
            return {"command": "move_ee", "intent": intent, "evidence": dict(clear, **(evidence or {})),
                    "args": {"position": dict(zip("xyz", position)), "rotation": dict(zip("wxyz", quat)),
                             "mode": mode, "speed_profile": speed}}

        hold = {"command": "hold", "intent": "observe", "args": {"samples": 1}}
        oid = obs["observation_id"]
        # --- the declared handover (checks named G6)
        view = await self.m4gb_step(client, "left-face", oid, {"left": move("left", *plan["left"], "plan", "transit")})
        self.check("G6: the left gripper turns to face the right arm (plan, reached)", view.get("ok") is True
                   and (view.get("arms") or {}).get("left", {}).get("reached") is True, view)
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "right-stage", oid, {"right": move(
            "right", plan["right_stage"], plan["right"][1], "plan", "transit")})
        self.check("G6: the right gripper turns to face the left gripper 15 cm away (plan, reached)",
                   view.get("ok") is True and (view.get("arms") or {}).get("right", {}).get("reached") is True, view)
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "right-face", oid, {"right": move("right", *plan["right"], "linear",
                                                                             "transit")})
        self.check("G6: the right gripper moves to 7 cm from the left gripper (linear, reached)",
                   view.get("ok") is True and (view.get("arms") or {}).get("right", {}).get("reached") is True, view)
        oid, before = await self.m4gb_latest(client, view)
        used = before["steps_used"]

        def approach(left_ev=None, right_ev=None):
            left = dict(hold, **({"evidence": left_ev} if left_ev else {}))
            right = {"command": "move_delta", "intent": "observe",
                     "args": {"dpos": list(plan["approach"]), "speed_profile": "precision"}}
            if right_ev:
                right["evidence"] = right_ev
            return {"left": left, "right": right}

        view = await self.m4gb_step(client, "approach-undeclared", oid, approach())
        error = view.get("action_error") or ""
        self.check("G6: the approach undeclared is refused by F-g before any motion (two-arm clearance, 25 mm), no step "
                   "used", view.get("ok") is False and "Two-arm clearance" in error and "25 mm" in error
                   and "declared handover" not in error and (view.get("obs") or {}).get("steps_used") == used, view)
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "approach-one-arm", oid, approach(
            right_ev={"handover": "left gripper holds the object"}))
        error = view.get("action_error") or ""
        self.check("G6: declared by the right arm only, the approach is refused as well (with the sentence that both "
                   "arms must declare), no step used", view.get("ok") is False and "Two-arm clearance" in error
                   and HANDOVER_HINT in error and (view.get("obs") or {}).get("steps_used") == used, view)
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "approach-bad-text", oid, approach(
            left_ev={"handover": "the other gripper takes the object"},
            right_ev={"handover": "left gripper holds the object"}))
        self.check("G6: a handover evidence that does not name the other gripper is refused by the runtime before "
                   "dispatch", view.get("ok") is False and view.get("error_code") == "INVALID_EVIDENCE"
                   and "right gripper" in (view.get("error") or "") and "obs" not in view, view)
        obs, _ = await self.call(client, "observe", {})
        oid = obs["observation_id"]
        self.check("G6: still no step used", obs.get("steps_used") == used, [obs.get("steps_used"), used])
        view = await self.m4gb_step(client, "approach-declared", oid, approach(
            left_ev={"handover": "right gripper takes the object from this gripper"},
            right_ev={"handover": "left gripper holds the object"}))
        self.check("G6: declared by both arms, the approach runs (the left arm holds, the right arm reaches), steps "
                   "used", view.get("ok") is True and set(view.get("arms") or {}) == {"left", "right"}
                   and (view.get("arms") or {}).get("right", {}).get("reached") is True
                   and (view.get("obs") or {}).get("steps_used", 0) > used, view)
        self.report["m4gb"]["declared_action_id"] = view.get("action_id")
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "retreat-undeclared", oid, {"right": {
            "command": "move_delta", "intent": "observe",
            "args": {"dpos": [-v for v in plan["approach"]], "speed_profile": "precision"}}})
        self.check("G6: the right arm moves back undeclared (moving apart is allowed), reached",
                   view.get("ok") is True and (view.get("arms") or {}).get("right", {}).get("reached") is True, view)
        oid, _ = await self.m4gb_latest(client, view)
        home = {arm: {"command": "home", "intent": "retreat", "args": {},
                      "evidence": {"clearance": "free air back to the initial pose"}} for arm in ("left", "right")}
        view = await self.m4gb_step(client, "home-1", oid, home)
        self.check("G6: home of both arms ok", view.get("ok") is True, view)
        oid, obs = await self.m4gb_latest(client, view)
        # --- the return reserve (checks named G4)
        view = await self.m4gb_step(client, "topdown", oid, topdown_actions())
        self.check("G4: paired top-down move, both reached", view.get("ok") is True
                   and all(a.get("reached") is True for a in (view.get("arms") or {}).values()), view)
        oid, obs = await self.m4gb_latest(client, view)
        budget = self.m4gb_budget("G4 after the move", obs)
        self.check("G4: away from the start poses the return needs steps and no release is counted yet",
                   budget.get("return", 0) > 0 and budget.get("releases") == 0 and budget.get("per_release") is None,
                   budget)
        close = {"command": "gripper", "intent": "grasp", "args": {"action": "close"}}
        for arm, evidence, counted in (
                ("left", {"release_requirement": "support",
                          "support": "free air below the gripper: a scripted release of nothing, counted as a goal "
                                     "release by its evidence"}, 1),
                ("right", {"release_requirement": "empty", "empty": "the fingers closed on nothing",
                           "object_location": "nothing was grasped", "opening_clearance": "free air around the fingers"},
                 1)):
            view = await self.m4gb_step(client, f"{arm}-close", oid, {arm: close})
            self.check(f"G4: {arm} close in free air, ok", view.get("ok") is True, view)
            oid, _ = await self.m4gb_latest(client, view)
            view = await self.m4gb_step(client, f"{arm}-open", oid, {arm: {
                "command": "gripper", "intent": "place", "args": {"action": "open"}, "evidence": evidence}})
            self.check(f"G4: {arm} open ok", view.get("ok") is True, view)
            oid, obs = await self.m4gb_latest(client, view)
            budget = self.m4gb_budget(f"G4 after the {arm} open", obs)
            self.check(f"G4: releases {counted} after the {arm} open ({evidence['release_requirement']} release)",
                       budget.get("releases") == counted, budget)
        # hold down to a spare of M4GB_SPARE (holds within the spare steps are not refused)
        spare = budget["spare"]
        burn = spare - M4GB_SPARE
        self.check(f"G4: more than {M4GB_SPARE} spare steps before the holds", burn > 0, budget)
        holds = [25] * (burn // 25) + ([burn % 25] if burn % 25 else [])
        for chunk in range(0, len(holds), 20):
            steps = [{arm: {"do": "hold", "samples": n} for arm in ("left", "right")} for n in holds[chunk:chunk + 20]]
            ran, _ = await self.call(client, "run", {"observation_id": oid, "steps": steps})
            self.check(f"G4: holds {chunk + 1}-{chunk + len(steps)} within the spare steps all run",
                       ran.get("ok") is True and ran.get("done") == len(steps), ran.get("stopped"))
            oid, obs = ran["obs"]["observation_id"], ran["obs"]
        budget = self.m4gb_budget("G4 at the reserve", obs)
        spare = budget.get("spare")
        self.check(f"G4: spare is {M4GB_SPARE} after the holds (one step either way when the estimate moved)",
                   isinstance(spare, int) and abs(spare - M4GB_SPARE) <= 1, budget)
        used = obs["steps_used"]
        # a run move that needs more than the spare steps: refused, no step used
        ran, _ = await self.call(client, "run", {"observation_id": oid, "steps": ["B d 0,0,.03 slow obs"]})
        self.report["m4gb"]["runs"].append({"name": "move beyond the spare steps", "view": ran})
        first = (ran.get("steps") or [{}])[0]
        self.check("G4: run: a move needing more than the spare steps is refused before any motion (Return reserve), "
                   "the run stops there, no step used",
                   ran.get("ok") is False and ran.get("done") == 0 and RESERVE_REFUSAL in (first.get("action_error") or "")
                   and ran["obs"]["steps_used"] == used, ran)
        oid = ran["obs"]["observation_id"]
        burst = {arm: {"command": "hold", "intent": "observe", "args": {"samples": spare + 1}}
                 for arm in ("left", "right")}
        view = await self.m4gb_step(client, "hold-undeclared", oid, burst)
        self.check(f"G4: an undeclared hold of spare + 1 = {spare + 1} steps is refused (Return reserve), no step used",
                   view.get("ok") is False and RESERVE_REFUSAL in (view.get("action_error") or "")
                   and view["obs"]["steps_used"] == used, view)
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "hold-no-return", oid, {arm: dict(a, evidence={
            "no_return": "scripted test: the declaration lets this hold run"}) for arm, a in burst.items()})
        self.check(f"G4: the same hold with evidence no_return runs ({spare + 1} steps)",
                   view.get("ok") is True and view["obs"]["steps_used"] == used + spare + 1, view)
        oid, obs = await self.m4gb_latest(client, view)
        used = obs["steps_used"]
        budget = self.m4gb_budget("G4 inside the reserve", obs)
        self.check("G4: spare is now below 0", isinstance(budget.get("spare"), int) and budget["spare"] < 0, budget)
        one = {arm: {"command": "hold", "intent": "observe", "args": {"samples": 1}} for arm in ("left", "right")}
        view = await self.m4gb_step(client, "hold1-undeclared", oid, one)
        self.check("G4: inside the reserve an undeclared hold of 1 is refused, no step used",
                   view.get("ok") is False and RESERVE_REFUSAL in (view.get("action_error") or "")
                   and view["obs"]["steps_used"] == used, view)
        oid, _ = await self.m4gb_latest(client, view)
        view = await self.m4gb_step(client, "hold1-final", oid, {arm: dict(a, evidence={
            "final": "scripted test: the last action before the return"}) for arm, a in one.items()})
        self.check("G4: the same hold with evidence final runs", view.get("ok") is True
                   and view["obs"]["steps_used"] == used + 1, view)
        oid, obs = await self.m4gb_latest(client, view)
        left_before = obs["steps_left"]
        view = await self.m4gb_step(client, "home-reserve", oid, home)
        self.check("G4: inside the reserve the home of both arms runs (never refused by the reserve), both reach",
                   view.get("ok") is True and all(a.get("reached") is True for a in (view.get("arms") or {}).values()),
                   view)
        self.check_home("G4 reserve home", steps_left_before=left_before)
        final, labels = await self.call(client, "observe", {})
        self.check_observation("final observe", final, labels)
        budget = self.m4gb_budget("G4 after the home", final)
        self.check("G4: back home the return needs 0 steps", budget.get("return") == 0, budget)
        self.report["steps_used"] = final["steps_used"]
        return final

    # ------------------------------------------------------------------ approximate home
    def latest_home(self):
        """The runtime's full record (action + observation) of the latest step whose arms carry approx_home."""
        for path in sorted((self.session / "native_tool_results").glob("*.json"), reverse=True):
            result = json.loads(path.read_text())
            records = [result] + [st for st in result.get("steps") or [] if isinstance(st, dict)]
            for record in reversed(records):
                arms = (record.get("action") or {}).get("arms") or {}
                if arms and all("approx_home" in a for a in arms.values()):
                    return record
        return None

    def check_home(self, name, steps_left_before=None):
        record = self.latest_home()
        self.check(f"{name}: a home of both arms with the approximate home (approx_home in both arm results)",
                   record is not None and set(record["action"]["arms"]) == {"left", "right"}, record and
                   list(record["action"]["arms"]))
        action = record["action"]
        arms = action["arms"]
        executed = int(action["low_level_actions"])
        after = int(record["observation"]["state"]["low_level_actions"])
        before = after - executed
        full = {arm: a["approx_home"].get("full_return_samples") for arm, a in arms.items()}
        summary = {"steps_before": before, "steps_after": after, "executed": executed, "full_return_samples": full,
                   "saved_vs_full_return": {arm: f - executed for arm, f in full.items() if f is not None},
                   "arms": {arm: {k: a["approx_home"].get(k) for k in ("tier", "method", "samples",
                                                                       "straight_full_samples", "predicted_end")}
                            | {"reached": a.get("target_reached"), "return_check": a.get("return_check"),
                               "trajectory_samples": a.get("trajectory_samples"),
                               "settle_samples": a.get("settle_samples")}
                            for arm, a in arms.items()},
                   "steps_left_before": steps_left_before}
        self.report.setdefault("approx_home", {})[name] = summary
        self.check(f"{name}: both arms reached the return region (official ee_pose inside the END margin)",
                   all(a.get("target_reached") is True and a.get("motion_outcome") == "reached"
                       and (a.get("return_check") or {}).get("inside_end_margin") is True for a in arms.values()),
                   summary["arms"])
        self.check(f"{name}: fewer control steps than the old full return (the larger of the arms' cuRobo plans to the "
                   f"exact start joints, which the old paired step took)",
                   all(f is not None for f in full.values()) and executed < max(full.values()), [executed, full])
        return record, summary

    async def burn_to(self, client, oid, left_now, target, name):
        """Holds of both arms until steps_left is `target`; returns (observation_id, steps_left). With
        OURS_RETURN_RESERVE on the holds run inside the return reserve on purpose, so they carry evidence final (with
        the switch off the request carries no evidence)."""
        burn = left_now - target
        self.check(f"{name}: enough steps left to hold down to {target}", burn >= 0, [left_now, target])
        holds = [25] * (burn // 25) + ([burn % 25] if burn % 25 else [])
        left = left_now
        declared = ({"evidence": {"final": "scripted budget test: holds that spend steps right before the final home"}}
                    if os.environ.get("OURS_RETURN_RESERVE") == "1" else {})
        for chunk in range(0, len(holds), 20):
            steps = [{arm: {"do": "hold", "samples": n} for arm in ("left", "right")} for n in holds[chunk:chunk + 20]]
            ran, _ = await self.call(client, "run", {"observation_id": oid, "steps": steps, **declared})
            self.check(f"{name}: holds {chunk + 1}-{chunk + len(steps)} done", ran.get("ok") is True
                       and ran.get("done") == len(steps), ran.get("stopped"))
            oid, left = ran["obs"]["observation_id"], ran["obs"]["steps_left"]
        self.check(f"{name}: steps_left is {target}", left == target, [left, target])
        return oid, left

    async def budget_test_free(self, client, oid):
        """The right arm away, the left arm back, FREE_TEST_STEPS_LEFT steps left: only the free plan fits."""
        name = "budget test (last resort)"
        moved, _ = await self.call(client, "step", {
            "action_id": "scripted-away", "observation_id": oid,
            "actions": {"right": {"command": "move_joints", "intent": "observe",       # precision speed: the runtime
                                  "args": {"joints": FREE_TEST_JOINTS["right"]}}}})       # takes no speed for joints
        self.check(f"{name}: right arm moved away (move_joints), reached", moved.get("ok") is True
                   and (moved.get("arms", {}).get("right") or {}).get("reached") is True, moved)
        oid, left = await self.burn_to(client, moved["obs"]["observation_id"], moved["obs"]["steps_left"],
                                       FREE_TEST_STEPS_LEFT, name)
        home, _ = await self.call(client, "step", {
            "action_id": "scripted-home-2", "observation_id": oid,
            "actions": {arm: {"command": "home", "intent": "retreat", "args": {}} for arm in ("left", "right")}})
        self.check(f"{name}: the home starts and is ok", home.get("ok") is True, home)
        record, second = self.check_home(name, steps_left_before=left)
        right = record["action"]["arms"]["right"]["approx_home"]
        restricted = {}                    # per tier, the fewest samples of the restricted (joints between) methods
        for a in right.get("attempts") or []:
            if a.get("samples") is not None and not str(a.get("method")).endswith("_free"):
                restricted[a["tier"]] = min(a["samples"], restricted.get(a["tier"], a["samples"]))
        restricted = [restricted[t] for t in ("standard", "last_chance") if t in restricted]
        full = [f for f in second["full_return_samples"].values() if f is not None]
        if os.environ.get("OURS_FREE_TOPT") == "1":
            # with F (OURS_FREE_TOPT) the home is re-timed and the right arm's restricted plan fits the steps left, so
            # the free plan is not the only one that fits; what the test is for stays checked: a full return does not
            # fit, the home stays within the steps left
            self.check(f"{name}: with F the right arm's home stayed within the {left} steps left, which the old full "
                       f"return ({max(full) if full else None}) exceeds", right.get("tier") in ("standard", "last_chance")
                       and full and max(full) > left and second["executed"] <= left,
                       [right.get("tier"), right.get("joint_range"), right.get("method"), restricted, full, left,
                        second["executed"]])
            return
        self.check(f"{name}: the right arm took a free plan (any joint values); its restricted plans ({restricted}) and "
                   f"the old full return ({max(full) if full else None}) need more than the {left} steps left; the "
                   f"step stayed within them", right.get("joint_range") == "free"
                   and right.get("tier") in ("standard", "last_chance") and len(restricted) == 2
                   and min(restricted) > left and full and max(full) > left
                   and second["executed"] <= left, [right.get("tier"), right.get("joint_range"), right.get("method"),
                                                     restricted, full, left, second["executed"]])

    async def budget_test_standard(self, client, oid, first):
        """The paired top-down move again; holds until steps_left is the first home's samples + 6 (below a full
        return); a home of both arms then starts and the return check accepts its end."""
        name = "second home (budget test)"
        moved, _ = await self.call(client, "step", {"action_id": "scripted-topdown-2", "observation_id": oid,
                                                    "actions": topdown_actions()})
        self.check("second paired top-down move: ok, both reached", moved.get("ok") is True and all(
            a.get("reached") is True for a in moved.get("arms", {}).values()), moved)
        need = max(a["samples"] for a in first["arms"].values()) + 6
        oid, left = await self.burn_to(client, moved["obs"]["observation_id"], moved["obs"]["steps_left"], need,
                                       "budget test")
        home, _ = await self.call(client, "step", {
            "action_id": "scripted-home-2", "observation_id": oid,
            "actions": {arm: {"command": "home", "intent": "retreat", "args": {}} for arm in ("left", "right")}})
        self.check("budget test: the home starts and is ok", home.get("ok") is True, home)
        _, second = self.check_home(name, steps_left_before=left)
        full = [f for f in second["full_return_samples"].values() if f is not None]
        self.check("budget test: steps_left before the home was below the old full return (that home would have been "
                   "refused for budget) and the approximate home stayed within it",
                   full and left < max(full) and second["executed"] <= left, [left, full, second["executed"]])

    async def approx_home_checks(self, client):
        _, first = self.check_home("first home")
        # settling: the official check keeps accepting while both arms are held
        oid = (await self.call(client, "observe", {}))[0]["observation_id"]
        held, _ = await self.call(client, "step", {
            "action_id": "scripted-hold", "observation_id": oid,
            "actions": {arm: {"command": "hold", "intent": "observe", "args": {"samples": 25}} for arm in ("left", "right")}})
        self.check("hold after the home: ok", held.get("ok") is True, held)
        oid = held["obs"]["observation_id"]
        model = json.loads((self.session / "approx_home.json").read_text()).get("planning_model") or {}
        if all((model.get(arm) or {}).get("chosen") == "public_urdf" for arm in ("left", "right")):
            await self.budget_test_free(client, oid)
        else:
            await self.budget_test_standard(client, oid, first)

    def check_captures(self):
        captures = sorted(p for p in (self.session / "captures").iterdir() if p.is_dir())
        bad = {}
        for directory in captures:
            # a raised measurement's match review image (agp_core/rgb_measure.py) is RGB, not depth
            files = {p.name for p in directory.iterdir() if not (p.name.startswith("match_") and p.suffix == ".png")}
            calibration = json.loads((directory / "calibration.json").read_text())
            problems = []
            if files != CAPTURE_FILES:
                problems.append(f"files {sorted(files)}")
            if any("depth" in name for name in files):
                problems.append("depth file")
            if any("depth" in cam for cam in calibration.values()):
                problems.append("depth entry in calibration.json")
            if abs(calibration["top"]["K"][0][0] - HEAD_FX) > 1e-9:
                problems.append(f"head fx {calibration['top']['K'][0][0]}")
            if problems:
                bad[directory.name] = problems
        self.check(f"{len(captures)} captures: RGB + public-model calibration only, no depth",
                   captures and not bad, bad)

    def check_preflight(self):
        records = [json.loads(p.read_text()) for p in sorted((self.session / "motion_diagnostics").glob("*.json"))]
        motions = [r for r in records if r.get("command") in MOTIONS]
        self.check(f"joint-trajectory preflight ok for all {len(motions)} arm motions",
                   motions and all((r.get("preflight") or {}).get("ok") is True for r in motions),
                   [(r.get("command"), r.get("preflight")) for r in motions
                    if not (r.get("preflight") or {}).get("ok")])
        self.report["motions"] = [{"arm": r.get("arm"), "command": r.get("command"), "outcome": r.get("motion_outcome"),
                                   "executed_samples": r.get("executed_samples"),
                                   "preflight_scope": (r.get("preflight") or {}).get("scope")} for r in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--name", choices=("smoke", "observe", "idle"), required=True)
    parser.add_argument("--fault", choices=("mcp_start", "mcp_crash", "agent_crash"))
    args = parser.parse_args()
    if args.fault in DEVELOPMENT_ONLY_FAULTS and PATHS.PACKAGED:
        parser.error(f"--fault {args.fault} needs {DEVELOPMENT_ONLY_FAULTS[args.fault]}, a test-only file of the "
                     f"development tree that the packaged runtime {ROOT} leaves out (scripts/package_policy.py)")
    session = args.session.resolve()
    code = 0
    script = None
    report = {"kind": f"scripted:{args.name}"}
    try:
        script = Script(session, args.name, args.fault)
        report = script.report
        asyncio.run(asyncio.wait_for(script.run(), timeout=1800))
        report["verified"] = True
    except BaseException as exc:
        detail = exc
        while isinstance(detail, BaseExceptionGroup) and detail.exceptions:   # MCP's task groups wrap errors
            detail = detail.exceptions[0]
        if isinstance(detail, CheckFailed):   # a check of the sequence failed
            code = 1
            report.update(verified=False, error=f"check failed: {detail}")
        else:                                 # MCP server or session failure
            code = 3
            report.update(verified=False, error=f"{type(detail).__name__}: {detail}",
                          traceback=traceback.format_exc()[-4000:])
    finally:
        report["fault"] = args.fault
        finished = {"kind": f"scripted:{args.name}", "returncode": code}
        if script is not None and script.mcp_process is not None:
            if script.mcp_process.poll() is None:          # still running: the driver ends it (not a crash)
                script.mcp_process.terminate()
                try:
                    script.mcp_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    script.mcp_process.kill()
            finished.update(script.attribution)            # --fault mcp_crash: who ended, as agp_core/run_agent.py
        (session / "scripted_report.json").write_text(json.dumps(report, indent=1, default=str))
        (session / "policy_finished.json").write_text(json.dumps(finished))
        print(json.dumps({"verified": report.get("verified"), "error": report.get("error"),
                          "checks": {k: v["pass"] for k, v in report.get("checks", {}).items()}}, indent=1),
              flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
