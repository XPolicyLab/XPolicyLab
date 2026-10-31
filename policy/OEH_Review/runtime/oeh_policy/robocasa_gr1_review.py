"""RoboCasa GR1's review port: OEH review mode (``oeh_policy.review``) over the released OpenWAM GR1 tabletop chunk.

The method is RoboDojo's ``review.ReviewMethod`` with GR1's action space (``action_space.RoboCasaGR1EEF33``). The
GR1's arms end in Fourier hands, not grippers, so the review layout has one field per arm, the pose, and the teacher
corrects the two arms only: a decision has no gripper field anywhere (``ActionSpace.gripper`` False). The hands and
the waist are carried: OpenWAM's rows (``student``, and the fallback) execute unchanged, hands and waist included,
and every row the harness or the teacher's corrections build (hold, edit, eef, home) holds both hands and the waist
at their last commands, the values this method's ledger keeps (``carried``: the first observation's measured values,
then each executed row's). What differs from RoboDojo besides:

- ``student`` rows are OpenWAM's native EEF33 rows, unchanged and unbounded: the rows the direct condition executes
  (GR1's ``action_limits`` is ``{"native_action": "eef33", "finite": true}`` for every condition, and the paired
  comparisons require it equal).
- The workspace box the harness's and the teacher's own rows start in and are checked against is the frozen
  ``review_workspace``, the box ``oeh_policy.robocasa_gr1_bounds`` derives at freeze from the pinned robosuite GR1
  model (each hand's kinematic reach around the waist).
- The packet shows each hand as one closure number (``hand_closure``) and the waist in degrees, and its proposal
  table has a hand-closure column per arm and the waist's three angles; the previous-result record lists the hand
  closures each executed row commanded.

The candidate is one fresh full OpenWAM chunk on the current observation (``robocasa_gr1.OpenWAMBackend.
review_chunk``), the official loop gets one row per call (``Runtime.next_action``) and the teacher's view is the
ego view. The teacher's prompt set is ``teacher/robocasa_gr1/``.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .action_space import RoboCasaGR1EEF33, check_review_row
from .profiles import PLACEHOLDER, check_box
from .review import ReviewMethod, eef_view, home_offset, rpy_deg

WORKSPACE_KEYS = ("frame", "quaternion", "left_xyz_bounds", "right_xyz_bounds", "bounds_source")
PROPOSAL_FRAME = ("GR1 robot base (pelvis, fixed in the scene); position m; roll/pitch/yaw deg (extrinsic xyz); "
                  "hand closure 0 open, 1 fully flexed; waist yaw/pitch/roll deg")
# Each Fourier hand's six joint targets (robosuite 1.5.1 grippers/fourier_{left,right}_hand.xml, in the order the
# released state and action use: robocasa's gather_robot_observations reverses unformat_gripper_space's pick, which
# gives format_action's order): the little, ring, middle and index fingers' flexion, the thumb's bend and the thumb's
# rotation. The closure averages the first five, each over its joint's upper limit (the intermediate finger joints'
# range 0-1.74, the thumb pitch joint's 0-1.22); the thumb's rotation is not a closing motion and is left out.
HAND_LIMITS = (1.74, 1.74, 1.74, 1.74, 1.22)
CLOSED = 0.5  # a closure at or above this reads as a closing hand in the diagnostics


def hand_closure(hand) -> float:
    """One hand's six joint values (rad) as a closure: 0 open (straight fingers), 1 fully flexed."""
    values = np.asarray(hand, dtype=float).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("a Fourier hand has six finite joint values")
    return float(np.mean(np.clip(values[:5] / np.asarray(HAND_LIMITS), 0.0, 1.0)))


def waist_deg(waist) -> dict:
    """The waist's yaw, pitch and roll (rad, EEF33's order) in degrees, one decimal."""
    values = np.asarray(waist, dtype=float).reshape(-1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("the GR1 waist has three finite joint values")
    return {name: round(math.degrees(float(value)), 1) for name, value in zip(("yaw", "pitch", "roll"), values)}


def check_review_workspace(value) -> dict:
    """The frozen review workspace: the GR1 robot-base frame, wxyz quaternions, each arm's finite 3x2 box and the
    pinned files it was derived from. An unfilled ``READ_FROM_PINNED`` placeholder fails: the box is never typed."""
    if not isinstance(value, Mapping) or set(value) != set(WORKSPACE_KEYS):
        raise ValueError(f"RoboCasa GR1 review mode requires review_workspace with exactly {sorted(WORKSPACE_KEYS)}")
    workspace = dict(value)
    if workspace["frame"] != "gr1_robot_base" or workspace["quaternion"] != "wxyz":
        raise ValueError("the RoboCasa GR1 review workspace is in the GR1 robot-base frame with wxyz quaternions")
    source = workspace["bounds_source"]
    if isinstance(source, str) and PLACEHOLDER in source:
        raise ValueError("review_workspace.bounds_source still needs its value from the pinned sources")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("review_workspace.bounds_source must cite the pinned files the bounds come from")
    for arm in ("left", "right"):
        workspace[arm + "_xyz_bounds"] = check_box(workspace[arm + "_xyz_bounds"], f"review_workspace.{arm}_xyz_bounds")
    return workspace


class RoboCasaGR1ReviewMethod(ReviewMethod):
    """``review.ReviewMethod`` on RoboCasa GR1: EEF33 in and out, the arms reviewed, the hands and the waist carried."""

    space = RoboCasaGR1EEF33()  # each instance binds its own, holding what its ledger carries

    def __init__(self, config: dict):
        super().__init__(config)
        self.workspace = check_review_workspace(config.get("review_workspace"))
        # The hands' and the waist's last commands, which a correction's rows hold (the space reads them through
        # ``held``); None until the first observation.
        self.carried = None
        self.space = RoboCasaGR1EEF33(hold=self.held)

    def held(self) -> dict:
        if self.carried is None:
            raise ValueError("the hands' and the waist's commands are unknown before the first observation")
        return {key: value.copy() for key, value in self.carried.items()}

    # -- the review port's hooks -----------------------------------------------------------------------------------

    def seed(self, state):
        """The first observation fixes the episode-start poses and the carried ledger: the measured hands and waist
        stand for the last command, as RoboDojo's measured opening does. Later observations never overwrite them."""
        first = self.home is None
        super().seed(state)
        if first:
            self.carried = self.space.carried_values(state["eef33"])

    def acknowledged(self, action):
        """An executed row's hands and waist are the last commands from now on (OpenWAM's, or the held ones)."""
        self.carried = self.space.carried_values(action)

    def bounds(self, arm: str):
        return self.workspace[arm + "_xyz_bounds"]

    def check_row(self, row: Mapping) -> dict:
        """A correction or hold row (review layout, poses only) inside the frozen workspace, unit quaternions."""
        checked = check_review_row(row, self.arms, self.gripper)
        for arm in self.arms:
            pose = checked[arm + "_ee_pose"]
            bounds = np.asarray(self.bounds(arm), dtype=float)
            if np.any(pose[:3] < bounds[:, 0]) or np.any(pose[:3] > bounds[:, 1]):
                raise ValueError("native action outside frozen workspace bounds")
            if abs(np.linalg.norm(pose[3:]) - 1) > 1e-3:
                raise ValueError("native action needs a unit wxyz quaternion")
        return checked

    def student_rows(self, candidate: list, steps: int) -> list:
        """The first ``steps`` EEF33 rows of OpenWAM's chunk exactly as it returned them (the runtime already checked
        each is 33 finite float32 values, GR1's native contract): copies, never clipped or renormalized."""
        return [np.asarray(row, dtype=np.float32).reshape(-1).copy() for row in candidate[:steps]]

    def executed_rows(self, rows: list, state: Mapping) -> list:
        """The offered EEF33 rows themselves: the previous-result record reads their hands."""
        return [self.space.eef33(row).copy() for row in rows]

    def describe_executed(self, decision_index, intent: Mapping, rows: list, executed: int, source: str,
                          trigger: str | None, assessment: Mapping | None) -> dict:
        """RoboDojo's record without a gripper column, plus the hand closures each executed row commanded."""
        record = super().describe_executed(decision_index, intent, rows, executed, source, trigger, assessment)
        closures = [{arm: round(hand_closure(np.asarray(row)[self.space.HAND[arm]]), 3) for arm in self.arms}
                    for row in rows[:executed]]
        ordered = {}
        for key, value in record.items():
            if key == "reason":
                ordered["executed_hand_closure"] = closures
            ordered[key] = value
        return ordered

    def packet_state(self, candidate: list | None, state: Mapping, truncated: Mapping | None = None) -> dict:
        """The packet's robot fields, in packet order: the measured hand poses with each hand's commanded and measured
        closure, the waist's commanded and measured angles, the episode-start poses, each arm's offset from them and
        the candidate as the teacher reads it."""
        measured = self.space.review_state(state)
        values = self.space.carried_values(state["eef33"])
        current = eef_view(measured, None, self.arms)
        for arm in self.arms:
            current[arm]["hand_closure_command"] = round(hand_closure(self.carried[arm + "_hand"]), 3)
            current[arm]["hand_closure_measured"] = round(hand_closure(values[arm + "_hand"]), 3)
        view = (self.proposal_view(candidate, measured, values["waist"]) if candidate else
                {"rows": 0, "table": [], "columns": [], "diagnostics": {}})
        if truncated:
            view["truncated"] = truncated
        return {"current_eef": current,
                "waist_deg": {"command": waist_deg(self.carried["waist"]), "measured": waist_deg(values["waist"])},
                "episode_start_eef": eef_view(self.home, None, self.arms),
                "home_offset": home_offset(measured, self.home, self.arms),
                "student_proposal": view}

    def proposal_view(self, rows: list, measured: Mapping, waist) -> dict:
        """OpenWAM's EEF33 rows as the teacher reads them: per arm the hand's pose (m, extrinsic xyz deg) and closure,
        then the waist (deg), plus plain arithmetic diagnostics from the measured poses and waist (rad): each arm's
        displacement, path and largest step, its closure range and the rows where the hand closes or opens again, and
        the waist's net change. Not a safety, collision, intent or success verdict."""
        space, arms = self.space, self.arms
        rows = [space.eef33(row) for row in rows]
        poses = [space.poses(row) for row in rows]
        closures = {arm: [hand_closure(row[space.HAND[arm]]) for row in rows] for arm in arms}
        waists = [waist_deg(row[space.WAIST]) for row in rows]
        table = []
        for index, pose in enumerate(poses):
            line = [index]
            for arm in arms:
                value = np.asarray(pose[arm + "_ee_pose"], dtype=float)
                line += [round(float(v), 4) for v in value[:3]] + rpy_deg(value[3:])
                line.append(round(closures[arm][index], 3))
            line += [waists[index][name] for name in ("yaw", "pitch", "roll")]
            table.append(line)
        diagnostics = {}
        for arm in arms:
            positions = np.asarray([np.asarray(pose[arm + "_ee_pose"], dtype=float)[:3] for pose in poses])
            start = np.asarray(measured[arm + "_ee_pose"], dtype=float)[:3]
            steps = np.linalg.norm(np.diff(np.vstack([start, positions]), axis=0), axis=1)
            values = closures[arm]
            diagnostics[arm] = {
                "net_displacement_m": [round(float(v), 4) for v in positions[-1] - start],
                "path_length_m": round(float(steps.sum()), 4),
                "max_step_m": round(float(steps.max()), 4),
                "hand_closure_range": [round(min(values), 3), round(max(values), 3)],
                "hand_closes_at_row": next((i for i, v in enumerate(values) if v >= CLOSED), None),
                "hand_opens_at_row": next((i for i, v in enumerate(values)
                                           if v < CLOSED and i and values[i - 1] >= CLOSED), None)}
        change = np.degrees(rows[-1][space.WAIST].astype(float) - np.asarray(waist, dtype=float))
        diagnostics["waist"] = {"net_change_deg": {name: round(float(value), 1)
                                                   for name, value in zip(("yaw", "pitch", "roll"), change)}}
        columns = ["i"] + [f"{arm}_{field}" for arm in arms
                           for field in ("x", "y", "z", "roll", "pitch", "yaw", "hand_closure")]
        columns += ["waist_yaw", "waist_pitch", "waist_roll"]
        return {"rows": len(rows), "columns": columns, "table": table, "diagnostics": diagnostics,
                "frame": PROPOSAL_FRAME}
