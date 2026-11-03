"""Embodiment profiles: the frozen robot contract a policy configuration selects.

A profile fixes, for fast and slow alike, the native action, the policy
observation, the shape of the admission limits, the slow persona and primitive
set, the gripper and quaternion conventions and the slow-brain cameras. The
benchmark decides the profile; a configuration naming another one fails
preflight (plan §3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The policy side of the benchmark registry (oeh_eval.benchmarks, which a test holds these to; the policy package
# imports nothing from the evaluator at load time). oeh-eval names LIBERO-Plus `libero_plus`; older oeh-policy files
# spell it `libero-plus`.
BENCHMARK_ALIASES = {"libero-plus": "libero_plus", "libero-pro": "libero_pro", "robocasa-gr1": "robocasa_gr1"}
# Benchmarks whose contract implements review mode (oeh_policy.review): the teacher may run only on these.
REVIEW_BENCHMARKS = ("robodojo", "libero", "libero_plus", "robotwin", "libero_pro", "robocasa_gr1", "vlabench",
                     "robocasa365", "ebench")
# Plan §1: a slow primitive emits at most 32 rows, each admitted and executed alone.
SLOW_CHUNK_ROWS = 32
PLACEHOLDER = "READ_FROM_PINNED"


@dataclass(frozen=True)
class Profile:
    name: str
    benchmarks: tuple[str, ...]
    native_action: str
    observation: str
    primitive_limits: tuple[str, ...]
    persona: str
    primitives: tuple[str, ...]
    arms: tuple[str, ...]
    gripper_open: float
    gripper_closed: float
    gripper_semantics: str
    quaternion: str
    cameras: tuple[str, ...]

    @property
    def persona_path(self) -> Path:
        return Path(__file__).with_name(self.persona)

    def openness(self, command: float) -> float:
        """0 = closed, 1 = open, whatever the native gripper sign."""
        return (float(command) - self.gripper_closed) / (self.gripper_open - self.gripper_closed)


SINGLE_ARM_EEF = Profile(
    name="single_arm_eef", benchmarks=("libero", "libero_plus", "libero_pro"),
    native_action="7D OSC delta [dxyz, drotvec, gripper] in [-1, 1] (robosuite OSC_POSE, control_delta)",
    observation="achieved EEF10 [robot0_eef_pos, rot6d(robot0_eef_quat xyzw), gripper open-scale]",
    primitive_limits=("chunk_rows", "bounds_frame", "workspace_bounds_m", "bounds_source"),
    persona="method_single_arm.txt",
    # One arm: OpenWAM-α's left end-effector block, as every single-arm slot map (oeh_policy.action_space).
    primitives=("vla", "move_relative", "gripper", "wait", "home"), arms=("left",),
    gripper_open=-1.0, gripper_closed=1.0,
    gripper_semantics="native sign (-1 open, +1 close); the ledger stores the last native command",
    quaternion="xyzw", cameras=("agentview", "eye_in_hand"))

DUAL_ARM_EE16 = Profile(
    name="dual_arm_ee16", benchmarks=("robodojo", "robotwin"),
    native_action="EE16 [l_xyz, l_quat_wxyz, l_grip, r_xyz, r_quat_wxyz, r_grip]",
    observation="per-arm EE xyz + quaternion wxyz and commanded gripper",
    primitive_limits=("chunk_rows", "frame", "quaternion", "left_xyz_bounds", "right_xyz_bounds",
                      "bounds_source"),
    persona="method.txt",
    primitives=("vla", "move_relative", "gripper", "wait", "home"), arms=("left", "right"),
    gripper_open=1.0, gripper_closed=0.0,
    gripper_semantics="1 open, 0 closed; the observed value is the previous command",
    quaternion="wxyz", cameras=("cam_head", "cam_left_wrist", "cam_right_wrist"))

# RoboCasa GR1 tabletop: the humanoid's two arms end in Fourier hands and it has a waist (oeh_policy.action_space.
# RoboCasaGR1EEF33). It runs the direct and review conditions only, so it has no slow persona or primitives (its
# persona_path is never read). No gripper: its "gripper" fields describe the hand closure the review packet shows.
GR1_EEF33 = Profile(
    name="gr1_eef33", benchmarks=("robocasa_gr1",),
    native_action=("EEF33 [l_xyz, l_rot6d, l_hand6, r_xyz, r_rot6d, r_hand6, waist3] in the robot-base frame, "
                   "solved to absolute arm joint targets by the released client's IK"),
    observation="EEF33 of the measured joints and grip sites, the ego view and the instruction",
    primitive_limits=(), persona="", primitives=(), arms=("left", "right"),
    gripper_open=0.0, gripper_closed=1.0,
    gripper_semantics=("no gripper: each hand's six joint targets (rad, 0 open) are OpenWAM's and are held by "
                       "corrections; the review packet reads them as a closure, 0 open, 1 fully flexed"),
    quaternion="rot6d (native); wxyz in the review layout", cameras=("ego_view",))

PROFILES = {profile.name: profile for profile in (SINGLE_ARM_EEF, DUAL_ARM_EE16, GR1_EEF33)}

# VLABench's Franka Panda: one absolute end-effector target per native step, world frame, which its evaluator turns into
# joint targets by IK (VLABench cf588fe evaluation/evaluator/base.py:147-158). Review mode only: the hybrid primitives
# are not implemented for it, so its primitive limits are empty (oeh_policy.vlabench).
SINGLE_ARM_EE = Profile(
    name="single_arm_ee", benchmarks=("vlabench",),
    native_action="8D absolute EE target [xyz, roll/pitch/yaw extrinsic xyz, finger1, finger2] in VLABench's world "
                  "frame (fingers 0.04 open, 0 closed)",
    observation="ee_state [xyz, quaternion wxyz, open flag] in the world frame, three RGB views, the instruction",
    primitive_limits=(), persona="method_single_arm.txt",
    primitives=("vla", "move_relative", "gripper", "wait", "home"), arms=("left",),
    gripper_open=1.0, gripper_closed=0.0,
    gripper_semantics="1 open, 0 closed; the observed value is the measured finger gap, the ledger keeps the command",
    quaternion="wxyz", cameras=("forward", "wrist", "right"))
PROFILES[SINGLE_ARM_EE.name] = SINGLE_ARM_EE


def normalize_benchmark(name) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("benchmark identity required")
    return BENCHMARK_ALIASES.get(name, name)


def profile_for(benchmark) -> Profile:
    name = normalize_benchmark(benchmark)
    for profile in PROFILES.values():
        if name in profile.benchmarks:
            return profile
    raise ValueError(f"no embodiment profile for benchmark {benchmark!r}")


def check_embodiment(value, benchmark) -> Profile:
    """The configuration's `embodiment` must be the profile its benchmark requires."""
    expected = profile_for(benchmark)
    if value is None:
        raise ValueError("embodiment profile required")
    if value not in PROFILES:
        raise ValueError(f"unknown embodiment profile {value!r}")
    if value != expected.name:
        raise ValueError(f"embodiment {value!r} does not match benchmark {benchmark!r}; "
                         f"{expected.name} required")
    return expected


def check_box(value, name: str) -> list:
    """A finite 3x2 [low, high] box. Unfilled pinned-source placeholders fail."""
    if isinstance(value, str) and PLACEHOLDER in value:
        raise ValueError(f"{name} still needs its value from the pinned sources")
    try:
        box = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a 3x2 box") from None
    if box.shape != (3, 2) or not np.isfinite(box).all() or np.any(box[:, 0] >= box[:, 1]):
        raise ValueError(f"finite robot-derived {name} required")
    return box.tolist()


def check_primitive_limits(profile: Profile, value) -> dict:
    """Frozen slow-primitive limits; the shape is the profile's."""
    if not isinstance(value, dict) or set(value) != set(profile.primitive_limits):
        raise ValueError(f"{profile.name} primitives require exactly {sorted(profile.primitive_limits)}")
    limits = dict(value)
    rows = limits["chunk_rows"]
    if type(rows) is not int or not 1 <= rows <= SLOW_CHUNK_ROWS:
        raise ValueError(f"slow chunk rows must be an integer in [1, {SLOW_CHUNK_ROWS}]")
    source = limits["bounds_source"]
    if isinstance(source, str) and PLACEHOLDER in source:
        raise ValueError("bounds_source still needs its value from the pinned sources")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("bounds_source must cite the pinned files the bounds come from")
    if profile is SINGLE_ARM_EEF:
        if limits["bounds_frame"] != "home_anchored_base_axes":
            raise ValueError("single-arm bounds are anchored at the first-seen EE pose, base axes")
        limits["workspace_bounds_m"] = check_box(limits["workspace_bounds_m"], "workspace_bounds_m")
        box = np.asarray(limits["workspace_bounds_m"])
        if np.any(box[:, 0] > 0) or np.any(box[:, 1] < 0):
            raise ValueError("home-anchored bounds must contain the first-seen pose")
    else:
        if limits["frame"] != "robotwin_world" or limits["quaternion"] != "wxyz":
            raise ValueError("RoboTwin primitives use native world-frame poses with wxyz quaternions")
        for arm in ("left", "right"):
            limits[arm + "_xyz_bounds"] = check_box(limits[arm + "_xyz_bounds"], arm + "_xyz_bounds")
    return limits


# RoboCasa365 (oeh_policy.robocasa365): one Panda arm on the Omron mobile base, robosuite's PandaOmron. It runs review
# mode only, so it has no slow persona, primitives or primitive limits; its native action also drives the base.
MOBILE_ARM_EEF12 = Profile(
    name="mobile_arm_eef12", benchmarks=("robocasa365",),
    native_action="12D [dxyz, drotvec (OSC_POSE delta in the mobile base frame), gripper_close, base forward, side, "
                  "yaw, torso, control_mode] (RoboCasa365 PandaOmron)",
    observation="RGB robot0_agentview_left, robot0_eye_in_hand, robot0_agentview_right; base world pose (xyzw), "
                "grip-site pose in the base frame (xyzw), finger joints",
    primitive_limits=(), persona="", primitives=(), arms=("left",),
    gripper_open=-1.0, gripper_closed=1.0,
    gripper_semantics="native gripper_close sign (+1 closes, -1 opens); the state carries the finger joints and the "
                      "ledger the last command",
    quaternion="xyzw", cameras=("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"))
PROFILES[MOBILE_ARM_EEF12.name] = MOBILE_ARM_EEF12


# EBench (GenManip lift2: two R5a arms on a lift and a mobile base). Review mode only: no slow primitives, so no
# primitive limits; the review workspace is per arm in each arm's own base frame (oeh_policy.ebench_review).
DUAL_ARM_MOBILE_EE = Profile(
    name="dual_arm_mobile_ee", benchmarks=("ebench",),
    native_action=("GenManip ee_pose dict: per arm [xyz, quat_wxyz, [finger, finger] m] in that arm's base frame, "
                   "and a relative base_motion [dx m, dy m, dyaw deg]"),
    observation="per-arm EE xyz + quaternion wxyz (arm base frames), four measured finger positions, base [x, y, yaw]",
    primitive_limits=(), persona="method.txt", primitives=(), arms=("left", "right"),
    gripper_open=1.0, gripper_closed=0.0,
    gripper_semantics="1 open, 0 closed (finger / 0.044 m); the observed value is the measured finger gap",
    quaternion="wxyz", cameras=("cam_overlook", "cam_left_wrist", "cam_right_wrist"))
PROFILES[DUAL_ARM_MOBILE_EE.name] = DUAL_ARM_MOBILE_EE
