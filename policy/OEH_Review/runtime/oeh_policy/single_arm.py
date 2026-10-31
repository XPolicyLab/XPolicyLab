"""Single-arm slow capabilities for LIBERO's 7D OSC delta controller.

robosuite's OSC_POSE with ``control_delta`` sets its goal to the pose achieved at
the start of a step plus the scaled delta: ``ee_pos + dx`` and ``R(dr) @ ee_ori``.
A precomputed chunk of deltas would therefore drift with the tracking lag. Each
primitive row is computed from the latest achieved EEF10 instead, so the goal
lands on a Cartesian reference that advances at most 5 mm and 25 mrad per row
(the dual-arm method's interpolation bounds) and then holds. The first row is
admitted with the slow decision, every later one just before its own submission.

The released client converts the raw EEF10 it receives; ``raw_eef10`` builds the
raw action that the pinned ``native_eef10_to_libero7d`` maps back onto a row.
"""
from __future__ import annotations

import math

import numpy as np

from .actions import TRANSITIONS, mode_rules
from .contracts import PHASES
from .gateway import FollowingRows
from .osc_controller import OscScale
from .profiles import SINGLE_ARM_EEF

STEP_M = .005
STEP_RAD = .025
MAX_MOVE_M = .030000001
REQUIRED = ("observation_id", "mode", "phase", "target", "execute_steps")
MODES = {"vla": {"subgoal"}, "move_relative": {"delta_xyz"}, "gripper": {"gripper_command"},
         "wait": set(), "home": set()}


def intent_schema(max_rows: int) -> dict:
    # check_intent requires exactly MODES[mode] besides REQUIRED; the brain is shown that rule (actions.mode_rules).
    rule, only = mode_rules({mode: tuple(sorted(fields)) for mode, fields in MODES.items()})
    return {"type": "object", "additionalProperties": False, "required": list(REQUIRED), "description": rule,
            "properties": {
                "observation_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "mode": {"type": "string", "enum": list(MODES)},
                "phase": {"type": "string", "enum": list(PHASES)},
                "target": {"type": "string", "minLength": 1, "maxLength": 128},
                "execute_steps": {"type": "integer", "minimum": 1, "maximum": max_rows},
                "subgoal": {"type": "string", "minLength": 1, "maxLength": 512, "description": only["subgoal"]},
                "delta_xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                              "description": only["delta_xyz"]},
                "gripper_command": {"type": "number", "minimum": -1, "maximum": 1,
                                    "description": only["gripper_command"]}}}


def rotvec_matrix(vector) -> np.ndarray:
    """Rodrigues: rotation vector -> matrix."""
    v = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def matrix_rotvec(matrix) -> np.ndarray:
    """Rotation matrix -> rotation vector, stable near 0 and pi."""
    r = np.asarray(matrix, dtype=float)
    angle = math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    if angle < 1e-6:
        return skew
    if math.pi - angle < 1e-4:
        symmetric = (r + np.eye(3)) / 2
        index = int(np.argmax(np.diag(symmetric)))
        axis = symmetric[:, index] / math.sqrt(max(symmetric[index, index], 1e-12))
        axis /= np.linalg.norm(axis)
        return axis * angle if axis @ skew >= 0 else -axis * angle
    return skew * (angle / math.sin(angle))


def rot6d_matrix(r6d) -> np.ndarray:
    """OpenWAM rot6d (first two rotation-matrix columns) -> matrix, by Gram-Schmidt."""
    a1, a2 = np.asarray(r6d[:3], dtype=float), np.asarray(r6d[3:6], dtype=float)
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    b2 = a2 - float(b1 @ a2) * b1
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def raw_eef10(native) -> list[float]:
    """Raw EEF10 that the released conversion maps onto a 7D OSC row.

    Position stays in normalized command space, the rotation vector is encoded as
    rot6d of its rotation, and the gripper flips back to OpenWAM's open scale.
    """
    row = np.asarray(native, dtype=float)
    rotation = rotvec_matrix(row[3:6])
    return np.concatenate([row[:3], rotation[:, 0], rotation[:, 1], [-row[6]]]).astype(np.float32).tolist()


def achieved(state) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(state, dtype=float)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError("LIBERO achieved EEF10 state required")
    return value[:3].copy(), rot6d_matrix(value[3:9])


class Plan:
    """A bounded Cartesian reference tracked row by row from the achieved pose."""

    def __init__(self, method, start, goal, grip, rows):
        (self.p0, self.r0), (goal_p, goal_r) = start, goal
        self.dp, self.turn = goal_p - self.p0, matrix_rotvec(goal_r @ self.r0.T)
        self.horizon = max(1, math.ceil(float(np.linalg.norm(self.dp)) / STEP_M),
                           math.ceil(float(np.linalg.norm(self.turn)) / STEP_RAD))
        self.method, self.grip, self.rows, self.index = method, grip, rows, 0

    def next(self, state):
        if self.index >= self.rows:
            raise RuntimeError("primitive chunk exhausted")
        self.index += 1
        fraction = min(1.0, self.index / self.horizon)
        reference_p = self.p0 + fraction * self.dp
        reference_r = rotvec_matrix(fraction * self.turn) @ self.r0
        position, rotation = achieved(state)
        physical = np.concatenate([reference_p - position, matrix_rotvec(reference_r @ rotation.T)])
        return self.method.codec.encode(np.concatenate([self.method.scale.to_native(physical), [self.grip]]))


class SingleArmMethod:
    """single_arm_eef slow capabilities; the fast path's VLA intents pass unchanged."""

    profile = SINGLE_ARM_EEF

    def __init__(self, cfg, codec):
        if codec is None or not callable(getattr(codec, "encode", None)):
            raise ValueError("single-arm primitives need the official OSC conversion")
        limits = cfg["primitives"]
        self.cfg, self.codec = cfg, codec
        self.rows = limits["chunk_rows"]
        self.bounds = np.asarray(limits["workspace_bounds_m"], dtype=float)
        self.scale = OscScale(cfg["osc_controller"])
        self.phase, self.target, self.subgoal = "grasp", "unspecified", None
        # As in actions.Method: every hybrid slow intent answers an escalation (an observation checkpoint)
        # while the fast path never advances the phase, so it may name any phase (the runtime logs it).
        self.any_phase = cfg.get("mode") == "hybrid"
        self.home = None
        # Previous-command semantics: the frozen settle action is the last native command.
        self.gripper_command = float(cfg["native"]["settle_action"][6])

    def seed(self, state):
        if self.home is None:
            self.home = achieved(state)

    def acknowledged(self, action):
        row = np.asarray(action, dtype=float)
        if row.shape != (7,) or not np.isfinite(row).all():
            raise ValueError("LIBERO native action must be finite 7D OSC")
        self.gripper_command = float(row[6])

    def check_intent(self, params):
        if not isinstance(params, dict) or set(REQUIRED) - params.keys():
            raise ValueError("unexpected or missing intent fields")
        mode = params["mode"]
        if mode not in MODES or params["phase"] not in PHASES:
            raise ValueError("unknown phase or capability mode")
        if set(params) != set(REQUIRED) | MODES[mode]:
            raise ValueError("parameters do not match selected capability")
        steps = params["execute_steps"]
        # A vla prefix is one row whatever the count: the released endpoint returns one
        # official action per call, and the subgoal then keeps driving the fast path.
        if type(steps) is not int or not 1 <= steps <= self.rows:
            raise ValueError("invalid execution prefix")
        for key in ("observation_id", "target"):
            if not isinstance(params[key], str) or not 0 < len(params[key]) <= 128:
                raise ValueError("invalid intent identity")
        if not self.any_phase and params["phase"] not in TRANSITIONS[self.phase]:
            raise ValueError("phase transition skips an observation checkpoint")
        if mode == "vla" and (not isinstance(params["subgoal"], str) or not params["subgoal"].strip()
                              or len(params["subgoal"]) > 512):
            raise ValueError("invalid VLA subgoal")

    def primitives(self, params, state):
        self.check_intent(params)
        mode = params["mode"]
        if mode == "vla":
            raise ValueError("VLA inference belongs to the model adapter")
        if self.home is None:
            raise ValueError("home pose has not been observed")
        start = achieved(state)
        goal, grip = start, self.gripper_command
        if mode == "move_relative":
            delta = np.asarray(params["delta_xyz"], dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > MAX_MOVE_M:
                raise ValueError("relative transport must be finite and at most 3 cm")
            goal = (start[0] + delta, start[1])
        elif mode == "gripper":
            command = params["gripper_command"]
            if isinstance(command, bool) or command not in (-1, 1):
                raise ValueError("gripper_command is the native sign: -1 open, +1 close")
            grip = float(command)
        elif mode == "home":
            if self.profile.openness(self.gripper_command) < .8:
                raise ValueError("open the gripper before returning home")
            goal = self.home
        if self.excess(goal[0]) > self.excess(start[0]) + 1e-9:
            raise ValueError("primitive target leaves the frozen workspace bounds")
        plan = Plan(self, start, goal, grip, params["execute_steps"])
        return FollowingRows([plan.next(state)], params["execute_steps"], plan.next)

    def excess(self, position) -> float:
        """Distance outside the home-anchored box (0 inside); a primitive may not grow it."""
        offset = np.asarray(position, dtype=float) - self.home[0]
        below = np.maximum(self.bounds[:, 0] - offset, 0)
        above = np.maximum(offset - self.bounds[:, 1], 0)
        return float(np.linalg.norm(below + above))
