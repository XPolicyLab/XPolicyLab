"""A fixed, non-learned check on every command before it takes effect.

The shield runs in the environment worker, outside the process that executes
model-written code, so programs cannot modify it. Every embodiment's shield has the
same interface (CommandShield): it judges one command against the measured state, and
the command passes unchanged or is rejected with the reasons; it never shortens, clamps
or otherwise alters a command, since a program whose command was quietly changed does
not learn it was wrong. A tick's `shield` field is its ShieldResult.

Shield (below) is the robot's. It sees absolute joint position commands and, when the
worker supplies forward kinematics, the end-effector positions those commands imply. It
catches commands that are bugs, and nothing else. Rejected are:

- a malformed command (wrong length, a value that is not finite);
- a joint outside its limits, or an end effector outside the workspace box,
  further than the robot already is (contact can push a measured joint or hand
  slightly out; holding there or moving back in is not a bug);
- a jump: an arm's joints more than max_joint_jump_rad, or its end effector more
  than max_cartesian_jump_m, from both the previous accepted command and the
  measured state. One control tick of a real motion moves far less; a jump is an
  IK solution on another branch, a target in the wrong frame, a controller that
  diverged. A smooth sequence is never rejected for running ahead of an arm that
  lags behind it, and a command from where the arm is (a hold, the next skill's
  first waypoint, a retreat) is never rejected for where the last one went.

Speed and acceleration are not the shield's: the skills time their own motions,
and a physical robot's controller enforces its own rate limits.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Protocol


PASS = "pass"
REJECTED = "rejected"


@dataclass
class ArmGroup:
    """One arm's indices in the command vector: its joints, and its gripper if it has one."""

    name: str
    joints: list[int]
    gripper: int | None = None


@dataclass
class ShieldProfile:
    """The limits the shield checks for one embodiment: the arm groups, joint bounds,
    jump limits and, optionally, the end-effector workspace box."""

    groups: list[ArmGroup]
    joint_lower: list[float]
    joint_upper: list[float]
    # Largest change of one arm joint target from the previous accepted command, in radians.
    max_joint_jump_rad: float
    # Axis-aligned world-frame box per end effector: {"x": [lo, hi], ...}.
    workspace: dict[str, list[float]] | None = None
    # Largest move of an end effector from the previous accepted command, in meters.
    max_cartesian_jump_m: float | None = None
    # A measured joint resting at a limit reads slightly past it; holding it is not a bug.
    joint_limit_tolerance_rad: float = 1e-3

    @classmethod
    def from_dict(cls, data: dict, joint_lower, joint_upper) -> "ShieldProfile":
        """A validated profile from a worker's shield settings (`groups` by name with their
        joints and gripper, `max_joint_jump_rad`, and optionally `workspace`,
        `max_cartesian_jump_m`, `joint_limit_tolerance_rad`) and the joint bounds."""
        groups = [ArmGroup(name, list(group["joints"]), group.get("gripper"))
                  for name, group in data["groups"].items()]
        size = len(joint_lower)
        if len(joint_upper) != size:
            raise ValueError("Joint bounds must have the same length")
        for group in groups:
            for index in group.joints + ([group.gripper] if group.gripper is not None else []):
                if not 0 <= index < size:
                    raise ValueError(f"Group {group.name} index {index} is outside the command")
        jump = data["max_joint_jump_rad"]
        if not (isinstance(jump, (int, float)) and math.isfinite(jump) and jump > 0):
            raise ValueError("max_joint_jump_rad must be a positive number")
        cartesian = data.get("max_cartesian_jump_m")
        if cartesian is not None and not (isinstance(cartesian, (int, float)) and math.isfinite(cartesian)
                                          and cartesian > 0):
            raise ValueError("max_cartesian_jump_m must be a positive number")
        workspace = data.get("workspace")
        if workspace is not None:
            for axis in "xyz":
                low, high = workspace[axis]
                if not low < high:
                    raise ValueError(f"workspace.{axis} must be [low, high]")
        tolerance = float(data.get("joint_limit_tolerance_rad", 1e-3))
        if not (math.isfinite(tolerance) and tolerance >= 0):
            raise ValueError("joint_limit_tolerance_rad must be a non-negative number")
        return cls(groups, [float(v) for v in joint_lower], [float(v) for v in joint_upper],
                   float(jump), workspace, None if cartesian is None else float(cartesian), tolerance)


@dataclass
class ShieldResult:
    """The verdict on one command: "pass" with the command unchanged, or "rejected" with
    no command and the reason (the failed check and the values it compared). The command
    is the embodiment's: a robot's joint vector."""

    verdict: str
    command: Any
    reasons: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The result as a plain dict, as a tick's `shield` field."""
        return {"verdict": self.verdict, "command": self.command, "reasons": self.reasons}


class CommandShield(Protocol):
    """The shield of one worker, whatever the embodiment: a fixed check of every command
    against the measured state (`current`) and what else the worker knows (`context`, such
    as a robot's forward kinematics), passing it unchanged or rejecting it with reasons."""

    def reset(self) -> None:
        """Forget what earlier commands of the episode established."""

    def check(self, command, current, **context) -> ShieldResult:
        """The verdict on `command`."""


Kinematics = Callable[[list[float]], dict[str, list[float]]]


class Shield:
    """The robot's command check of one worker (a CommandShield). It keeps the last
    accepted command, against which (and the measured state) jumps are judged until
    `reset`."""

    def __init__(self, profile: ShieldProfile):
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        """Forget the last accepted command: the next is judged from the measured state."""
        self._previous = None

    def check(self, command, current, fk: Kinematics | None = None) -> ShieldResult:
        """Judge an absolute joint command against the measured joints `current`: shape,
        finiteness, joint limits and joint jumps, and with `fk` (joints to each group's
        end-effector position) the workspace box and Cartesian jumps. The first failed check
        rejects it; a passed command becomes the previous command."""
        profile = self.profile
        size = len(profile.joint_lower)
        if len(command) != size or len(current) != size:
            return self._reject({"check": "shape", "expected": size, "got": len(command)})
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in command):
            return self._reject({"check": "finite"})
        command = [float(v) for v in command]
        current = [float(v) for v in current]
        tolerance = profile.joint_limit_tolerance_rad
        for index, value in enumerate(command):
            low, high = profile.joint_lower[index], profile.joint_upper[index]
            beyond = max(low - value, value - high)
            if beyond > max(low - current[index], current[index] - high, 0.0) + tolerance:
                return self._reject({"check": "joint_limit", "index": index, "requested": value,
                                     "measured": current[index], "limit": [low, high]})
        previous = self._previous if self._previous is not None else current
        for group in profile.groups:
            jumps = [(abs(command[i] - previous[i]), abs(command[i] - current[i]), i) for i in group.joints]
            if jumps and min(max(j[0] for j in jumps), max(j[1] for j in jumps)) > profile.max_joint_jump_rad:
                last, measured, index = max(jumps)
                return self._reject({"check": "joint_jump", "group": group.name, "index": index,
                                     "from_last_command": last, "from_measured": measured,
                                     "limit": profile.max_joint_jump_rad})
        if fk is not None and (profile.workspace or profile.max_cartesian_jump_m):
            points, measured_points = fk(command), None
            before = fk(previous) if profile.max_cartesian_jump_m else None
            for group in profile.groups:
                point = points[group.name]
                if profile.workspace and self._outside(point) > 0:
                    measured_points = measured_points or fk(current)
                    if self._outside(point) > self._outside(measured_points[group.name]):
                        return self._reject({"check": "workspace", "group": group.name, "position": point,
                                             "measured": measured_points[group.name]})
                if before is not None:
                    last = math.dist(point, before[group.name])
                    if last > profile.max_cartesian_jump_m:
                        measured_points = measured_points or fk(current)
                        measured = math.dist(point, measured_points[group.name])
                        if measured > profile.max_cartesian_jump_m:
                            return self._reject({"check": "cartesian_jump", "group": group.name,
                                                 "from_last_command": last, "from_measured": measured,
                                                 "limit": profile.max_cartesian_jump_m})
        self._previous = command
        return ShieldResult(PASS, list(command))

    def _outside(self, point) -> float:
        """How far a point is outside the workspace box (0 inside), in meters."""
        return math.sqrt(sum(max(low - value, value - high, 0.0) ** 2
                             for value, (low, high) in zip(point, (self.profile.workspace[a] for a in "xyz"))))

    @staticmethod
    def _reject(reason: dict) -> ShieldResult:
        return ShieldResult(REJECTED, None, [reason])
