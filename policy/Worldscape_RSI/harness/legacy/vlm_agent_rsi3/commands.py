"""Discrete command vocabulary of the policy.

The same grammar as ``harness/core/commands.py``, which drives RoboTwin 2.0,
with 15 degree rotation steps and orientation presets added. The vocabulary is deliberately tiny so that a
multimodal LLM can drive the robot by emitting one short line per action::

    left move z -5        # translate the left fingertips 5 cm down (world frame)
    right rotate yaw 30   # rotate the right gripper 30 deg about world Z, through its fingertips
    left point down       # snap the left gripper to the fingers-down preset
    left gripper close
    right home
    wait
    done

Translation magnitudes are in centimetres and rotation magnitudes in degrees;
both are clipped to :data:`LIMITS` so that a single command can never fling
the arm across the workspace. Standard library only: the policy server parses
replies with it and the simulation client executes the parsed commands.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

ARMS = ("left", "right")
MOVE_AXES = ("x", "y", "z")
ROT_AXES = ("roll", "pitch", "yaw")
POINT_PRESETS = ("down", "down5in", "down15", "down30", "down45", "down60", "down75", "forward", "down30y90")
GRIPPER_STATES = {"open": 1.0, "close": 0.0, "closed": 0.0, "half": 0.5}
KINDS = ("push3d", "sync", "move", "move3d", "pose3d", "rotate", "point", "gripper", "home", "homeplan", "wait", "done")

# Safety limits applied to a single command.
LIMITS = {
    "move_cm": 20.0,
    "rotate_deg": 90.0,
}
# Rotations come in 15 degree steps (15, 30, 45, 60, 75, 90); a request is rounded to the nearest step,
# never to zero.
ROTATE_STEP_DEG = 15.0


def rotation_step(value: float) -> float:
    """The 15 degree step nearest to ``value`` (already clipped), keeping its sign and never returning 0."""
    magnitude = abs(float(value))
    level = max(ROTATE_STEP_DEG, round(magnitude / ROTATE_STEP_DEG) * ROTATE_STEP_DEG)
    return math.copysign(level, value) if value else 0.0


@dataclass
class Command:
    """One discrete command; ``kind`` is one of :data:`KINDS`."""

    kind: str
    arm: Optional[str] = None
    axis: Optional[str] = None
    value: Optional[float] = None
    raw: str = ""
    clipped: bool = False
    rounded: bool = False

    def text(self) -> str:
        if self.kind == "sync":
            return "sync " + " | ".join(Command.from_dict(v).text() for v in self.value)
        if self.kind == "pose3d":
            return f"{self.arm} pose3d " + " ".join(f"{v:+.2f}" for v in self.value) + f" {self.axis}"
        if self.kind in ("move3d", "push3d"):
            return f"{self.arm} {self.kind} " + " ".join(f"{v:+.2f}" for v in self.value)
        if self.kind == "move":
            return f"{self.arm} move {self.axis} {self.value:+.1f}"
        if self.kind == "rotate":
            return f"{self.arm} rotate {self.axis} {self.value:+.1f}"
        if self.kind == "gripper":
            return f"{self.arm} gripper {self.value:.2f}"
        if self.kind in ("home", "homeplan"):
            return f"{self.arm} {self.kind}"
        if self.kind == "point":
            return f"{self.arm} point {self.axis}"
        return self.kind

    def to_dict(self) -> dict:
        """Wire form: the simulation client rebuilds the command from this."""
        return {"kind": self.kind, "arm": self.arm, "axis": self.axis, "value": self.value,
                "raw": self.raw, "clipped": self.clipped, "rounded": self.rounded, "text": self.text()}

    @classmethod
    def from_dict(cls, data: Any) -> "Command":
        if not isinstance(data, dict):
            raise CommandError(f"command must be an object, got {type(data).__name__}")
        kind = data.get("kind")
        if kind not in KINDS:
            raise CommandError(f"unknown command kind {kind!r}")
        if kind == "sync":
            children = [cls.from_dict(v) for v in data.get("value", [])]
            _validate_sync(children)
            return cls("sync", value=[v.to_dict() for v in children], raw=str(data.get("raw", "")))
        arm = data.get("arm")
        if kind in ("wait", "done"):
            arm = None
        elif arm not in ARMS:
            raise CommandError(f"{kind} needs an arm, got {arm!r}")
        axis = data.get("axis")
        value = data.get("value")
        if kind == "pose3d":
            value = _vector(value)
            if axis not in POINT_PRESETS:
                raise CommandError("pose3d requires an orientation preset")
        elif kind in ("move3d", "push3d"):
            value = _vector(value)
            axis = None
        elif kind == "move":
            if axis not in MOVE_AXES:
                raise CommandError(f"move axis must be x, y or z, got {axis!r}")
            value = _finite(value, LIMITS["move_cm"])
        elif kind == "rotate":
            if axis not in ROT_AXES:
                raise CommandError(f"rotate axis must be roll, pitch or yaw, got {axis!r}")
            value = _finite(value, LIMITS["rotate_deg"])
        elif kind == "point":
            if axis not in POINT_PRESETS:
                raise CommandError(f"point preset must be one of {POINT_PRESETS}, got {axis!r}")
            value = None
        elif kind == "gripper":
            value = _finite(value, 1.0)
            if value < 0.0:
                raise CommandError("gripper value must be in [0, 1]")
            axis = None
        else:
            axis, value = None, None
        return cls(kind, arm, axis, value, raw=str(data.get("raw", "")), clipped=bool(data.get("clipped", False)),
                   rounded=bool(data.get("rounded", False)))


def _finite(value: Any, limit: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CommandError(f"command value must be a number, got {value!r}") from None
    if isinstance(value, bool) or not math.isfinite(number) or abs(number) > limit + 1e-9:
        raise CommandError(f"command value {value!r} is not a finite number within the limit {limit}")
    return number


class CommandError(ValueError):
    pass


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CommandError("move3d requires three finite centimetre offsets")
    result = [_finite(v, 40.0) for v in value]
    if math.sqrt(sum(v*v for v in result)) > 40.0 + 1e-9:
        raise CommandError("move3d displacement must be <=40cm; split longer routes")
    return result


_NUM = r"([-+]?\d+(?:\.\d+)?)"
_RE_MOVE = re.compile(rf"^(left|right)\s+move\s+([xyz])\s*{_NUM}\s*(cm)?$", re.I)
_RE_ROT = re.compile(rf"^(left|right)\s+rotate\s+(roll|pitch|yaw)\s*{_NUM}\s*(deg)?$", re.I)
_RE_GRIP = re.compile(rf"^(left|right)\s+gripper\s+(open|close|closed|half|{_NUM})$", re.I)
_RE_HOME = re.compile(r"^(left|right)\s+home$", re.I)
_RE_POINT = re.compile(r"^(left|right)\s+point\s+(down(?:5in|15|30|45|60|75)?|down30y90|forward)$", re.I)
_RE_WAIT = re.compile(r"^wait$", re.I)
_RE_DONE = re.compile(r"^done$", re.I)


def parse_command(line: str) -> Command:
    """Parse one command line. Raises :class:`CommandError` on unknown syntax."""
    text = line.strip().strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    # tolerate trailing comments
    text = text.split("#", 1)[0].strip().rstrip(".").strip()
    if not text:
        raise CommandError("empty command")

    if text.lower().startswith("sync "):
        children = [parse_command(v) for v in text[5:].split("|")]
        _validate_sync(children)
        return Command("sync", value=[v.to_dict() for v in children], raw=line)
    parts = text.lower().split()
    if len(parts) >= 2 and parts[1] == "pose3d":
        if len(parts) != 6 or parts[0] not in ARMS or parts[5] not in POINT_PRESETS:
            raise CommandError("syntax: <arm> pose3d <dx_cm> <dy_cm> <dz_cm> <preset>")
        return Command("pose3d", parts[0], parts[5], _vector(parts[2:5]), raw=line)
    if len(parts) == 2 and parts[0] in ARMS and parts[1] == "homeplan":
        return Command("homeplan", parts[0], raw=line)
    if len(parts) >= 2 and parts[1] in ("move3d", "push3d"):
        if len(parts) != 5 or parts[0] not in ARMS:
            raise CommandError("syntax: <left|right> move3d <dx_cm> <dy_cm> <dz_cm>")
        return Command(parts[1], parts[0], value=_vector(parts[2:]), raw=line)

    m = _RE_MOVE.match(text)
    if m:
        arm, axis, value = m.group(1).lower(), m.group(2).lower(), float(m.group(3))
        clipped = abs(value) > LIMITS["move_cm"]
        value = max(-LIMITS["move_cm"], min(LIMITS["move_cm"], value))
        return Command("move", arm, axis, value, raw=line, clipped=clipped)

    m = _RE_ROT.match(text)
    if m:
        arm, axis, value = m.group(1).lower(), m.group(2).lower(), float(m.group(3))
        clipped = abs(value) > LIMITS["rotate_deg"]
        value = max(-LIMITS["rotate_deg"], min(LIMITS["rotate_deg"], value))
        return Command("rotate", arm, axis, value, raw=line, clipped=clipped)

    m = _RE_GRIP.match(text)
    if m:
        arm, state = m.group(1).lower(), m.group(2).lower()
        if state in GRIPPER_STATES:
            value = GRIPPER_STATES[state]
        else:
            value = max(0.0, min(1.0, float(state)))
        return Command("gripper", arm, None, value, raw=line)

    m = _RE_HOME.match(text)
    if m:
        return Command("home", m.group(1).lower(), raw=line)

    m = _RE_POINT.match(text)
    if m:
        return Command("point", m.group(1).lower(), m.group(2).lower(), raw=line)

    if _RE_WAIT.match(text):
        return Command("wait", raw=line)
    if _RE_DONE.match(text):
        return Command("done", raw=line)

    raise CommandError(f"unrecognised command: {line!r}")


def step_rotation(command: Command) -> Command:
    """The model's rotations come in 15 degree steps: round a rotate command to the nearest one."""
    if command.kind != "rotate" or command.value is None:
        return command
    stepped = rotation_step(command.value)
    if stepped == command.value:
        return command
    return Command(command.kind, command.arm, command.axis, stepped, raw=command.raw, clipped=command.clipped,
                   rounded=True)


def parse_command_list(lines: Any) -> tuple[list[Command], list[str]]:
    """Parse the model's command lines; returns ``(commands, errors)``. Rotations are rounded to 15 deg steps."""
    commands: list[Command] = []
    errors: list[str] = []
    if isinstance(lines, str):
        lines = [lines]
    for line in lines or []:
        if not isinstance(line, str):
            errors.append(f"command is not a string: {line!r}")
            continue
        try:
            commands.append(step_rotation(parse_command(line)))
        except CommandError as exc:
            errors.append(str(exc))
    return commands, errors


GRAMMAR_HELP = """\
Command grammar (one command per string, case-insensitive):
  Additional preset down30y90: exactly down30 followed by world yaw +90, combined with pose3d into ONE planned target; jaw axis along -Y. Same planner and timing.
  <arm> pose3d <dx> <dy> <dz> <preset>  one cuRobo motion combining WORLD TCP displacement (cm,norm<=40) and orientation preset. Use for empty-gripper approaches; changes orientation and resets yaw.
  <arm> homeplan               use cuRobo to plan back to the recorded initial wrist pose. No unplanned fallback; inspect failures and choose a clear intermediate pose if necessary.
  <arm> move3d <dx> <dy> <dz>   single planned WORLD displacement in cm, preserving orientation; vector length <=40cm.
                                Uses the same cuRobo table/self-collision checks and settling. Tabletop objects are NOT collision models; choose a clear route.
  <arm> move <axis> <cm>        translate that gripper's FINGERTIP CENTRE along a WORLD axis, keeping its orientation.
                                <arm> is left|right, <axis> is x|y|z, <cm> is a signed number, |cm| <= 20.
  <arm> rotate <axis> <deg>     rotate that gripper about a WORLD axis through its own fingertip centre.
                                <axis> is roll (about x) | pitch (about y) | yaw (about z); <deg> is a signed
                                multiple of 15 up to 90 (15, 30, 45, 60, 75, 90; other values are rounded to
                                the nearest step).
  <arm> point down|down15|down30|down45|down60|down75|forward
                                snap the gripper to a preset orientation: fingers pointing straight down,
                                tilted towards +y by 15/30/45/60/75 deg, or straight forward (+y). Use
                                "rotate yaw" afterwards to spin the finger-closing axis.
  Think about whether the current grasp or placement needs a particular angle, and choose the rotation
  from what you observe.
  <arm> gripper open|close      open or close the fingers of that arm (also accepts a number 0..1, 1 = open).
  <arm> home                    send that arm back to its initial pose. If the planner finds no path the arm
                                retreats along a direct joint path instead, so home is the way out of a stuck pose.
  wait                          let physics settle for a moment without moving.
  done                          declare the task complete (or impossible) and stop.
"""


def _validate_sync(children):
    if len(children) != 2 or {v.arm for v in children} != {"left", "right"}:
        raise CommandError("sync needs exactly one left and one right command")
    allowed = {"push3d", "move", "move3d", "pose3d", "point", "rotate", "homeplan", "gripper"}
    if any(v.kind not in allowed for v in children):
        raise CommandError("unsupported synchronized command")
    if any(v.kind == "gripper" for v in children) and not all(v.kind == "gripper" for v in children):
        raise CommandError("sync grippers only with grippers; no mixed grasp and motion")
