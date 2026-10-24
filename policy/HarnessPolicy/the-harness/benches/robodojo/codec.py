"""Pure RoboDojo action and observation conversion.

This module must stay importable without Isaac, torch, or a GPU. The worker
adapter uses it at the process boundary.
"""

import math

from harness_policy.core.codec import EpisodeCursor, data_url, encode_png, finite_bounds, validate_position


ACTION_NAME = "joint_position"
SUPPORTED_TASK = "stack_bowls"
ENV_CONFIG = "arx_x5"
ROBOT_CONFIG = "dual_x5"
STACK = "robodojo-isaac51"
LAYOUT_PACK_SEED = 0
ARM_DIMS = (6, 6)
GRIPPER_DIMS = (1, 1)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
DEFAULT_INSTRUCTION = "Stack the three bowls together."
GRIPPER_COMMAND_BOUNDS = (0.0, 1.0)
OPEN_GRIPPER = 1.0
SUCCESS_THRESHOLD = 1.0 - 1e-3


def policy_observation(obs_manager, env_idx_list=(0,)):
    """The frames the official client reads for this RoboDojo checkout.

    Commit 82e1ba3 renders once more before the read, fixing a one-frame lag.
    Simate was trained on the earlier observation. The benchmark's 2026-09-22
    note puts that fix within 1% success, so a checkout from before 82e1ba3 has
    no render_for_capture and the read stays the one the checkpoint learned.
    """
    render = getattr(obs_manager, "render_for_capture", None)
    if render is not None:
        render()
    return obs_manager.get_obs(env_idx_list=list(env_idx_list))

__all__ = [
    "ACTION_NAME",
    "ARM_DIMS",
    "CAMERAS",
    "DEFAULT_INSTRUCTION",
    "ENV_CONFIG",
    "EpisodeCursor",
    "GRIPPER_COMMAND_BOUNDS",
    "GRIPPER_DIMS",
    "LAYOUT_PACK_SEED",
    "OPEN_GRIPPER",
    "ROBOT_CONFIG",
    "STACK",
    "SUCCESS_THRESHOLD",
    "SUPPORTED_TASK",
    "action_dict",
    "clamp_position",
    "data_url",
    "encode_png",
    "episode_outcome",
    "finite_bounds",
    "hold_position",
    "instruction_text",
    "nudge_position",
    "policy_observation",
    "tool_schema",
    "validate_position",
]


def hold_position(action_terms, joint_position, gripper_open=OPEN_GRIPPER):
    """Keep the current arm targets and command each gripper to ``gripper_open``.

    ``joint_position`` is aligned with ``action_terms``. Arm names can repeat
    across the two X5 arms, so components are addressed by index.
    """
    opening = _gripper_command(gripper_open)
    command = []
    index = 0
    for term in action_terms:
        size = _term_size(term)
        chunk = _slice(joint_position, index, size, term["term"])
        if term["command"] == "gripper_normalized":
            command.extend([opening] * size)
        elif term["command"] == "absolute_joint":
            command.extend(_finite_numbers(chunk, f"{term['term']} joint position"))
        else:
            raise ValueError(f"Unknown action command {term['command']!r}")
        index += size
    if index != len(joint_position):
        raise ValueError("Joint position length does not match the action layout")
    return command


def nudge_position(
    action_terms,
    joint_position,
    low,
    high,
    step,
    delta_radians,
    gripper_open=OPEN_GRIPPER,
):
    """Move one arm joint a small amount and command every gripper open.

    Arm component ``step % arm_count`` receives ``+delta_radians`` on even steps
    and ``-delta_radians`` on odd steps. The result is clamped to the command
    bounds. Gripper command ``1`` is open for the X5 normalization.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if type(delta_radians) is bool or type(delta_radians) not in (int, float):
        raise ValueError("delta_radians must be a positive finite number")
    if not math.isfinite(delta_radians) or delta_radians <= 0:
        raise ValueError("delta_radians must be a positive finite number")
    command = hold_position(action_terms, joint_position, gripper_open=gripper_open)
    bounds = finite_bounds(low, high)
    if len(command) != len(bounds):
        raise ValueError("Action layout does not match the action bounds")
    arm_indexes = [
        index
        for index, term in _indexed_terms(action_terms)
        if term["command"] == "absolute_joint"
    ]
    if not arm_indexes:
        raise ValueError("Action has no arm joints to nudge")
    chosen = arm_indexes[step % len(arm_indexes)]
    direction = 1.0 if step % 2 == 0 else -1.0
    lower, upper = bounds[chosen]
    command[chosen] = min(upper, max(lower, command[chosen] + direction * float(delta_radians)))
    return clamp_position(command, bounds)


def clamp_position(values, bounds):
    """Clamp a command onto finite bounds and validate it."""
    if not isinstance(values, list) or len(values) != len(bounds):
        raise ValueError(f"position must contain {len(bounds)} finite numbers")
    clamped = []
    for value, (lower, upper) in zip(values, bounds, strict=True):
        if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("position values must be finite numbers")
        clamped.append(min(upper, max(lower, float(value))))
    return validate_position({"position": clamped}, bounds)


def action_dict(action_terms, command):
    """Pack a flat command into the native per-arm action dictionary."""
    payload = {}
    index = 0
    for term in action_terms:
        size = _term_size(term)
        key = term.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Action term {term.get('term')!r} has no native key")
        if key in payload:
            raise ValueError(f"Duplicate action key {key}")
        payload[key] = _finite_numbers(_slice(command, index, size, term["term"]), key)
        index += size
    if index != len(command):
        raise ValueError("Command length does not match the action layout")
    return payload


def tool_schema(bounds, layout):
    count = len(bounds)
    lines = [
        "Absolute dual-X5 joint-position command for one RoboDojo policy step.",
        "position has one finite number per action component, in tool order.",
        "Arm components are joint position targets in radians.",
        "Gripper components are normalized openings in [0, 1].",
        "For the X5, 0 maps to the gripper scale minimum and 1 maps to the maximum;",
        "the task counts a normalized opening at or above 0.8 as open.",
        "One command is one policy step, which the simulator interpolates.",
        "Components:",
    ]
    index = 0
    for item in layout:
        joints = item["joints"]
        if len(joints) != item["size"]:
            raise ValueError(f"Action term {item['term']!r} does not match its joint names")
        for joint in joints:
            lower, upper = bounds[index]
            lines.append(
                f"{index}: {item['term']} {joint} {item['command']} key {item['key']} "
                f"in [{lower:.6g}, {upper:.6g}]"
            )
            index += 1
    if index != count:
        raise ValueError("Action layout does not match the action bounds")
    return [{
        "type": "function",
        "name": ACTION_NAME,
        "description": " ".join(lines),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": count,
                    "maxItems": count,
                },
            },
            "required": ["position"],
            "additionalProperties": False,
        },
    }]


def episode_outcome(ended, native_success, process_score, step_count, step_limit):
    """Map RoboDojo's terminal flags and process score into harness fields.

    ``native_success`` starts true and only means the task succeeded after the
    episode has ended. A process score of 0 is a recorded result, not success.
    Official success is the reward flag ``reward > 1 - 1e-3``. The returned
    reward is 1 on that flag and otherwise ``process_score / 100``, matching
    the score RoboDojo stores for an unsuccessful episode. Reaching
    ``step_limit`` without that flag is truncation.
    """
    if type(ended) is not bool or type(native_success) is not bool:
        raise ValueError("Episode flags must be booleans")
    if type(step_count) is not int or isinstance(step_count, bool) or step_count < 0:
        raise ValueError("step_count must be a nonnegative integer")
    if type(step_limit) is not int or isinstance(step_limit, bool) or step_limit < 1:
        raise ValueError("step_limit must be a positive integer")
    if type(process_score) is bool or type(process_score) not in (int, float) or not math.isfinite(process_score):
        raise ValueError("process_score must be a finite number")
    success = ended and native_success
    reward = 1.0 if success else float(process_score) / 100.0
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    terminated = success
    truncated = (not success) and (ended or step_count >= step_limit)
    return reward, success, terminated, truncated


def instruction_text(value, default=DEFAULT_INSTRUCTION):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if not isinstance(default, str) or not default.strip():
        raise ValueError("Instruction is empty")
    return default


def _gripper_command(value):
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("gripper_open must be a finite number in [0, 1]")
    number = float(value)
    lower, upper = GRIPPER_COMMAND_BOUNDS
    if number < lower or number > upper:
        raise ValueError("gripper_open must be a finite number in [0, 1]")
    return number


def _term_size(term):
    size = term.get("size")
    if type(size) is not int or isinstance(size, bool) or size < 1:
        raise ValueError(f"Action term {term.get('term')!r} has an invalid size")
    return size


def _slice(values, index, size, label):
    if not isinstance(values, list) or index + size > len(values):
        raise ValueError(f"{label} command is shorter than the action layout")
    return list(values[index:index + size])


def _finite_numbers(values, label):
    numbers = []
    for value in values:
        if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{label} values must be finite numbers")
        numbers.append(float(value))
    return numbers


def _indexed_terms(action_terms):
    index = 0
    for term in action_terms:
        for _joint in range(_term_size(term)):
            yield index, term
            index += 1
