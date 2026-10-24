"""R0 direct control and R1 fixed skills: the model picks one action per call.

The driver turns each <action> into a fixed program over the same kernel API,
so R0 shares the planner, shield, and traces with the programmable rungs and
differs only in what the model may decide. Supported actions:

  {"type": "move", "arm": "left_arm", "position": [x, y, z] (fingertips),
   "orientation": [qw, qx, qy, qz] of link6 (optional, default: keep), "gripper": g (optional)}
  {"type": "gripper", "arm": "right_arm", "value": g}      0 closed .. 1 open
  {"type": "observe"}                                        new images, no motion
  {"type": "wait", "seconds": s}
With learned skills also:
  {"type": "policy", "name": "simate" (optional: the agent's first policy),
   "instruction": "..." (optional: the task's own), "arms": [...], "seconds": s}
With the fixed skill vocabulary (R1, Tab. 3; the orchestration of Zhang et al.
2026a and Galanti et al. 2026) also skills that ground an object name themselves:
  {"type": "locate", "object": "red cup"}
  {"type": "pick", "object": "red cup", "arm": "left_arm" (optional)}
  {"type": "place", "arm": "left_arm", "target": "blue plate" | "position": [x, y, z],
   "clearance": c (optional)}
pick and place are the session's pick() and place().
Giving up is <finish>.
"""

import json
import math


ARMS = ("left_arm", "right_arm")
MOVE_SECONDS = 10.0
# Default clearance of the place skill (the held object's bottom above the surface).
PLACE_CLEARANCE_M = 0.01


class ActionError(ValueError):
    pass


def _number_list(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ActionError(f"{name} must be a list of {size} numbers")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(item) for item in numbers):
        raise ActionError(f"{name} must be finite")
    return numbers


def _arm(action):
    arm = action.get("arm")
    if arm not in ARMS:
        raise ActionError(f"arm must be one of {list(ARMS)}")
    return arm


def _gripper(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ActionError("gripper must be between 0 (closed) and 1 (open)")
    return value


def _name(action, key):
    value = action.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ActionError(f"{key} must be a short object description")
    return value.strip()


# The skills are fixed programs over the kernel API, so R1 shares perception,
# grasp planning, motion, and shield with the other rungs. A skill that ends with
# an unhandled event stops and reports it like any action.
_LOCATE = """import json
__obs = get_observation()
__found = localize_3d(__obs, {name})
if __found is None:  # the views disagree or only one sees it: head camera on a plane 2 cm above the table
    __flat = localize(__obs, {name}, z=TABLE_Z + 0.02)
    __found = {{"position": __flat[0]["position"], "single_view": True}} if __flat else None
print(json.dumps({{"object": {name}, "position": None if __found is None else [round(v, 4) for v in __found["position"]],
                  "single_view": None if __found is None else __found["single_view"]}}))
"""

_PICK = """import json
print(json.dumps(pick({name}, arm={arm})))
"""

_PLACE = """import json
print(json.dumps(place({target}, {arm}, clearance={clearance})))
"""


def _skill_program(action, kind):
    if kind == "locate":
        return _LOCATE.format(name=json.dumps(_name(action, "object")))
    if kind == "pick":
        arm = action.get("arm")
        if arm is not None:
            _arm(action)
        return _PICK.format(name=json.dumps(_name(action, "object")), arm=repr(arm))
    arm = _arm(action)
    clearance = float(action.get("clearance", PLACE_CLEARANCE_M))
    if not 0.0 <= clearance <= 0.3:
        raise ActionError("clearance must be between 0 and 0.3 m")
    if ("target" in action) == ("position" in action):
        raise ActionError("place needs exactly one of target (an object description) or position")
    target = (_number_list(action["position"], 3, "position") if "position" in action
              else json.dumps(_name(action, "target")))
    return _PLACE.format(target=target, arm=json.dumps(arm), clearance=clearance)


def action_program(text: str, allow_policy: bool = False, allow_skills: bool = False) -> str:
    """Validate an action and return the program that executes it."""
    try:
        action = json.loads(text)
    except json.JSONDecodeError as error:
        raise ActionError(f"action is not valid JSON: {error}") from None
    if not isinstance(action, dict):
        raise ActionError("action must be a JSON object")
    kind = action.get("type")
    report = ("__s = get_state()\n"
              "print(json.dumps({'left_fingertips': [round(v, 4) for v in fingertip_of(__s['left_ee_pose'])], "
              "'left_orientation': [round(v, 4) for v in __s['left_ee_pose'][3:]], "
              "'right_fingertips': [round(v, 4) for v in fingertip_of(__s['right_ee_pose'])], "
              "'right_orientation': [round(v, 4) for v in __s['right_ee_pose'][3:]], "
              "'grippers': [round(__s['joint_position'][6], 3), round(__s['joint_position'][13], 3)]}))\n")
    if kind == "move":
        arm = _arm(action)
        position = _number_list(action.get("position"), 3, "position")
        orientation = action.get("orientation")
        if orientation is not None:
            orientation = _number_list(orientation, 4, "orientation")
        side = arm.split("_")[0]
        # `position` is where the fingertips go; the link6 target follows from the orientation.
        lines = ["import json, math",
                 f"__current = get_state()['{side}_ee_pose']",
                 f"__orientation = {orientation!r} or list(__current[3:])",
                 f"__trace = move_eef('{arm}', link6_for_tip({position!r}, __orientation), max_seconds={MOVE_SECONDS})",
                 f"__reached = fingertip_of(get_state()['{side}_ee_pose'])",
                 f"print('fingertip arrival_error_m', round(math.dist(__reached, {position!r}), 4))"]
        if "gripper" in action:
            value = _gripper(action["gripper"])
            call = "close_gripper" if value == 0.0 else "open_gripper"
            args = f"'{arm}'" if value == 0.0 else f"'{arm}', width={value}"
            lines.append(f"{call}({args})")
        return "\n".join(lines) + "\n" + report
    if kind == "gripper":
        arm = _arm(action)
        value = _gripper(action.get("value"))
        call = f"close_gripper('{arm}')" if value == 0.0 else f"open_gripper('{arm}', width={value})"
        return f"import json\n{call}\n" + report
    if kind == "observe":
        return "import json\n" + report
    if kind == "wait":
        seconds = float(action.get("seconds", 1.0))
        if not 0.0 < seconds <= 10.0:
            raise ActionError("wait seconds must be in (0, 10]")
        return f"import json\nhold({seconds})\n" + report
    if kind == "policy" and allow_policy:
        name = action.get("name")
        instruction = action.get("instruction")
        if (name is not None and (not isinstance(name, str) or not name.strip())) or (
                instruction is not None and (not isinstance(instruction, str) or not instruction.strip())):
            raise ActionError("a policy's name and instruction, when given, must be non-empty")
        arms = action.get("arms", list(ARMS))
        arms = [arms] if isinstance(arms, str) else arms
        if not arms or any(arm not in ARMS for arm in arms):
            raise ActionError(f"arms must be a subset of {list(ARMS)}")
        seconds = float(action.get("seconds", 8.0))
        if not 0.0 < seconds <= 30.0:
            raise ActionError("policy seconds must be in (0, 30]")
        instruction = "None" if instruction is None else json.dumps(instruction)
        # Without a name, the agent's policy (its first): the prompt names none, so R1
        # follows whichever policy the agent configures.
        name = "list_policies()[0]['name']" if name is None else json.dumps(name)
        return (f"import json\nprint(run_policy({name}, {instruction}, "
                f"arms={arms!r}, max_seconds={seconds}, from_anywhere=True))\n" + report)
    if kind in ("locate", "pick", "place") and allow_skills:
        return _skill_program(action, kind) + report
    kinds = ("move, gripper, observe, wait" + (", policy" if allow_policy else "")
             + (", locate, pick, place" if allow_skills else ""))
    raise ActionError(f"type must be one of {kinds}")
