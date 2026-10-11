"""pick: point the gripper down, go above a point, descend, close, lift. Five basic commands in one.

If the point is out of reach for a gripper pointing straight down, the tool retries once with the gripper
tilted 45 degrees forward (`down45`), which reaches about 15 cm farther forward (+y)."""
import numpy as np

from roboshell.server.core import BadRequest, tool_rotation

TOOL = {
    "name": "pick",
    "commands": [
        {"name": "pick", "budget": True, "help": "grasp from above at a world point",
         "args": [
             {"name": "arm", "positional": True, "choices": ["left", "right"]},
             {"name": "x", "type": "float", "required": True, "help": "world x of the grasp point (m)"},
             {"name": "y", "type": "float", "required": True, "help": "world y of the grasp point (m)"},
             {"name": "z", "type": "float", "required": True, "help": "world z of the grasp point (m)"},
             {"name": "open", "type": "str", "default": "x", "choices": ["x", "y"], "help": "world axis along which the fingers open"},
             {"name": "approach", "type": "str", "default": "down", "choices": ["down", "down45"], "help": "gripper direction: straight down, or 45 degrees forward-down (reaches farther forward)"},
             {"name": "clearance", "type": "float", "default": 0.10, "help": "height above the grasp point to approach from (m)"},
             {"name": "lift", "type": "float", "default": 0.10, "help": "how far to lift after closing (m)"},
         ]},
    ],
}


def run(api, command, args):
    arm = api.arm(args["arm"])
    stages = []
    goal = np.array([args["x"], args["y"], args["z"]])

    def stage(name, target):
        feedback = {}
        code = api.move_tcp(arm, target.copy(), feedback)
        stages.append({"stage": name, "approach": approach, "plan_ok": feedback.get("plan_ok"), "error_m": feedback.get("error_m")})
        return code, feedback

    def failed(code, feedback):
        return {"stages": stages, "plan_ok": False, "plan_fail_reason": feedback.get("plan_fail_reason"), "plan_detail": feedback.get("plan_detail")}, code

    presets = [args["approach"]] + (["down45"] if args["approach"] == "down" else [])
    for approach in presets:
        # 1. turn in place (a rotation combined with a long move often has no straight-line solution)
        target = arm.tcp()
        target[:3, :3] = tool_rotation(approach, args["open"], target[:3, :3])
        code, feedback = stage("point", target)
        if code != 0 or api.over:
            return failed(code, feedback)
        # 2. above the point
        target[:3, 3] = goal + np.array([0.0, 0.0, args["clearance"]])
        code, feedback = stage("approach", target)
        if code == 0 and not api.over:
            break
        if feedback.get("plan_fail_reason") != "ik_unreachable" or approach == presets[-1] or api.over:
            return failed(code, feedback)
    api.set_gripper(arm, 1.0)
    # 3. descend
    target[:3, 3] = goal
    code, feedback = stage("descend", target)
    if code != 0 or api.over:
        return failed(code, feedback)
    # 4. close
    api.set_gripper(arm, 0.0)
    # 5. lift
    target[:3, 3] = goal + np.array([0.0, 0.0, args["lift"]])
    code, feedback = stage("lift", target)
    reached = arm.tcp()
    final = [s for s in stages if s["approach"] == approach]
    return {"stages": stages, "plan_ok": all(s["plan_ok"] for s in final), "approach": approach,
            "reached_tcp": {"pos": [round(float(v), 4) for v in reached[:3, 3]]}, "gripper": arm.gripper()}, code
