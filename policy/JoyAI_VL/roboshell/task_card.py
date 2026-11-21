"""Assemble evaluator instructions and current-episode robot observations."""


def task_card(first):
    from vendor.robot15.hexashell.motion.chain import slide
    from vendor.robot15.hexashell.motion.driver import display_angles
    from vendor.robot15.hexashell.state import CAMERAS
    from vendor.robot15.hexashell.world import ARMS, GRASP_REACH

    lines = ["# Task card", "", "## Instruction from the evaluator", "", first.instruction]
    if first.step_limit is not None:
        lines += ["", f"Step budget: {first.step_limit}; used: {first.step or 0}."]
    lines += ["", "## Current robot observations", "",
              "Start pose of each arm (robot.home() returns to this pose):"]
    for arm in ARMS:
        reading = first.arms[arm]
        pose = slide(reading.flange, GRASP_REACH)
        pitch, yaw, roll = display_angles(pose[:3, :3])
        xyz = ", ".join(f"{value:.3f}" for value in pose[:3, 3])
        lines.append(f"- {arm}: grasp point ({xyz}), pitch {pitch:.0f} yaw {yaw:.0f} roll {roll:.0f}, "
                     f"gripper {reading.gripper:.2f}")
    cameras = [camera for camera in CAMERAS if camera in first.images]
    lines += ["", f"Initial pictures: {', '.join(cameras)}."]
    return "\n".join(lines) + "\n"


def info():
    return {"variant": "joyai-vl-agent", "instruction_source": "evaluator observation",
            "context": ["instruction", "step budget", "robot state", "initial RGB images"]}
