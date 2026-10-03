"""Reviewed code-policy example using only the declared primitive API."""

from PhysicalRSI_core.infra.storage import digest

from .skill_proposal import validate_policy_source


SOURCE = """def policy(robot, memory):
    target = memory.get("target")
    arm = memory.get("arm", "right")
    scene = robot.scene()
    objects = scene.get("objects", {})
    if target not in objects:
        raise ValueError("The requested target is not in the current observation")
    pose = objects[target]["pose"]
    robot.approach(
        arm=arm,
        position_m=pose[:3],
        quaternion_wxyz=pose[3:7],
    )
    robot.gripper(arm=arm, command=1.0)
    return {"target": target, "state": "grasped"}
"""


def describe() -> dict:
    """Return a portable demo artifact; execution still requires a native runtime."""
    validate_policy_source(SOURCE)
    return {
        "schema": "physicalrsi.robodojo.code-policy-demo/v1",
        "source": SOURCE,
        "source_sha256": digest(SOURCE),
        "memory": {"target": "declared_by_task", "arm": "right"},
        "memory_mode": "task-scoped exploration memory",
        "qualification": False,
        "layout_access": False,
        "scope": "reviewed example; native primitive execution required",
    }
