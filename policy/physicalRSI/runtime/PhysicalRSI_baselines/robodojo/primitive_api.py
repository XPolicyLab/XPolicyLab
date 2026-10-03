"""Policy-visible API derived from the frozen native primitive assembly."""

import inspect
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest

from .primitives import RoboDojoPrimitives

SCHEMA = "physicalrsi.robodojo.primitive-api/v1"
ASSEMBLER = "PhysicalRSI_baselines.robodojo.primitive_assembly:assemble"


def describe(spec):
    if spec["entrypoint"] != ASSEMBLER:
        raise ValueError("API description requires the native primitive assembler")
    if spec["schema"] == "physicalrsi.robodojo.code-policy/v1":
        configurations = {"*": spec["configuration"]}
    elif spec["schema"] == "physicalrsi.robodojo.code-policy/v2":
        configurations = spec["task_configurations"]
    else:
        raise ValueError("Unsupported policy assembly specification")
    tasks = {}
    for task, configuration in configurations.items():
        controls = configuration["primitives"]
        perception = configuration["perception"]
        if (
            perception["entrypoint"]
            != "PhysicalRSI_baselines.robodojo.perception_factory:create"
        ):
            raise ValueError("Describe a custom perception API explicitly")
        targets = perception["configuration"]["services"]["targets"]["spec"]
        tasks[task] = dict(
            scene_object_names=deepcopy(targets["queries"]),
            dimensions=deepcopy(controls["dimensions"]),
            joint_limits=deepcopy(controls["joint_limits"]),
            max_joint_step=controls["max_joint_step"],
            timeout_s=controls.get("timeout_s", 30),
            max_waypoints=controls.get("max_waypoints", 500),
            gripper_steps=controls.get("gripper_steps", 5),
        )
    methods = {}
    descriptions = {
        "scene": "Return world-frame metre cuboids in objects, with observation identity. Estimates from RGB; not hidden geometry or a success signal. Unresolved targets raise an error.",
        "reobserve": "Return current joint state. Arm coordinates are radians; gripper commands are normalized.",
        "approach": "Plan and execute a bounded path for left or right arm to a world-frame position_m and unit quaternion_wxyz; preserve the other arm and grippers. May fail due to perception, planning or execution limits.",
        "gripper": "Execute normalized command in [0,1] on left or right gripper; preserve other joints. Completion does not prove a grasp.",
    }
    for name, description in descriptions.items():
        signature = inspect.signature(getattr(RoboDojoPrimitives, name))
        signature = signature.replace(
            parameters=[p for p in signature.parameters.values() if p.name != "self"]
        )
        methods[name] = dict(
            signature="robot." + name + str(signature),
            description=description,
            returns="scene: frame, unit, objects{name:{pose[x,y,z,w,x,y,z],dims[x,y,z]}}, observation"
            if name == "scene"
            else "{state: joint arrays}; no action_steps field is exposed",
        )
    result = dict(
        schema=SCHEMA,
        policy_spec_sha256=digest(spec),
        implementation_sha256=file_digest(Path(__file__)),
        methods=methods,
        task_configurations=tasks,
        shared_configuration=spec["schema"] == "physicalrsi.robodojo.code-policy/v1",
        policy_signature="def policy(robot, memory)",
        memory="JSON lessons from development; never evaluation layouts or answers",
        failure="Do not automatically retry an uncertain physical action; propagate the failure for reconciliation.",
    )
    # Keep the old digest only on old manifests.  New proposal artifacts and
    # task integrations expose the neutral policy-assembly name exclusively.
    if spec["schema"].startswith("physicalrsi.robodojo.code-policy/"):
        result["policy_spec_sha256"] = digest(spec)
    return result


def for_task(api, task):
    """Only expose this task's configuration to a proposal; preserve custom APIs."""
    if api.get("schema") != SCHEMA:
        return deepcopy(api)
    result = deepcopy(api)
    configurations = result.pop("task_configurations")
    key = "*" if result.pop("shared_configuration") else task
    if key not in configurations:
        raise ValueError("No primitive API configuration for task: " + task)
    result.update(task=task, configuration=configurations[key])
    return result
