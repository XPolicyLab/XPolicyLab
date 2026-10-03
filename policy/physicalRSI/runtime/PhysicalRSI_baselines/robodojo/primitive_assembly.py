"""Owned dual-arm primitive assembly using observed RGB and supervised planners.

Calibration, perception and service dependencies come from the frozen policy assembly.
No simulator objects or layout files are available through this interface.
"""

import importlib
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest, identifier

from .collision_world import ObservedCollisionScene
from .perception import ObservedObstacles
from .planner_bridge import WorldArmPlanner
from .planner_service import PlannerService, check_factory
from .primitives import RoboDojoPrimitives


def assemble(channel, *, configuration, identity, own, pool=None):
    config = configuration
    if set(config["planners"]) != {"left", "right"}:
        raise ValueError("Both arm planner configurations required")
    if channel.dimensions != config["primitives"]["dimensions"]:
        raise ValueError("Primitive and actor dimensions differ")
    perception = config["perception"]
    for name, sha in perception["files"].items():
        if file_digest(Path(name)) != sha:
            raise ValueError("Perception dependency changed")
    module, name = perception["entrypoint"].split(":")
    factory = getattr(importlib.import_module(module), name)
    check_factory(factory, perception)
    version = perception.get("factory_version", 1)
    if version == 1:
        estimate = factory(configuration=perception["configuration"], own=own)
    elif version == 2:
        estimate = factory(
            configuration=perception["configuration"],
            own=own,
            identity=dict(identity),
            pool=pool,
        )
    else:
        raise ValueError("Unsupported perception factory version")
    obstacles = ObservedObstacles(
        channel=channel, estimate=estimate, max_age_s=perception["max_age_s"]
    )
    root = (
        Path(config["output"])
        / identifier(identity["episode"])
        / identifier(identity["actor"])
    )
    bridges = {}
    for arm in ("left", "right"):
        entry = config["planners"][arm]
        service = own(
            PlannerService(
                entry["spec"],
                python=entry["python"],
                environment=entry["environment"],
                output=root / arm,
                pool=pool,
                resources=entry["resources"],
                startup_timeout_s=entry["startup_timeout_s"],
            )
        )
        service.__enter__()
        bridges[arm] = WorldArmPlanner(service, arm=arm, **entry["calibration"])

    def spheres(*, arm, state, deadline):
        return bridges[arm].collision_spheres(state=state, deadline=deadline)

    scene = ObservedCollisionScene(
        base_poses={arm: bridge.base_pose for arm, bridge in bridges.items()},
        obstacles=obstacles,
        arm_spheres=spheres,
        padding_m=config["padding_m"],
        max_obstacles=config["max_obstacles"],
    )
    for bridge in bridges.values():
        bridge.scene_provider = scene

    def plan(**arguments):
        arm = arguments["arm"]
        if arm not in bridges:
            raise ValueError("Unknown arm planner")
        return bridges[arm](**arguments)

    return RoboDojoPrimitives(
        observe=channel.observe,
        step=channel.step,
        plan=plan,
        perceive=obstacles.observe,
        **config["primitives"],
    )


def resource_requirements(configuration):
    """Sum services held for an actor's lifetime; None means a custom graph."""
    perception = configuration["perception"]
    if (
        perception.get("factory_version", 1) != 2
        or perception["entrypoint"]
        != "PhysicalRSI_baselines.robodojo.perception_factory:create"
    ):
        return None
    entries = [
        *configuration["planners"].values(),
        *perception["configuration"]["services"].values(),
    ]
    total = {}
    for entry in entries:
        for key, amount in entry["resources"].items():
            if type(amount) is not int or amount < 1:
                raise ValueError(
                    "Positive integer service resource requirements required"
                )
            total[key] = total.get(key, 0) + amount
    return total


assemble.resource_requirements = resource_requirements
