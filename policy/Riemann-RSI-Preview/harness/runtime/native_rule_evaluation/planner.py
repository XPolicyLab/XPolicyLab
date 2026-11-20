"""Evaluator-owned pinned cuRobo resources; only declared numeric tools escape."""
from __future__ import annotations
from typing import Any
import numpy as np
PLANNER_TABLE_CLEARANCE_BELOW_X5_BASE = 0.025
PLANNER_TABLE_THICKNESS = 0.05

def configure_planner_table(env: Any) -> dict[str, Any]:
    """Remove the core planner's legacy absolute-world table dependency.

    RobotManager constructs CuRobo with a legacy absolute-world table
    expression.  That happens before this task-specific teacher is created and
    silently assumes a fixed world-frame table height.  Replace the resulting
    scene in every CuRobo solver with the equivalent X5-base-relative collision
    plane.
    """
    from curobo._src.geom.types import Cuboid, SceneCfg
    collision_top_z = -PLANNER_TABLE_CLEARANCE_BELOW_X5_BASE
    collision_center_z = collision_top_z - 0.5 * PLANNER_TABLE_THICKNESS

    def table_scene() -> SceneCfg:
        return SceneCfg(cuboid=[Cuboid(name='table', pose=[0.0, 0.0, collision_center_z, 1.0, 0.0, 0.0, 0.0], dims=[3.0, 3.0, PLANNER_TABLE_THICKNESS])])
    configured = []
    seen = set()
    for planner in env.robot_manager.planner.values():
        if id(planner) in seen:
            continue
        seen.add(id(planner))
        for solver_name in ('motion_planner', 'motion_planner_batch', 'ik_solver'):
            getattr(planner, solver_name).update_world(table_scene())
        configured.append(planner.yml_path)
    return {'source': 'fixed_x5_base_relative_mount_geometry', 'collision_top_relative_to_x5_base_m': collision_top_z, 'collision_table_thickness_m': PLANNER_TABLE_THICKNESS, 'configured_planner_count': len(configured), 'planner_configs': configured}
