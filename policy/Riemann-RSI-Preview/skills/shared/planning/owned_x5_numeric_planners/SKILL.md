---
name: owned_x5_numeric_planners
description: "Verifies hash-pinned official runtime and public X5 static resources, then constructs fresh CuRobo planners from the declared public URDF/config and writes a generated planner YAML under output_root. It returns public robot descriptions, planner mapping, and joint limits without accepting a simulator environment or scene/task handle."
---

# Owned x5 numeric planners

## Function

Verifies hash-pinned official runtime and public X5 static resources, then constructs fresh CuRobo planners from the declared public URDF/config and writes a generated planner YAML under output_root. It returns public robot descriptions, planner mapping, and joint limits without accepting a simulator environment or scene/task handle.

## Inputs

- resources: dict with official_runtime {path, commit, files}, public_robot_urdf, public_robot_usd, public_robot_config, public_planner_robot_config, public_robot_meshes, public_robot_usd_layers, public_camera, public_robot_parameters, planner_robot_names, public_joint_limits
- output_root: filesystem path for owned-x5-public-planner.yml
- public_robot_parameters: arm_joint_names, finger_joint_names, gripper_scale, sign, mimic
- planner_robot_names: mapping of arm names to planner robot names

## Outputs

- verify_static_resources(resources): the same resources dict after pinned-resource checks
- create(resources, output_root): tuple (descriptions, planners, limits), with PublicRobotDescription objects and CuroboPlanner objects keyed by robot name
- writes owned-x5-public-planner.yml with resolved asset_root_path and urdf_path

## Callable interface

Implementation: [implementation.py](implementation.py).

- `verify_static_resources(resources)`
- `create(resources, output_root)`

## Constraints

- Requires Path(sys.prefix) == configured OFFICIAL_PREFIX; official_runtime.path must equal OFFICIAL_ROOT and commit must equal OFFICIAL_COMMIT, checked against git rev-parse HEAD
- Rejects symlinked or SHA256-mismatched official files, robot URDF/USD/config/planner config, meshes, USD layers, and camera files
- Imports env.planner_manager.curobo_planner only if its resolved file is OFFICIAL_ROOT/env/planner_manager/curobo_planner.py; constructs planners with dt=0.004, table_height=0.0, zero origin, and a 3x3x0.05 m table cuboid
- Planner device_cfg.device must be cuda:0; parent must bind an owned cuda:0 lease; no simulator RobotManager, Task, scene, articulation, reward, or layout inputs
