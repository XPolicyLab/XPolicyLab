---
name: declared_x5_numeric_tools
description: "Loads pinned public X5 numeric resource declarations and optionally applies task-declared overrides, then creates fresh cuRobo planner, IK and robot descriptor objects from hashed public URDF, mesh, config and planner files. It writes a declared-x5.yml into a new output directory and returns descriptors without simulator, articulation, object-handle, finger-state or reward data."
---

# Declared x5 numeric tools

## Function

Loads pinned public X5 numeric resource declarations and optionally applies task-declared overrides, then creates fresh cuRobo planner, IK and robot descriptor objects from hashed public URDF, mesh, config and planner files. It writes a declared-x5.yml into a new output directory and returns descriptors without simulator, articulation, object-handle, finger-state or reward data.

## Inputs

- resources: optional dict overriding public_robot_urdf, public_robot_config, public_planner_robot_config, public_robot_parameters, planner_robot_names or public_joint_limits
- output_root: new directory for declared-x5.yml
- arm_joints_only: bool selecting arm-only planners with table_height=-.025

## Outputs

- declarations: manifest dict, optionally with overridden resource fields
- namespace with robot_list, planner mapping, ik_solver mapping and process_name helper

## Callable interface

Implementation: [implementation.py](implementation.py).

- `declarations(resources=None)`
- `create(resources, output_root, *, arm_joints_only=False)`

## Constraints

- output_root is created with exist_ok=False; existing path raises
- Pinned URDF/config/planner hashes and robot mesh hashes must match; unexpected curobo_planner.py path raises ValueError
- Planner device must be cuda:0; dt=.004; non-arm-only planners add a 3x3x.05 m table cuboid at z=-.05
- Joint names fall back to the pinned public YAML when task parameters omit them; task-declared joint-limit precision and arm-only variant are preserved
