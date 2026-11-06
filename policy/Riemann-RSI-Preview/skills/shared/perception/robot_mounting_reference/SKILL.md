---
name: robot_mounting_reference
description: "Validates exactly two own-robot FK-derived 7-element bases, requires upright rotations and z agreement within 0.005 m, and returns the mean observed mounting height or explicit uncertainty."
---

# Robot mounting reference

## Function

Validates exactly two own-robot FK-derived 7-element bases, requires upright rotations and z agreement within 0.005 m, and returns the mean observed mounting height or explicit uncertainty.

## Inputs

- bases: dict with exactly keys 'left' and 'right'; each value is a 7-element own-robot FK-derived pose (position and rotation)
- pose[2] is used as height in metres for each side

## Outputs

- {'status':'observed','height_m': mean z in metres,'source':'current_own_robot_pose_and_declared_URDF','semantics':'robot_mounting_reference_not_an_object_or_table_measurement','object_support_requires_task_contract':true}
- {'status':'uncertain','height_m':null,'reason':'independent_own_mounts_disagree_or_are_tilted'}

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(bases)`

## Constraints

- Raises ValueError unless bases is a dict with exactly left/right, each pose has shape (7,), all values are finite, and max absolute position component is <= 3.
- Returns uncertain when height range exceeds 0.005 m or any rotation_matrix(pose[3:])[2,2] < 0.999.
- Not a measured table height, object support plane, or substitute for nominal/scene state; support use requires a declared task contract.
