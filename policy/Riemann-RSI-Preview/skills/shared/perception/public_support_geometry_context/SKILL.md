---
name: public_support_geometry_context
description: "run gates before-contact retention on passive_view and no evidence of an issued closure, delegates to retained_public_support, and optionally emits a fresh current-geometry record when observed support qualifies. possible_own_closure scans command_memory for actions_executed > 0 and a finite last_gripper_command in [0,.9)."
---

# Public support geometry context

## Function

run gates before-contact retention on passive_view and no evidence of an issued closure, delegates to retained_public_support, and optionally emits a fresh current-geometry record when observed support qualifies. possible_own_closure scans command_memory for actions_executed > 0 and a finite last_gripper_command in [0,.9).

## Inputs

- api: restricted PublicRobotAPI from the public_phase_execution adapter, checked by exact type
- previous: optional same-episode support record validated by validate_geometry
- passive_view: explicit bool

## Outputs

- result dict from retained_public_support.run with memory, support and reason fields
- validated current record with height_estimate_m, height_uncertainty_m, camera_alignments, own_base_poses and source CURRENT_SOURCE when fresh observed geometry qualifies
- ExecutionError when api type or passive_view is invalid

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`
- `possible_own_closure(api)`
- `run(api, previous=None, passive_view=False)`

## Constraints

- Code is adapter-bound: PublicRobotAPI and retained_public_support are imported dependencies
- Retention requires prior source SOURCE and before_contact = passive_view and not possible_own_closure(api)
- Fresh current geometry is tagged current source and reason fresh_public_support_geometry_not_before_contact_permission, never before-contact permission, finger measurement or contact confirmation
- No default height or saved layout is produced; qualified geometry is not holding, occupancy or success
