---
name: public_support_geometry_context_observed_start
description: "run applies the same before-contact retention gate, delegates to retained_public_support, and optionally emits a fresh current-geometry record when observed support qualifies. It is the observed-start adapter variant of the support-context logic."
---

# Public support geometry context observed start

## Function

run applies the same before-contact retention gate, delegates to retained_public_support, and optionally emits a fresh current-geometry record when observed support qualifies. It is the observed-start adapter variant of the support-context logic.

## Inputs

- api: restricted PublicRobotAPI from the observed_start_phase_execution adapter, checked by exact type
- previous: optional same-episode support record validated by validate_geometry
- passive_view: explicit bool

## Outputs

- result dict from retained_public_support.run with memory, support and reason fields
- validated current record with height_estimate_m, height_uncertainty_m, camera_alignments, own_base_poses and source CURRENT_SOURCE when fresh observed geometry qualifies
- ExecutionError when api type or passive_view is invalid

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`
- `run(api, previous=None, passive_view=False)`

## Constraints

- Code is adapter-bound: PublicRobotAPI comes from observed_start_phase_execution, while possible_own_closure is imported from public_support_geometry_context
- Retention requires prior source SOURCE and before_contact = passive_view and not possible_own_closure(api)
- Fresh current geometry is tagged current source and reason fresh_public_support_geometry_not_before_contact_permission, never before-contact permission, finger measurement or contact confirmation
- No default height or saved layout is produced; qualified geometry is not holding, occupancy or success
