---
name: public_circular_target_reference
description: "Captures and validates a bounded family of circular targets from delivered RGB, retains it only across qualifying camera-only inspection commands, and checks current image-row compatibility before reuse."
---

# Public circular target reference

## Function

Captures and validates a bounded family of circular targets from delivered RGB, retains it only across qualifying camera-only inspection commands, and checks current image-row compatibility before reuse.

## Inputs

- Current family geometry, head-image target row, delivered image and support reference.
- For reuse, command memory, declared view/home skill names and the reference field.

## Outputs

- Defensively copied reference with target centres/radii, cross-view coverage, image digest and optional partial-row evidence.
- A compatible flag or a reconstructed family from the validated retained reference.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_row_correspondence(evidence, points, pixels, radii, reference, indices, cameras)`
- `validate(value)`
- `capture(family, row, image, support_reference_m)`
- `retained(memory, view_skill, home_skill, reference_field)`
- `compatible(value, row, support_reference_m)`
- `family(value)`

## Constraints

- Requires 2-8 finite targets, bounded radii, positive target height above the reference and independent head/wrist coverage. Partial single-wrist families require tighter current-row correspondence evidence.
- Retention stops across an intervening executed non-view action or a non-inspection home action, and requires a completed public settle phase.
- Compatibility requires support-reference change at most 0.005 m and matched head centres within four pixels; retained geometry is explicitly previous delivered RGB, never hidden state.
