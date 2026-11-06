---
name: normalized_gripper_ramp
description: "Computes dense normalized gripper command sequences with linear interpolation over the first 90% of the requested steps and a final hold; optionally resamples the dense sequence at the declared observation interval with fractional linear replay."
---

# Normalized gripper ramp

## Function

Computes dense normalized gripper command sequences with linear interpolation over the first 90% of the requested steps and a final hold; optionally resamples the dense sequence at the declared observation interval with fractional linear replay.

## Inputs

- start: finite normalized public gripper command in [0,1]
- target: finite normalized command value, clipped to [0,1]
- steps: integer command budget from 1 to 6400; unchanged: boolean forcing a constant target
- interval: integer 1 to 40; time_scale: finite 0.1 to 4 applied only when includes_arm_targets is true; includes_arm_targets: boolean

## Outputs

- dense_commands returns a new numpy float array of length steps
- sampled_commands returns a new numpy float array of ceil(len(dense)/stride) sampled command values, where stride is interval or interval*time_scale

## Callable interface

Implementation: [implementation.py](implementation.py).

- `dense_commands(start, target, steps, unchanged=False)`
- `sampled_commands(start, target, steps, unchanged=False, interval=10, time_scale=1.0, includes_arm_targets=False)`

## Constraints

- Raises ValueError for invalid steps, unchanged, interval, includes_arm_targets, time_scale, non-finite start/target, or start outside [0,1]; target is clipped rather than rejected.
- unchanged=True or interpolation_steps <= 1 returns the target repeated for all steps; otherwise interpolation_steps is int(steps*0.9).
- Pure own-command clock: no environment or object information is read, and normalized values are command responses, not physical finger-joint measurements, width, or contact.
- Returns command values only; it does not return actions or claim success, and a registered public tool supplies actual replay and receipts.
