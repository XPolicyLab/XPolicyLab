---
name: sampled_gripper_dwell
description: "commands samples a bounded dense normalized own-gripper command interpolation on the native action clock. It preserves the original official bridge rule for gripper-only versus arm-target sequences."
---

# Sampled gripper dwell

## Function

commands samples a bounded dense normalized own-gripper command interpolation on the native action clock. It preserves the original official bridge rule for gripper-only versus arm-target sequences.

## Inputs

- start and target: finite normalized own commands in [0, 1]
- dense_steps: integer in [1, 400]
- interval: integer in [1, 40]
- time_scale: finite float in [.1, 4]; includes_arm_targets: bool

## Outputs

- numpy array of sampled normalized command values starting from start and ending at target interpolation fractions
- sample count max(1, ceil(dense_steps / stride)), where stride is interval*time_scale when includes_arm_targets is true and interval otherwise

## Callable interface

Implementation: [implementation.py](implementation.py).

- `commands(start, target, dense_steps, interval=10, time_scale=1.0, includes_arm_targets=False)`

## Constraints

- raises ValueError for non-integer dense_steps/interval, non-bool includes_arm_targets, non-finite start/target/time_scale, or values outside the declared ranges
- pure function reads no environment or object and cannot change the official action budget
- returned values are own normalized commands, not true finger positions, force, contact, or success
