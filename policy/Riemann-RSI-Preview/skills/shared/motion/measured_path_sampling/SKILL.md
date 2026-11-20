---
name: measured_path_sampling
description: "Resamples a planned six-joint own-arm path by arc-length maximum joint step and minimum control count, then checks a declared phase profile against the remaining action budget. It returns sampled joint positions with endpoint dwell steps or a budget refusal."
---

# Measured path sampling

## Function

Resamples a planned six-joint own-arm path by arc-length maximum joint step and minimum control count, then checks a declared phase profile against the remaining action budget. It returns sampled joint positions with endpoint dwell steps or a budget refusal.

## Inputs

- path array shape (N, 6) with finite values and 1..20000 rows; current 6-vector with finite values
- profile one of contact, carry, free, home, contact_correction, view
- remaining_actions integer in 0..3600 for run
- maximum_joint_step, speed, and minimum_controls for resample, with positive finite step/speed and integer minimum_controls in 1..250

## Outputs

- run returns {'status': 'Success', 'positions': sampled (<=250, 6) array, 'settle_steps': profile dwell} or {'status': 'budget', 'positions': [], 'settle_steps': 0}
- resample returns sampled joint positions with at most 250 controls

## Callable interface

Implementation: [implementation.py](implementation.py).

- `resample(path, current, maximum_joint_step=0.075, speed=1.8, minimum_controls=5)`
- `run(path, current, profile, remaining_actions)`

## Constraints

- Raises ValueError for invalid path/current shapes, nonfinite values, invalid step/speed/minimum_controls, more than 250 controls, or a route exceeding the per-phase action budget.
- A zero-length path repeats the last position minimum_controls times.
- Does not execute actions, acquire observations, change an official clock, inspect an environment, or establish payload contact.
- Profile settings preserve distinct maximum joint steps, speeds, and settle steps, including the original 250-control phase ceiling.
