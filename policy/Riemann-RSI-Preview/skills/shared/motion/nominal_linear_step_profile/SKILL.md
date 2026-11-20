---
name: nominal_linear_step_profile
description: "Returns a smaller nominal Cartesian displacement request using ramp-up acceleration * dt^2 and a distance-based braking envelope. It does not guarantee actual acceleration or payload-force bounds."
---

# Nominal linear step profile

## Function

Returns a smaller nominal Cartesian displacement request using ramp-up acceleration * dt^2 and a distance-based braking envelope. It does not guarantee actual acceleration or payload-force bounds.

## Inputs

- original_step, previous_step, distance, tolerance, dt, acceleration, cruise_step numeric values
- distance is current public own-pose distance; tolerance is the unchanged caller tolerance

## Outputs

- scalar displacement limit no greater than original_step, cruise_step, previous_step + acceleration * dt^2, braking_step, or distance

## Callable interface

Implementation: [implementation.py](implementation.py).

- `next_step(original_step, previous_step, distance, tolerance, *, dt, acceleration, cruise_step)`

## Constraints

- Raises ValueError unless all values are finite, original_step/tolerance/dt/acceleration/cruise_step are positive, and previous_step/distance are nonnegative.
- Braking uses max(0, distance - 0.5 * tolerance) to keep a half-tolerance margin.
- Does not select trajectories, alter endpoints, emit actions, or change caller stopping, collision planning, joint limits, or action budget.
- Current consumers are store_tools_in_toolbox phases pliers_lift and pliers_transfer; other phases retain their original controller.
