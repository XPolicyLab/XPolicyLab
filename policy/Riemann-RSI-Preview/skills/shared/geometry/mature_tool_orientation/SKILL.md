---
name: mature_tool_orientation
description: "Builds an orthonormal downward tool orientation from a supplied 2D horizontal direction and tilt, then converts it to a scalar-first quaternion with mat_to_quat_wxyz. The result is an own-robot command orientation; it does not obtain object orientation, size, layout or contact measurement."
---

# Mature tool orientation

## Function

Builds an orthonormal downward tool orientation from a supplied 2D horizontal direction and tilt, then converts it to a scalar-first quaternion with mat_to_quat_wxyz. The result is an own-robot command orientation; it does not obtain object orientation, size, layout or contact measurement.

## Inputs

- horizontal_direction: finite 2-element vector with norm at least 1e-8
- tilt_deg: tilt angle in degrees, default 25.0; float conversion is applied without an explicit finite check

## Outputs

- scalar-first quaternion (w, x, y, z) as float32 from mat_to_quat_wxyz

## Callable interface

Implementation: [implementation.py](implementation.py).

- `downward_tool_x(horizontal_direction, tilt_deg=25.0)`

## Constraints

- Raises ValueError when horizontal_direction is not a finite 2-vector or has norm below 1e-8
- Projects world Z against the tool axis and normalizes with a 1e-12 floor before forming the rotation columns
- Extracted without teacher class, scene access or planner dependency
