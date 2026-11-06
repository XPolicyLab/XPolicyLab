---
name: orientation_quaternion
description: "Converts a 3x3 rotation matrix to a normalized scalar-first quaternion using trace/diagonal branch selection. The function returns float32 and canonicalizes w to be nonnegative."
---

# Orientation quaternion

## Function

Converts a 3x3 rotation matrix to a normalized scalar-first quaternion using trace/diagonal branch selection. The function returns float32 and canonicalizes w to be nonnegative.

## Inputs

- matrix: array-like reshaped to 3x3; no explicit finite or orthonormality validation in this function

## Outputs

- float32 quaternion (w, x, y, z) normalized with a 1e-12 norm floor

## Callable interface

Implementation: [implementation.py](implementation.py).

- `mat_to_quat_wxyz(matrix)`

## Constraints

- Callers retain original validation and normalization; this function does not read task or object state
- Negates quaternion when w < 0
- Extracted without teacher class, scene access or planner dependency
