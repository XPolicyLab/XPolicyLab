---
name: finite_rotation_pose
description: "Copies and validates finite numeric 7D XYZ/scalar-first-wxyz poses and normalizes scalar-first quaternions. Normalization rejects zero, NaN, or Inf quaternions and chooses a deterministic sign representative."
---

# Finite rotation pose

## Function

Copies and validates finite numeric 7D XYZ/scalar-first-wxyz poses and normalizes scalar-first quaternions. Normalization rejects zero, NaN, or Inf quaternions and chooses a deterministic sign representative.

## Inputs

- quaternion: length-4 finite scalar-first (w, x, y, z) array
- pose: length-7 finite array [x, y, z, qw, qx, qy, qz]

## Outputs

- normalized float64 unit quaternion with first numerically significant component positive
- copied float64 7D pose with normalized rotation

## Callable interface

Implementation: [implementation.py](implementation.py).

- `normalize_quat_wxyz(quaternion)`
- `validate_pose(pose)`

## Constraints

- Raises ValueError for wrong shape, nonfinite values, or a zero quaternion; does not repair malformed input.
- Scales by max absolute component before norm to avoid overflow.
- Returns new arrays; input arrays are copied.
- No environment handle, simulator state, layout, object dimensions, or scorer is read.
