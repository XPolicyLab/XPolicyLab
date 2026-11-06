---
name: public_plane_memory
description: "validate enforces the exact public support-hypothesis schema, finite measured scalars, bounded support evidence and camera-transform validity, then returns a deep copy. compatible re-validates a stored hypothesis and checks it against current transform, focal length, support pixels and sizes with fixed tolerances."
---

# Public plane memory

## Function

validate enforces the exact public support-hypothesis schema, finite measured scalars, bounded support evidence and camera-transform validity, then returns a deep copy. compatible re-validates a stored hypothesis and checks it against current transform, focal length, support pixels and sizes with fixed tolerances.

## Inputs

- value: dict with exactly support_plane_m, edge_m, camera_transform, focal_pixels, support_pixels, support_sizes_px and cap_top_m
- camera_transform: 4x4 finite array whose last row is [0,0,0,1] and whose 3x3 rotation is orthonormal to atol 1e-4
- support_pixels: 1-8 finite image centers in a 640x480 image; support_sizes_px: matching positive sizes <=640
- compatible args: value, transform, focal, pixels, sizes

## Outputs

- validate returns a deep-copied value
- compatible returns a bool for shape, transform, focal, center-distance and size-difference checks
- ValueError on schema, finiteness, range or transform violations

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`
- `compatible(value, transform, focal, pixels, sizes)`

## Constraints

- support_plane_m, edge_m and focal_pixels must be finite int/float; edge_m must satisfy 0 < edge_m <= .2; focal_pixels > 0
- cap_top_m may be None or a finite int/float; pixels must be in image bounds and sizes in (0,640]
- compatible requires transform allclose atol 1e-5 rtol 0, focal isclose abs_tol 1e-5 rel_tol 0, center distance <= max(2, old_size*.1), and size difference <= old_size*.15
- The payload excludes layout, object state and score fields; no physical finger width or contact claim is made
