---
name: wrist_contact_pose
description: "Builds a 7-value wrist command from a finite observed contact point, a target height, and a declared forward tool offset, using a default downward WXYZ quaternion unless one is supplied. The returned xyz is the contact point with z set to height and shifted backward along the orientation's first rotation-matrix column by forward."
---

# Wrist contact pose

## Function

Builds a 7-value wrist command from a finite observed contact point, a target height, and a declared forward tool offset, using a default downward WXYZ quaternion unless one is supplied. The returned xyz is the contact point with z set to height and shifted backward along the orientation's first rotation-matrix column by forward.

## Inputs

- center: finite length-3 current observed contact target
- height: finite target z value
- forward: finite offset in the inclusive range 0 to 0.3, default 0.155
- quaternion: optional finite length-4 WXYZ orientation with norm in [0.99, 1.01], default [0.5, -0.5, 0.5, 0.5]

## Outputs

- seven-element ndarray ordered x, y, z, qw, qx, qy, qz
- wrist xyz equals point minus rotation_matrix(orientation)[:, 0] * forward after setting point[2] to height
- orientation is copied from the supplied or default quaternion

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(center, height, forward=0.155, quaternion=None)`

## Constraints

- Raises ValueError for wrong sizes, non-finite values, quaternion norm outside 0.99-1.01, or forward outside 0-0.3.
- This is a pure function with no simulator, task identity, object dimensions, file access, or outcome query.
- It is not a reachability or contact test; the default quaternion is the mature downward vertical tool frame.
