---
name: declared_pickup_kinematics
description: "Loads the declared six-joint robot URDF and provides own FK, root inference, TCP/public-pose transforms, and local least-squares IK with original seed and winding settings. run() verifies configs/mature-trajectory.json and the URDF SHA-256 before returning OwnToolModel."
---

# Declared pickup kinematics

## Function

Loads the declared six-joint robot URDF and provides own FK, root inference, TCP/public-pose transforms, and local least-squares IK with original seed and winding settings. run() verifies configs/mature-trajectory.json and the URDF SHA-256 before returning OwnToolModel.

## Inputs

- declared own robot URDF from configs/mature-trajectory.json (robot_urdf and robot_urdf_sha256)
- six finite public joints; measured end-effector pose; target_tcp, seed, and root 4x4 matrices
- solve retries 0..2, tolerance 0.002..0.008 m, rotation_tolerance 0.02..0.20 rad

## Outputs

- OwnToolModel with eef(), root(), tcp(), public_pose(), and solve()
- solve returns a six-joint solution or None; no robot action or collision certificate

## Callable interface

Implementation: [implementation.py](implementation.py).

- `OwnToolModel: __init__(self, source); eef(self, joints); root(self, joints, measured_ee); tcp(self, joints, root); public_pose(self, tcp); solve(self, target_tcp, seed, root, retries=0, tolerance=0.005, rotation_tolerance=0.2)`
- `run()`

## Constraints

- Reads only the contained declared URDF (not a symlink, <= 2 MiB) and rejects a SHA-256 mismatch in run().
- FK requires exactly six finite joints; IK validates finite 4x4 targets/root and 6-vector seed and caps queries at 512.
- IK bounds each joint to max(lower, seed-pi)..min(upper, seed+pi), uses position plus 0.16-weighted rotation residual, deterministic bend seeds and RNG 9017, max_nfev=100.
- Fixed TCP offset is 0.145 along local x; no task asset, object mesh, real finger state, or simulator handle is available.
