---
name: garment_settle_fold
description: "Runs either the prehome released-cloth settle path or the lower settling path: the prehome branch binds the current public six-joint snapshot and emits sixteen identical controls, while the lower branch builds wrist goals from planned contact poses or current target poses and samples a contact wrist path."
---

# Garment settle fold

## Function

Runs either the prehome released-cloth settle path or the lower settling path: the prehome branch binds the current public six-joint snapshot and emits sixteen identical controls, while the lower branch builds wrist goals from planned contact poses or current target poses and samples a contact wrist path.

## Inputs

- api: PublicRobotAPI used for command_memory(), instruction(), targets(), and wrist-path sampling
- request: stage request passed to GarmentProgram or ReleasedClothSettle
- recorded same-instruction public state from observed_garment_mount_state.recorded(api.command_memory(), api.instruction())
- validated sixteen-command allowance (CHUNK=16) for the prehome branch

## Outputs

- ReleasedClothSettle program following 'home' when phase='prehome_settle', part=3, and route is None; it renames phase 0 to 'released_cloth_settle_before_origin' and endpoint_reached always returns True
- GarmentProgram following 'release' after sampled_wrist_path succeeds for an allowed lower branch
- unavailable(api, request, current) when the lower branch is not allowed or sampled_wrist_path raises ValueError; the latter sets current['phase']='recover'

## Callable interface

Implementation: [implementation.py](implementation.py).

- `ReleasedClothSettle: __init__(self, api, request, current, controls); endpoint_reached(self)`
- `run(api, request)`

## Constraints

- prehome branch requires phase 'prehome_settle', part 3, and route None; it sets left_ee_joint_state and right_ee_joint_state to [1.] and returns CHUNK identical deep-copied bindings; it adds no pose or contact endpoint gate
- lower part 2 goal uses fold_arc_wrist(pose_matrix(contact_poses[side]), destinations[:2]-contacts[:2], pi) and lowers z by 0.003; other lower goals use targets()[side+'_ee_pose'] and lower z by 0.063
- sampled_wrist_path is called with GarmentOwnModel() and contact=True; ValueError marks the phase recover and returns unavailable rather than clamping or guessing
- normalized jaw commands held at 1 are commands, not measured physical finger width or contact; missing evidence remains uncertain and no asset, object pose, contact sensor, or official success substitute is used
