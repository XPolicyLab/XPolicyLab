---
name: garment_clear_wrist
description: "After release, builds candidate wrist-clearance goals for the route arms: root-relative high outward park and lower camera-clear park for parts 0/1, plus a smoothed vertical lift option, then returns a ClearGarment for the first candidate sampled_wrist_path accepts. If the first outward candidate fails Cartesian sampling, solves its endpoint joints and uses sampled_joint_path; review increments part, clears the route/start_wrists/attempts, and advances to view or prehome_settle."
---

# Garment clear wrist

## Function

After release, builds candidate wrist-clearance goals for the route arms: root-relative high outward park and lower camera-clear park for parts 0/1, plus a smoothed vertical lift option, then returns a ClearGarment for the first candidate sampled_wrist_path accepts. If the first outward candidate fails Cartesian sampling, solves its endpoint joints and uses sampled_joint_path; review increments part, clears the route/start_wrists/attempts, and advances to view or prehome_settle.

## Inputs

- api: PublicRobotAPI
- request: stage request with skill and steps
- current: recorded public state with phase, attempts, part, and route.arms
- api.targets(), api.own_joint_states(), and GarmentOwnModel roots

## Outputs

- ClearGarment carrying accepted wrist-path or fallback joint-path controls
- unavailable GarmentProgram when the clear gate fails or no candidate succeeds
- mutations in review_completion: part incremented, chunk and attempts reset to 0, route and start_wrists cleared, phase set to view or prehome_settle

## Callable interface

Implementation: [implementation.py](implementation.py).

- `ClearGarment: review_completion(self)`
- `run(api, request)`

## Constraints

- allowed requires current.phase == 'clear', current.attempts < 3, and a non-null route
- Root-relative park offsets are [±0.06, 0.15, 0.275] m high and [±0.06, 0.15, 0.18] m lower; vertical candidates try heights 0.050, 0.035, 0.020, and 0.010 m
- Vertical solve rejects a candidate on ValueError or any per-sample joint change above 0.30; the joint-path fallback is used only for the first outward candidate when part is 0 or 1
- review_completion is adapter-bound to self.current; missing evidence remains uncertain and no asset dimensions, object pose, contact sensor, or official success flag are substituted
