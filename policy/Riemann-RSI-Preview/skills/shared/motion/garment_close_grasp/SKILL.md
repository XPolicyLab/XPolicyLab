---
name: garment_close_grasp
description: "Checks the recorded close phase and samples the original sixteen-command gripper clock (twelve close commands plus four small lift-probe commands) for the route arms, returning a CloseGarment. On sampling ValueError it sets current.phase to recover and returns an unavailable GarmentProgram; completion records current ee poses as start_wrists and advances to lift."
---

# Garment close grasp

## Function

Checks the recorded close phase and samples the original sixteen-command gripper clock (twelve close commands plus four small lift-probe commands) for the route arms, returning a CloseGarment. On sampling ValueError it sets current.phase to recover and returns an unavailable GarmentProgram; completion records current ee poses as start_wrists and advances to lift.

## Inputs

- api: PublicRobotAPI
- request: stage request with skill and steps
- current: recorded public state with phase, attempts, and route.arms

## Outputs

- CloseGarment carrying the sampled gripper controls
- unavailable GarmentProgram when the close gate fails or sampled_gripper_clock raises ValueError
- mutations on completion: start_wrists set from api.targets() ee poses, phase set to lift, chunk reset to 0

## Callable interface

Implementation: [implementation.py](implementation.py).

- `CloseGarment: review_completion(self)`
- `run(api, request)`

## Constraints

- allowed requires current.phase == 'close', current.attempts < 3, and a non-null route
- The public normalized jaw command is not measured finger width, physical contact, or a contact sensor
- review_completion is adapter-bound to self.current/self.api; missing evidence remains uncertain and no official success flag is substituted
- Standard and randomized garment modes use the same implementation
