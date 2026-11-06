---
name: garment_approach_contact
description: "Checks the recorded same-instruction state is in the approach phase, samples a sixteen-command own-joint pregrasp from route.approach_joints using nearest valid winding, and returns a GarmentProgram configured to continue with contact. If the phase/attempt/route gate fails, returns an unavailable GarmentProgram instead."
---

# Garment approach contact

## Function

Checks the recorded same-instruction state is in the approach phase, samples a sixteen-command own-joint pregrasp from route.approach_joints using nearest valid winding, and returns a GarmentProgram configured to continue with contact. If the phase/attempt/route gate fails, returns an unavailable GarmentProgram instead.

## Inputs

- api: PublicRobotAPI
- request: stage request with skill and steps
- current: recorded public state from api.command_memory() and api.instruction(), including phase, attempts, and route.approach_joints

## Outputs

- GarmentProgram carrying the sampled approach controls and following='contact'
- unavailable GarmentProgram when allowed(current, 'approach') is false

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- allowed requires current.phase == 'approach', current.attempts < 3, and a non-null route
- Sampling or GarmentProgram validation errors propagate; the stage does not catch them
- State, own endpoints, and image review are validated by observed_garment_mount_state; missing evidence remains uncertain
- No asset dimensions, object pose, contact sensor, or official success flag are substituted; standard and randomized modes use the same implementation
