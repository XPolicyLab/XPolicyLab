---
name: incremental_tool_tilt
description: "Builds a PhasedProgram that rotates an already checked tool about its current robot tool point in bounded increments, with dwell and inspect phases between increments and optional restoration of the starting orientation."
---

# Incremental tool tilt

## Function

Builds a PhasedProgram that rotates an already checked tool about its current robot tool point in bounded increments, with dwell and inspect phases between increments and optional restoration of the starting orientation.

## Inputs

- api: PublicRobotAPI providing targets() and instruction()
- request['side'] selecting the end-effector pose
- request['axis'] as a unit public robot axis and request['angle'] in radians
- request['dwell'] for inspect phases and request['return_upright'] for optional restore

## Outputs

- PhasedProgram with alternating incremental_tilt_N pose phases and inspect_during_tilt_N dwell phases
- optional 'restore_starting_grasp_orientation' phase when return_upright is true
- precondition true only when 'pour' appears in api.instruction().lower()

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- increment count is max(1, ceil(abs(angle)/(pi/12))), so each step is at most pi/12 radians
- poses are computed by relative_rotation(initial, axis, angle*(index+1)/count) about the current robot tool point; no physical container pivot, cup radius, capacity, object pose, or scorer state is inferred
- each dwell consumes the same official action budget as other commands; a finished tilt is not evidence that contents entered the receiver
- check current source, receiver, and spills after the segment before continuing or transporting
