---
name: garment_command_execution
description: "Executes paired-arm garment controls as complete sixteen-command segments, merging both arm targets and normalized gripper commands into an observed phase program. Reviews own-arm tracking and visual evidence to update the current fold phase or bounded recovery state."
---

# Garment command execution

## Function

Executes paired-arm garment controls as complete sixteen-command segments, merging both arm targets and normalized gripper commands into an observed phase program. Reviews own-arm tracking and visual evidence to update the current fold phase or bounded recovery state.

## Inputs

- Public robot API, a stage request with skill and available steps, and the validated same-episode garment state.
- Optional controls containing six own-joint targets and one normalized gripper command for each arm, plus the following stage.

## Outputs

- A GarmentProgram exposing sampled targets, peer-arm commands and per-frame observation handling.
- Updated garment progress and tracking/visual uncertainty for the next workflow stage.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `GarmentProgram: __init__(self, api, request, current, controls=None, following=None); snapshot(self); sampled_target(self, observation); additional_joint_commands(self, unused_observation); additional_gripper_commands(self, unused_observation); observe(self, observation); endpoint_reached(self); recovery_affordable(self); continue_after_shortfall(self); shortfall_phase(self); review_completion(self); feedback(self, observation, budget=False)`
- `allowed(current, phase)`
- `unavailable(api, request, current)`

## Constraints

- Controls must contain complete sixteen-command chunks within the request budget; joint targets must obey the declared own-robot limits and gripper commands must lie in [0, 1].
- An initial sampled joint jump greater than 0.30 rad holds the current joints. Endpoint checks use stage-specific bounds; transfer stages check travel relative to the commanded travel.
- Gripper commands and image following do not establish measured physical contact. Stage completion and recovery progress are separate from official task success.
