---
name: normalized_command_clock
description: "NormalizedCommandClock extends PhasedProgram and turns already-sampled normalized opening commands into one pose phase per command against the current side_ee_pose target, preserving commanded joint targets and nonrepeatable command memory. run constructs this clock for a PublicRobotAPI request."
---

# Normalized command clock

## Function

NormalizedCommandClock extends PhasedProgram and turns already-sampled normalized opening commands into one pose phase per command against the current side_ee_pose target, preserving commanded joint targets and nonrepeatable command memory. run constructs this clock for a PublicRobotAPI request.

## Inputs

- api must be PublicRobotAPI; request dict with skill string length 1..64, side left/right, steps integer 1..640, and optional reason
- openings list of 1..40 finite normalized command values in [0, 1]
- segment_end_indices sorted unique integers starting at >=1 and ending at len(openings)

## Outputs

- run returns a NormalizedCommandClock; feedback returns the superclass feedback dict plus image_measurements.normalized_command_clock diagnostics
- diagnostics include prepared_native_commands, issued_native_commands, segment_end_indices, sampling_reapplied False, arm_trajectory_replanned False, and official_success_claimed False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `NormalizedCommandClock: __init__(self, api, request, openings, segment_end_indices); feedback(self, observation, budget=False)`
- `run(api, request, openings, segment_end_indices)`

## Constraints

- Raises ExecutionError for invalid API type, request fields, opening count/range, or segment end indices.
- Issues each element once as an official native command; does not interpolate, resample, or replan the arm trajectory.
- Normalized opening commands are not physical finger width or contact readings; feedback does not claim true fingers/contact or official success.
- Held joints come from the last own command or initial public own joints when no command memory exists.
