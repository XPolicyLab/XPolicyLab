---
name: mature_feedback_execution_planned_start
description: "Builds validated move, grip, and hold phase/profile pairs and runs a finite mature feedback schedule through a planned-start PhasedProgram over the restricted PublicRobotAPI. The program replays official joint positions without resampling, generates gripper commands with the shared dense/official sampler, and advances only after public execution feedback."
---

# Mature feedback execution planned start

## Function

Builds validated move, grip, and hold phase/profile pairs and runs a finite mature feedback schedule through a planned-start PhasedProgram over the restricted PublicRobotAPI. The program replays official joint positions without resampling, generates gripper commands with the shared dense/official sampler, and advances only after public execution feedback.

## Inputs

- MatureFeedbackProgram(api, request, schedule, precondition=True, measurements=None)
- schedule: list or tuple of 1-16 (phase, profile) pairs produced by move_phase, grip_phase, or hold_phase
- move_phase(name, pose, opening, precise=True, tool_goal=None, attachment=None, tolerance=0.004 m, rotation_tolerance_degrees=3.0, time_scale=1.4, operation='pose')
- grip_phase(name, pose, opening, seconds=0.28, lift_settle=False) and hold_phase(name, pose, opening, actions=1)

## Outputs

- phase/profile tuples whose phase has name, checked pose copy, opening, minimum_steps, and require_pose; profile has precision, target, optional tool_goal/attachment, tolerances, time_scale, operation, and dense_steps
- MatureFeedbackProgram with status, finished, failed, per-phase reports, correction counts, and completed_nominal_phases
- execution_receipt() returns a validated riemann-mature-public-feedback-replay-v1 receipt

## Callable interface

Implementation: [implementation.py](implementation.py).

- `move_phase(name, pose, opening, precise=True, tool_goal=None, attachment=None, tolerance=0.004, rotation_tolerance_degrees=3.0, time_scale=1.4, operation='pose')`
- `grip_phase(name, pose, opening, seconds=0.28, lift_settle=False)`
- `hold_phase(name, pose, opening, actions=1)`
- `MatureFeedbackProgram: __init__(self, api, request, schedule, precondition=True, measurements=None); phase_time_scale(self, unused_phase); trajectory_operation(self, unused_phase); accept_trajectory(self, unused_phase, response, unused_current_joints); command_opening(self, unused_observation, unused_steps); observe(self, observation); execution_receipt(self)`

## Constraints

- move_phase rejects non-bool precise, operation outside pose/joint, tolerance outside (0, 0.04] m, rotation_tolerance_degrees outside (0, 20], time_scale outside [0.1, 4], or exactly one of tool_goal and attachment set
- grip_phase requires finite seconds in [0.04, 0.64]; lift_settle caps duration at 0.12 s; hold_phase requires integer actions in [1, 16]
- constructor requires 1-16 phases and len(profiles)+2*precise profiles <= 40; missing ramp metadata, replay beyond sampled ramp, or mismatched dense/official counts raise ExecutionError
- precise phases stop within tolerance or after at most two corrections, otherwise set uncertain/failed; gripper commands are normalized values, not physical finger width or contact, and tool tracking uses wrist pose plus explicit attachment hypothesis
