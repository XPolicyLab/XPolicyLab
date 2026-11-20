---
name: mature_feedback_execution
description: "MatureFeedbackProgram composes 1-16 move, grip, and hold phases and executes them through a restricted PublicRobotAPI/PhasedProgram, using public pose tracking corrections and the same dense/official sampler for joint and gripper commands. It records a bounded public-feedback receipt; completion is not an official task success."
---

# Mature feedback execution

## Function

MatureFeedbackProgram composes 1-16 move, grip, and hold phases and executes them through a restricted PublicRobotAPI/PhasedProgram, using public pose tracking corrections and the same dense/official sampler for joint and gripper commands. It records a bounded public-feedback receipt; completion is not an official task success.

## Inputs

- MatureFeedbackProgram(api, request, schedule, precondition=True, measurements=None), where schedule is 1-16 (phase, profile) pairs from move_phase, grip_phase, or hold_phase
- move_phase(name, pose, opening, precise=True, tool_goal=None, attachment=None, tolerance=.004, rotation_tolerance_degrees=3., time_scale=1.4, operation='pose') with checked 7-vector poses and normalized opening
- grip_phase(name, pose, opening, seconds=.28, lift_settle=False) and hold_phase(name, pose, opening, actions=1)
- restricted PublicRobotAPI observations, targets, command_memory execution_feedback, and official action acknowledgments

## Outputs

- Program state fields status/finished/failed and per-phase reports; completed/finished is not official task success
- execution_receipt returns an exact dict with schema riemann-mature-public-feedback-replay-v1, action acknowledgments, completed nominal phases, maximum corrections, dense/official gripper counts, trajectory_sampling_reapplied false, and gripper_feedback_is_contact_sensor false
- Action/target updates contain own arm joint states plus one-element normalized gripper commands sampled from the ramp; no extra hidden holds are appended

## Callable interface

Implementation: [implementation.py](implementation.py).

- `move_phase(name, pose, opening, precise=True, tool_goal=None, attachment=None, tolerance=0.004, rotation_tolerance_degrees=3.0, time_scale=1.4, operation='pose')`
- `grip_phase(name, pose, opening, seconds=0.28, lift_settle=False)`
- `hold_phase(name, pose, opening, actions=1)`
- `validate_receipt(value)`
- `MatureFeedbackProgram: __init__(self, api, request, schedule, precondition=True, measurements=None); phase_time_scale(self, unused_phase); trajectory_operation(self, unused_phase); accept_trajectory(self, unused_phase, response, unused_current_joints); command_opening(self, unused_observation, unused_steps); observe(self, observation); execution_receipt(self)`

## Constraints

- schedule length is 1-16; len(profiles)+2*sum(precise) must be <=40; move tolerance is 0<tolerance<=.04 m, rotation_tolerance_degrees is 0<value<=20 deg, time_scale is .1-4; tool_goal and attachment must both be present or both absent
- grip_phase seconds must be finite .04-.64; lift_settle caps duration at .12 s; dense_steps=max(10, round(duration/.004)) and official_steps=ceil(dense_steps/10); hold_phase actions is int 1-16
- A moving gripper ramp uses sampled_commands(start, opening, dense_steps, unchanged=opening==last_target, interval=10, time_scale=profile time_scale) and must match official_steps; command_opening rejects hidden holds beyond the sampled sequence
- A precise phase may correct at most twice after the initial attempt; diverged or over-tolerance tracking returns uncertain; receipt counters are bounded and official_gripper_samples >= official_action_acknowledgments; normalized gripper values are commands, not physical finger width or contact
