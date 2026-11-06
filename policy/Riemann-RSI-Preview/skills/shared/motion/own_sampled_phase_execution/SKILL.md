---
name: own_sampled_phase_execution
description: "Subclass of public_phase_execution.PhasedProgram that validates and replays bounded declared six-joint own-arm samples on the official action clock. It stops fail-closed when the first sampled target jumps more than 0.8 rad from current own-joint feedback."
---

# Own sampled phase execution

## Function

Subclass of public_phase_execution.PhasedProgram that validates and replays bounded declared six-joint own-arm samples on the official action clock. It stops fail-closed when the first sampled target jumps more than 0.8 rad from current own-joint feedback.

## Inputs

- api, request, phases, precondition, measurements, and optional samples mapping phase index to finite 6-joint arrays
- observation['state'][side + '_arm_joint_state'] as a finite 6-vector
- phase metadata minimum_steps and require_pose for each sampled phase

## Outputs

- self.own_joint_samples with validated copied arrays
- sampled_target returning a 6-vector copy or the current joint vector on a fail-closed jump
- trajectory_operation returning 'joint' if phase_index is in joint_targets else 'pose'
- observe may report 'stalled' and set failed=True, finished=True, status='uncertain'

## Callable interface

Implementation: [implementation.py](implementation.py).

- `OwnSampledPhaseProgram: __init__(self, api, request, phases, samples=None, precondition=True, measurements=None); sample_kinematics(self); sampled_target(self, observation); prepare_sampled_phase(self, unused_observation); trajectory_operation(self, phase); trajectory_joint_target(self, unused_phase); observe(self, observation)`

## Constraints

- Class sets preserve_commanded_joint_targets=True, remember_nonrepeatable_commands=True, motion_time_scale=1.0, joint_targets={}, geometry={}, and stop_after_held_action=False.
- Samples are rejected unless index is an int in range, values.ndim is 2, shape equals (phases[index]['minimum_steps'], 6), length is <=16, values are finite and within own model lower/upper limits, and phases[index]['require_pose'] is False.
- No second Cartesian planner or resampling is applied to samples; cold approach and home retain the actual public cuRobo joint trajectory.
- Own-URDF solutions and normalized gripper targets are commands, not measured contact or real finger positions; a first-action target jump over 0.8 rad stops after the held action.
