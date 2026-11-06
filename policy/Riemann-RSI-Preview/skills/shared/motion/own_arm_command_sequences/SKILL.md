---
name: own_arm_command_sequences
description: "Adapter-bound helper methods read the own robot's real end pose as a 7-vector, plan normalized gripper controls, merge control sequences by holding last targets, and execute timed single- or both-gripper holds. The capability selects no task, grasp, object, or workflow."
---

# Own arm command sequences

## Function

Adapter-bound helper methods read the own robot's real end pose as a 7-vector, plan normalized gripper controls, merge control sequences by holding last targets, and execute timed single- or both-gripper holds. The capability selects no task, grasp, object, or workflow.

## Inputs

- Own robot object accepted by self.rm.get_real_endpose, with env_idx_list=[0] and is_relative=True
- normalized_value float clipped to [0.0, 1.0] for gripper control
- steps int for dense control count, or seconds float used with self.env.dt for holds
- Optional robot whose arm_name is used for gripper planning; defaults to self.robot

## Outputs

- 7-element float32 own end-effector pose array
- list of control dicts from self.rm.plan_endeffector_joint
- merged control list with shorter sequences holding their last target
- acknowledged execution through self._execute

## Callable interface

Implementation: [implementation.py](implementation.py).

- `OwnArmCommandSequences: exception or data type`

## Constraints

- Methods depend on adapter-provided self.rm, self.robot, self.robots, self.env.dt, and self._execute.
- Gripper planning calls self.rm.plan_endeffector_joint with env_idx=0, arm_tag=robot.arm_name, need_plan=True, and control_step_num=max(1, int(steps)); hold steps are max(10, round(seconds / self.env.dt)).
- _merge_controls iterates to the longest sequence and reuses each shorter sequence's last control dict.
- No task, grasp, object pose, scene, layout, reward, or scorer is read; normalized gripper targets are commands, not measured physical finger width or contact.
