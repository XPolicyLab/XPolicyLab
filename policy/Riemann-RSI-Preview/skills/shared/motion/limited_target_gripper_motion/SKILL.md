---
name: limited_target_gripper_motion
description: "Provides gripper control planning and execution for a class-bound runtime: BaseGripperMotion clips normalized values and delegates to rm.plan_endeffector_joint, while GripperMotion persists commanded jaw targets, reasserts idle jaws in official mode, and submits controls through the official bridge."
---

# Limited target gripper motion

## Function

Provides gripper control planning and execution for a class-bound runtime: BaseGripperMotion clips normalized values and delegates to rm.plan_endeffector_joint, while GripperMotion persists commanded jaw targets, reasserts idle jaws in official mode, and submits controls through the official bridge.

## Inputs

- normalized_value, steps, and optional robot or arm for _gripper_controls; _grip also takes arm, value, and phase
- caller-bound self attributes args, robots, robot, _gripper_targets, phase, rm, env, and _official_bridge
- env provides public dt for step counts; rm provides declared plan_endeffector_joint

## Outputs

- list of control dictionaries from rm.plan_endeffector_joint with env_idx=0, arm_tag=robot.arm_name, need_plan=True, and control_step_num=max(1,int(steps))
- repeated target returns one planned command repeated max(1,int(steps)) times
- _execute submits non-empty controls to _official_bridge.execute(controls, phase) and returns its result; _grip returns nothing

## Callable interface

Implementation: [implementation.py](implementation.py).

- `BaseGripperMotion: exception or data type`
- `GripperMotion: exception or data type`

## Constraints

- normalized_value is clipped to [0,1]; steps are at least 1 and settle steps are max(10, round(seconds/env.dt))
- unchanged commanded target plans once and repeats the command to hold the commanded target continuously across segment boundaries; official mode reasserts idle jaws from commanded positions while explicit active ramps override them
- official _lift_settle caps gripper settle seconds at min(seconds, .12); empty controls are not submitted; missing _official_bridge fails rather than falling back to env.step
- methods are compiled in their class definitions and depend on adapter-bound self globals; no finger getter, width inference, or new 15 mm jaw gate is added; normalized or commanded values are not physical finger width or contact
