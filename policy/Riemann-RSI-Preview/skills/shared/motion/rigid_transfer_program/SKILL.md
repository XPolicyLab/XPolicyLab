---
name: rigid_transfer_program
description: "Provides RigidMotion, RigidClock, and RigidHome phased programs plus motion, clock, home, clear_view, and unavailable helpers that execute recorded rigid-transfer stages with declared own-arm trajectories and normalized command clocks. It validates recorded public transfer state, binds the own side, and performs fresh public-image review after lift and verification."
---

# Rigid transfer program

## Function

Provides RigidMotion, RigidClock, and RigidHome phased programs plus motion, clock, home, clear_view, and unavailable helpers that execute recorded rigid-transfer stages with declared own-arm trajectories and normalized command clocks. It validates recorded public transfer state, binds the own side, and performs fresh public-image review after lift and verification.

## Inputs

- api: restricted public robot API with command_memory, instruction, targets, home_targets, home_joint_states, trajectory_available, and observe
- request: dict restricted to skill, side, steps, reason; side must be auto or the selected own arm and steps is an action budget
- stage: one of pregrasp, contact, lift, carry, align, place, retract, close, release, reset, verify, finish, home, view, clear
- current: validated public transfer record from recorded(api.command_memory(), api.instruction()); current['scene'] may be None

## Outputs

- RigidMotion, RigidClock, RigidHome, or unavailable RigidClock with status precondition_unmet
- feedback image_measurements containing public_rigid_transfer_state, rigid_visual_review, and rigid_command_phases_completed
- rigid_motion_execution record with backend, phase, native_actions_issued, maximum_native_actions, trajectory_completed, and explicit false contact/finger/official-success flags
- visual_review evidence with status uncertain, current_surface_follows, or current_image_placement and position_error_m/colour_difference

## Callable interface

Implementation: [implementation.py](implementation.py).

- `own_opening(api, side)`
- `actual_request(request, side)`
- `visual_review(api, current, following)`
- `feedback_state(program, result)`
- `RigidClock: __init__(self, api, request, current, openings, boundaries, following=None, reviewer=None, precondition=True); observe(self, observation); feedback(self, observation, budget=False)`
- `RigidMotion: __init__(self, api, request, current, target, phase, opening, maximum=160, following=None, reviewer=None); trajectory_operation(self, phase); observe(self, observation); record_action(self); feedback(self, observation, budget=False)`
- `unavailable(api, request, current)`
- `clear_view(api, request, current)`
- `motion(api, request, stage)`
- `clock(api, request, stage)`
- `RigidHome: __init__(self, api, request, current); trajectory_operation(self, phase); trajectory_joint_target(self, phase); observe(self, observation); feedback(self, observation, budget=False)`
- `home(api, request)`

## Constraints

- actual_request rejects extra request keys, non-left/right sides, and side mismatches; an unavailable scene preserves an explicit allowed left/right request rather than silently choosing left
- RigidMotion requires type(maximum) int in 1..320 and uses min(request['steps'], maximum); trajectory_available is a precondition and failures remain uncertain/precondition outcomes
- pose_tolerance_m is .006 m; lift visual review accepts only current_surface_follows, verify accepts only current_image_placement, using position threshold .025 + min(uncertainty, .025) m and colour_difference < 100
- normalized gripper values and trajectory completion are own commands/execution feedback, not measured finger width, contact, grasp, or official success; programs depend on supplied api and self state with no standalone invocation
