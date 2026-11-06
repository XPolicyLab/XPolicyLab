---
name: public_observation_command_clock
description: "Provides a blocking-policy dense joint-control sampler and public history/clock that issue one exchange per sampled action, update retained RGB/state history, and enforce a local action budget. It preserves fractional arm stride, pure gripper/hold duration, sparse target fill and declared normalized gripper command geometry."
---

# Public observation command clock

## Function

Provides a blocking-policy dense joint-control sampler and public history/clock that issue one exchange per sampled action, update retained RGB/state history, and enforce a local action budget. It preserves fractional arm stride, pure gripper/hold duration, sparse target fill and declared normalized gripper command geometry.

## Inputs

- environment with robot_manager target robots and step_lim, initial/current public observation, and output_root
- exchange(frame) callable, finite positive time_scale, controls or normalized command, and phase
- declared robot gripper_scale and gripper_move sign/mimic metadata
- PublicHistory-created output directory that must not already exist

## Outputs

- normalized_actions, action_sim_steps and action_status logs
- PublicHistory image/state histories, phases and current_public_observation
- PublicActionBudgetReached when the local action count reaches environment.step_lim
- normalized action dict from imported normalized_action for execute_normalized

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PublicActionBudgetReached: exception or data type`
- `color_frame(frame)`
- `PublicHistory: __init__(self, initial, output_root); update(self, observation, *, phase)`
- `array(value)`
- `PublicCommandClock: __init__(self, environment, history, exchange, *, time_scale); action_count(self); execute_normalized(self, command, phase); execute(self, controls, phase)`

## Constraints

- No simulator, reward, terminal-success or layout reference; local action counter is only a command budget and official evaluator owns actual termination and scoring
- Gripper seed conversion uses normalized public target and declared mimic geometry; it does not estimate physical aperture or contact
- Arm trajectories retain original fractional stride; pure gripper and hold sequences retain original duration; sparse targets retain previous targets
- PublicHistory uses exist_ok=False for output_root; execute_normalized rejects pose targets; exactly two declared X5 target arms/four targets are required
