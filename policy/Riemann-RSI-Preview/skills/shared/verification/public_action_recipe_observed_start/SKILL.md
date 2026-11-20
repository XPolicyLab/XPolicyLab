---
name: public_action_recipe_observed_start
description: "Validates bounded same-episode public-reference recipe jobs, steps, and compositions for the observed-start variant, returning deep copies of accepted values. Provides initial_job, movement, command, and compose helpers that construct and validate job, step, or recipe records."
---

# Public action recipe observed start

## Function

Validates bounded same-episode public-reference recipe jobs, steps, and compositions for the observed-start variant, returning deep copies of accepted values. Provides initial_job, movement, command, and compose helpers that construct and validate job, step, or recipe records.

## Inputs

- job: dict exactly matching initial_job fields, including bounded substage, held_side, own_openings, pose fields, identities, labels, logical_completed, and thread counters
- step: dict with tool, side, target, opening, commands, maximum_actions, and visual_check
- recipe: None or dict with source 'current_public_reference_recipe', 1-48 steps, after, substage, complete_item, held_side_after, thread_target_after, thread_rotation_after, and reset_thread_after
- movement(tool, side, target, opening, maximum=160, visual_check='none'), command(tool, side, values, visual_check='none'), or compose(...) constructor arguments

## Outputs

- initial_job(): new dict with substage 'ready', held_side None, empty lists, zero counters, and initialized receiver and thread fields
- job(), step(), and recipe() return deep copies of validated inputs; recipe(None) returns None
- movement() and command() return validated step dicts; command() sets opening from the last sampled command and maximum_actions from the command count
- compose() returns a validated recipe dict with source 'current_public_reference_recipe'

## Callable interface

Implementation: [implementation.py](implementation.py).

- `initial_job()`
- `checked_pose(value)`
- `job(value)`
- `step(value)`
- `recipe(value)`
- `movement(tool, side, target, opening, maximum=160, visual_check='none')`
- `command(tool, side, values, visual_check='none')`
- `compose(steps, substage='ready', complete_item=True, after='view', held_side_after='unchanged', thread_target_after=None, thread_rotation_after=None, reset_thread_after=False)`

## Constraints

- Rejects unknown or missing fields, invalid tool kinds, side values, visual_check values, non-finite numbers, and malformed bounded values with ValueError
- checked_pose requires a 7-value finite pose with max absolute value <= 4 and last-three norm in [0.99, 1.01]
- Motion tools must have target and commands None; command tools must have target None, 1-40 commands, and maximum_actions equal to command count
- own_openings retain only issued normalized own commands in [0, 1], not finger or contact measurements; substage completion is not official success; receiver_anchor is validated by observed_rigid_transfer_observed_start.implementation.anchor and identity descriptors by public_action_recipe descriptor
