---
name: return_episode_home
description: "Returns a PhasedProgram that restores the selected arm to its own episode-home EE pose and then opens its gripper via pose_phase('return_to_public_episode_home', pose, 1, 4)."
---

# Return episode home

## Function

Returns a PhasedProgram that restores the selected arm to its own episode-home EE pose and then opens its gripper via pose_phase('return_to_public_episode_home', pose, 1, 4).

## Inputs

- api: adapter exposing home_targets()
- request: mapping with side key

## Outputs

- PhasedProgram containing one pose_phase named 'return_to_public_episode_home' with arguments (pose, 1, 4)

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- Reads pose from api.home_targets()[request['side'] + '_ee_pose']; it must be the arm's own pose captured at first public observation.
- Does not use a task-specific preset, another episode's home, an object pose, or simulator state.
- Pose arrival is robot feedback only and does not prove manipulation-task completion or official success.
