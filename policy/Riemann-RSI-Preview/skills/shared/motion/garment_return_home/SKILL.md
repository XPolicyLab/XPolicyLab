---
name: garment_return_home
description: "Returns both own arms simultaneously to api.home_joint_states() over sixteen sampled joint commands and sets both gripper command channels to [1.]. It is available only in the home phase of part 3."
---

# Garment return home

## Function

Returns both own arms simultaneously to api.home_joint_states() over sixteen sampled joint commands and sets both gripper command channels to [1.]. It is available only in the home phase of part 3.

## Inputs

- api PublicRobotAPI and request with a validated sixteen-command allowance
- recorded same-instruction state with phase home and part 3
- api.home_joint_states()

## Outputs

- GarmentProgram with sixteen controls and following set to hold, or an unavailable result

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- Requires current['phase'] == 'home' and current['part'] == 3
- Sets both left and right *_ee_joint_state values to [1.] on every command
- Only the unchanged official evaluator decides success; the policy check is not official success
- Uses sampled_joint_path within the bounded sixteen-command chunk
