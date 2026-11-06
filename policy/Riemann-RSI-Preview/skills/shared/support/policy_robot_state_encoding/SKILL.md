---
name: policy_robot_state_encoding
description: "Builds the 16-dimensional float32 state by concatenating left_ee_pose, the first left_ee_joint_state scalar, right_ee_pose, and the first right_ee_joint_state scalar. Missing keys or a non-16 length are errors."
---

# Policy robot state encoding

## Function

Builds the 16-dimensional float32 state by concatenating left_ee_pose, the first left_ee_joint_state scalar, right_ee_pose, and the first right_ee_joint_state scalar. Missing keys or a non-16 length are errors.

## Inputs

- obs: mapping whose 'state' contains left_ee_pose, left_ee_joint_state, right_ee_pose, right_ee_joint_state
- pose arrays are flattened and only the first joint-state scalar per side is used

## Outputs

- np.ndarray with shape (16,) and dtype float32

## Callable interface

Implementation: [implementation.py](implementation.py).

- `_state_from_obs(obs: dict)`

## Constraints

- Defined as the Model._state_from_obs static method; the adapter supplies obs
- Missing state key raises KeyError naming the missing key and available keys
- Total concatenated length must equal _STATE_DIM=16 or ValueError is raised
- Jaw scalars are SDK/public joint values, not measured contact or physical finger width
