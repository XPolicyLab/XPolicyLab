---
name: upright_transfer
description: "Implements UprightTransfer, which builds a selected arm's target dict from current public left/right end-effector poses and joint states and steps through source-high, destination-high and destination poses using a requested delta and clearance. RelativeRotation builds a target by rotating the selected arm's current quaternion around a declared axis by an angle and reports angular error."
---

# Upright transfer

## Function

Implements UprightTransfer, which builds a selected arm's target dict from current public left/right end-effector poses and joint states and steps through source-high, destination-high and destination poses using a requested delta and clearance. RelativeRotation builds a target by rotating the selected arm's current quaternion around a declared axis by an angle and reports angular error.

## Inputs

- observation['state'][side+'_ee_pose']: finite 7-vector XYZ and WXYZ quaternion
- observation['state'][side+'_ee_joint_state']: own joint-state vector
- request['side'], request['delta'] (3-vector), request['clearance'] for UprightTransfer
- request['axis'] (3-vector) and request['angle'] in radians for RelativeRotation

## Outputs

- proposal(observation) returns a deepcopy of the current target pose and joint-state dict for both sides
- UprightTransfer.finished: true once all three phases are reached
- RelativeRotation.error: angle in radians; finished when error <=0.02
- Invalid orientation raises ExecutionError

## Callable interface

Implementation: [implementation.py](implementation.py).

- `unit_vector(value, size, label)`
- `current_targets(observation)`
- `UprightTransfer: __init__(self, observation, request); proposal(self, observation)`
- `axis_rotation(quaternion, axis, angle)`
- `RelativeRotation: __init__(self, observation, request); proposal(self, observation)`

## Constraints

- Orientations must be finite, nonzero and are normalized; phase advance uses Euclidean translation distance <=0.004 in pose units.
- UprightTransfer sets the selected side joint-state[0] to 0.0 and keeps the other arm's copied targets.
- RelativeRotation uses quaternion multiplication and holds translation; it does not choose a pivot or insertion orientation.
- No collision checking is provided by this primitive.
