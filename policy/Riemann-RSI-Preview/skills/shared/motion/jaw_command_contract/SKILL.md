---
name: jaw_command_contract
description: "Validates one normalized public jaw command against an authenticated static robot configuration and the robot's declared command-pair mimic, returning the normalized target and corresponding drive targets without issuing an action."
---

# Jaw command contract

## Function

Validates one normalized public jaw command against an authenticated static robot configuration and the robot's declared command-pair mimic, returning the normalized target and corresponding drive targets without issuing an action.

## Inputs

- robot: public robot with command_pair(), gripper_scale, and gripper_sign
- normalized_value: finite value in [0, 1]
- config_path and config_sha256 for the authenticated YAML configuration
- parameters: expected gripper_scale, sign, and mimic

## Outputs

- dict with normalized_target, drive_targets, actual_aperture=None, contact_evidence='unknown', and action_issued=False
- ValueError for SHA mismatch, configuration mismatch, invalid finite ranges, invalid pair, non-finite or out-of-range normalized value, or mimic mismatch

## Callable interface

Implementation: [implementation.py](implementation.py).

- `load_command_ranges(config_path, expected_sha256, parameters)`
- `validate_command_pair(robot, pair, ranges)`
- `validated_normalized_target(robot, value, ranges)`
- `run(robot, normalized_value, *, config_path, config_sha256, parameters)`

## Constraints

- config bytes must hash to expected_sha256; gripper_scale must be finite shape (2,) with scale[0] < scale[1] and match parameters; mimic multiplier must be finite and nonzero
- declared ranges must be finite shape (2,2) with lower < upper; command_pair endpoints must match ranges within atol 1e-8; input pair must be finite and within ranges plus or minus 1e-8 without clamping
- normalized_value must be finite and within [0,1]; drive targets are not physical joint positions and may lie below a physical joint-position lower limit while preserving the normalized close command
- actual_aperture stays None and contact_evidence stays 'unknown'; successful validation issues zero simulation actions and claims no measurement
