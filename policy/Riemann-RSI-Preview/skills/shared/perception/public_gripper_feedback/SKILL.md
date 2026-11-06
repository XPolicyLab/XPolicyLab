---
name: public_gripper_feedback
description: "Reads one normalized public left or right EE joint-state scalar from the current observation or recorded history and returns frozen GripperEvidence with explicit uncertainty. It also converts normalized command targets into declared command coordinates and supplies static finger-range bounds for conservative sweep geometry."
---

# Public gripper feedback

## Function

Reads one normalized public left or right EE joint-state scalar from the current observation or recorded history and returns frozen GripperEvidence with explicit uncertainty. It also converts normalized command targets into declared command coordinates and supplies static finger-range bounds for conservative sweep geometry.

## Inputs

- observation dict containing state[side + '_ee_joint_state'] as one normalized scalar in [0, 1]
- side string 'left' or 'right', optionally suffixed with '_arm'
- optional requested_target normalized scalar in [0, 1]
- recorder with state_left_gripper or state_right_gripper history for from_history

## Outputs

- GripperEvidence or as_dict with side, normalized_applied_target, requested_target, possible_closing_obstruction, contact='unknown', physical_aperture_m=None, finger_joint_positions=None, source='official_public_ee_joint_state', and sdk_normalized_target_limit=0.2
- command_joint_target returns an interpolated declared command coordinate from lower, upper, and sign; it is never a measured finger width
- geometry_joint_candidates returns (0.0, max(abs(lower), abs(upper))) as static finger geometry bounds

## Callable interface

Implementation: [implementation.py](implementation.py).

- `GripperEvidence: as_dict(self)`
- `observe(observation, side, *, requested_target=None)`
- `from_history(recorder, side, *, requested_target=None)`
- `command_joint_target(normalized, *, lower=-0.01, upper=0.044, sign=1)`
- `geometry_joint_candidates(*, lower=-0.01, upper=0.044)`

## Constraints

- Accepts one-element list/tuple feedback, converting array-like inputs with tolist first; rejects Boolean, nonnumeric, nonfinite or out-of-range scalar values.
- When requested_target == 0, possible_closing_obstruction is bool(value > 1e-6); for other or absent requests it is None. None of these outcomes proves physical contact or an empty grasp.
- from_history uses the last already recorded public sample and raises ValueError when no current public gripper observation is available
- command_joint_target and geometry_joint_candidates validate finite lower < upper and, for command_joint_target, sign in (-1, 1); no simulator, robot manager, reward, task label, or layout is accessed, and no actions, waits, or retries are issued
