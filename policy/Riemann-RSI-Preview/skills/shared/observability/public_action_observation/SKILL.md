---
name: public_action_observation
description: "Defines an immutable current Observation with allowlisted state and three RGB views, binds actions to observation id/step, and validates fresh bound joint actions. Also exposes normalized gripper feedback, opening hypotheses, and adapter-context payload evidence."
---

# Public action observation

## Function

Defines an immutable current Observation with allowlisted state and three RGB views, binds actions to observation id/step, and validates fresh bound joint actions. Also exposes normalized gripper feedback, opening hypotheses, and adapter-context payload evidence.

## Inputs

- state mapping exactly STATE_KEYS derived from imported ARMS: per arm _arm_joint_state (6), _ee_pose (7), _ee_joint_state (1 in [0,1])
- vision mapping exactly CAMERAS imported from skills.shared.geometry.x5_public_frame_model, each 480x640x3 uint8
- step native control int in [0,1050]; previous_summary string <=5000 characters
- bound dict with observation_id/physical_step/action plus latest Observation; payload_evidence context with obs, frame, target, grips, reader.tools

## Outputs

- Observation with observation_id=physical_step=step, copied state/images, per-camera SHA-256 evidence, and preflight metadata
- bind returns dict observation_id, physical_step, plain action
- validate_bound_action returns action dict or raises ValueError for stale/invalid actions
- normalized_feedback returns float; public_finger_openings returns copy of rig.maximum_openings; opening_hypotheses returns (zeros(2), nominal(2), maximum_openings copy); payload_evidence returns evidence dict

## Callable interface

Implementation: [implementation.py](implementation.py).

- `plain(value)`
- `Observation: create(cls, state, vision, step, previous_summary=''); bind(self, action)`
- `validate_bound_action(bound, latest)`
- `normalized_feedback(state, arm)`
- `public_finger_openings(state, arm, rig)`
- `opening_hypotheses(state, arm, rig)`
- `payload_evidence(context, arm, phase)`

## Constraints

- Rejects key allowlist mismatches, wrong shapes/nonfinite values, ee_joint_state outside [0,1], images not 480x640x3 uint8, invalid step, previous_summary >5000, or total >4,000,000 bytes; does not drop a camera
- Stores no historical images; preflight reports image_count 3, historical_images 0, API_context_tokens None, deterministic controller
- Normalized singleton feedback is previous control, not measured finger width/contact; payload_evidence sets measured_finger_pair False and measured_width_m None
- payload_evidence depends on adapter-supplied context and reader.tools.verify_digit_identity; missing tools or ValueError/RuntimeError returns base evidence, and a confirmed current RGB digit beyond .055 m sets state False; no official success is claimed
