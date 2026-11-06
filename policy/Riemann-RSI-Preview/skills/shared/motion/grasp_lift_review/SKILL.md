---
name: grasp_lift_review
description: "Reviews a test lift by comparing the selected colour's unique RGB region and own end-effector translation before and after the 'test_lift_keep_orientation' phase; it records visual_lift_feedback and can mark the invoking program uncertain."
---

# Grasp lift review

## Function

Reviews a test lift by comparing the selected colour's unique RGB region and own end-effector translation before and after the 'test_lift_keep_orientation' phase; it records visual_lift_feedback and can mark the invoking program uncertain.

## Inputs

- observation passed to observe()
- api: public regions(colour), image(), and targets() for own end-effector pose
- request['colour'] and self.side from the invoking phase program
- before-lift snapshot captured only while the phase name is 'test_lift_keep_orientation'

## Outputs

- visual_lift_feedback['robot_translation_m'] set to measured end-effector translation in metres
- visual_lift_feedback status set to 'ambiguous', 'visible_change', or 'no_visible_follow' with head_displacement_fraction and head_area_ratio when evaluated
- self.status set to 'uncertain' for ambiguous or non-following evidence; no action or return value

## Callable interface

Implementation: [implementation.py](implementation.py).

- `GraspTestLift: __init__(self, *args, **kwargs); observe(self, observation)`

## Constraints

- requires exactly one prior and one current colour region and unchanged image shape; otherwise status is ambiguous and program status uncertain
- movement below 0.015 m or region-count or shape mismatch makes the result ambiguous; with valid regions, displacement >= 0.004 or abs(area ratio - 1) >= 0.10 gives visible_change, otherwise no_visible_follow and uncertain
- visible image change is not an attachment measurement, contact signal, or proof of task success
- issues no actions of its own and inherits the invoking phase program's budget
