---
name: focus_public_relation_view
description: "Proposes a raised wrist-camera view for an unresolved public scene relation, alternating preferred sides and selecting a camera from the recorded unqualified views."
---

# Focus public relation view

## Function

Proposes a raised wrist-camera view for an unresolved public scene relation, alternating preferred sides and selecting a camera from the recorded unqualified views.

## Inputs

- Public robot API and validated rigid-transfer progress with a view hint, support memory and own tool offset.

## Outputs

- A dictionary containing side and proposed pose, or None when the view preconditions or clearance checks fail.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, current)`

## Constraints

- Runs only in the view phase without a current scene, with fewer than four view attempts, available own bases and both issued gripper targets at least 0.9.
- Requires the proposed tool tip to remain at least 0.06 m above the retained support hypothesis. Returns a proposal without issuing robot commands.
