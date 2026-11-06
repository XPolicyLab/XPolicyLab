---
name: visible_block_grasp
description: "VisibleBlockGrasp builds a grasp-lift controller only when the instruction mentions blocks or cubes and the requested colour has a current RGB surface reconstruction and trajectory availability. It uses PublicVerticalGraspSelector and grasp_phases to form open, approach, descend, close, settle and test-lift phases, and degrades to precondition or uncertain states when geometry or reachability is missing."
---

# Visible block grasp

## Function

VisibleBlockGrasp builds a grasp-lift controller only when the instruction mentions blocks or cubes and the requested colour has a current RGB surface reconstruction and trajectory availability. It uses PublicVerticalGraspSelector and grasp_phases to form open, approach, descend, close, settle and test-lift phases, and degrades to precondition or uncertain states when geometry or reachability is missing.

## Inputs

- api: restricted adapter with instruction(), targets(), geometry_memory() and trajectory_available()
- request: colour, side in auto/left/right, and steps action allowance
- instruction text: used to detect block/cube semantics and instructed stack roles
- current public RGB surface geometry for the requested colour and support-plane hypothesis

## Outputs

- run(api, request) returns a VisibleBlockGrasp controller
- grasp_selection: reachable event or no_reachable_candidate with requested_side and candidate_limit
- observe(observation) updates public_geometry_update only when finished and status completed
- status/finished/failed reflect precondition, uncertainty or completion states

## Callable interface

Implementation: [implementation.py](implementation.py).

- `VisibleBlockGrasp: __init__(self, api, request); observe(self, observation)`
- `run(api, request)`

## Constraints

- Requires steps >= 22; otherwise status becomes precondition_unmet and finished true.
- Protects an instructed stack bottom colour from this cube-specific grasp and rejects non-block/cube instructions via semantic check.
- All phase poses must be within 0.65 translation of the initial pose; pose_tolerance_m is 0.05.
- Exhausted reachability sets status uncertain and failed; missing geometry is a precondition failure; closed commands or image change do not certify attachment or contact.
