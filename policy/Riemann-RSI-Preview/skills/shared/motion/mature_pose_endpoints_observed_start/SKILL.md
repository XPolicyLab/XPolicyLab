---
name: mature_pose_endpoints_observed_start
description: "Extends the observed-start PhasedProgram with an own-EE translation endpoint check and one same-target retry phase named with an _endpoint_retry suffix. It inserts the retry after a stalled endpoint when budget allows, otherwise reports uncertain/failed, and adjusts cycle accounting for repeated press_down retries."
---

# Mature pose endpoints observed start

## Function

Extends the observed-start PhasedProgram with an own-EE translation endpoint check and one same-target retry phase named with an _endpoint_retry suffix. It inserts the retry after a stalled endpoint when budget allows, otherwise reports uncertain/failed, and adjusts cycle accounting for repeated press_down retries.

## Inputs

- MaturePoseEndpoints(*args, **kwargs) forwarding to observed-start PhasedProgram
- observation: public observation passed to api.observe and used for own EE error check
- request: must include steps; phases must include name, require_pose, minimum_steps, and pose
- optional recover_plan_failure(observation) method on the adapter

## Outputs

- inserted phase copy named <phase>_endpoint_retry after a stalled non-retry endpoint, or uncertain/failed status when retry cannot be inserted
- record_action decrements issued_cycles when a press_down_*_endpoint_retry phase issues its first action
- status, finished, failed, and phase reports via inherited PhasedProgram

## Callable interface

Implementation: [implementation.py](implementation.py).

- `MaturePoseEndpoints: __init__(self, *args, **kwargs); record_action(self); observe(self, observation)`

## Constraints

- pose_tolerance_m is 0.05 and rotation_tolerance_radians is pi; observe aborts on translation distance > 0.05 m only after trajectory_pending is false and minimum_steps reached
- one same-target retry per endpoint phase; no newly invented retreat; insertion requires fewer than 40 phases and request['steps'] - actions >= 2
- a retry phase name ending _endpoint_retry first tries recover_plan_failure if available, else marks uncertain/failed
- endpoint feedback is not object contact, identity, grasp, or official success; repeated down-target repair is not another completed button cycle
