---
name: garment_recover_contact
description: "Performs bounded recovery for only the selected own arms after uncertain visual following or tracking: it opens and retreats by .08, .04, or .02 m, then falls back to the qualified approach joints. On completion it resets the route to a fresh view while retaining the anchor."
---

# Garment recover contact

## Function

Performs bounded recovery for only the selected own arms after uncertain visual following or tracking: it opens and retreats by .08, .04, or .02 m, then falls back to the qualified approach joints. On completion it resets the route to a fresh view while retaining the anchor.

## Inputs

- api PublicRobotAPI and request with a validated sixteen-command allowance
- recorded state with phase recover, a non-None route, and attempts below 3
- route arms and approach_joints

## Outputs

- RecoverGarment program with sixteen controls or an unavailable result
- updated phase view with route=None, start_wrists=None, chunk=0, and attempts=0

## Callable interface

Implementation: [implementation.py](implementation.py).

- `RecoverGarment: observe(self, observation); review_completion(self)`
- `run(api, request)`

## Constraints

- Unavailable unless phase is recover, route is present, and attempts are less than 3
- Only the selected route arms are opened and retreated
- Retreat order is .08, .04, .02 m before joint fallback; the joint fallback still uses the native sixteen-command clock
- RecoverGarment methods rely on inherited GarmentProgram self/current/status; a completed recovery is the bounded retry
