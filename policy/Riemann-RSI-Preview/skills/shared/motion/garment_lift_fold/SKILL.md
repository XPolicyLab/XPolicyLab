---
name: garment_lift_fold
description: "Runs one lift/carry stage: for part 2 it replays the initial paired body semicircle, otherwise it raises both wrist goals by .075 m through sampled_wrist_path. Completion review uses lifted_surface_feedback to choose carry, bounded continuation, or recover with uncertain status."
---

# Garment lift fold

## Function

Runs one lift/carry stage: for part 2 it replays the initial paired body semicircle, otherwise it raises both wrist goals by .075 m through sampled_wrist_path. Completion review uses lifted_surface_feedback to choose carry, bounded continuation, or recover with uncertain status.

## Inputs

- api PublicRobotAPI and request with a validated sixteen-command allowance
- recorded same-instruction state from observed_garment_mount_state
- current part, route, contact_poses, destinations, contacts, and start_wrists

## Outputs

- LiftGarment program with sixteen controls or an unavailable result
- updated phase and status, including carry, recover, or uncertain

## Callable interface

Implementation: [implementation.py](implementation.py).

- `LiftGarment: review_completion(self)`
- `run(api, request)`

## Constraints

- Requires allowed(current, 'lift') and non-None start_wrists
- Part 2 anchors the body arc on route contact_poses rather than start_wrists; other parts add .075 m to the target wrist z position
- Fresh head and wrist image evidence is required before transporting the folded surface; missing evidence remains uncertain
- LiftGarment methods rely on inherited GarmentProgram self/current/status; do not substitute asset dimensions, object pose, contact sensor, or official success flag
