---
name: garment_lower_contact
description: "Reuses or plans the same garment contact route and produces a sixteen-command Cartesian descent to its contact poses, with part 2 refining hem contacts from current wrist RGB first. If Cartesian sampling fails, it falls back to a bounded joint interpolation to solved own-joint endpoints."
---

# Garment lower contact

## Function

Reuses or plans the same garment contact route and produces a sixteen-command Cartesian descent to its contact poses, with part 2 refining hem contacts from current wrist RGB first. If Cartesian sampling fails, it falls back to a bounded joint interpolation to solved own-joint endpoints.

## Inputs

- api PublicRobotAPI and request with a validated sixteen-command allowance
- recorded state containing part, anchor, and route
- current garment landmarks and current_garment_contacts route output when no qualified route is present

## Outputs

- GarmentProgram with sixteen controls and following set to close, or an unavailable result
- updated anchor and route in the recorded state

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- Requires allowed(current, 'contact') and a planned route whose arms match the recorded route
- Does not re-read the head image when a qualified anchor/route is already recorded; recovery performs a fresh view only after an actual lower failure
- Fallback solves contact goals with position_tolerance .006 and uses sampled_joint_path within the same sixteen-command bound
- Missing current contact or own endpoint error requests bounded recovery; normalized jaw commands are not physical finger measurements
