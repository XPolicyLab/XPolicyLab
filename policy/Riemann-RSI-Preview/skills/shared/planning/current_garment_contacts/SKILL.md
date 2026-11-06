---
name: current_garment_contacts
description: "Builds a validated own-pad contact route for one bounded garment stage from current anchor landmarks: part 0/1 plans one cuff and part 2 plans paired hem contacts with a searched separated body fold. Contact candidates use five wrist tilts, two closing signs, real inverse solves and paired left-by-right cost search; refine re-plans paired hem contacts at the measured open approach and returns a validated route plus sampled controls or None."
---

# Current garment contacts

## Function

Builds a validated own-pad contact route for one bounded garment stage from current anchor landmarks: part 0/1 plans one cuff and part 2 plans paired hem contacts with a searched separated body fold. Contact candidates use five wrist tilts, two closing signs, real inverse solves and paired left-by-right cost search; refine re-plans paired hem contacts at the measured open approach and returns a validated route plus sampled controls or None.

## Inputs

- api: bound adapter supplying own_joint_states(), targets() and helpers used by imported kernels
- anchor: current garment landmarks including lateral_axis, longitudinal_axis, center, height_m and named cuff/hem/shoulder/chest points
- part: integer 0 for left cuff, 1 for right cuff, 2 for paired hem/body
- refine route: arms, contacts, contact_poses and destinations used with anchor

## Outputs

- validated route dict with sorted arms, contacts, destinations, contact_poses, approach_joints, carry_chunks=2 and body_chunks=2
- refine returns (validated route, sampled_wrist_path controls) or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `rotated_axis(axis, degrees)`
- `contact_choices(api, model, side, contact, closing, destination, body)`
- `contact_choice(api, model, side, contact, closing, destination, body)`
- `separated_arc(model, roots, left, right)`
- `retry_contacts(api, model, route, points)`
- `refine(api, route, anchor)`
- `run(api, anchor, part)`

## Constraints

- part must be an int 0..2; missing current outline support raises ValueError
- Body planning searches nine closing-axis rotations, sorts the left x right candidate product by cost, and requires wrist clearance >=.105 m plus sampled fold-arc separation
- Cuff planning shifts 22 mm inward, uses interior_contact, and searches rotated closing axes only when the initial axis yields no options; right cuff destination is shifted .035 m along the lateral axis
- contact_choices uses tilts (0,20,-20,40,-40) degrees and signs -1/+1, rejects |high_joints[4]|>2.4, and validates body fold-arc joint delta <=.22; endpoint qualification is not a collision or contact certificate
