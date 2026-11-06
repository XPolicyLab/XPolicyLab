---
name: garment_carry_fold
description: "On each carry request, builds the next chunk for either a paired-hem body arc anchored at route.contact_poses or a lateral cuff carry anchored at current.start_wrists, selecting the chunk count on chunk 0 and interpolating remaining cuff goals from the current target. Returns a CarryGarment whose review advances chunks, skips intermediate visual review, and performs the final cuff lifted_surface_feedback check."
---

# Garment carry fold

## Function

On each carry request, builds the next chunk for either a paired-hem body arc anchored at route.contact_poses or a lateral cuff carry anchored at current.start_wrists, selecting the chunk count on chunk 0 and interpolating remaining cuff goals from the current target. Returns a CarryGarment whose review advances chunks, skips intermediate visual review, and performs the final cuff lifted_surface_feedback check.

## Inputs

- api: PublicRobotAPI
- request: stage request with skill and steps
- current: recorded public state with phase, attempts, part, chunk, route.arms/contacts/destinations, and start_wrists
- route.contact_poses for part 2; api.targets() and own_joint_states() for pose feedback

## Outputs

- CarryGarment with controls for the next body-arc or wrist-path chunk
- unavailable GarmentProgram when the carry gate fails or sampling raises ValueError
- mutations to current.route.body_chunks or carry_chunks, current.chunk, current.phase, current.status, and self.visual during review

## Callable interface

Implementation: [implementation.py](implementation.py).

- `CarryGarment: review_completion(self)`
- `run(api, request)`

## Constraints

- allowed requires current.phase == 'carry', current.attempts < 3, a non-null route, and current.start_wrists is not None
- Cuff goals add 0.075 m in z to start_wrists and apply XY destinations minus contacts; body goals use contact_poses and the same XY displacement
- Intermediate chunks return without visual review; part 2 goes directly to lower after its final chunk, while the final cuff chunk requires lifted_surface_feedback status 'current_surface_follows_own_pad' or continues to lower only when continue_after_shortfall() is true, otherwise sets recover/uncertain
- review_completion is adapter-bound to self.current/self.api/self.visual; missing evidence remains uncertain and no asset dimensions, object pose, contact sensor, or official success flag are substituted
