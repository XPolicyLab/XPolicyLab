---
name: container_return_contract
description: "Validates a bounded public container anchor and current release-image evidence for a container-return command. MINIMUM declares exact minimum step counts for named phases."
---

# Container return contract

## Function

Validates a bounded public container anchor and current release-image evidence for a container-return command. MINIMUM declares exact minimum step counts for named phases.

## Inputs

- anchor dict with cups list length labels, blocks dict keyed by labels, order permutation of labels, finite support_plane_m, and cap_geometry
- labels tuple default ('red','green','blue')
- release dict with status in ambiguous/visible_near_commanded_goal/not_at_visible_goal, contact_claimed False, official_success_claimed False, and head_distance_fraction for non-ambiguous statuses

## Outputs

- validate_anchor returns a deep-copied validated anchor
- validate_release returns the release dict unchanged
- MINIMUM maps cover_first/cover_second/cover_third/uncover_red/uncover_green/uncover_blue/home_left/home_right/settle to integer minimums

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_anchor(value, labels=('red', 'green', 'blue'))`
- `validate_release(value)`

## Constraints

- Cap geometry requires .005 < height_m - support_plane_m <= .3, size_m in [.005,.12], sorted unique observed_head_indices of length 2..3, and allowed wrist_views; fewer than three head indices requires both wrist views
- Cup/block goals must be finite 3-vectors with max absolute coordinate <=3
- Non-ambiguous release needs numeric finite head_distance_fraction in [0,2]; ambiguous must omit it; contact_claimed and official_success_claimed must both be False
- No current simulator pose, task identity, layout, contact, real finger value, or score is accepted; a home is an intended return goal, not a moving object's current location
