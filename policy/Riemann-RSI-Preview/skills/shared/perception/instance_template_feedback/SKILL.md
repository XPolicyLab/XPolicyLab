---
name: instance_template_feedback
description: "Captures a bounded local masked template from an initial RGB image plus an empty background, then matches it in a current same-camera image and reports observed or unknown appearance persistence. matched_mask returns the observed footprint only when the match state is observed."
---

# Instance template feedback

## Function

Captures a bounded local masked template from an initial RGB image plus an empty background, then matches it in a current same-camera image and reports observed or unknown appearance persistence. matched_mask returns the observed footprint only when the match state is observed.

## Inputs

- initial, background and current: same-camera uint8 three-channel images with consistent channel order.
- mask: Boolean observed-instance mask of initial.shape[:2] with at least 20 true pixels.
- margin: bounded integer pixel search margin, default 14, range 0..64.
- reference from capture and observation from match for matched_mask.

## Outputs

- capture returns a reference dict or None when the mask is too small or empty-table similarity is at least 0.80.
- match returns state observed/unknown with reason, similarity, empty_similarity and foreground_similarity; bbox and translation are returned only when observed.
- matched_mask returns a Boolean HxW mask or raises ValueError when the observation is not observed.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `capture(initial, background, mask, *, margin=14)`
- `match(reference, current)`
- `matched_mask(reference, observation)`

## Constraints

- Search is local around the earlier bbox plus margin and requires explicit uint8 RGB/BGR images with matching dimensions and a Boolean mask.
- Observed requires similarity at least 0.90 and foreground excess at least 0.80; otherwise the result is unknown.
- Receives no environment, layout, scorer, task ID or object state; it cannot certify identity under arbitrary rotation, occlusion, contact, grasp or delivery.
- Similarities are RGB confidence bounds, not physical width or contact gates; ambiguity is UNKNOWN.

## Provenance

Motivated by PyRUA_RSI PR213's object-specific `still_there` and `find_lost`; uses the project masked-template method with an explicit empty-background control.
