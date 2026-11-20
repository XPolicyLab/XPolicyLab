---
name: material_glyph_outline
description: "Segments current RGB views, ranks all ten public digit outlines with planar templates and optional public USD extrusion, and associates independently ranked views into confirmed, uncertain, or view_conflict digit records. verify_digit_identity uses expected_xy only for anchoring and expected_digit only after candidate detection, raising RecognitionNeedsObservation when current RGB cannot verify the expected identity."
---

# Material glyph outline

## Function

Segments current RGB views, ranks all ten public digit outlines with planar templates and optional public USD extrusion, and associates independently ranked views into confirmed, uncertain, or view_conflict digit records. verify_digit_identity uses expected_xy only for anchoring and expected_digit only after candidate detection, raising RecognitionNeedsObservation when current RGB cannot verify the expected identity.

## Inputs

- Current 640x480 RGB images keyed by CAMERAS, with camera rig/calibration, frame.mounting_z, measured robot state, observation_id, physical_step, and evidence.
- Digit bank records and masks for all ten public digits, plus material_profile 'fixed' or 'random'.
- Observed component data with pixel mask, bbox, centroid, hue, and center_xy; a placed flag selects support height and template yaw range.
- For verify_digit_identity: expected_digit and expected_xy, where expected_xy restricts anchors and expected_digit is excluded from candidate generation and ranking.

## Outputs

- Detection report dict with score_semantics='template_similarity_not_probability', objects containing status confirmed/uncertain/view_conflict, per-view candidates and reasons, topology, occlusion fractions, component_counts, thresholds, and edge_proposals.
- Successful verification returns a detection dict plus report with status 'verified'; otherwise RecognitionNeedsObservation carries status identity_conflict, uncertain, or pose_uncertain and a reobservation request.
- request_reobservation returns suggested_clear_arms, remaining_native_controls=max(0,1050-physical_step), next_input string, and action_execution='existing_checked_active_vision_generator_only'.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `RecognitionNeedsObservation: __init__(self, message, report, scene=None)`
- `components(mask, min_area=20)`
- `material_mask(image, extended=False)`
- `hole_areas(mask)`
- `DigitTools: __init__(self, bank, rig=None, material_profile='fixed'); segment(self, image, camera, frame, state, *, placed=False); random_segment(self, image, camera, frame, state, placed=False); segment_mask(self, mask, hsv, camera, frame, state, placed=False); project_candidate(self, candidate, camera, plane_z); volume_rank(self, item, camera, frame, placed=False); rank(self, item, camera, frame, state, *, placed=False, occlusion=None); detect_digit_candidates(self, obs, frame, *, expected_xy=None, placed=False, inspection_xy=None); request_reobservation(self, report, reason); verify_digit_identity(self, obs, frame, expected_digit, expected_xy, *, placed=False)`
- `qualify_public_finger_overlap(views)`

## Constraints

- DigitTools is stateful and bound to self.bank, self.rig, and shared imports; construction requires exactly ten public digit shapes, duplicate material-family records must have identical masks, and invalid material_profile raises ValueError.
- Planar ranking searches yaw angles -60..60 degrees unplaced and -90..90 degrees placed; public_USD_extrusion is used only under the coded low-similarity or low-margin gate, and its winner must remain among the top two planar digits.
- Finger occlusion is recorded as public_opening_hypotheses_not_measured_width; qualify_public_finger_overlap clears only robot_finger_occlusion under near-normal planar gates with independent reliable camera agreement and does not establish width, contact, or grasp success.
- Scores are template similarities, not probabilities; confirmed means current-RGB consensus only, no historical view is used, and the tool cannot execute robot actions.
