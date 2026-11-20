---
name: glyph_scene_observation
description: "Reads current RGB scene evidence to locate the fixed public pad row and detect/confirm descending digit identities, raising RecognitionNeedsObservation with a report and provisional obstacle scene when evidence is uncertain. Reader.reacquire re-verifies a given digit near expected_xy, while SceneObservation.read records events and writes detections.json."
---

# Glyph scene observation

## Function

Reads current RGB scene evidence to locate the fixed public pad row and detect/confirm descending digit identities, raising RecognitionNeedsObservation with a report and provisional obstacle scene when evidence is uncertain. Reader.reacquire re-verifies a given digit near expected_xy, while SceneObservation.read records events and writes detections.json.

## Inputs

- Reader(bank, material_profile='fixed', rig=None); bank is passed to DigitTools.
- obs with vision['cam_head'], observation_id and physical_step; frame with mounting_z and projection/pixel support.
- read(obs, frame, family=None, pads=None, inspection_xy=None); pad helpers accept optional expected_count.
- reacquire(obs, frame, digit, expected_xy, family=None, placed=False).

## Outputs

- dict with digits, pads, family=0, observation_id, physical_step and recognition_status='independently_confirmed'.
- Reader.last_report and RecognitionNeedsObservation carrying a report plus provisional scene with digits and pads.
- detections.json written by SceneObservation.read.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `current_identity_support(obj)`
- `Reader: __init__(self, bank, material_profile='fixed', rig=None); pad_ring_centers(self, image, frame, expected_count=None); pad_edge_centers(self, image, frame, expected_count=None); pad_centers(self, image, frame, expected_count=None); read(self, obs, frame, *, family=None, pads=None, inspection_xy=None); reacquire(self, obs, frame, digit, expected_xy, *, family=None, placed=False)`
- `SceneObservation: read(self, inspection_xy=None)`
- `read_scene(context)`

## Constraints

- Pad count is independent current-RGB evidence; fixed public spacing is 0.085 m with 4/5 counts, and pad memory reuses only independently measured fixed-geometry text, not old images for current digit recognition.
- Mismatched pad count/position, missing or duplicate digit identities, or insufficient pad-boundary support cause uncertain status and reobservation.
- In material_profile='random', weak features and non-clear features near pad centers are retained as weak/pad candidates rather than counted as confirmed digits; unconfirmed component positions are obstacles, never identities.
- SceneObservation.read and read_scene are adapter-bound: they depend on self.obs, self.frame, self.reader, self.event, self.directory or context.read.
