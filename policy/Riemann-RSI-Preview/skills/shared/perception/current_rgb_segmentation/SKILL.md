---
name: current_rgb_segmentation
description: "Runs SAM3 on the three current 480x640 RGB images to return bounded per-camera segmentation regions, either native bottle/cup labels or prompted labels. Caches results only while all three current image byte hashes are unchanged and returns deep copies."
---

# Current rgb segmentation

## Function

Runs SAM3 on the three current 480x640 RGB images to return bounded per-camera segmentation regions, either native bottle/cup labels or prompted labels. Caches results only while all three current image byte hashes are unchanged and returns deep copies.

## Inputs

- api: exactly PublicRobotAPI; other types raise ExecutionError
- prompts for prompted(): list or tuple of 1-3 unique strings, each 1-128 printable characters
- current images: three uint8 RGB frames shaped 480x640x3 from the restricted API
- configuration: configs/native-perception.json and manifest declaring SAM3 and resource://models/sam3-robodojo-v21

## Outputs

- run(api): dict per camera with bottle and cup region lists; head cup may fall back to mug or wine glass prompts
- prompted(api,prompts): dict per camera of region lists
- regions: outline, bbox, area, centroid, confidence, clipped flag, camera, and source sam3_current_public_rgb_mask
- statistics(): inference, cache, retained prompt-set, and current-RGB-only counters

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_prompts(prompts)`
- `observed_regions(image, masks, scores, maximum=12)`
- `CurrentRGBSegmentation: __init__(self); current_frame(self, api); cached_current(self, key); remember_current(self, key, found); prompted(self, api, prompts); current(self, api)`
- `backend()`
- `run(api)`
- `prompted(api, prompts)`
- `statistics()`

## Constraints

- validation requires SAM3, maximum 12 regions per label, and the declared public model reference; missing GPU lease raises ExecutionError
- masks below confidence 0.45, areas outside 40 to 30% of the image, small boxes, or extra contours are filtered; each mask keeps up to two contours
- cache is cleared on any current camera pixel change, retains at most eight prompt sets, and returns deep copies; clipped regions are aiming-only
- module-level CAMERAS and cached backend singleton bind the adapter; model is lazy-loaded once and no history images, geometry, pose, or task identity are retained
