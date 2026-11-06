---
name: rgb_text_masks
description: "ArraySegmenter loads an authenticated pretrained SAM3 model from an explicit manifest, runs prompted segmentation on an explicit native RGB array, and returns at most eight deduplicated Boolean masks with confidence scores and prompts. The module-level run uses a cached backend keyed by manifest path."
---

# Rgb text masks

## Function

ArraySegmenter loads an authenticated pretrained SAM3 model from an explicit manifest, runs prompted segmentation on an explicit native RGB array, and returns at most eight deduplicated Boolean masks with confidence scores and prompts. The module-level run uses a cached backend keyed by manifest path.

## Inputs

- image: explicit uint8 RGB array with shape (480, 640, 3)
- prompts: list or tuple of 1 to 3 semantic description strings, each length 1 to 128
- manifest_path: explicit SAM3 manifest path
- minimum_score: keyword-only .45 default or .2 weaker-candidate threshold

## Outputs

- list of at most 8 dicts with mask as a 480x640 Boolean array, score as float, and prompt string
- empty list when no masks meet score, area, or overlap filters

## Callable interface

Implementation: [implementation.py](implementation.py).

- `ArraySegmenter: __init__(self, manifest_path); run(self, image, prompts, *, minimum_score=0.45)`
- `backend(manifest_path)`
- `run(image, prompts, manifest_path, *, minimum_score=0.45)`

## Constraints

- Manifest must declare model SAM3; every listed file is SHA-256 verified inside the resolved asset root, with no directory walk, download, or install.
- Requires cuda:0; the model loads once through lru_cache(maxsize=1) and segmentation uses inference_mode with CUDA autocast bfloat16.
- Accepts only minimum_score .45 or .2; masks require score >= threshold, 30 <= area <= 30% of 640x480, and IoU <= 0.8 against earlier results, and processing stops after the first prompt that yields results.
- Masks are uncertain visual regions, not object identity or metric geometry; lower-confidence masks must not by itself establish task identity, possession, metric pose, or successful execution.
