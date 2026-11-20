---
name: sam3_nesting_image_model
description: "image_configuration validates the canonical segmentation config and model manifest and returns config with cuda_device=0 plus manifest. PublicSAM3ImageModel authenticates pinned files, builds SAM3 on the leased CUDA device, and prompted_masks returns up to twelve masks and scores per appearance prompt."
---

# Sam3 nesting image model

## Function

image_configuration validates the canonical segmentation config and model manifest and returns config with cuda_device=0 plus manifest. PublicSAM3ImageModel authenticates pinned files, builds SAM3 on the leased CUDA device, and prompted_masks returns up to twelve masks and scores per appearance prompt.

## Inputs

- root: optional model root passed to image_configuration; defaults to Path(__file__).resolve().parents[4]
- rgb: explicit current public image with shape (480,640,3) and dtype uint8
- prompts: list of 1-3 appearance prompt strings, each length 1-64

## Outputs

- image_configuration: (config dict with cuda_device=0, manifest dict) after canonical config/manifest checks
- prompted_masks: list with one tuple per prompt; each tuple is Boolean masks shaped (<=12,480,640) and finite confidence scores

## Callable interface

Implementation: [implementation.py](implementation.py).

- `image_configuration(root=None)`
- `PublicSAM3ImageModel: __init__(self); prompted_masks(self, rgb, prompts)`

## Constraints

- Requires config schema 'riemann-current-rgb-segmentation-v1', model 'SAM3', maximum_regions_per_label 12, history_retained false, manifest 'configs/models/sam3.json', and resource reference 'resource://models/sam3-robodojo-v21'.
- Authenticates exactly declared manifest files by byte size and SHA-256 once per instance; rejects unsafe member paths and changed files; no directory or NAS scan.
- Builds SAM3 from local sam3.pt with load_from_HF=False on cuda:0; confidence_threshold is 0.45 and initialization errors propagate.
- PublicSAM3ImageModel methods depend on instance state (self.torch, self.device, self.model, self.processor); each prompted_masks call resets image/prompt state and discards it.
