---
name: current_instance_collection
description: "Collects up to 12 deduplicated current metric instances for one to three bounded semantic prompts, optionally overwriting each anchor description with a canonical label. Also maintains support memory, derives a missing-head hint, and selects a qualified upper opening from receiver regions."
---

# Current instance collection

## Function

Collects up to 12 deduplicated current metric instances for one to three bounded semantic prompts, optionally overwriting each anchor description with a canonical label. Also maintains support memory, derives a missing-head hint, and selects a qualified upper opening from receiver regions.

## Inputs

- api: restricted public robot API used by the shared support, instance, and opening implementations
- prompts: list of 1-3 bounded semantic description strings; other lengths raise ValueError
- observed_support: dict with height_estimate_m and height_uncertainty_m in meters
- receiver/diagnostics: receiver dict with regions and anchor height; diagnostics list passed to instance queries

## Outputs

- support(api, previous): support memory dict; remember_support mutates current['support_memory'] and returns the observed memory
- collect(...): list of at most 12 instance items with anchor, surface, and associated regions; centers within 0.015 m are deduplicated
- missing_hint(...): first unqualified head hint dict with source current_public_head_semantic_hint, or None
- upper_opening(...): highest qualified boundary frame or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `support(api, previous=None)`
- `remember_support(api, current)`
- `collect(api, prompts, observed_support, canonical=None, require_supported=True, diagnostics=None)`
- `missing_hint(diagnostics, receiver_outlines=())`
- `upper_opening(api, receiver, observed_support)`

## Constraints

- prompts must be a list of length 1-3; canonical, when provided, mutates item['anchor']['description']
- deduplication uses Euclidean center distance <0.015 m and the result is truncated to 12
- closed-mask-line boundaries require current_upper_edges_agree true and z separation from support of at least 3 times combined position/height uncertainty
- other openings require boundary z at or above support height plus 0.45 times receiver anchor height
