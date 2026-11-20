---
name: public_region_stability
description: "run validates native current RGB, an explicit current-image boolean region and optional same-format previous RGB, then counts pixels whose any RGB channel differs by more than 15 inside the region. It returns None when previous is None."
---

# Public region stability

## Function

run validates native current RGB, an explicit current-image boolean region and optional same-format previous RGB, then counts pixels whose any RGB channel differs by more than 15 inside the region. It returns None when previous is None.

## Inputs

- current: 480x640x3 uint8 public RGB
- previous: None or same-shape, same-dtype public RGB
- region: 480x640 bool mask in current-image coordinates; must be nonempty

## Outputs

- int count of changed pixels inside region
- None when previous is None
- ValueError on invalid shapes, dtypes, non-native images or empty region

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(current, previous, region)`

## Constraints

- Shape must be exactly 480x640x3 for current and region 480x640 bool
- Previous must match current shape and dtype
- Change threshold is abs difference >15 on any channel
- The count is not contact, object state or success and does not replace an independent visibility check
