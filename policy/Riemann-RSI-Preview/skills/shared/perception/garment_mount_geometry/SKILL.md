---
name: garment_mount_geometry
description: "Derives a garment reference datum from current own-robot roots and fits or retains RGB garment landmarks across sleeve and torso folds. Flat fitting, second-sleeve reuse and torso hem-only rereading preserve their distinct reference semantics."
---

# Garment mount geometry

## Function

Derives a garment reference datum from current own-robot roots and fits or retains RGB garment landmarks across sleeve and torso folds. Flat fitting, second-sleeve reuse and torso hem-only rereading preserve their distinct reference semantics.

## Inputs

- Public robot API, current action epoch, own joint/EEF state and the declared fixed URDF.
- Current head RGB and an optional previous garment anchor; read_with_datum also receives a validated mounting datum, reuse flag and phase.

## Outputs

- Validated own-root mounting datum and action/RGB provenance records.
- A garment observation with anchor, support memory, optional alternatives and observed/not_observed status.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `current_mounting_datum(api, epoch)`
- `provenance(api, epoch)`
- `torso_read_with_datum(api, previous, datum, image, empty, retry=True)`
- `read_with_datum(api, datum, previous=None, reuse=False, phase='flat')`

## Constraints

- The mean own-root Z is a static task datum, explicitly not an observed support surface; its surface error bound remains unknown.
- Torso rereading changes the two hem landmarks while retaining the previous garment axes and other landmarks; incoherent readings may retain the previous qualified anchor with a fallback reason.
- Flat fitting uses current segmentation and public templates, including bounded appearance retries. Retained appearance and landmark hypotheses do not prove cloth contact or official fold success.
