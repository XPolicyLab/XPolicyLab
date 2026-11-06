---
name: public_dense_plane_alignment
description: "Estimates a horizontal support-plane hypothesis by aligning current head/wrist image texture over candidate heights. Preserves the qualified ordinary alignment, then tries colour-independent patch matching and an antialiased fallback only if needed."
---

# Public dense plane alignment

## Function

Estimates a horizontal support-plane hypothesis by aligning current head/wrist image texture over candidate heights. Preserves the qualified ordinary alignment, then tries colour-independent patch matching and an antialiased fallback only if needed.

## Inputs

- Public robot API with current native uint8 RGB from head and both wrists, camera geometry and own-robot bases.
- Explicit Boolean colour-independent and antialias modes for lower-level helpers.

## Outputs

- Observed or uncertain status, optional plane normal/distance/height uncertainty, per-camera alignment records and qualification reason.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `image_fields(image, color_independent=False, antialiased=False)`
- `plane_projection(api, camera, support_height)`
- `ranked_correlation(source, destination, warp, robust=False, for_search=False)`
- `correlation(source, destination, warp, robust=False)`
- `qualified_alignments(records)`
- `current_alignment(api, color_independent=False, antialiased=False)`
- `run(api)`

## Constraints

- Searches 31 heights within 0.15 m of the own mounting reference, then 25 heights within 0.012 m of each coarse optimum. Mounting height centres the search; it does not supply the surface answer.
- Requires a unique bounded correlation peak and agreement from two qualified camera pairs, or one strong pair with independent camera corroboration. Missing texture or camera reference stays uncertain.
- Correlation scores describe image alignment and official_success_claimed remains False.
