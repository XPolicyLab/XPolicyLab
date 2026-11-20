---
name: active_surface_view
description: "Builds a bounded own-wrist view program for a selected colour from the current head image and, only after a completed three-phase view, attempts scene geometry from the fresh RGB and measured wrist pose. If the colour target lacks stereo geometry, it marks status uncertain and measurement_ambiguous instead of accepting a pose."
---

# Active surface view

## Function

Builds a bounded own-wrist view program for a selected colour from the current head image and, only after a completed three-phase view, attempts scene geometry from the fresh RGB and measured wrist pose. If the colour target lacks stereo geometry, it marks status uncertain and measurement_ambiguous instead of accepting a pose.

## Inputs

- api: public adapter providing image(), geometry_memory(), and the PhasedProgram/measurement interfaces used by imported helpers
- request: dict with side ('left' or 'right') and colour used to build view phases

## Outputs

- ActiveSurfaceView instance with public_geometry_update (deep-copied geometry memory plus support_plane_m) or None, reviewed flag, status, and measurements

## Callable interface

Implementation: [implementation.py](implementation.py).

- `ActiveSurfaceView: __init__(self, api, request); observe(self, observation)`
- `run(api, request)`

## Constraints

- Both arms must have open public commands and the chosen wrist must be near observed home before aiming; do not scan while possibly carrying an object or near a stack.
- Camera aiming uses a provisional own-robot reference only to change the view; it is not a measured object position, depth estimate, or grasp destination.
- Geometry is accepted only after consistent triangulation from the new current RGB and measured own wrist pose; a changed view without such evidence remains uncertain.
- No environment render call, depth, object pose, task metadata, extra camera, or unbudgeted observation is available; motion and dwell consume the episode action budget.
