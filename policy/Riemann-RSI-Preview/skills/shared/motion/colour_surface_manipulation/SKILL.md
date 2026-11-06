---
name: colour_surface_manipulation
description: "Constructs validated grasp, carry, and placement phase lists from image-derived surface, source, and support data, using a downward tool frame and declared fingertip offset. Carry planning attempts a closed-orientation transfer and falls back to low, midpoint, and destination carry phases only on a cuRobo plan_path failure."
---

# Colour surface manipulation

## Function

Constructs validated grasp, carry, and placement phase lists from image-derived surface, source, and support data, using a downward tool frame and declared fingertip offset. Carry planning attempts a closed-orientation transfer and falls back to low, midpoint, and destination carry phases only on a cuRobo plan_path failure.

## Inputs

- surface: mapping with point, edge_m, and yaw_radians; initial is a seven-value XYZ-plus-WXYZ pose; optional selection supplies grasp, engagement, and approach copies
- grasp_phases(initial, surface, selection=None, *, lift_clearance=0.055): optional lift clearance
- placement_phases(initial, source, support, engagement, api=None, request=None, plane=None, *, carry_clearance=None): source and support mappings with point and edge_m; engagement is measured face engagement
- CarryRecipe(api, request, plane): api must expose trajectory_available() and plan_trajectory(side, pose, plane, steps, 1.4); request provides side and steps

## Outputs

- wrist_for_surface returns a seven-value wrist pose and the engagement computed as surface point z minus adjusted point z
- grasp_phases returns a phase list ordered open, approach, descend, close command, settle command, test lift, plus engagement
- placement_phases returns a phase list, the support tip height support point z + source edge_m, and the CarryRecipe preflight result

## Callable interface

Implementation: [implementation.py](implementation.py).

- `wrist_for_surface(surface, orientation=None)`
- `grasp_phases(initial, surface, selection=None, *, lift_clearance=0.055)`
- `CarryRecipe: __init__(self, api, request, plane)`
- `placement_phases(initial, source, support, engagement, api=None, request=None, plane=None, *, carry_clearance=None)`

## Constraints

- Commands use grip values 0 or 1 as commands only; they do not prove attachment, released support, or task success, so callers must inspect visible attachment and released support.
- Default grasp orientation is [0.5, -0.5, 0.5, 0.5] rotated about z by surface yaw_radians; wrist xyz shifts by 0.155 along rotation_matrix(orientation)[:, 0], and engagement is measured face z minus adjusted point z.
- Carry fallback triggers only on RuntimeError containing 'cuRobo plan_path failed'; _carry_upright requires abs(dot(source_quat, destination_quat)) >= 0.9999, and placement clearance defaults to 0.035 or 0.055 from support and source heights unless carry_clearance is supplied.
- Functions construct bounded phases only; if api is None or trajectory_available() is false preflight stays None; they do not query an environment, select a task policy, read a scorer, or report official success; a stalled pose aborts rather than releasing off-target.
