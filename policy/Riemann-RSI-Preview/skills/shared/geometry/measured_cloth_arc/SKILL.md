---
name: measured_cloth_arc
description: "Computes a bounded chunk of proposed own-wrist 4x4 targets along a constant-orientation semicircle using current visible displacement and declared fold angles. run validates the exact request keys, calls chunk_angles for 16 smoothstep samples, and applies fold_arc_wrist per angle."
---

# Measured cloth arc

## Function

Computes a bounded chunk of proposed own-wrist 4x4 targets along a constant-orientation semicircle using current visible displacement and declared fold angles. run validates the exact request keys, calls chunk_angles for 16 smoothstep samples, and applies fold_arc_wrist per angle.

## Inputs

- request: dict with exactly start_wrist, displacement_xy, start_angle, end_angle, chunk_index, chunk_count
- start_wrist: finite 4x4 own wrist frame with bottom row [0,0,0,1] and orthonormal determinant-1 rotation
- displacement_xy: finite 2- or 3-element current visible displacement; only first two components are used
- start_angle/end_angle: finite radians in [0, pi]; chunk_index int 0 <= index < chunk_count; chunk_count int 1..8

## Outputs

- dict with own_wrist_targets: 16 4x4 own-wrist targets as nested lists
- robot_actions_issued: 0
- public_cloth_lift_confirmed: False and official_success_claimed: False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `fold_arc_wrist(start, displacement, angle)`
- `chunk_angles(start_angle, end_angle, chunk_index, chunk_count)`
- `run(request)`

## Constraints

- fold_arc_wrist copies start orientation and translates XY by 0.5*(1-cos(angle))*displacement_xy; Z adds 0.012 + radius*sin(angle), where radius=0.5*norm(displacement_xy)
- chunk_angles samples fraction=(i+1)/16 and smooth=fraction^2*(3-2*fraction), yielding angle=start_angle+(end_angle-start_angle)*(chunk_index+smooth)/chunk_count
- Displacement must come from current visible hem/shoulder correspondence, not stored garment layout or physical-asset dimensions; the 0.012 vertical offset is own-command clearance, not measured cloth size
- Issues no robot action and does not attest cloth contact, lift or success; eventual stage must check original continuous own IK, two-arm separation, public lift uncertainty, release, withdrawal and home budgets
