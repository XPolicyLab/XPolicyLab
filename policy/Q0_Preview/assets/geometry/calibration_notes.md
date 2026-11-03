# Geometry quick reference

Use this file for units, common calculations and fixed robot dimensions.
Full helper contracts are in `/work/geometry_interfaces.md`; reference measurements are
in `/work/geometry_evidence.md`. Read only the file needed for the current decision.
Each file fits in one tool response when printed alone.

The helpers use supplied observations and caller-selected pixels or points. They do
not locate objects or check task success. Current gripper state is command feedback,
not measured opening or contact. Robot pose accuracy does not establish object pose
or contact-point accuracy.

### Inputs, outputs and units

`np`, `Image`, `plt`, `obs` and `geom` are preloaded. Use `print()` to return values.
World positions/displacements are finite numeric three-vectors in metres; lists and
NumPy arrays are accepted. Pixel coordinates are `(u,v)=(x,y)` in the original image,
with origin at the upper left. NumPy indexing is `[y,x]`. Rotation vectors use radians.
Camera names: `cam_head`, `cam_left_wrist`, `cam_right_wrist`.

### Common calculations

Use `obs.state()` for full-precision robot state and calibration. Build a pose target
with `geom.target_from_state(obs.state()['state']['left'])` (or `'right'`), then use
`geom.translate_points` or `geom.rotate_points`; preserve the physical finger order.
Use `geom.grasp_reference_for` to place the intended finger region at a working point.

For image geometry, use `geom.camera_ray` and `geom.project_camera` with a camera name.
Use `geom.plane_pixel_scale` for a known surface; `geom.pixel_scale` depends on depth.
For two-view triangulation, match the same feature and verify its reprojection in
both images. A small ray residual alone does not validate the match.
For a historical image, pass the same `observation_id` to the camera helper.

## Fixed geometry and its source

These dimensions came from the ARX X5 robot mesh and robot-only calibration probes.
The archived `robot_mesh_metrology.json` uses the same `ARX.usd` asset as the current
checkout: SHA-256 `5723a835bf5627a935a8dabb3a44447725bd1465243ce791e221b3e990121d63`.
The robot configuration sets `gripper_bias=0.145` m and finger joint range
`[-0.01,0.044]` m. Geometry values are not per-scene object measurements.

### Reference frames and dimensions

The flange frame has axes `(+X tool, +Y jaw, +Z tool cross jaw)`.
The grasp reference is flange-local `[0.145,0,0]` m. Offsets below are relative to
the grasp reference in the same local axes:

| Finger-region midpoint | Offset in metres |
| --- | --- |
| Front tip | `[0.01257,-0.000002,-0.00014012]` |
| Distal flat-pad bounding box | `[0.00507,-0.000002,-0.00144044]` |

`grasp_reference_for` applies these full offsets, including their small lateral
components. Tip and pad references differ by 7.5 mm along the tool axis; choose the
region that fits the object's thickness and available support clearance. Putting
the pad centre on a very thin object can place the foremost tips below its support.
The pad bounding-box centre is a geometric reference, not a contact-pressure estimate.

| Additional dimension | Approximate value |
| --- | --- |
| Distal flat inner face, flange-local X | 0.14257–0.15757 m |
| Flat-face axial length | 15.00 mm |
| Flat-face rear edge behind grasp reference | 2.43 mm |
| Front edge width / jaw-direction thickness | 10.00 / 1.50 mm |
| Finger thickness / lateral width at flange X=0.145 m | 5.61 / 16.98 mm |
| Finger thickness / lateral width at flange X=0.150 m | 3.98 / 14.20 mm |
| Complete finger extent: tool / jaw / lateral | 86.00 / 37.35 / 61.00 mm |

A visible bend or throat need not coincide with the flat-face rear edge. The wider
rear finger structure is included in the complete extent above.

### Opening and table baseline

Full-open distal inner gap is about 88.8 mm. Historical empty-close probes stopped
near 0.8 mm. For a hypothetical physical normalized opening `g`, the clamped finger
position is `q_mm=max(0,-10+54*g)` and mesh gap is approximately `2*q_mm+0.807` mm.
The current command-derived nominal gap is instead
`max(0, 2*(-10+54*clip(g,0,1)) + 0.800)` mm. In particular, command zero reports zero
nominal gap; it does not report the physical empty-stop gap. The 0.007 mm mesh/rounded
offset difference applies at corresponding joint positions. No measured gap is
available to the current policy.

The evaluated scene configuration sets table centre Z to 0.740 m and thickness to
0.050 m, giving the fixed top plane `z=0.740+0.050/2=0.765` m. This is the support
surface, not the unknown top of an object resting on it.

