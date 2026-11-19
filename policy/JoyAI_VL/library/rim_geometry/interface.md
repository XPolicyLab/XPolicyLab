## Tool: rim_geometry

Use it to measure a level circle in 3D (the rim of a round container, a round opening or hole, the top edge of a cylinder or disc): its centre, radius and height.

`robo fit-rim --views '{"head":[[u,v],...],"wrist_r":[[u,v],...]}' --zmin Z --zmax Z [--tolerance 0.003] [--snap 4]`

Free: nothing moves and no simulation time is used.

- `--views`: 2 or 3 cameras (`head`, `wrist_l`, `wrist_r`), each with 6–64 different pixels on the same physical circular edge. The pixels do not have to be the same points in the different views, but in each view they must be spread around the circle, with no gap larger than half of it. The `points` of a `robo boundary_candidates` candidate can be used as they are. The cameras must be at least 3 cm apart.
- `--zmin`, `--zmax`: world heights in m between which the circle lies, for example the table height and 0.30 m above it. The best height must lie inside this range, which may be at most 1 m wide.
- `--snap` (0–6 pixels, default 4): first moves each pixel to the nearest strong image edge within this distance, across the edge. Every pixel needs such an edge. 0 uses the pixels as given.
- `--tolerance` (0.0005–0.01 m, default 0.003): largest allowed distance of a point from the fitted circle; also at most 10% of the radius.

Output: `center_world` [x, y, z] in m (z is the height of the edge), `radius_m`, `normal_world`, `max_residual_m`, `rms_residual_m`, `cameras` and, per camera, the snapped pixels (`refinement`).

The circle must be level and have a radius of 5 mm to 20 cm. Fails with `geometry_fit_failed` and a `plan_detail` (exit code 2, nothing moved), for example: the pixels are not on one edge or not on the same edge in each view (`inconsistent_multiview_boundary`, `inconsistent_boundary`); they cover too little of the circle (`insufficient_boundary_coverage`, `insufficient_view_coverage`); the best height is at the end of the range (`height_outside_bracket`: widen the range); two heights fit about equally well (`ambiguous_height`: use cameras that see the circle from more different directions); no edge near a pixel when snapping (`boundary_not_supported_by_rgb`: use better pixels or `--snap 0`).
