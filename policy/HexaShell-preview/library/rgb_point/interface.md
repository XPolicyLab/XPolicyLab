## Tool: rgb_point

Use it to find the 3D world position of a point that you see in a camera image (a spot on a surface, the top of an object, a corner).

`robo rgb_point --u U --v V [--camera C] [one mode below]`

Free: nothing moves and no simulation time is used. U, V: pixel column and row in the latest image of camera C (default `head`). Cameras: `head`, `wrist_l`, `wrist_r` (`cam_head`, `cam_left_wrist`, `cam_right_wrist` also work).

Modes:

- Automatic (no other arguments): measures the height of the surface around the pixel from the other two cameras. It matches a small square patch around the pixel (`--radius R`, half size in pixels, 2–15, default 6) and assumes that this patch is flat and horizontal. If the patch is plain (almost one colour), it uses the outline of the one-coloured region that contains the pixel instead; that outline must lie in one horizontal plane (for example the flat top of something seen from above) and must not touch the image border. Heights between `--z_min` and `--z_max` are searched (m, default 0.50 and 1.05; the true height must lie inside).
- `--z Z`: the point lies on the horizontal plane at world height Z (m), for example the table height from `robo table_plane`. Nothing is measured; a wrong Z gives a wrong point.
- `--camera2 C2 --u2 U2 --v2 V2`: (U2, V2) is the same physical point in camera C2. The two pixel rays are intersected.
- `--reference R`: R is the `reference` text printed by `robo rgb_ray` for the same physical point, taken earlier from a different camera position (for example before you moved a wrist). The point must not have moved since then. Put R in single quotes.

Output: `point_world` [x, y, z] in m and `method`. Automatic mode also gives `matched_cameras` and, per camera, the result or the reason it failed (`pairs`). The two-ray modes give `ray_gap_m` (at most 0.003 m) and `crossing_sine` (at least 0.05).

`robo rgb_ray --u U --v V [--camera C]`

Free. Prints `reference`: the ray of this pixel (camera origin and direction) as JSON text, for `--reference` later.

Fails with `plan_ok` false, a `plan_fail_reason` and exit code 1 (nothing moved) when the pixel is outside the image, when no other camera gives a clear and unique match (`no_unique_rgb_surface_match`, `ambiguous_surface_match`, `nonplanar_or_ambiguous_patch`, `camera_surface_disagreement`; `diagnostics` gives the reason per camera), when the two rays miss each other by more than 3 mm (`rays_disagree`: the pixels are not the same point) or are nearly parallel.

Tips: pick a pixel well inside one flat face, at least `radius` pixels from its edges; sloped faces, edges and blank areas fail. A pixel on a top face gives the top surface, not the centre of the object. The result is not checked against the arms: a pixel on a gripper measures the gripper.
