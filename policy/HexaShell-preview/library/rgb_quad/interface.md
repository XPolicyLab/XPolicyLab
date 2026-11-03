## Tool: rgb_quad

Use it to measure a flat face with four straight sides (the top of a box, a tray, a pad or a lid) in 3D: its corners, centre, side lengths, direction and tilt.

`robo rgb_quad --regions '[[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax]]' [--camera head] [--other wrist_r]`

Free: nothing moves and no simulation time is used. The first rectangle is in the image of `--camera` (default `head`), the second in the image of `--other` (default `wrist_r`); the two cameras must differ. Cameras: `head`, `wrist_l`, `wrist_r`. Rectangles are in pixels, at least 16 pixels wide and high.

Each rectangle must contain the whole outline of the same face, with at least 3 pixels of free space between the outline and the rectangle border, and no second complete four-sided outline. The sides must be straight, clearly visible and not covered by anything, at least 12 pixels long in the image, and the corners must not be sharper than 20 degrees. The tool finds the four sides in each image, pairs the corners of the two views and triangulates them.

Output:

- `vertices_xyz`: the four corners [x, y, z] in m, in order around the face.
- `center_xyz`: their mean. `edge_lengths_m`: lengths of the sides 0-1, 1-2, 2-3, 3-0.
- `plane_fit`: `normal`, `max_residual_m`, `tilt_from_horizontal_deg`.
- `matched_pixels`: per corner [u, v, u2, v2] in the two images; `quality`: stereo checks per corner.

The direction of a side in the horizontal plane is atan2(y1 - y0, x1 - x0) of its two corners.

Fails with `plan_fail_reason` `quadrilateral_unavailable` and a `detail` (exit code 2, nothing moved) when a rectangle holds no outline or more than one ("found N"), the outline touches the rectangle border, a side is curved, broken or hidden, the corner pairing is not unique, the 3D corners are not flat within 5 mm, or the stereo geometry is weak (ray angle below 3 degrees, ray gap above 8 mm, reprojection above 2 pixels, or more than 15 mm change per pixel of error). Then draw tighter rectangles, or move a wrist camera so that it sees the face better. It does not measure thickness or hidden parts.
