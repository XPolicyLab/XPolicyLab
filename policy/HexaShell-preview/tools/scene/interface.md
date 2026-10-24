## Tool: scene

Three commands that find and measure the objects on the table from the camera images. They use no simulation time and the robot does not move.

### `robo segment [--camera head|wrist_l|wrist_r] [--min_area PX]`

Finds the object regions in one camera image: everything that is not table and not robot. The default camera is `head`.
It also writes `obs/seg_<camera>.png`: the image with each region outlined and its id written on it. The robot is tinted blue. The area that does not count (off the table, or seen too flat) is darkened. Look at this image before you trust a region.

The output lists `objects`, sorted by x from left to right. Each object has:

- `id`: the number to use with `robo locate --id`. An id belongs to this camera and this image only. After the robot moves, run `robo segment` again.
- `bbox` [u0, v0, u1, v1], `centroid_px` [u, v], `area_px`: where the region is in the image, in pixels.
- `color`: the main colour; `colors`: every colour that covers at least 15 % of the region; `mean_rgb`.
- `bottom_xy` [x, y]: the point of the region nearest to the camera, on the table. For an object that stands on the table this is a point on its near bottom edge, not its centre.
- `table_xy` [x, y]: where the ray through the region's centre pixel meets the table. It lies behind the object (farther from the camera), more so for tall objects. Use it only as a rough position.
- `touches_border`: the region runs off the image, so part of the object is not seen.
- `near_robot`: the region touches the robot in the image. Part of it may be hidden, or the robot holds it.
- `clipped`: the region ran into the background behind the table and was cut there.

`gripper_seen` gives the opening of each gripper as seen in its own wrist camera (0 closed, 1 open). It is only given when the commanded opening in `obs/state.json` is below 1. If it is clearly larger than the commanded value, the fingers are stopped by something they hold.

Limits:
- Two touching objects of clearly different colours become two regions. Touching objects of the same colour stay one region.
- An object with parts of very different colours can become two regions.
- Objects with the colour of the table (wood, brown) and very dark objects in shadow can be missed or found only in part.
- Objects behind the line of the two arm bases are not listed, and neither is the edge of the table.
- `--min_area` (default 40 pixels) drops smaller regions. Use a smaller value for small objects far from the camera.

### `robo locate --camera C --id N [--camera_b auto|head|wrist_l|wrist_r]`
### `robo locate --camera C --u U --v V [--camera_b ...]`

Measures one object in 3D from two or more cameras. Give the id from the latest `robo segment --camera C`, or a pixel (U, V) on the object in camera C.
It finds the same object in another camera and compares the two views on horizontal planes at many heights. With `--camera_b auto` (default) it uses every other camera that sees the object.

Output:

- `top_z`: world z of the top of the object, in meters.
- `footprint_xy` [x, y]: the centre of the object's top face. For an object with straight sides (a box, a cylinder, a bowl) this is the centre of the object.
- `top_xyz`: [x, y, top_z] of that centre.
- `extents`: `length` and `width` of the top face (meters, length is the longer side) and `height` of the top above the table.
- `yaw_deg`: the direction of the long side in the xy plane, in degrees from +x toward +y, between -90 and 90. It means little when length and width are about the same.
- `cameras`: the cameras that were used. `matched_ids`: the id of the object in each other camera. `pair_heights`: the height found with each other camera.
- `iou`: how well the views agree at the measured height (1 is perfect, below 0.5 is weak). `conf`: 0 to 1, higher with more cameras and better agreement.
- `lifted`: present and true when the object does not reach down to the table: it is held, or it stands on another object. `base_height` is then roughly where its bottom is, above the table.
- `partly_seen`: present when another camera sees only part of the object (image border or robot in the way).
- `notes`: why a camera was not used, or that the cameras disagree on the height.

When it fails (exit code 2), `reason` says why. The most common reason is that only one camera sees the object: in the start pose the wrist cameras look forward and do not see the table near the robot. `hint` then says where to put a gripper so that its wrist camera sees the object: pointing down with the fingers opening along x, a wrist camera sees the table from about 0.04 m ahead of the TCP to 0.30 m behind it (toward the robot). After the move, run `robo segment` and `robo locate` again.

Limits:
- The measurement is best for objects with a flat top. For a round or uneven top, `top_z` is near the highest part, and `footprint_xy` is the centre of the outline seen from above.
- For a thin object lying flat, `top_z` is close to the table and the height is only good to a few millimeters.
- For an object of irregular shape (a mug with a handle, a brush, a tool), `footprint_xy` is the centre of what both cameras see from above, not a grasp point.
- An object partly hidden in one camera gives a smaller `iou` and a less exact result. Check `partly_seen` and `notes`.
- When two cameras see the whole object, `footprint_xy` is usually within 1 cm.

### `robo pixel_world --camera C --u U --v V [--z Z]`

Gives the world point `xyz` where the ray of pixel (U, V) meets the horizontal plane at height Z. Without `--z` the plane is the table top. Use it for a point you know lies at that height.
