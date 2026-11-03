# Geometry helper interfaces

Units, common calculations and robot dimensions: `/work/geometry_reference.md`.
World coordinates use metres; pixels use original-image `(u,v)`. Rotation vectors
use radians. Read this file alone to keep the complete contracts in one response.

### Observations and images

| Function | Input and result |
| --- | --- |
| `obs.state(observation_id=None)` | Full observation dict; arms under `['state']['left'/'right']`, matrices under `['calibration']`. |
| `obs.image(camera='cam_head', observation_id=None)` | Independent writable `uint8` RGB NumPy array, shape `(H,W,3)`. |
| `obs.image_path(camera='cam_head', observation_id=None)` | Absolute filename for the original camera JPEG; do not guess filenames. |
| `obs.save_crop(camera, box, path, observation_id=None)` | Save native-resolution PNG/JPEG under `/work`; return metadata. `box=[x0,y0,x1,y1]`: integer original pixels, exclusive upper bounds. |

`save_crop` rejects invalid boxes/paths and crops over a 1,000-pixel long edge;
it does not clip, resize or pad. Returned metadata: `path`, `source_observation_id`,
`camera`, `bbox_xyxy`, `source_size_wh`, `size_wh`, `original_pixel_offset_xy`,
`scale_xy`, `source_image_sha256`, `sha256`. A copy is saved as `<path>.crop.json`.
Native-crop mapping: `original_uv = crop_uv + original_pixel_offset_xy`, scale `[1,1]`.
Resizing or overwriting the image requires a new mapping.

Pass saved paths in `run_python.return_images` for same-call display (limits in tool
schema). Display errors do not undo Python execution. Use NumPy slicing for manual
crops, or `Image.fromarray(array)` for Pillow operations.

### Camera geometry

Camera functions default to the latest observation. Keyword `observation_id='obs_...'`
selects a historical frame without changing that default; pair it with the same image ID.

| Function | Return value and meaning |
| --- | --- |
| `geom.camera_ray(camera, u, v)` | `(origin, direction)`: two NumPy three-vectors in world coordinates; direction is not normalized. |
| `geom.project_camera(camera, world_point)` | NumPy `[u,v]`; `None` behind the camera. Pixels may fall outside the image. |
| `geom.pixel_scale(camera, world_point)` | Dict: `depth_m`, `horizontal_mm_per_pixel`, `vertical_mm_per_pixel`. Scale at the supplied depth on a frontoparallel plane. |
| `geom.plane_pixel_scale(camera, u, v, surface_point, surface_normal)` | Dict: `point` (world intersection), `du_world_m`, `dv_world_m` (world displacement for one pixel right/down on that plane). Rejects a zero normal, parallel ray or intersection behind the camera. |
| `geom.projection_jacobian(camera, world_point, epsilon_m=0.0001)` | NumPy `(2,3)` Jacobian in pixels/metre, camera held fixed; positive `epsilon_m`. |

Historical-frame example (use an observation ID from the conversation):
```python
old_id = 'obs_00003'  # substitute the intended observation ID
pixels = obs.image('cam_left_wrist', observation_id=old_id)
origin, direction = geom.camera_ray('cam_left_wrist', 320, 240, observation_id=old_id)
print(origin, direction)
```
Reading an old image alone does not switch calibration. The harness configures both
helpers for each new planning observation.

| Other calculation | Return value |
| --- | --- |
| `geom.at_height(origin, direction, z)` | NumPy world point at supplied height `z`; rejects parallel rays and backward intersections. |
| `geom.triangulate(p1, d1, p2, d2)` | `(point, miss_distance_m)`: closest-line midpoint and residual; `p1,p2` are origins and `d1,d2` directions. Rejects zero/parallel rays; verify that the point projects in front of both cameras and matches the same feature. A low residual alone does not validate correspondence. |
| `geom.fit_circle_px(xs, ys)` | `(circle, rms_px)`, with NumPy `circle=[u_center,v_center,radius_px]`; needs at least three matching finite boundary pixels. |

### Pose targets and finger regions

`target` is a single-arm action dict: `target_points` shape `(2,3)`,
`orientation_point` shape `(3,)`, optional `contact_constraint`. It is not an observed
arm or action group. Helpers reject malformed/nonfinite coordinates; schema and
controller validation still apply.

| Function | Return value and behavior |
| --- | --- |
| `geom.target_from_state(arm_state)` | New target from one observed arm. Copies ordered `gripper_pose_points`; puts `orientation_point` 0.1 m behind their midpoint along `tool_axis`. No gripper commands or contact constraint. |
| `geom.translate_points(target, displacement)` | New target: all three points translated by the world displacement in metres. |
| `geom.rotate_points(target, pivot, rotation_vector)` | New target dict rotated about a world pivot. Rotation-vector direction is the world rotation axis; its norm is the angle in radians. This is not an Euler-angle triple. |
| `geom.grasp_reference_for(working_point, tool_axis, jaw_axis, region='front_tip')` | NumPy grasp-reference three-vector placing the selected finger-region midpoint at `working_point`; not a full target. |

Transforms preserve inputs and finger order; other fields, including a fixed-world
contact plane and local working-point offset, remain unchanged.
For `grasp_reference_for`, supported regions are `front_tip`,
`distal_pad_bbox_center` and `grasp_reference`. Axes must be nonzero/nonparallel;
the helper normalizes the tool axis and uses the jaw-axis component perpendicular to it.
