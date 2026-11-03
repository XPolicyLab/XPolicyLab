# Observation interface for evaluator integration

Q0-Preview receives observations exclusively through the standard XPolicyLab `update_obs` call. The evaluator bridge in [deploy.py](../deploy.py) forwards `TASK_ENV.get_obs()`; it does not query simulator camera managers or reconstruct camera calibration. [runtime/remote_observation.py](../runtime/remote_observation.py) is the existing validation boundary.

Camera calibration is a required evaluator integration point. Whether a particular official RoboDojo deployment exposes these fields must be confirmed with its maintainers. Enabling configuration flags alone does not add an interface to a version that does not implement it. If the interface is absent, the evaluator must populate the fields below before normal policy evaluation can run. Q0-Preview does not substitute identity matrices, stale wrist poses or guessed calibration.

## Fields

All three cameras are required: `cam_head`, `cam_left_wrist`, `cam_right_wrist`.

| Observation path | Shape / type | Meaning |
| --- | --- | --- |
| `vision[camera].color` | `(H, W, 3)`, `uint8` RGB at the model boundary | Current camera image; XPolicyLab decodes transport images before `update_obs` |
| `vision[camera].intrinsic_matrix` | `(3, 3)`, finite numeric matrix | Pinhole intrinsics in pixels for this image's original resolution |
| `vision[camera].extrinsic_matrix` | `(4, 4)`, finite numeric matrix | Current camera-to-world rigid transform using the camera-axis convention below |
| `state.left_arm_joint_state`, `state.right_arm_joint_state` | Arrays matching shared robot dimensions | Arm joint positions |
| `state.left_ee_joint_state`, `state.right_ee_joint_state` | One scalar each, in `[0, 1]` | Normalized gripper commands/state supplied by the standard interface |
| `state.left_ee_pose`, `state.right_ee_pose` | Seven values each | World position in metres, followed by quaternion `[w, x, y, z]` |
| `instruction` | String or list of strings | Task instruction; `instructions` is also accepted |
| `additional_info.frequency` | Positive finite number | Control frequency; supported evaluation uses 25 Hz |

`extrinsics_matrix` is accepted as an alias for `extrinsic_matrix`. If both are present they must agree. The intrinsic matrix must be nonsingular; the extrinsic matrix must have an orthonormal rotation, positive determinant and homogeneous final row `[0, 0, 0, 1]`. All matrices must describe the same world frame as the end-effector poses.

## Coordinate and timing contract

Pixels use the original image's `(u, v)` coordinates: `u` increases rightward and `v` downward. Intrinsics use the ordinary pinhole optical convention. The extrinsic matrix, however, transforms **USD/OpenGL camera coordinates** to world coordinates: camera `+X` right, `+Y` up, `-Z` forward. Translation is in metres.

The existing geometry helper computes a world ray as:

```text
optical_direction = inverse(K) @ [u, v, 1]
camera_gl_direction = optical_direction * [1, -1, -1]
world_origin = T_camera_gl_to_world[:3, 3]
world_direction = T_camera_gl_to_world[:3, :3] @ camera_gl_direction
```

Consequently, a world-to-camera view matrix cannot be passed directly. A camera-to-world matrix in OpenCV optical axes also needs conversion at the evaluator boundary:

```text
T_camera_gl_to_world = T_camera_optical_to_world @ diag(1, -1, -1, 1)
```

If the evaluator starts with a world-to-optical-camera transform, invert it first. These are coordinate-frame conversions; they do not change RGB channel order.

Each matrix must correspond to the image captured in the same observation. Wrist extrinsics must be refreshed after arm motion; episode-start matrices cannot be reused. If the evaluator resizes or crops an image, it must update the intrinsics accordingly. Matrix validation can detect malformed transforms but cannot detect an otherwise valid matrix from the wrong frame or timestamp; that synchronization is the evaluator's responsibility.

## Evaluator-side hook

For RoboDojo versions implementing these observation options, merge the following into the benchmark workspace's `env_cfg/arx_x5.yml` without replacing its other settings:

```yaml
observation:
  vision:
    intrinsic_matrix: true
    extrinsic_matrix: true
```

If those options are unavailable, the equivalent public integration point is the returned observation:

```python
# Evaluator-side pseudocode. RGB, K and T must come from the same capture.
observation["vision"][camera_name].update({
    "intrinsic_matrix": K_pixels,
    "extrinsic_matrix": T_camera_gl_to_world,
})
```

No new policy-side simulator API is required. Provide matrices through the existing observation payload before it crosses the XPolicyLab transport boundary. Preserve the fields through serialization and use the shared image codec for encoded RGB. Depth, object poses, simulator managers, reward and success information are not required by this interface.

## Missing fields and verification

Normal evaluation rejects missing calibration with an error such as:

```text
Missing authorized calibration: cam_left_wrist.extrinsic_matrix
```

This happens during observation validation, before the observation reaches the planning loop.

On the evaluator host, inspect a real observation: all three images and matrices must be present, a known visible world point must reproject to the expected pixel, and wrist matrices must track camera motion across captures.
