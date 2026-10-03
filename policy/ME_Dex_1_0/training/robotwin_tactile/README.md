# RoboTwin tactile data

`tactile_force_field.py` is the simulator-side collector used to add aligned
three-axis tactile force fields to RoboTwin demonstrations.

The collector does not reconstruct tactile values from an existing video file.
To create the Clean50 training set, replay the original Clean50 task episodes
with the same task, instruction, and seed, enable the collector in RoboTwin,
and save the tactile payload together with the RGB and qpos streams.

## Integration

Copy `tactile_force_field.py` to the RoboTwin environment at
`envs/utils/tactile_force_field.py`. The RoboTwin base task must construct the
collector when `tactile_force_field: true`, call `record_physics_step()` once
per simulator step, and flush the completed frame payload when an episode frame
is written. The existing RoboTwin data writer can keep its normal RGB/qpos
fields; the collector adds the `tactile_force_field` group.

Use the following configuration for the Clean50 replay:

```yaml
collect_data: true
tactile_force_field: true
tactile_force_field_config:
  spec_version: TFA-2.0
  height: 10
  width: 14
  grid_x_range: [-0.005, 0.070]
  grid_z_range: [-0.016, 0.011]
  mesh_paths:
    - assets/embodiments/aloha-agilex/urdf/aloha_maniskill_sim/meshes/link7.STL
    - assets/embodiments/aloha-agilex/urdf/aloha_maniskill_sim/meshes/link8.STL
    - assets/embodiments/aloha-agilex/urdf/aloha_maniskill_sim/meshes/link7.STL
    - assets/embodiments/aloha-agilex/urdf/aloha_maniskill_sim/meshes/link8.STL
  native_frame_physics_steps: 15
  completed_frame_queue_size: 64
```

Each saved frame contains `force_canonical` with shape `[4, 3, 10, 14]` and
channels `[normal, shear_u, shear_v]` in Newtons. The collector also records
the support mask and frame interval. The training loader consumes the two
observed frames and sixteen future frames from these fields; normalization is
applied by the model's tactile encoder at load time.

The original RGB/qpos data and the tactile fields should be released together
as one dataset so that frame indices and timestamps remain aligned.
