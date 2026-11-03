# Data preparation

WorldScape Policy can train directly from raw HDF5 episodes or from an existing
LeRobot v2 dataset. The scripts in this directory generate the native metadata
expected by the corresponding dataset adapters; they do not change model
checkpoints.

Run commands from the repository root. Install the training dependencies first:

```bash
pip install -e ".[train]"
```

## Choose a preparation path

- Official RoboDojo ARX-X5 HDF5: use the statistics and index tools below.
  They read the original episodes in place; no format conversion is needed.
- Raw `episode.hdf5` files: use `convert_hdf5_to_native_meta.py`. It writes only
  a `meta/` directory and does not materialize Parquet or MP4 files.
- Existing LeRobot v2 data: use `convert_lerobot_to_native_meta.py`. It scans the
  existing Parquet and video files and augments or refreshes their metadata.
- To fully convert HDF5 into LeRobot Parquet and MP4 instead, use
  `convert_hdf5_to_lerobot.py`.

## Official RoboDojo HDF5

The tools assume the official dataset layout
`<root>/<task>/arx_x5/data/episode_*.hdf5`. They read the `state/` and `action/`
groups with `left_arm_joint_states`, `left_ee_joint_states`,
`right_arm_joint_states`, `right_ee_joint_states` in 6+1+6+1 order.
Actions are absolute stored targets; joint velocities are not included.
Indexing also reads `instruction` and the lengths of `vision/cam_head/colors`,
`vision/cam_left_wrist/colors`, and `vision/cam_right_wrist/colors`, without
reading image payloads. An existing `meta/episodes.jsonl` is honored; otherwise
episodes are discovered recursively in sorted path order.

Run from `worldscape-policy/` after installing training dependencies:

```bash
export DATA_ROOT=/path/to/RoboDojo
export PREPARED_DIR=/path/to/robodojo_prepared
python -m tools.data.compute_robodojo_zscore_stats \
    --data-root "$DATA_ROOT" --output "$PREPARED_DIR/zscore_stats.json" --workers 4
python -m tools.data.compute_robodojo_joint_stats \
    --data-root "$DATA_ROOT" --output "$PREPARED_DIR/q99_stats.json" --workers 4
python -m tools.data.build_robodojo_index \
    --data-root "$DATA_ROOT" --stats-path "$PREPARED_DIR/zscore_stats.json" \
    --output "$PREPARED_DIR/index_h24_c1_uniform_stride1.npz" \
    --action-horizon 24 --max-chunk-size 1 --step-stride 1 --metadata-workers 4
```

Z-score uses every raw frame, float32 loader inputs and float64 accumulation,
with population standard deviation (`ddof=0`) and a `[-5, 5]` clip contract.
Constant dimensions are rejected because the training z-score loader requires
positive standard deviations. Quantile statistics contain `q01` and `q99`
per dimension. By default `--max-samples 2000000` divides an approximate sample
budget equally per episode, sampling without replacement with `--seed 62`.
Short episodes use all rows; each episode contributes at least one row.
Use `--max-samples 0` for exact all-frame quantiles; all sampled state/action
arrays are held in memory. Parallel workers preserve the sampling order.

Each tool validates its output and writes atomically. Existing outputs require
`--overwrite`. The index uses the same temporal packing as training; its paths
and geometry must match training. It can be shared between z-score and q99
normalization. To check an existing index and source file sizes/mtimes:

```bash
python -m tools.data.build_robodojo_index \
    --data-root "$DATA_ROOT" --stats-path "$PREPARED_DIR/zscore_stats.json" \
    --output "$PREPARED_DIR/index_h24_c1_uniform_stride1.npz" --verify-only
```

The adapter's `process_data.sh RoboDojo <ckpt_name> arx_x5 joint` runs all
three steps together and prints the training environment variables. Set
`ROBODOJO_MODEL=flash` (default, H24/C1) or `ROBODOJO_MODEL=memory` (H24/C24)
for this wrapper and use the same selection when launching training.
The standalone Python index tool defaults to H24/C1; pass `--max-chunk-size 24`
to prepare a memory index directly.

## Raw HDF5 metadata

### Expected input

By default, the converter recursively discovers `**/episode.hdf5`. Each episode
is expected to contain:

- `observations/end_pose` with shape `[T, 14]`
- `observations/qpos` with shape `[T, 14]`
- Camera streams corresponding to:
  - `observation.camera.head`
  - `observation.camera.left`
  - `observation.camera.right`

The state/action layout defaults to a 20-dimensional bimanual EEF vector:

```text
[left_pos(3), left_rot6d(6), left_gripper(1),
 right_pos(3), right_rot6d(6), right_gripper(1)]
```

An optional `task_id_to_prompt.json` at the dataset root can map source task IDs
to training prompts.

Example layout:

```text
dataset/
├── task_id_to_prompt.json
├── task_a/episode_0/episode.hdf5
├── task_a/episode_1/episode.hdf5
└── task_b/episode_0/episode.hdf5
```

### Basic command

```bash
python tools/data/convert_hdf5_to_native_meta.py \
    --dataset-path /path/to/dataset \
    --embodiment agilex
```

This writes `/path/to/dataset/meta/`.

To write metadata elsewhere:

```bash
python tools/data/convert_hdf5_to_native_meta.py \
    --dataset-path /path/to/dataset \
    --output-path /path/to/output \
    --embodiment agilex
```

### Task prompts and input overrides

```bash
python tools/data/convert_hdf5_to_native_meta.py \
    --dataset-path /path/to/dataset \
    --task-map /path/to/task_id_to_prompt.json \
    --default-task "complete the demonstrated task" \
    --fps 30 \
    --force
```

Useful options:

- `--glob`: episode discovery pattern; default `**/episode.hdf5`.
- `--sources-list`: text file containing one HDF5 path per line.
- `--max-episodes`: process only the first N discovered episodes.
- `--state-keys` / `--action-keys`: JSON mappings from semantic names to
  `[start, end]` slices.
- `--video-keys`: JSON mapping from camera names to HDF5 keys.
- `--arm-order`: interpret source poses as `left_first` or `right_first`.
- `--align-to-first-frame`: subtract the first EEF state from state and action.
- `--skip-stats`: omit `meta/stats.json`.
- `--absolute-paths`: store absolute episode paths in `episodes.jsonl`.
- `--force`: replace existing generated metadata files.

The HDF5 statistics currently include all episode frames.

## Existing LeRobot v2 metadata

### Expected input

The LeRobot converter expects an existing dataset containing:

```text
dataset/
├── data/
├── videos/
└── meta/info.json
```

`meta/info.json` is used to discover episode Parquet files, state/action
columns, camera features, annotations, FPS, and chunk layout.

### Basic command

```bash
python tools/data/convert_lerobot_to_native_meta.py \
    --dataset-path /path/to/lerobot_dataset \
    --embodiment agilex
```

By default, metadata is written in place. Existing generated files are retained
unless `--force` is supplied.

### Explicit EEF layout

```bash
python tools/data/convert_lerobot_to_native_meta.py \
    --dataset-path /path/to/lerobot_dataset \
    --state-keys '{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}' \
    --action-keys '{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}' \
    --task-key annotation.language.language_instruction \
    --force
```

Without explicit state/action mappings, the converter creates one modality
entry spanning the detected column.

### Relative-action statistics

To additionally generate `meta/relative_stats_dreamzero.json`:

```bash
python tools/data/convert_lerobot_to_native_meta.py \
    --dataset-path /path/to/lerobot_dataset \
    --state-keys '{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}' \
    --action-keys '{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}' \
    --relative-action-keys left_pos left_rot6d right_pos right_rot6d \
    --action-horizon 24 \
    --force
```

Other useful options:

- `--fps`: override the FPS stored in `meta/info.json`.
- `--output-path`: copy the complete dataset before updating its metadata.
- `--force`: overwrite generated metadata. When `--output-path` already exists,
  this option removes and recreates that output directory.

## Generated metadata

Both preparation paths produce the metadata consumed by native dataset loaders:

- `meta/info.json`: dataset dimensions, paths, FPS, feature schema, and counts.
- `meta/modality.json`: semantic state/action slices, video keys, and annotations.
- `meta/embodiment.json`: canonical robot embodiment.
- `meta/tasks.jsonl`: task index and prompt definitions.
- `meta/episodes.jsonl`: episode indices, lengths, and associated tasks.
- `meta/stats.json`: state/action statistics used by normalization.

After generation, point the training dataset root at the prepared dataset.
HDF5 metadata is used by `worldscape_hdf5_text`, `worldscape_hdf5_goal`, and
`worldscape_hdf5_demo`; LeRobot metadata is used by the corresponding
`worldscape_lerobot_*` datasets.

## Verification

Inspect the generated files before launching a long training run:

```bash
ls -lh /path/to/dataset/meta
python -m json.tool /path/to/dataset/meta/info.json >/dev/null
python -m json.tool /path/to/dataset/meta/modality.json >/dev/null
python -m json.tool /path/to/dataset/meta/stats.json >/dev/null
```

Check that episode counts, FPS, camera keys, state/action dimensions, task
prompts, and embodiment match the intended training configuration.
