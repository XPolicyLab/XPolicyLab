# RoboDojo data contract

## Version and acquisition

Source: [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main), using the latest `main` branch at download time. Default export: `data/RoboDojo_lerobot_v30_video`; `RoboDojo_lerobot_v21_video` is also accepted. Task, episode and frame counts come from the downloaded data.

Use `bash download_checkpoint.sh data-v30` (or `data-v21`) to download `meta/`, Parquet state/action rows and original MP4 videos. Model checkpoints also follow the latest official `main` branch. Conversion reads local files; rerun the download command to fetch dataset updates. `--source-revision` is an optional provenance label recorded as supplied, and defaults to null for local data of unknown revision.

## Selection and split

- The official `dlc` subset is excluded. In the [official task metadata](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/blob/main/data/RoboDojo_lerobot_v21_video/meta/tasks.jsonl), its instruction is `Arrange the letters to spell "RoboDojo" in a row.` All remaining tasks and episodes are included by default. Every row of each selected episode is retained, including terminal rows. No success-based filtering, frame skipping, interpolation, deduplication, clipping or unit conversion is applied.
- Task indices and instructions are read directly from `meta/tasks.jsonl` (v2.1) or the instruction-indexed `meta/tasks.parquet` (v3.0). Parquet data rows select instructions through their original `task_index`; the `dlc` exclusion is the only instruction-based task filter.
- `--tasks 3 6` selects source task indices 3 and 6. Explicitly selecting a `dlc` index raises an error. Output directories use `task_<index>`, and training records use `robodojo_task_<index>`; neither requires a simulator task-name mapping.
- All selected episodes are used for training. This adapter does not create a held-out split; evaluation uses simulator rollouts.
- Conversion writes `dataset.json` and `episodes.jsonl` for the actual selection, including task instructions, counts, output JSONL hashes, excluded task indices and excluded episode counts. Format checks still require the supported 25 Hz joint+gripper schema and shared robot dimensions.

Use the published checkpoint when evaluating its reported results; training on the latest dataset is not guaranteed to reproduce those weights.

## Cameras, state and action

| Source LeRobot video key | JSONL | Model label | Online XPolicyLab camera |
| --- | --- | --- | --- |
| `observation.images.cam_high` | `images_1` | `Head` | `cam_head` (aliases `cam_high`, `cam_third_view`) |
| `observation.images.cam_left_wrist` | `images_2` | `Left wrist` | `cam_left_wrist` (alias `left_wrist`) |
| `observation.images.cam_right_wrist` | `images_3` | `Right wrist` | `cam_right_wrist` (alias `right_wrist`) |

Official videos are 640×480 at 25 FPS and remain unchanged during conversion. Images are RGB throughout conversion, training and inference.

`observation.state[t]` becomes `state[t]`; `action[t]` becomes `action[t]` with no temporal shift. Both use this order:

```text
0..5   left_joint_0 .. left_joint_5   : six left arm positions, radians
6      left_joint_6                  : left gripper opening, dimensionless [0,1]
7..12  right_joint_0 .. right_joint_5 : six right arm positions, radians
13     right_joint_6                 : right gripper opening, dimensionless [0,1]
```

Gripper 0 is closed and 1 is open; values are copied without clipping. Each arm has six joint values and one gripper value. Actions are **absolute targets**, not state deltas, velocities, end-effector poses or millimeters. A training action chunk starts at the same row `t`, then uses `action[t:t+50]`, extending with the last action near the episode end. This chunk padding is a training transform; it does not append dataset rows.

## Time alignment and history

Within each episode, rows must have contiguous `frame_index = 0..N−1`, and `timestamp = frame_index / 25` (seconds, checked within floating-point tolerance). Each state/action row is paired with all three camera frames at that source index.

- v2.1 has a video and Parquet file per episode; video frame index equals `frame_index`.
- v3.0 concatenates episodes into files. For each camera, the converter reads `videos/<key>/chunk_index`, `file_index` and `from_timestamp` from episode metadata; video frame index is `round(from_timestamp × 25) + frame_index`. Parquet rows are selected by `episode_index`, not by file boundaries.
- Training history uses only head frames at `t−500, t−475, …, t−25`, oldest first. Slots before episode start are empty. There is no future-frame access or crossing between episodes.
- Evaluation observes every executed action step, retaining a head frame every 25 steps, and resets history for each episode/evaluation scope. At the default 25-action replanning interval this matches the 1 FPS training grid.

## Output

```text
data/RoboDojo-robodojo-mem-arx_x5-joint/
  dataset.json                 # actual source/version/split/counts/dimensions
  episodes.jsonl               # actual task/episode/frame list and output hashes
  media -> <LeRobot root>      # original videos remain here
  jsonl/index_cache.json       # absolute JSONL paths and row counts
  jsonl/task_<index>/episode_<global_index>.jsonl
```

Each row is a dexdataset record:

```json
{
  "images_1": {"type": "video", "url": "videos/.../file-000.mp4", "frame_idx": 0},
  "images_2": {"type": "video", "url": "videos/.../file-000.mp4", "frame_idx": 0},
  "images_3": {"type": "video", "url": "videos/.../file-000.mp4", "frame_idx": 0},
  "state": ["14 original numeric values"],
  "action": ["14 original numeric values"],
  "prompt": "Exact source task instruction",
  "is_robot": true,
  "task_name": "robodojo_task_3"
}
```

The array strings above are schematic; actual rows contain 14 numbers. Relative video URLs resolve against `media/`. Each episode JSONL remains separate even when its source videos are shared with other v3.0 episodes. The upstream sampler uses `N−1` start positions per episode; counts depend on the downloaded dataset and selected task indices. No adapter-side downsampling is performed.

Rerunning conversion removes obsolete generated episode files and rebuilds the index. Source videos remain unchanged.

After relocating the source, remove the output directory's old `media` symlink (only the link, leaving the source videos intact), then rerun conversion with the new `--source` path and the intended `--output` path. The converter does not replace an existing or broken `media` symlink. If only the output moves, rerun conversion with its new `--output` path to refresh the generated absolute paths. Preserve the original task selection when reconverting.
