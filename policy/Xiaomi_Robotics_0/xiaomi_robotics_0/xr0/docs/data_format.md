# Training Data Format

Each episode consists of **one JSON annotation file** and **videos of captured from multiple views** (ego, wrist-left, wrist-right cameras).

## Directory Layout

```
data/
├── json/
│   ├── episode_001.json
│   ├── episode_002.json
│   └── ...
└── videos/
    ├── episode_001_ego.mp4
    ├── episode_001_wrist_left.mp4
    ├── episode_001_wrist_right.mp4
    └── ...
```

Videos can be placed anywhere as long as the paths in the JSON files are correct. Paths in JSON can be either **absolute** or **relative to the working directory** where training is launched (the project root).

## JSON Annotation Structure

Each JSON file describes a single episode with the following top-level fields:

```jsonc
{
  "trajectory_type": "success",       // "success", "ongoing", or "invalid"
  "time": "2026-01-01_09_00_01",      // episode timestamp (used as identifier)
  "num_frames": 5997,                 // total number of frames in this episode

  "instruction": {
    "general": [
      {
        "images": [                    // dot-paths into the "observations" field
          "observations.ego",
          "observations.wrist_left",
          "observations.wrist_right"
        ],
        "conversations": [
          {
            "from": "human",
            "value": "The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n<YOUR TASK DESCRIPTION>"
          },
          {
            "from": "gpt",
            "value": "<bot></bot>"
          }
        ]
      }
    ]
  },

  "observations": {                    // video references for each camera view
    "ego":         [{"path": "data/videos/episode_001_ego.mp4",         "start": 0, "end": 5997, "fps": 30, "crop_bbox": null}],
    "wrist_left":  [{"path": "data/videos/episode_001_wrist_left.mp4",  "start": 0, "end": 5997, "fps": 30, "crop_bbox": null}],
    "wrist_right": [{"path": "data/videos/episode_001_wrist_right.mp4", "start": 0, "end": 5997, "fps": 30, "crop_bbox": null}]
  },

  "proprios": {                        // per-frame proprioceptive state, each is [num_frames, D]
    "left_ee_pos":      [[x, y, z], ...],                  // [N, 3] left end-effector position
    "left_ee_rotm":     [[r00, r01, ..., r22], ...],       // [N, 9] left end-effector rotation matrix (flattened 3x3)
    "left_arm_joint":   [[j0, j1, ..., j5], ...],          // [N, 6] left arm joint angles
    "left_gripper_pos": [[g], ...],                        // [N, 1] left gripper position
    "right_ee_pos":     [[x, y, z], ...],                  // [N, 3]
    "right_ee_rotm":    [[r00, r01, ..., r22], ...],       // [N, 9]
    "right_arm_joint":  [[j0, j1, ..., j5], ...],          // [N, 6]
    "right_gripper_pos":[[g], ...]                         // [N, 1]
  },

  "actions": {                         // per-frame action targets, same structure as proprios
    "left_ee_pos":      [[x, y, z], ...],                  // [N, 3]
    "left_ee_rotm":     [[r00, r01, ..., r22], ...],       // [N, 9]
    "left_arm_joint":   [[j0, j1, ..., j5], ...],          // [N, 6]
    "left_gripper_pos": [[g], ...],                        // [N, 1]
    "right_ee_pos":     [[x, y, z], ...],                  // [N, 3]
    "right_ee_rotm":    [[r00, r01, ..., r22], ...],       // [N, 9]
    "right_arm_joint":  [[j0, j1, ..., j5], ...],          // [N, 6]
    "right_gripper_pos":[[g], ...]                         // [N, 1]
  }
}
```

## Field Details

| Field | Description |
|-------|-------------|
| `trajectory_type` | `"success"` = completed task; `"ongoing"` = in-progress (partial episode); `"invalid"` = failed episode (actions are masked during training) |
| `num_frames` | Must equal the length of every array in `proprios` and `actions`, and the frame count of every referenced video |
| `instruction.general[].images` | Dot-paths (e.g. `"observations.ego"`) that map `<image>` placeholders in the conversation to the corresponding video views |
| `instruction.general[].conversations` | A dialogue: the human turn contains the multi-view prompt with `<image>` placeholders and a task description; the gpt turn is always `"<bot></bot>"` |
| `observations.*.path` | Path to the video file. Can be absolute or relative to the project root |
| `proprios` | Current proprioceptive readings at each frame |
| `actions` | Target proprioceptive values at each frame. The training pipeline computes **relative actions** (target - current) internally |

## 32-D Action Space

The model operates in a fixed 32-dimensional action space. During training, raw `proprios` and `actions` are converted into relative deltas and packed into a `(action_length, 32)` tensor:

| Dimensions | Component | Description |
|------------|-----------|-------------|
| 0--2 | `left_ee_pos` | Left end-effector position delta (3D) |
| 3--5 | `left_ee_aa` | Left end-effector rotation delta as axis-angle (3D) |
| 6 | `left_gripper` | Left gripper delta (1D) |
| 7--12 | `left_joint` | Left arm joint deltas (6D) |
| 13 | — | Reserved (always 0) |
| 14--16 | `right_ee_pos` | Right end-effector position delta (3D) |
| 17--19 | `right_ee_aa` | Right end-effector rotation delta as axis-angle (3D) |
| 20 | `right_gripper` | Right gripper delta (1D) |
| 21--26 | `right_joint` | Right arm joint deltas (6D) |
| 27--31 | — | Reserved (always 0) |

## Data Pipeline

The data pipeline (`mibot/data/datasets/json_dataset.py`) performs the following at training time:

1. **Frame sampling** — For each training sample, a random frame is selected. An action chunk of length `action_length` (default: 30) frames is extracted starting from that frame.
2. **Image extraction** — The image frame is decoded from each of the three videos (ego, wrist-left, wrist-right) via `decord`.
3. **Prompt construction** — A multi-view prompt is assembled with the three camera images for Qwen3-VL input.
4. **Relative action computation** — Target positions/rotations are converted to deltas relative to the current frame's proprioceptive state. Rotation deltas are computed as axis-angle representations.
5. **Normalization** — Actions are normalized using per-timestep `mean` and `std` statistics specified in the data config file. Both have shape `(action_length, 32)`.
6. **State composition** — The current frame's gripper positions and joint angles are packed into a `(1, 32)` state vector.
7. **Action masking** — A binary mask of shape `(action_length, 32)` is generated: valid timesteps are 1, padding (when the episode is shorter than `action_length`) is 0, and `"invalid"` episodes are fully masked.
