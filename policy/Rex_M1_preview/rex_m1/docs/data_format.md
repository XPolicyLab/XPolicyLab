# Training Data Format

Each episode uses one JSON annotation and three synchronized videos. Absolute video paths are supported; relative video paths are resolved from the directory where training is started.

## Example

```json
{
  "trajectory_type": "success",
  "time": "2026-01-01_09_00_01",
  "num_frames": 1000,
  "instruction": {
    "general": [
      {
        "images": [
          "observations.ego",
          "observations.wrist_left",
          "observations.wrist_right"
        ],
        "conversations": [
          {
            "from": "human",
            "value": "The following observations are captured from multiple views.\n# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\nGenerate robot actions for the task:\nload the washer"
          },
          {"from": "gpt", "value": ""}
        ]
      }
    ]
  },
  "observations": {
    "ego": [
      {
        "path": "data/videos/episode_001_ego.mp4",
        "start": 0,
        "crop_bbox": null
      }
    ],
    "wrist_left": [
      {
        "path": "data/videos/episode_001_left.mp4",
        "start": 0,
        "crop_bbox": null
      }
    ],
    "wrist_right": [
      {
        "path": "data/videos/episode_001_right.mp4",
        "start": 0,
        "crop_bbox": null
      }
    ]
  },
  "proprios": {
    "left_ee_pos": [],
    "left_ee_rotm": [],
    "left_arm_joint": [],
    "left_gripper_pos": [],
    "right_ee_pos": [],
    "right_ee_rotm": [],
    "right_arm_joint": [],
    "right_gripper_pos": [],
    "waist_pos": []
  },
  "actions": {
    "left_ee_pos": [],
    "left_ee_rotm": [],
    "left_gripper_pos": [],
    "right_ee_pos": [],
    "right_ee_rotm": [],
    "right_gripper_pos": [],
    "waist_pos": [],
    "base_vel": []
  }
}
```

`num_frames`, `instruction.general`, the three entries referenced by `images`, and every array shown under `proprios` and `actions` are required. These arrays contain one entry per frame. Rotation matrices are flattened to 9 values, and arm joints may contain up to 7 values.

Each video entry requires `path`. `start` defaults to `0`, and `crop_bbox` defaults to no crop.

Each `<image>` placeholder in `instruction.general` consumes the next entry listed in `images`.

The empty assistant turn completes the prompt template; its value is filled by the data loader.

`crop_bbox` is either `null` or `(top, left, height, width)`. Crops extending outside the frame are zero-padded.

## Action Layout

Actions are converted to relative deltas and packed into `(30, 60)` tensors.

| Dimensions | Value |
|---|---|
| 0-2 | left end-effector position |
| 3-5 | left end-effector axis-angle rotation |
| 6 | left gripper |
| 8-10 | right end-effector position |
| 11-13 | right end-effector axis-angle rotation |
| 14 | right gripper |
| 16 | waist |
| 17-19 | base velocity |
| remaining | reserved |

The data config supplies per-step `mean`/`std` for these packed relative actions and `q01`/`q99` for the packed robot state.
