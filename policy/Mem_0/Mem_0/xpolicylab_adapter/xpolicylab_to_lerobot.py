"""
Convert XPolicyLab trajectory HDF5 directly into the Mem_0 LeRobot training
format (one step -- no RMBench-format intermediate).

DP-style entrypoint (called by ../process_data.sh):

    python xpolicylab_to_lerobot.py <dataset_name> <task_name> <env_cfg_type> \
        <expert_data_num> <action_type> --task_type {M1,Mn} [--instruction "..."] \
        [--language_annotation PATH]

Reads:  <ROOT>/data/<dataset_name>/<task_name>/<env_cfg_type>/data/episode_*.hdf5
        via XPolicyLab.utils.load_file.load_hdf5 (default sample: data/RoboDojo/test_data/arx_x5)
Writes: <upstream>/lerobot_datasets/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>

State/action are packed with XPolicyLab's dual-arm joint convention (14-dim:
[LA(6),LGrip,RA(6),RGrip]) and expanded to Mem_0's 16-dim model layout
([LA(6),pad,RA(6),pad,LGrip,RGrip]). Every head-camera frame is standardized to
RGB HWC (240, 320, 3).

Sub-task annotation (Mem_0 trains on `subtask` language + `subtask_end`):

- ``--task_type M1`` (single-stage): one instruction for the whole episode
  (--instruction, else <task_name>); subtask_end=1 on the final 8 frames.
- ``--task_type Mn`` (multi-stage): per-segment sub-task language consumed from a
  language_annotation.json in the RMBench reference format
  (``{"episode_<i>": [[subtask_text, duration], ...]}``), auto-discovered next to the
  dataset or given via --language_annotation. The SAME sub-task keeps an identical
  instruction across its frames; subtask_end=1 within 8 frames of each segment boundary.

That annotation is produced upstream by the VLM caption operator
(~/Desktop/zijian/ego/Ego-X_Operator/operators/caption), which merges same-objective
runs into one captioned task (one identical instruction per sub-task). The XPolicyLab
sample data does not ship one, so an Mn run needs the annotation provided.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)               # policy/Mem_0/Mem_0
ROOT_DIR = os.path.abspath(os.path.join(UPSTREAM_DIR, "..", "..", "..",".."))  # repo root
ANNOTATIONS_ROOT = os.path.join(UPSTREAM_DIR, "language_annotations")
for p in (ROOT_DIR, UPSTREAM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from XPolicyLab.utils.load_file import load_hdf5  # noqa: E402
from XPolicyLab.utils.process_data import (  # noqa: E402
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)

STD_W, STD_H = 320, 240
SUBTASK_END_WINDOW = 8  # frames before a (sub)task boundary flagged subtask_end=1

STATE_NAMES = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "right_joint_0", "right_joint_1", "right_joint_2",
    "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6",
    "left_gripper", "right_gripper",
]

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "observation.image.head_camera": {
        "dtype": "video", "shape": (STD_H, STD_W, 3),
        "names": ["height", "width", "channels"],
    },
    "subtask": {"dtype": "string", "shape": (1,), "names": ["subtask_annotation"]},
    "global_task": {"dtype": "string", "shape": (1,), "names": ["global_task_annotation"]},
    "subtask_end": {"dtype": "int32", "shape": (1,), "names": ["subtask_end_flag"]},
    "episode_id": {"dtype": "int32", "shape": (1,), "names": ["episode_id"]},
}


def _packed14_to_model16(packed14: np.ndarray) -> np.ndarray:
    """[LA(6),LGrip,RA(6),RGrip] -> Mem_0 model layout [LA(6),pad,RA(6),pad,LGrip,RGrip]."""
    out = np.zeros((packed14.shape[0], 16), dtype=np.float32)
    out[:, 0:6] = packed14[:, 0:6]      # left arm
    out[:, 6] = 0.0                     # left arm pad
    out[:, 7:13] = packed14[:, 7:13]    # right arm
    out[:, 13] = 0.0                    # right arm pad
    out[:, 14] = packed14[:, 6]         # left gripper
    out[:, 15] = packed14[:, 13]        # right gripper
    return out


def _decode_rgb(img_bit) -> np.ndarray:
    """Encoded bytes -> RGB HWC (240, 320, 3) uint8 (decode_image_bit already returns RGB)."""
    img = decode_image_bit(img_bit)
    assert img.ndim == 3 and img.shape[-1] == 3, f"Expected HxWx3, got {img.shape}"
    img = cv2.resize(img, (STD_W, STD_H), interpolation=cv2.INTER_AREA)
    assert img.shape == (STD_H, STD_W, 3)
    return img


def _segment_boundaries(episode_annotation, episode_length: int):
    """[[text, duration], ...] -> [(start, end, text)] consecutive segments (clamped)."""
    boundaries, cur = [], 0
    for text, duration in episode_annotation:
        start = cur
        end = min(cur + int(duration) - 1, episode_length - 1)
        boundaries.append((start, end, text))
        cur = end + 1
        if cur >= episode_length:
            break
    if boundaries:  # ensure the final segment reaches the last frame
        s, _e, t = boundaries[-1]
        boundaries[-1] = (s, episode_length - 1, t)
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description="XPolicyLab HDF5 -> Mem_0 LeRobot dataset")
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("env_cfg_type", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", type=str, help="'joint' (Mem_0 default) or 'ee'")
    parser.add_argument("--task_type", choices=["M1", "Mn"], required=True,
                        help="M1: single-stage; Mn: multi-stage with per-segment sub-tasks")
    parser.add_argument("--instruction", default=None,
                        help="M1 instruction / Mn global task (defaults to <task_name>)")
    parser.add_argument("--language_annotation", default=None,
                        help="Mn segmentation JSON {episode_<i>:[[text,duration],...]}; "
                             "auto-discovered at language_annotations/<dataset>/<task>/<env_cfg>/ if omitted")
    parser.add_argument("--camera", default="cam_head",
                        help="vision camera key holding the head view (default cam_head)")
    args = parser.parse_args()

    # Heavy dependency imported lazily so --help works without lerobot installed.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    global_task = args.instruction or args.task_name
    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    assert len(robot_action_dim_info["arm_dim"]) == 2, (
        f"Mem_0 expects a dual-arm robot; env_cfg_type={args.env_cfg_type} gave "
        f"arm_dim={robot_action_dim_info['arm_dim']}."
    )

    load_dir = os.path.join(ROOT_DIR, "data", args.dataset_name, args.task_name, args.env_cfg_type)
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(
            f"Source data dir not found: {load_dir}\n"
            "Expected data/<dataset_name>/<task_name>/<env_cfg_type>/data/episode_*.hdf5 "
            "(default sample: data/RoboDojo/test_data/arx_x5)."
        )

    # Mn sub-task annotation (reference format). XPolicyLab sample ships none.
    annotations = {}
    if args.task_type == "Mn":
        ann_path = args.language_annotation or os.path.join(
            ANNOTATIONS_ROOT, args.dataset_name, args.task_name, args.env_cfg_type,
            "language_annotation.json")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(
                f"Mn task needs sub-task annotations but {ann_path} is missing.\n"
                "Generate it first with the in-project VLM segmenter:\n"
                "    bash segment_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num>\n"
                "(xpolicylab_adapter/segment_language_annotation.py), or pass --language_annotation "
                'to an existing {"episode_<i>": [[text, duration], ...]} file.'
            )
        annotations = json.loads(Path(ann_path).read_text(encoding="utf-8"))

    out_name = f"{args.dataset_name}-{args.task_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    out_root = Path(UPSTREAM_DIR) / "lerobot_datasets" / out_name
    if out_root.exists():
        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=out_name, fps=30, features=FEATURES, root=out_root, use_videos=True,
    )

    written, total_frames = 0, 0
    bar = tqdm(range(args.expert_data_num), desc=f"convert {out_name} [{args.task_type}]",
               unit="ep", dynamic_ncols=True)
    for episode_idx in bar:
        load_path = os.path.join(load_dir, "data", f"episode_{episode_idx:07d}.hdf5")
        if not os.path.isfile(load_path):
            tqdm.write(f"[convert] skip missing {load_path}")
            continue

        data = load_hdf5(load_path)
        state14 = np.asarray(
            pack_robot_state(data, args.action_type, robot_action_dim_info,
                             source_type="dataset", state_type="state"), dtype=np.float32)
        action14 = np.asarray(
            pack_robot_state(data, args.action_type, robot_action_dim_info,
                             source_type="dataset", state_type="action"), dtype=np.float32)
        state16 = _packed14_to_model16(state14)
        action16 = _packed14_to_model16(action14)
        colors = data["vision"][args.camera]["colors"]
        episode_length = state16.shape[0]

        boundaries = None
        if args.task_type == "Mn":
            ep_ann = annotations.get(f"episode_{episode_idx}")
            if not ep_ann:
                tqdm.write(f"[convert] skip episode {episode_idx}: no annotation entry")
                continue
            boundaries = _segment_boundaries(ep_ann, episode_length)

        for frame_idx in range(episode_length):
            if args.task_type == "Mn":
                subtask, subtask_end = global_task, 0
                for start, end, text in boundaries:
                    if start <= frame_idx <= end:
                        subtask = text  # identical text for every frame of this sub-task
                        subtask_end = 1 if (end - frame_idx) < SUBTASK_END_WINDOW else 0
                        break
            else:  # M1
                subtask = global_task
                subtask_end = 1 if (episode_length - frame_idx) <= SUBTASK_END_WINDOW else 0

            dataset.add_frame(
                {
                    "observation.state": state16[frame_idx],
                    "action": action16[frame_idx],
                    "observation.image.head_camera": _decode_rgb(colors[frame_idx]),
                    "subtask": subtask,
                    "global_task": global_task,
                    "subtask_end": np.array([subtask_end], dtype=np.int32),
                    "episode_id": np.array([episode_idx], dtype=np.int32),
                },
                task=args.task_name,
            )
        dataset.save_episode()
        written += 1
        total_frames += episode_length
        bar.set_postfix(frames=episode_length, total=total_frames, episodes=written)
    bar.close()

    if written == 0:
        raise RuntimeError(f"No episodes converted from {load_dir}; check expert_data_num / paths.")
    tqdm.write(f"[convert] wrote {written} episodes / {total_frames} frames -> {out_root}")


if __name__ == "__main__":
    main()
