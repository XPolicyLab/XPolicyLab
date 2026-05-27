"""Convert XPolicyLab HDF5 episodes into a LeRobot v2.1 dataset for FastWAM.

Single-task usage (legacy, preserved):
    python process_data.py <dataset> <task> <env_cfg> <num> <action_type>

Multi-task / cotrain usage (mirrors LDA_1B):
    --task-name accepts a comma-separated list, e.g. "stack_bowls,press_by_number".
    All listed tasks are merged into one LeRobot dataset with continuous
    episode/frame indices. Each episode's instruction is resolved from its
    HDF5; identical instructions collapse to a single tasks.jsonl entry while
    distinct ones each get their own task_index.

    --dataset-id overrides the output folder name. For a single task it
    defaults to "<dataset>-<task>-<env_cfg>-<num>-<action_type>"; for a
    merged multi-task run it defaults to "cotrain_dataset".

Image standard: HWC RGB uint8, resized to (240, 320, 3) before encoding.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm import tqdm

from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


CAMERA_MAP = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _read_hdf5(path):
    def read_obj(obj):
        if isinstance(obj, h5py.Dataset):
            value = obj[()]
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8", errors="replace")
            try:
                return value.item()
            except Exception:
                return value
        return {key: read_obj(child) for key, child in obj.items()}

    with h5py.File(path, "r") as f:
        return read_obj(f)


def _resolve_source_root(project_root, dataset_name, task_name, env_cfg_type):
    candidate = project_root / "final_data" / dataset_name / task_name / env_cfg_type
    if (candidate / "data").is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find XPolicyLab trajectory directory: {candidate}")


def _decode_rgb(image_bits):
    image = decode_image_bit(image_bits)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected decoded HWC image, got {image.shape}")
    # XPolicyLab preview videos for RoboDojo/test_data/arx_x5 match this decoded
    # array when it is treated as RGB. Do not apply OpenCV's usual BGR->RGB swap.
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected RGB image shape (240, 320, 3), got {image.shape}")
    return image


def _write_video(video_path, frames_rgb, fps):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        video_path,
        np.asarray(frames_rgb, dtype=np.uint8),
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
    )


def _feature_stats(array):
    array = np.asarray(array, dtype=np.float32)
    return {
        "min": array.min(axis=0),
        "max": array.max(axis=0),
        "mean": array.mean(axis=0),
        "std": array.std(axis=0),
        "count": np.asarray([array.shape[0]], dtype=np.int64),
    }


def _processor_stats(array):
    stats = _feature_stats(array)
    q01 = np.quantile(array, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(array, 0.99, axis=0).astype(np.float32)
    return {
        "global_min": stats["min"],
        "global_max": stats["max"],
        "global_mean": stats["mean"],
        "global_std": stats["std"],
        "global_q01": q01,
        "global_q99": q99,
        "global_count": stats["count"],
        "stepwise_min": stats["min"][None, :],
        "stepwise_max": stats["max"][None, :],
        "stepwise_mean": stats["mean"][None, :],
        "stepwise_std": stats["std"][None, :],
        "stepwise_q01": q01[None, :],
        "stepwise_q99": q99[None, :],
        "stepwise_count": stats["count"],
    }


def _episode_stats(frame_dict):
    return {
        "action": _feature_stats(frame_dict["action"]),
        "observation.state": _feature_stats(frame_dict["observation.state"]),
    }


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _write_info(dataset_root, fps, action_dim, total_episodes, total_frames, total_tasks):
    # task_index is the only task-related index the FastWAM dataloader requires:
    # LeRobotDataset.__getitem__ reads `task_index` -> tasks.jsonl[idx] to set `task`.
    # `coarse_task_index` is optional and (with drop_high_level_prob=1.0 default in
    # configs/data/robotwin.yaml) never affects the encoded prompt. We drop the legacy
    # coarse_task / coarse_quality / quality slots from the original FastWAM
    # process_data.py because their fixed-index design (task_index=1, coarse=0, etc.)
    # only worked for single-task, single-instruction datasets and breaks once we merge
    # multiple tasks into one cotrain dataset. Keeping just `task_index` matches LDA_1B
    # and the lerobot v2.1 minimum schema.
    features = {
        "observation.images.cam_high": {"dtype": "video", "shape": [3, 240, 320], "names": ["channel", "height", "width"], "info": None},
        "observation.images.cam_left_wrist": {"dtype": "video", "shape": [3, 240, 320], "names": ["channel", "height", "width"], "info": None},
        "observation.images.cam_right_wrist": {"dtype": "video", "shape": [3, 240, 320], "names": ["channel", "height", "width"], "info": None},
        "observation.state": {"dtype": "float32", "shape": [action_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "xpolicylab",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": int(total_tasks),
        "total_videos": total_episodes * 3,
        # We always write every episode under chunk-000/, so chunks_size must be
        # >= total_episodes; otherwise LeRobot's `ep_idx // chunks_size` chunk
        # routing points at non-existent chunk-001/, chunk-002/, ... dirs and
        # `LeRobotDataset.__init__` falls back to `snapshot_download(repo_id=<local path>)`
        # which raises HFValidationError. lerobot v2.1 doesn't require any specific
        # value here; the published RoboTwin/LIBERO datasets pre-shard files into
        # multiple chunk dirs to match the 1000 default, but a flat layout works
        # as long as `chunks_size` reflects it.
        "total_chunks": 1 if total_episodes else 0,
        "chunks_size": max(1000, int(total_episodes)),
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with (dataset_root / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def _resolve_instruction(episode_data: dict, fallback: str) -> str:
    inst = episode_data.get("instruction")
    if isinstance(inst, bytes):
        inst = inst.decode("utf-8", errors="replace")
    if isinstance(inst, str) and inst.strip():
        return inst.strip()
    return fallback


def _resolve_dataset_id(args, task_names: List[str]) -> str:
    if args.dataset_id:
        return args.dataset_id
    if len(task_names) == 1:
        return (
            f"{args.dataset_name}-{task_names[0]}-{args.env_cfg_type}"
            f"-{args.expert_data_num}-{args.action_type}"
        )
    return "cotrain_dataset"


def convert(args):
    policy_dir = Path(__file__).resolve().parents[1]
    project_root = Path(args.project_root).resolve()

    task_names = [t.strip() for t in str(args.task_name).split(",") if t.strip()]
    if not task_names:
        raise ValueError("--task-name resolved to an empty task list.")

    # Gather episodes from every task, capped per task by expert_data_num.
    # Each job remembers its source task so the instruction fallback is correct.
    episode_jobs: List[Tuple[Path, str]] = []
    for task_name in task_names:
        source_root = _resolve_source_root(project_root, args.dataset_name, task_name, args.env_cfg_type)
        task_episodes = sorted((source_root / "data").glob("episode_*.hdf5"))[: int(args.expert_data_num)]
        if len(task_episodes) < int(args.expert_data_num):
            raise FileNotFoundError(
                f"Requested {args.expert_data_num} episodes for task '{task_name}', "
                f"found {len(task_episodes)} in {source_root / 'data'}"
            )
        episode_jobs.extend((ep, task_name) for ep in task_episodes)

    dataset_id = _resolve_dataset_id(args, task_names)
    output_base = policy_dir / "data" / dataset_id
    dataset_root = output_base / "lerobot"

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)

    robot_dim = get_robot_action_dim_info(args.env_cfg_type)
    action_dim = sum(robot_dim["arm_dim"]) + sum(robot_dim["ee_dim"])

    episodes_records = []
    episode_stats_records = []
    # Deduplicate instructions: each unique string -> its own task_index in tasks.jsonl.
    tasks_index: dict[str, int] = {}
    all_actions = []
    all_states = []
    global_index = 0
    fps = None

    bar = tqdm(episode_jobs, desc=f"convert {dataset_id}", unit="ep", dynamic_ncols=True)
    for episode_index, (hdf5_path, fallback_task) in enumerate(bar):
        data = _read_hdf5(hdf5_path)
        instruction = _resolve_instruction(data, fallback=fallback_task)
        if instruction not in tasks_index:
            tasks_index[instruction] = len(tasks_index)
        task_index_value = tasks_index[instruction]

        fps = int(data.get("additional_info", {}).get("frequency", args.fps))
        state_all = pack_robot_state(data, args.action_type, robot_dim, source_type="dataset", state_type="state").astype(np.float32)
        action_all = pack_robot_state(data, args.action_type, robot_dim, source_type="dataset", state_type="action").astype(np.float32)
        length = min(state_all.shape[0], action_all.shape[0])

        camera_frames = {key: [] for key in CAMERA_MAP}
        for i in range(length):
            for fastwam_key, xpl_key in CAMERA_MAP.items():
                camera_frames[fastwam_key].append(_decode_rgb(data["vision"][xpl_key]["colors"][i]))

        for fastwam_key, frames in camera_frames.items():
            video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{fastwam_key}" / f"episode_{episode_index:06d}.mp4"
            _write_video(video_path, frames, fps)

        frame_dict = {
            "timestamp": np.arange(length, dtype=np.float32) / float(fps),
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + length, dtype=np.int64),
            "task_index": np.full(length, task_index_value, dtype=np.int64),
            "observation.state": [row.astype(np.float32) for row in state_all[:length]],
            "action": [row.astype(np.float32) for row in action_all[:length]],
        }
        pd.DataFrame(frame_dict).to_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet", index=False)

        episodes_records.append({"episode_index": episode_index, "tasks": [instruction], "length": length, "raw_file_name": hdf5_path.name})
        episode_stats_records.append({"episode_index": episode_index, "stats": _episode_stats({"action": action_all[:length], "observation.state": state_all[:length]})})
        all_actions.append(action_all[:length])
        all_states.append(state_all[:length])
        global_index += length
        bar.set_postfix(frames=length, total=global_index, tasks=len(tasks_index))
    bar.close()

    tasks_jsonl = [
        {"task_index": idx, "task": text}
        for text, idx in sorted(tasks_index.items(), key=lambda kv: kv[1])
    ]

    _write_info(dataset_root, fps or args.fps, action_dim, len(episode_jobs), global_index, len(tasks_jsonl))
    _write_jsonl(dataset_root / "meta" / "tasks.jsonl", tasks_jsonl)
    _write_jsonl(dataset_root / "meta" / "episodes.jsonl", episodes_records)
    _write_jsonl(dataset_root / "meta" / "episodes_stats.jsonl", episode_stats_records)

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    dataset_stats = {
        "action": {"default": _processor_stats(all_actions)},
        "state": {"default": _processor_stats(all_states)},
        "num_episodes": len(episode_jobs),
        "num_transition": int(global_index),
    }
    with (output_base / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2, default=_json_default)

    tqdm.write(
        f"[FastWAM] dataset_id={dataset_id} | tasks={len(task_names)} | "
        f"episodes={len(episode_jobs)} | frames={global_index} | unique_instructions={len(tasks_index)}"
    )
    tqdm.write(f"[FastWAM] wrote LeRobot dataset: {dataset_root}")
    tqdm.write(f"[FastWAM] wrote dataset stats:   {output_base / 'dataset_stats.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("task_name", help="task name, or comma-separated list to merge into one dataset")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", choices=["joint", "ee"])
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--dataset-id", default=None, help="override output folder name (default: per-task data_key, or 'cotrain_dataset' for multi-task)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
