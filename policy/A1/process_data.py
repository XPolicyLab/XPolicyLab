#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 episodes into a LeRobot v2.1 dataset for A1.

The converter follows the same on-disk layout used by FastWAM-style v2.1
exports: parquet frame metadata plus per-camera mp4 videos. Images are stored
as HWC RGB uint8. XPolicyLab compressed frames are decoded by OpenCV as BGR, so
we explicitly convert BGR -> RGB before writing videos.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, pack_robot_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_ALIASES = {
    "cam_head": "observation.images.cam_head",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _resolve_source_root(project_root: Path, bench_name: str, task_name: str, env_cfg_type: str) -> Path:
    candidates = [
        project_root / "final_data" / bench_name / task_name / env_cfg_type,
        project_root / "data" / bench_name / task_name / env_cfg_type,
        project_root / "data" / task_name / env_cfg_type,
        project_root.parent / "final_data" / bench_name / task_name / env_cfg_type,
        project_root.parent / "data" / bench_name / task_name / env_cfg_type,
        project_root.parent / "data" / task_name / env_cfg_type,
    ]
    for candidate in candidates:
        if (candidate / "data").is_dir():
            return candidate
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find XPolicyLab trajectory directory. Checked:\n  {checked}")


def _decode_rgb(image_bits) -> np.ndarray:
    image = decode_image_bit(image_bits)
    if image is None:
        raise ValueError("Failed to decode compressed image bytes.")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected decoded HWC image, got {image.shape}")
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    # decode_image_bit uses cv2.imdecode(..., IMREAD_COLOR), which returns BGR.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _write_video(video_path: Path, frames_rgb: Iterable[np.ndarray], fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        video_path,
        np.asarray(list(frames_rgb), dtype=np.uint8),
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
    )


def _feature_stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    return {
        "min": array.min(axis=0),
        "max": array.max(axis=0),
        "mean": array.mean(axis=0),
        "std": array.std(axis=0),
        "q01": np.quantile(array, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(array, 0.99, axis=0).astype(np.float32),
        "count": np.asarray([array.shape[0]], dtype=np.int64),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _write_info(dataset_root: Path, fps: int, action_dim: int, total_episodes: int, total_frames: int, total_tasks: int) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [action_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for feature_name in CAMERA_ALIASES.values():
        features[feature_name] = {
            "dtype": "video",
            "shape": [3, 240, 320],
            "names": ["channels", "height", "width"],
            "info": None,
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "xpolicylab",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": int(total_episodes * len(CAMERA_ALIASES)),
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


def _instruction_at(data: dict, frame_idx: int, fallback: str) -> str:
    candidates = []
    if "instructions" in data:
        candidates.append(data["instructions"])
    if "instruction" in data:
        candidates.append(data["instruction"])

    for candidate in candidates:
        if isinstance(candidate, np.ndarray):
            if candidate.ndim == 0:
                candidate = candidate.item()
            elif len(candidate):
                candidate = candidate[min(frame_idx, len(candidate) - 1)]
        elif isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[min(frame_idx, len(candidate) - 1)]

        if isinstance(candidate, bytes):
            candidate = candidate.decode("utf-8", errors="replace")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback


def _load_episode(ep_path: Path, action_type: str, robot_action_dim_info: dict) -> dict[str, Any]:
    data = load_hdf5(ep_path)
    state = pack_robot_state(
        data,
        action_type,
        robot_action_dim_info,
        source_type="dataset",
        state_type="state",
    ).astype(np.float32)
    action = pack_robot_state(
        data,
        action_type,
        robot_action_dim_info,
        source_type="dataset",
        state_type="action",
    ).astype(np.float32)
    return {"raw": data, "state": state, "action": action}


def _episode_fps(data: dict, default_fps: int) -> int:
    frequency = data.get("additional_info", {}).get("frequency", default_fps)
    try:
        return int(frequency)
    except Exception:
        return int(default_fps)


def _resolve_dataset_id(args, task_names: list[str]) -> str:
    if args.repo_id:
        return args.repo_id
    if args.dataset_id:
        return args.dataset_id
    if len(task_names) == 1:
        return f"{args.bench_name}-{task_names[0]}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
    return f"{args.bench_name}-cotrain-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"


def convert(args) -> None:
    project_root = Path(args.project_root).resolve()
    task_names = [task.strip() for task in str(args.task_name).split(",") if task.strip()]
    if not task_names:
        raise ValueError("task_name resolved to an empty task list")

    episode_jobs: list[tuple[Path, str]] = []
    for task_name in task_names:
        source_root = _resolve_source_root(project_root, args.bench_name, task_name, args.env_cfg_type)
        ep_files = sorted((source_root / "data").glob("episode_*.hdf5"))[: int(args.expert_data_num)]
        if len(ep_files) < int(args.expert_data_num):
            raise FileNotFoundError(
                f"Requested {args.expert_data_num} episodes for task '{task_name}', "
                f"found {len(ep_files)} in {source_root / 'data'}"
            )
        episode_jobs.extend((ep_file, task_name) for ep_file in ep_files)

    dataset_id = _resolve_dataset_id(args, task_names)
    output_base = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).resolve().parent / "data"
    dataset_root = output_base / dataset_id
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)

    robot_action_dim_info = get_robot_action_dim_info(args.env_cfg_type)
    action_dim = sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])

    episodes_records: list[dict[str, Any]] = []
    episode_stats_records: list[dict[str, Any]] = []
    tasks_index: dict[str, int] = {}
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    global_index = 0
    dataset_fps = int(args.fps)

    for episode_index, (ep_file, fallback_task) in enumerate(tqdm(episode_jobs, desc=f"convert {dataset_id}", unit="ep")):
        ep = _load_episode(ep_file, args.action_type, robot_action_dim_info)
        data = ep["raw"]
        state = ep["state"]
        action = ep["action"]
        length = min(state.shape[0], action.shape[0])
        fps = _episode_fps(data, args.fps)
        dataset_fps = fps

        camera_frames = {feature_name: [] for feature_name in CAMERA_ALIASES.values()}
        vision = data.get("vision", {})
        for frame_idx in range(length):
            for source_name, feature_name in CAMERA_ALIASES.items():
                if source_name not in vision or "colors" not in vision[source_name]:
                    raise KeyError(f"Missing vision/{source_name}/colors in {ep_file}")
                camera_frames[feature_name].append(_decode_rgb(vision[source_name]["colors"][frame_idx]))

        for feature_name, frames in camera_frames.items():
            video_path = dataset_root / "videos" / "chunk-000" / feature_name / f"episode_{episode_index:06d}.mp4"
            _write_video(video_path, frames, fps)

        fallback_instruction = args.instruction or fallback_task
        episode_instruction = _instruction_at(data, 0, fallback_instruction)
        if episode_instruction not in tasks_index:
            tasks_index[episode_instruction] = len(tasks_index)
        task_index_value = tasks_index[episode_instruction]

        frame_dict = {
            "timestamp": np.arange(length, dtype=np.float32) / float(fps),
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + length, dtype=np.int64),
            "task_index": np.full(length, task_index_value, dtype=np.int64),
            "observation.state": [row.astype(np.float32) for row in state[:length]],
            "action": [row.astype(np.float32) for row in action[:length]],
        }
        pd.DataFrame(frame_dict).to_parquet(
            dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            index=False,
        )

        episodes_records.append({
            "episode_index": episode_index,
            "tasks": [episode_instruction],
            "length": int(length),
            "raw_file_name": ep_file.name,
        })
        episode_stats_records.append({
            "episode_index": episode_index,
            "stats": {
                "action": _feature_stats(action[:length]),
                "observation.state": _feature_stats(state[:length]),
            },
        })
        all_actions.append(action[:length])
        all_states.append(state[:length])
        global_index += length

    tasks_records = [
        {"task_index": idx, "task": text}
        for text, idx in sorted(tasks_index.items(), key=lambda item: item[1])
    ]
    _write_info(dataset_root, dataset_fps, action_dim, len(episode_jobs), global_index, len(tasks_records))
    _write_jsonl(dataset_root / "meta" / "tasks.jsonl", tasks_records)
    _write_jsonl(dataset_root / "meta" / "episodes.jsonl", episodes_records)
    _write_jsonl(dataset_root / "meta" / "episodes_stats.jsonl", episode_stats_records)

    all_actions_arr = np.concatenate(all_actions, axis=0)
    all_states_arr = np.concatenate(all_states, axis=0)
    action_stats = _feature_stats(all_actions_arr)
    state_stats = _feature_stats(all_states_arr)
    dataset_stats = {
        "state": state_stats,
        "action": action_stats,
        "actions": action_stats,
        "num_episodes": len(episode_jobs),
        "num_transition": int(global_index),
    }
    with (dataset_root / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False, default=_json_default)

    tqdm.write(
        f"[A1 process_data] dataset_id={dataset_id} | tasks={len(task_names)} | "
        f"episodes={len(episode_jobs)} | frames={global_index} | unique_instructions={len(tasks_records)}"
    )
    tqdm.write(f"[A1 process_data] wrote LeRobot v2.1 dataset: {dataset_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XPolicyLab HDF5 data to LeRobot v2.1 format for A1.")
    parser.add_argument("bench_name")
    parser.add_argument("task_name", help="task name, or comma-separated list to merge")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("action_type", choices=["joint"])
    parser.add_argument("--repo_id", default=None, help="backward-compatible output dataset id override")
    parser.add_argument("--dataset-id", default=None, help="output dataset id override")
    parser.add_argument("--mode", choices=["video"], default="video", help="A1 writes LeRobot v2.1 video datasets")
    parser.add_argument("--instruction", default=None, help="fallback instruction when the HDF5 has none")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
