#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CAMERA_MAP = {
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
    "observation.images.cam_third_view": "cam_third_view",
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=_json_default)


def _write_jsonl(path: Path, records: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": int(values.shape[0]),
    }


def _resolve_source_dirs(root: Path, bench_name: str, task_names: list[str], env_cfg_type: str) -> list[tuple[str, Path]]:
    jobs = []
    for task_name in task_names:
        candidates = [
            root / "data" / bench_name / task_name / env_cfg_type,
            root / "final_data" / bench_name / task_name / env_cfg_type,
            root / "data" / task_name / env_cfg_type,
        ]
        for candidate in candidates:
            if (candidate / "data").is_dir():
                jobs.append((task_name, candidate))
                break
        else:
            checked = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Could not find XPolicyLab HDF5 directory for task={task_name!r}. Checked:\n  {checked}"
            )
    return jobs


def _normalise_instruction(value: Any, fallback: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _decode_frame_rgb(raw, decode_image_bit, width: int, height: int, input_color_space: str) -> np.ndarray:
    # XPolicyLab color observations are semantically RGB. Encoded image bytes are
    # decoded through OpenCV here, whose native output is BGR, so convert them
    # explicitly instead of depending on decode_image_bit's OpenCV convention.
    raw_array = np.asarray(raw)
    if raw_array.ndim == 3 and raw_array.shape[-1] == 3:
        frame = raw_array
    elif raw_array.ndim == 4 and raw_array.shape[0] == 1 and raw_array.shape[-1] == 3:
        frame = raw_array[0]
    elif isinstance(raw, (bytes, bytearray, np.bytes_, np.void)):
        encoded = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image frame")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif raw_array.ndim == 1 and raw_array.dtype == np.uint8:
        frame = cv2.imdecode(np.ascontiguousarray(raw_array), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image frame")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = decode_image_bit(raw)
        if frame is None:
            raise ValueError("Failed to decode image frame")
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {frame.shape}")
    if input_color_space.lower() == "bgr":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if (frame.shape[1], frame.shape[0]) != (width, height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame.astype(np.uint8))


def _write_video(path: Path, frames_rgb: list[np.ndarray], fps: int, codec: str):
    if not frames_rgb:
        raise ValueError(f"No frames to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _vector_semantics(action_type: str, robot_dim: dict[str, list[int]]) -> list[str]:
    arm_dims = list(robot_dim["arm_dim"])
    ee_dims = list(robot_dim["ee_dim"])
    if len(arm_dims) == 1:
        prefixes = [("", arm_dims[0], ee_dims[0])]
    elif len(arm_dims) == 2:
        prefixes = [("left_", arm_dims[0], ee_dims[0]), ("right_", arm_dims[1], ee_dims[1])]
    else:
        prefixes = [(f"arm{i}_", arm_dim, ee_dim) for i, (arm_dim, ee_dim) in enumerate(zip(arm_dims, ee_dims))]

    names: list[str] = []
    if action_type == "joint":
        for prefix, arm_dim, ee_dim in prefixes:
            names.extend(f"{prefix}arm_joint_{i}" for i in range(arm_dim))
            names.extend(f"{prefix}ee_joint_{i}" for i in range(ee_dim))
    elif action_type == "ee":
        pose_names = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        for prefix, _arm_dim, ee_dim in prefixes:
            names.extend(f"{prefix}ee_pose_{name}" for name in pose_names)
            names.extend(f"{prefix}ee_joint_{i}" for i in range(ee_dim))
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")
    return names


def _feature_schema(
    width: int,
    height: int,
    fps: int,
    action_names: list[str],
    state_names: list[str],
    video_codec: str,
) -> dict[str, Any]:
    action_dim = len(action_names)
    state_dim = len(state_names)
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in CAMERA_MAP:
        features[key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": video_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    return features



def convert(args: argparse.Namespace):
    xpl_root = Path(args.xpolicylab_root).resolve()
    sys.path.insert(0, str(xpl_root.parent))
    sys.path.insert(0, str(xpl_root))

    from XPolicyLab.utils.load_file import load_hdf5
    from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, pack_robot_state

    task_names = [t.strip() for t in str(args.task_names).split(",") if t.strip()]
    if not task_names:
        raise ValueError("--task-names resolved to an empty list")

    source_dirs = _resolve_source_dirs(xpl_root, args.bench_name, task_names, args.env_cfg_type)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    robot_dim = get_robot_action_dim_info(args.env_cfg_type)
    vector_names = _vector_semantics(args.action_type, robot_dim)
    expected_dim = len(vector_names)

    episode_jobs: list[tuple[str, Path]] = []
    for task_name, source_root in source_dirs:
        episode_files = sorted((source_root / "data").glob("episode_*.hdf5"))
        if args.expert_data_num > 0:
            episode_files = episode_files[: args.expert_data_num]
        if not episode_files:
            raise FileNotFoundError(f"No episode_*.hdf5 files under {source_root / 'data'}")
        episode_jobs.extend((task_name, path) for path in episode_files)

    tasks_index: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    total_frames = 0
    fps_value = int(args.fps) if args.fps else None

    for episode_index, (fallback_task, episode_path) in enumerate(
        tqdm(episode_jobs, desc="convert xpolicylab hdf5", unit="ep", dynamic_ncols=True)
    ):
        data = load_hdf5(str(episode_path))
        instruction = _normalise_instruction(
            data.get("instructions", data.get("instruction")), fallback=fallback_task
        )
        if instruction not in tasks_index:
            tasks_index[instruction] = len(tasks_index)
        task_index = tasks_index[instruction]

        ep_fps = int(data.get("additional_info", {}).get("frequency", fps_value or 25))
        if fps_value is None:
            fps_value = ep_fps
        elif ep_fps != fps_value:
            print(f"[WARN] {episode_path.name} fps={ep_fps} differs from dataset fps={fps_value}; using {fps_value}")

        state_all = pack_robot_state(
            data, args.action_type, robot_dim, source_type="dataset", state_type="state"
        ).astype(np.float32)
        action_all = pack_robot_state(
            data, args.action_type, robot_dim, source_type="dataset", state_type="action"
        ).astype(np.float32)
        if state_all.shape[-1] != expected_dim or action_all.shape[-1] != expected_dim:
            raise ValueError(
                f"{episode_path.name}: packed dims state={state_all.shape[-1]}, "
                f"action={action_all.shape[-1]}, expected={expected_dim}"
            )

        length = min(state_all.shape[0], action_all.shape[0])
        vision = data.get("vision", {})
        for output_key, source_key in CAMERA_MAP.items():
            if source_key not in vision or "colors" not in vision[source_key]:
                raise KeyError(f"{episode_path.name}: missing vision/{source_key}/colors")
            length = min(length, len(vision[source_key]["colors"]))

        rows = []
        for frame_idx in range(length):
            rows.append(
                {
                    "observation.state": state_all[frame_idx].astype(np.float32),
                    "action": action_all[frame_idx].astype(np.float32),
                    "timestamp": float(frame_idx) / float(fps_value),
                    "frame_index": int(frame_idx),
                    "episode_index": int(episode_index),
                    "index": int(total_frames + frame_idx),
                    "task_index": int(task_index),
                }
            )

        parquet_path = output_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        for output_key, source_key in CAMERA_MAP.items():
            frames = [
                _decode_frame_rgb(
                    vision[source_key]["colors"][i],
                    decode_image_bit,
                    args.image_width,
                    args.image_height,
                    args.input_color_space,
                )
                for i in range(length)
            ]
            video_path = output_dir / "videos" / "chunk-000" / output_key / f"episode_{episode_index:06d}.mp4"
            _write_video(video_path, frames, fps=fps_value, codec=args.video_codec)

        ep_states = state_all[:length]
        ep_actions = action_all[:length]
        all_states.append(ep_states)
        all_actions.append(ep_actions)
        episodes_meta.append(
            {
                "episode_index": int(episode_index),
                "tasks": [instruction],
                "length": int(length),
                "raw_file_name": episode_path.name,
                "raw_task_name": fallback_task,
            }
        )
        episode_stats.append(
            {
                "episode_index": int(episode_index),
                "stats": {
                    "observation.state": _stats(ep_states),
                    "action": _stats(ep_actions),
                    "task_index": _stats(np.full((length, 1), task_index, dtype=np.int64)),
                },
            }
        )
        total_frames += length

    tasks_jsonl = [
        {"task_index": idx, "task": text}
        for text, idx in sorted(tasks_index.items(), key=lambda item: item[1])
    ]
    all_states_arr = np.concatenate(all_states, axis=0)
    all_actions_arr = np.concatenate(all_actions, axis=0)

    info = {
        "codebase_version": "v2.1",
        "robot_type": args.env_cfg_type,
        "fps": int(fps_value or 25),
        "total_episodes": int(len(episode_jobs)),
        "total_frames": int(total_frames),
        "total_tasks": int(len(tasks_jsonl)),
        "total_videos": int(len(episode_jobs) * len(CAMERA_MAP)),
        "total_chunks": 1 if episode_jobs else 0,
        "chunks_size": max(1000, int(len(episode_jobs))),
        "splits": {"train": f"0:{len(episode_jobs)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _feature_schema(
            args.image_width,
            args.image_height,
            int(fps_value or 25),
            action_names=vector_names,
            state_names=vector_names,
            video_codec=args.video_codec,
        ),
    }
    _write_json(output_dir / "meta" / "info.json", info)
    _write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks_jsonl)
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_meta)
    _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episode_stats)
    _write_json(
        output_dir / "meta" / "stats.json",
        {
            "observation.state": _stats(all_states_arr),
            "action": _stats(all_actions_arr),
        },
    )
    _write_json(
        output_dir / "meta" / "xpolicylab_info.json",
        {
            "bench_name": args.bench_name,
            "task_names": task_names,
            "env_cfg_type": args.env_cfg_type,
            "action_type": args.action_type,
            "robot_action_dim_info": robot_dim,
            "action_dim": expected_dim,
            "state_dim": expected_dim,
            "action_names": vector_names,
            "state_names": vector_names,
            "packing_order": "left arm, left end-effector, right arm, right end-effector" if len(robot_dim["arm_dim"]) == 2 else "arm, end-effector",
            "camera_map": CAMERA_MAP,
            "image_size": [args.image_width, args.image_height],
            "input_color_space": args.input_color_space,
            "source_dirs": [{"task_name": t, "path": str(p)} for t, p in source_dirs],
        },
    )
    print(
        f"[GigaWorldPolicy] wrote LeRobot v2.1 dataset: {output_dir} | "
        f"episodes={len(episode_jobs)} frames={total_frames} action_dim={expected_dim}"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert XPolicyLab HDF5 episodes to LeRobot v2.1 for GigaWorldPolicy")
    parser.add_argument("--xpolicylab-root", required=True)
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--task-names", required=True, help="One task or comma-separated tasks. For single task this is usually ckpt_name.")
    parser.add_argument("--env-cfg-type", required=True)
    parser.add_argument("--expert-data-num", required=True, type=int, help="Episodes per task; <=0 means all available.")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--input-color-space", default="rgb", choices=["rgb", "bgr"])
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
