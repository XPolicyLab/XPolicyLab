"""Convert XPolicyLab HDF5 episodes into LeRobot v2.1 datasets that LDA's
`gr00t_lerobot` loader can consume directly.

Expected input layout (raw task dirs; comma-separated to merge):
    <root>/data/<dataset_name>/<raw_task>/<env_cfg_type>/data/episode_*.hdf5
    (legacy fallback: <root>/final_data/...)

Output layout (one dataset per invocation):
    <policy_dir>/data/<dataset_id>/
        data/chunk-NNN/episode_XXXXXX.parquet
        videos/chunk-NNN/video.<cam>/episode_XXXXXX.mp4
    Episodes are split across chunks by episode_index // chunks_size (LeRobot v2.1).
        meta/{info.json, modality.json, episodes.jsonl,
              episodes_stats.jsonl, tasks.jsonl, stats.json}

`--raw-task-dirs` (alias `--task-name`) accepts a comma-separated list; all tasks
are merged into one LeRobot dataset with continuous episode/frame indices.
`--expert-data-num` caps episodes per task, so N tasks yield up to N*num episodes.

Default dataset_id (README §4.2):
    single-task: "<dataset_name>-<ckpt_name>-<env_cfg_type>-<action_type>"
    multi-task:  "cotrain"

Override with --dataset-id. XPOLICYLAB_DATASET_ID in train.sh must match.

Image standard: HWC RGB uint8, resized to (240, 320, 3) before encoding.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
)


# LDA trains single-view (num_views=1); ArxX5DataConfig only reads video.cam_head, so
# only the head camera is converted. Add the wrist views back here if a multi-view config
# is ever used (the model's num_views must then match the number of video_keys).
CAMERA_NAMES = ("cam_head",)
IMG_HEIGHT = 240
IMG_WIDTH = 320
DEFAULT_FPS = 25  # arx_x5.yml collect_freq
VIDEO_CODEC = "mp4v"
VIDEO_PIX_FMT = "yuv420p"
CHUNK_SIZE = 1000  # LeRobot v2.1 default


def _episode_chunk(episode_index: int, chunk_size: int = CHUNK_SIZE) -> int:
    return episode_index // chunk_size


def _total_chunks(num_episodes: int, chunk_size: int = CHUNK_SIZE) -> int:
    if num_episodes == 0:
        return 0
    return (num_episodes + chunk_size - 1) // chunk_size


def _standardize_frame(raw_rgb: np.ndarray) -> np.ndarray:
    # decode_image_bit returns frames already in RGB order; only resize here.
    img = cv2.resize(raw_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    if img.shape != (IMG_HEIGHT, IMG_WIDTH, 3):
        raise ValueError(f"Expected frame shape ({IMG_HEIGHT}, {IMG_WIDTH}, 3), got {img.shape}.")
    return img.astype(np.uint8)


def _decode_camera_frames(episode: Dict[str, Any], camera_name: str) -> np.ndarray:
    camera = episode["vision"][camera_name]
    raw = camera.get("colors", camera.get("color"))
    if raw is None:
        raise KeyError(f"Camera '{camera_name}' has no 'colors'/'color' field.")
    # Decode each frame from bytes (already RGB) and resize to (H, W, 3).
    decoded = decode_image_bit(raw)
    if decoded.ndim == 3:
        decoded = decoded[None]
    return np.stack([_standardize_frame(frame) for frame in decoded], axis=0)


def _write_mp4(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (IMG_WIDTH, IMG_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _build_state_columns(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "state.left_arm": np.asarray(state["left_arm_joint_states"], dtype=np.float32),
        "state.left_gripper_close": np.asarray(state["left_ee_joint_states"], dtype=np.float32),
        "state.right_arm": np.asarray(state["right_arm_joint_states"], dtype=np.float32),
        "state.right_gripper_close": np.asarray(state["right_ee_joint_states"], dtype=np.float32),
        "state.left_ee_pose": np.asarray(state["left_ee_poses"], dtype=np.float32),
        "state.right_ee_pose": np.asarray(state["right_ee_poses"], dtype=np.float32),
    }


def _build_action_columns(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "action.left_arm": np.asarray(action["left_arm_joint_states"], dtype=np.float32),
        "action.left_gripper_close": np.asarray(action["left_ee_joint_states"], dtype=np.float32),
        "action.right_arm": np.asarray(action["right_arm_joint_states"], dtype=np.float32),
        "action.right_gripper_close": np.asarray(action["right_ee_joint_states"], dtype=np.float32),
    }


def _frame_rows(
    state_cols: Dict[str, np.ndarray],
    action_cols: Dict[str, np.ndarray],
    n_frames: int,
    episode_index: int,
    global_offset: int,
    task_index: int,
    fps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dt = 1.0 / float(fps)
    for i in range(n_frames):
        row: Dict[str, Any] = {}
        for key, arr in state_cols.items():
            row[key] = arr[i].tolist()
        for key, arr in action_cols.items():
            row[key] = arr[i].tolist()
        row["timestamp"] = float(i * dt)
        row["frame_index"] = int(i)
        row["episode_index"] = int(episode_index)
        row["index"] = int(global_offset + i)
        row["task_index"] = int(task_index)
        rows.append(row)
    return rows


def _accumulate_stats(acc: Dict[str, Dict[str, Any]], key: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1] if arr.ndim >= 2 else 1)
    bucket = acc.setdefault(
        key,
        {
            "count": 0,
            "sum": np.zeros(flat.shape[-1], dtype=np.float64),
            "sum_sq": np.zeros(flat.shape[-1], dtype=np.float64),
            "min": np.full(flat.shape[-1], np.inf, dtype=np.float64),
            "max": np.full(flat.shape[-1], -np.inf, dtype=np.float64),
            "samples": [],
        },
    )
    bucket["count"] += flat.shape[0]
    bucket["sum"] += flat.sum(axis=0)
    bucket["sum_sq"] += np.square(flat).sum(axis=0)
    bucket["min"] = np.minimum(bucket["min"], flat.min(axis=0))
    bucket["max"] = np.maximum(bucket["max"], flat.max(axis=0))
    bucket["samples"].append(flat)


def _finalize_stats(acc: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, list]]:
    stats: Dict[str, Dict[str, list]] = {}
    for key, bucket in acc.items():
        merged = np.concatenate(bucket["samples"], axis=0)
        mean = bucket["sum"] / bucket["count"]
        var = np.maximum(bucket["sum_sq"] / bucket["count"] - mean ** 2, 0.0)
        std = np.sqrt(var)
        q01 = np.percentile(merged, 1, axis=0)
        q99 = np.percentile(merged, 99, axis=0)
        stats[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": bucket["min"].tolist(),
            "max": bucket["max"].tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "count": int(bucket["count"]),
        }
    return stats


def _features_schema(
    state_cols: Dict[str, np.ndarray],
    action_cols: Dict[str, np.ndarray],
    fps: int,
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for key, arr in {**state_cols, **action_cols}.items():
        features[key] = {"dtype": "float32", "shape": [int(arr.shape[-1])]}
    for cam in CAMERA_NAMES:
        features[f"video.{cam}"] = {
            "dtype": "video",
            "shape": [IMG_HEIGHT, IMG_WIDTH, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": IMG_HEIGHT,
                "video.width": IMG_WIDTH,
                "video.codec": VIDEO_CODEC,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def _modality_json(state_cols: Dict[str, np.ndarray], action_cols: Dict[str, np.ndarray]) -> Dict[str, Any]:
    def block(prefix: str, cols: Dict[str, np.ndarray]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for full_key, arr in cols.items():
            short = full_key[len(prefix) + 1 :]
            out[short] = {
                "start": 0,
                "end": int(arr.shape[-1]),
                "original_key": full_key,
            }
        return out

    return {
        "state": block("state", state_cols),
        "action": block("action", action_cols),
        "video": {cam: {"original_key": f"video.{cam}"} for cam in CAMERA_NAMES},
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def _dataset_tag(dataset_name: str, ckpt_name: str, env_cfg_type: str, action_type: str) -> str:
    return f"{dataset_name}-{ckpt_name}-{env_cfg_type}-{action_type}"


def _resolve_source_dir(
    root: Path,
    dataset_name: str,
    raw_task: str,
    env_cfg_type: str,
    legacy_data_roots: Iterable[str],
) -> Path:
    candidates = [root / "data"] + [root / name for name in legacy_data_roots]
    tried: List[Path] = []
    for data_root in candidates:
        source_dir = data_root / dataset_name / raw_task / env_cfg_type / "data"
        tried.append(source_dir)
        if source_dir.exists():
            return source_dir
    raise FileNotFoundError(
        "Source directory does not exist. Tried:\n  " + "\n  ".join(str(p) for p in tried)
    )


def _resolve_instruction(episode: Dict[str, Any], fallback: str) -> str:
    inst = episode.get("instruction")
    if isinstance(inst, bytes):
        inst = inst.decode("utf-8", errors="replace")
    if isinstance(inst, str) and inst.strip():
        return inst.strip()
    return fallback


def convert(args: argparse.Namespace) -> None:
    root = Path(args.root_dir).resolve()
    policy_dir = Path(args.policy_dir).resolve()
    raw_task_dirs = [t.strip() for t in args.raw_task_dirs.split(",") if t.strip()]
    if not raw_task_dirs:
        raise ValueError("--raw-task-dirs resolved to an empty task list")

    legacy_roots = [x.strip() for x in args.legacy_data_roots.split(",") if x.strip()]

    # Gather episodes from every raw task dir, capped per task by expert_data_num.
    episode_jobs: List[Tuple[Path, str]] = []
    for raw_task in raw_task_dirs:
        source_dir = _resolve_source_dir(
            root, args.dataset_name, raw_task, args.env_cfg_type, legacy_roots
        )
        task_episodes = sorted(source_dir.glob("episode_*.hdf5"))[: int(args.expert_data_num)]
        if not task_episodes:
            raise FileNotFoundError(f"No episode_*.hdf5 files under {source_dir}")
        episode_jobs.extend((ep, raw_task) for ep in task_episodes)

    if args.dataset_id:
        dataset_id = args.dataset_id
    elif len(raw_task_dirs) == 1:
        dataset_id = _dataset_tag(
            args.dataset_name, args.ckpt_name, args.env_cfg_type, args.action_type
        )
    else:
        dataset_id = "cotrain"
    output_root = policy_dir / "data" / dataset_id

    # Sanity-check robot dim for the requested action_type.
    robot_info = get_robot_action_dim_info(args.env_cfg_type)
    expected_action_dim = sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"])

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    fps = int(args.fps) if args.fps else DEFAULT_FPS
    tasks_index: Dict[str, int] = {}
    episodes_meta: List[Dict[str, Any]] = []
    episodes_stats: List[Dict[str, Any]] = []
    stats_acc: Dict[str, Dict[str, Any]] = {}
    total_frames = 0
    sample_state_cols: Dict[str, np.ndarray] | None = None
    sample_action_cols: Dict[str, np.ndarray] | None = None

    episode_bar = tqdm(
        episode_jobs,
        desc=f"convert {dataset_id}",
        unit="ep",
        dynamic_ncols=True,
    )
    for episode_index, (episode_path, fallback_instruction) in enumerate(episode_bar):
        episode = load_hdf5(str(episode_path))
        state_cols = _build_state_columns(episode["state"])
        action_cols = _build_action_columns(episode["action"])
        n_frames = next(iter(state_cols.values())).shape[0]
        for key, arr in {**state_cols, **action_cols}.items():
            if arr.shape[0] != n_frames:
                raise ValueError(f"{episode_path.name}: column {key} length {arr.shape[0]} != {n_frames}")

        packed_action_dim = (
            action_cols["action.left_arm"].shape[-1]
            + action_cols["action.left_gripper_close"].shape[-1]
            + action_cols["action.right_arm"].shape[-1]
            + action_cols["action.right_gripper_close"].shape[-1]
        )
        if packed_action_dim != expected_action_dim:
            raise ValueError(
                f"{episode_path.name}: packed action dim {packed_action_dim} "
                f"!= env_cfg_type {args.env_cfg_type} expected {expected_action_dim}"
            )

        instruction = _resolve_instruction(episode, fallback=fallback_instruction)
        if instruction not in tasks_index:
            tasks_index[instruction] = len(tasks_index)
        task_index = tasks_index[instruction]

        rows = _frame_rows(
            state_cols,
            action_cols,
            n_frames=n_frames,
            episode_index=episode_index,
            global_offset=total_frames,
            task_index=task_index,
            fps=fps,
        )
        chunk_index = _episode_chunk(episode_index)
        data_dir = output_root / "data" / f"chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        for cam in CAMERA_NAMES:
            frames = _decode_camera_frames(episode, cam)
            if frames.shape[0] != n_frames:
                raise ValueError(
                    f"{episode_path.name}: camera {cam} produced {frames.shape[0]} frames vs {n_frames} states"
                )
            video_path = (
                output_root
                / "videos"
                / f"chunk-{chunk_index:03d}"
                / f"video.{cam}"
                / f"episode_{episode_index:06d}.mp4"
            )
            _write_mp4(video_path, frames, fps=fps)

        episode_stats_buckets: Dict[str, Dict[str, Any]] = {}
        for key, arr in {**state_cols, **action_cols}.items():
            _accumulate_stats(stats_acc, key, arr)
            _accumulate_stats(episode_stats_buckets, key, arr)

        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": int(n_frames),
            }
        )
        episodes_stats.append(
            {
                "episode_index": episode_index,
                "stats": _finalize_stats(episode_stats_buckets),
            }
        )
        total_frames += n_frames
        sample_state_cols = state_cols
        sample_action_cols = action_cols
        episode_bar.set_postfix(frames=n_frames, total=total_frames, tasks=len(tasks_index))

    episode_bar.close()
    assert sample_state_cols is not None and sample_action_cols is not None
    stats = _finalize_stats(stats_acc)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "arx_x5",
        "total_episodes": len(episodes_meta),
        "total_frames": int(total_frames),
        "total_tasks": len(tasks_index),
        "total_videos": len(episodes_meta) * len(CAMERA_NAMES),
        "total_chunks": _total_chunks(len(episodes_meta)),
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features_schema(sample_state_cols, sample_action_cols, fps),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (meta_dir / "modality.json").write_text(
        json.dumps(_modality_json(sample_state_cols, sample_action_cols), indent=2), encoding="utf-8"
    )
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as fp:
        for task_text, task_idx in sorted(tasks_index.items(), key=lambda kv: kv[1]):
            fp.write(json.dumps({"task_index": task_idx, "task": task_text}) + "\n")
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as fp:
        for entry in episodes_meta:
            fp.write(json.dumps(entry) + "\n")
    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as fp:
        for entry in episodes_stats:
            fp.write(json.dumps(entry) + "\n")

    tqdm.write(
        f"[convert] wrote {len(episodes_meta)} episodes / {total_frames} frames "
        f"to {output_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="workspace root containing data/")
    parser.add_argument("--policy-dir", required=True, help="policy/LDA_1B directory for output")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--ckpt-name", required=True, help="artifact ckpt_name (README §4.2)")
    parser.add_argument(
        "--raw-task-dirs",
        default=None,
        help="raw HDF5 task dir(s) under data/<dataset>/; comma-separated to merge",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="deprecated alias for --raw-task-dirs",
    )
    parser.add_argument("--env-cfg-type", required=True)
    parser.add_argument("--expert-data-num", required=True)
    parser.add_argument("--action-type", required=True, choices=("joint", "ee"))
    parser.add_argument("--dataset-id", default=None, help="override output folder name")
    parser.add_argument(
        "--legacy-data-roots",
        default="final_data",
        help="comma-separated legacy roots tried after data/ (default: final_data)",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()
    raw = args.raw_task_dirs or args.task_name
    if not raw:
        parser.error("one of --raw-task-dirs or --task-name is required")
    args.raw_task_dirs = raw
    convert(args)


if __name__ == "__main__":
    main()
