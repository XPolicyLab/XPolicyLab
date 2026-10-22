#!/usr/bin/env python
"""Convert raw EgoDex MP4+HDF5 episodes into a LeRobot v3 dataset for WSABase.

This converter is tailored to the current WSABase training design:

- keep the single egocentric RGB stream as `observation.image`
- keep the language task as `task`
- write dummy zero `observation.state` / `action` placeholders

The default "fast" mode avoids per-frame decode + re-encode. It writes LeRobot
metadata/parquet files directly and reuses the original MP4 files by hard-link,
symlink, or copy.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import datasets
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("opencv-python-headless is required for EgoDex conversion.") from exc

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("h5py is required for EgoDex conversion.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_FEATURES,
    DEFAULT_VIDEO_PATH,
    create_empty_dataset_info,
    get_hf_features_from_features,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import get_video_info

DEFAULT_FPS = 30
DEFAULT_EPISODES_PER_CHUNK = 1000
IMAGENET_MEAN = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
IMAGENET_STD = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw EgoDex MP4+HDF5 episodes into a local LeRobot v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing EgoDex task folders and paired *.hdf5/*.mp4 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the LeRobot dataset. Must not already exist.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Logical dataset name stored in LeRobot metadata. Defaults to output directory name.",
    )
    parser.add_argument(
        "--repo-layout",
        type=str,
        choices=["single", "task_dir"],
        default="single",
        help=(
            "How to organize converted LeRobot repos. "
            "'single' writes one dataset under output-dir; "
            "'task_dir' writes one dataset per relative EgoDex task directory under output-dir/task_rel."
        ),
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="egodex_v",
        help="robot_type written into meta/info.json.",
    )
    parser.add_argument(
        "--dummy-dim",
        type=int,
        default=2,
        help="Dimension of the dummy state/action placeholder vectors.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Source EgoDex video FPS before frame subsampling.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep one frame every N frames. Effective dataset FPS becomes fps / frame_stride.",
    )
    parser.add_argument(
        "--task-regex",
        type=str,
        default=None,
        help="Optional regex applied to the relative task path for filtering episodes.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most this many paired episodes after filtering.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Optional cap on converted frames per episode after stride.",
    )
    parser.add_argument(
        "--conversion-mode",
        type=str,
        choices=["fast", "reencode"],
        default="fast",
        help="Use direct metadata writing plus linked/copied MP4s, or the slower re-encode path.",
    )
    parser.add_argument(
        "--video-file-mode",
        type=str,
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="How fast mode materializes MP4 files inside the LeRobot dataset.",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=DEFAULT_EPISODES_PER_CHUNK,
        help="Number of episodes stored under each chunk directory in fast mode.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="LeRobot async image-writer processes used before video encoding in re-encode mode.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="LeRobot async image-writer threads used before video encoding in re-encode mode.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=64,
        help="Number of episodes to accumulate before LeRobot batch video encoding in re-encode mode.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Emit an info log after every N converted episodes.",
    )
    return parser.parse_args()


def normalize_h5_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return normalize_h5_attr(value.item())
    return str(value)


def normalize_task_text(value) -> str:
    return normalize_h5_attr(value).strip()


def select_task_description(h5_root: h5py.File, fallback_task: str) -> str:
    attrs = h5_root.attrs
    if "llm_description" not in attrs:
        return normalize_task_text(fallback_task)

    llm_type = normalize_h5_attr(attrs.get("llm_type", ""))
    if llm_type == "reversible":
        which = normalize_h5_attr(attrs.get("which_llm_description", "1"))
        if which == "2" and "llm_description2" in attrs:
            return normalize_task_text(attrs["llm_description2"])
    return normalize_task_text(attrs["llm_description"])


def discover_episode_pairs(
    input_root: Path,
    *,
    task_regex: str | None,
    max_episodes: int | None,
) -> list[tuple[Path, Path, str]]:
    pattern = re.compile(task_regex) if task_regex else None
    pairs: list[tuple[Path, Path, str]] = []

    for h5_path in sorted(input_root.rglob("*.hdf5")):
        mp4_path = h5_path.with_suffix(".mp4")
        if not mp4_path.is_file():
            logging.warning("Skipping %s because paired MP4 is missing.", h5_path)
            continue

        task_rel = str(h5_path.parent.relative_to(input_root)).replace("\\", "/")
        if pattern and not pattern.search(task_rel):
            continue

        pairs.append((h5_path, mp4_path, task_rel))
        if max_episodes is not None and len(pairs) >= max_episodes:
            break

    return pairs


def task_rel_to_repo_id(task_rel: str) -> str:
    safe = task_rel.replace("\\", "/").strip("/")
    safe = safe.replace("/", "__")
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", safe)
    return safe


def get_video_shape(mp4_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(mp4_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read frame size from video: {mp4_path}")

    return height, width


def build_features(height: int, width: int, dummy_dim: int) -> dict[str, dict]:
    dummy_names = [f"dummy_{i}" for i in range(dummy_dim)]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (dummy_dim,),
            "names": dummy_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dummy_dim,),
            "names": dummy_names,
        },
        "observation.image": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def iter_progress(items: Iterable, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def make_zero_vector_stats(dim: int, count: int) -> dict[str, np.ndarray]:
    zero = np.zeros((dim,), dtype=np.float32)
    return {
        "min": zero.copy(),
        "max": zero.copy(),
        "mean": zero.copy(),
        "std": zero.copy(),
        "count": np.array([count], dtype=np.int64),
        "q01": zero.copy(),
        "q10": zero.copy(),
        "q50": zero.copy(),
        "q90": zero.copy(),
        "q99": zero.copy(),
    }


def make_image_placeholder_stats(count: int) -> dict[str, np.ndarray]:
    zeros = np.zeros((3, 1, 1), dtype=np.float32)
    ones = np.ones((3, 1, 1), dtype=np.float32)
    return {
        "min": zeros.copy(),
        "max": ones.copy(),
        "mean": IMAGENET_MEAN.copy(),
        "std": IMAGENET_STD.copy(),
        "count": np.array([count], dtype=np.int64),
        "q01": zeros.copy(),
        "q10": zeros.copy(),
        "q50": IMAGENET_MEAN.copy(),
        "q90": ones.copy(),
        "q99": ones.copy(),
    }


def resolve_materialization_order(mode: str) -> list[str]:
    if mode == "auto":
        return ["hardlink", "symlink", "copy"]
    return [mode]


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def materialize_video_file(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in resolve_materialization_order(mode):
        try:
            safe_unlink(target)
            if attempt == "hardlink":
                os.link(source, target)
            elif attempt == "symlink":
                target.symlink_to(source.resolve())
            elif attempt == "copy":
                shutil.copy2(source, target)
            else:  # pragma: no cover - impossible with argparse choices
                raise ValueError(f"Unsupported video-file mode: {attempt}")
            return attempt
        except Exception as exc:  # pragma: no cover - depends on filesystem
            last_error = exc
            safe_unlink(target)

    raise RuntimeError(f"Failed to materialize video {source} -> {target}: {last_error}") from last_error


def get_episode_info(
    h5_path: Path,
    task_rel: str,
    *,
    frame_stride: int,
    max_frames_per_episode: int | None,
) -> tuple[str, np.ndarray]:
    with h5py.File(h5_path, "r") as root:
        episode_task = select_task_description(root, fallback_task=task_rel)
        episode_len = int(root["/transforms/camera"].shape[0])

    if episode_len <= 0:
        raise ValueError(f"Episode has zero frames: {h5_path}")

    kept_source_indices = np.arange(0, episode_len, frame_stride, dtype=np.int64)
    if max_frames_per_episode is not None:
        kept_source_indices = kept_source_indices[: max_frames_per_episode]

    if kept_source_indices.size == 0:
        raise ValueError(f"No frames were selected for episode: {h5_path}")

    return episode_task, kept_source_indices


def make_episode_rows(
    *,
    episode_index: int,
    task_index: int,
    kept_source_indices: np.ndarray,
    fps: int,
    dummy_state: np.ndarray,
    global_frame_offset: int,
) -> tuple[dict[str, np.ndarray], int]:
    output_frames = int(kept_source_indices.size)
    timestamps = kept_source_indices.astype(np.float32) / np.float32(fps)

    rows = {
        "timestamp": timestamps,
        "frame_index": np.arange(output_frames, dtype=np.int64),
        "episode_index": np.full(output_frames, episode_index, dtype=np.int64),
        "index": np.arange(global_frame_offset, global_frame_offset + output_frames, dtype=np.int64),
        "task_index": np.full(output_frames, task_index, dtype=np.int64),
        "observation.state": np.tile(dummy_state[None, :], (output_frames, 1)),
        "action": np.tile(dummy_state[None, :], (output_frames, 1)),
    }
    return rows, output_frames


def append_rows(buffer: dict[str, list[np.ndarray]], rows: dict[str, np.ndarray]) -> None:
    for key, value in rows.items():
        buffer[key].append(value)


def flush_data_chunk(
    buffer: dict[str, list[np.ndarray]],
    *,
    output_dir: Path,
    chunk_index: int,
    hf_features: datasets.Features,
) -> int:
    if not buffer:
        return 0

    data_dict: dict[str, np.ndarray] = {}
    for key, pieces in buffer.items():
        data_dict[key] = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]

    data_path = output_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_index, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_dict(data_dict, features=hf_features, split="train").to_parquet(str(data_path))
    return int(data_dict["index"].shape[0])


def flush_episode_chunk(episodes_rows: list[dict], *, output_dir: Path, chunk_index: int) -> int:
    if not episodes_rows:
        return 0

    columns = {key: [row[key] for row in episodes_rows] for key in episodes_rows[0]}
    episode_path = output_dir / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_index, file_index=0)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_dict(columns, split="train").to_parquet(str(episode_path))
    return len(episodes_rows)


def convert_episode_reencode(
    dataset: LeRobotDataset,
    h5_path: Path,
    mp4_path: Path,
    task_rel: str,
    *,
    frame_stride: int,
    max_frames_per_episode: int | None,
    dummy_state: np.ndarray,
) -> tuple[int, str]:
    episode_task, kept_source_indices = get_episode_info(
        h5_path,
        task_rel,
        frame_stride=frame_stride,
        max_frames_per_episode=max_frames_per_episode,
    )
    selected_frames = set(int(idx) for idx in kept_source_indices.tolist())

    capture = cv2.VideoCapture(str(mp4_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    frame_index = 0
    output_frames = 0
    while frame_index <= int(kept_source_indices[-1]):
        ok, frame_bgr = capture.read()
        if not ok:
            logging.warning(
                "Video ended early for %s at source frame %d (expected at least %d frames).",
                mp4_path,
                frame_index,
                int(kept_source_indices[-1]) + 1,
            )
            break

        if frame_index not in selected_frames:
            frame_index += 1
            continue

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = {
            "observation.image": image_rgb,
            "observation.state": dummy_state.copy(),
            "action": dummy_state.copy(),
            "task": episode_task,
        }
        dataset.add_frame(frame)
        output_frames += 1
        frame_index += 1

    capture.release()

    if output_frames == 0:
        raise ValueError(f"No frames were converted for episode: {h5_path}")

    dataset.save_episode()
    return output_frames, episode_task


def convert_fast(args: argparse.Namespace, pairs: list[tuple[Path, Path, str]], repo_id: str) -> None:
    first_h, first_w = get_video_shape(pairs[0][1])
    core_features = build_features(first_h, first_w, args.dummy_dim)
    features = {**core_features, **DEFAULT_FEATURES}

    effective_fps = args.fps / args.frame_stride
    if abs(effective_fps - round(effective_fps)) > 1e-6:
        raise ValueError(
            f"fps/frame_stride must be an integer for LeRobot metadata, got {args.fps}/{args.frame_stride}"
        )
    effective_fps = int(round(effective_fps))

    info = create_empty_dataset_info(
        codebase_version="v3.0",
        fps=effective_fps,
        features=features,
        use_videos=True,
        robot_type=args.robot_type,
        chunks_size=args.episodes_per_chunk,
    )
    try:
        info["features"]["observation.image"]["info"] = get_video_info(pairs[0][1])
    except Exception as exc:  # pragma: no cover - depends on video backend
        logging.warning("Could not probe source video info from %s: %s", pairs[0][1], exc)

    write_info(info, args.output_dir)

    task_to_index: dict[str, int] = {}
    data_buffer: dict[str, list[np.ndarray]] = defaultdict(list)
    episodes_buffer: list[dict] = []
    video_mode_counts: dict[str, int] = defaultdict(int)
    hf_features = get_hf_features_from_features(features)
    dummy_state = np.zeros((args.dummy_dim,), dtype=np.float32)

    current_chunk_index = 0
    total_frames = 0
    total_episodes = 0

    for episode_index, (h5_path, mp4_path, task_rel) in enumerate(
        iter_progress(pairs, desc="Converting EgoDex episodes"), start=0
    ):
        chunk_index = episode_index // args.episodes_per_chunk
        if chunk_index != current_chunk_index:
            flush_data_chunk(data_buffer, output_dir=args.output_dir, chunk_index=current_chunk_index, hf_features=hf_features)
            flush_episode_chunk(episodes_buffer, output_dir=args.output_dir, chunk_index=current_chunk_index)
            data_buffer = defaultdict(list)
            episodes_buffer = []
            current_chunk_index = chunk_index

        task_name, kept_source_indices = get_episode_info(
            h5_path,
            task_rel,
            frame_stride=args.frame_stride,
            max_frames_per_episode=args.max_frames_per_episode,
        )
        if task_name not in task_to_index:
            task_to_index[task_name] = len(task_to_index)
        task_index = task_to_index[task_name]

        rows, episode_frames = make_episode_rows(
            episode_index=episode_index,
            task_index=task_index,
            kept_source_indices=kept_source_indices,
            fps=args.fps,
            dummy_state=dummy_state,
            global_frame_offset=total_frames,
        )
        append_rows(data_buffer, rows)

        video_chunk_index = chunk_index
        video_file_index = episode_index % args.episodes_per_chunk
        dst_video_path = args.output_dir / DEFAULT_VIDEO_PATH.format(
            video_key="observation.image",
            chunk_index=video_chunk_index,
            file_index=video_file_index,
        )
        actual_video_mode = materialize_video_file(mp4_path, dst_video_path, args.video_file_mode)
        video_mode_counts[actual_video_mode] += 1

        episodes_buffer.append(
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": episode_frames,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + episode_frames,
                "data/chunk_index": chunk_index,
                "data/file_index": 0,
                "videos/observation.image/chunk_index": video_chunk_index,
                "videos/observation.image/file_index": video_file_index,
                "videos/observation.image/from_timestamp": 0.0,
                "videos/observation.image/to_timestamp": float(episode_frames / effective_fps),
                "meta/episodes/chunk_index": chunk_index,
                "meta/episodes/file_index": 0,
            }
        )

        total_frames += episode_frames
        total_episodes += 1

        if (episode_index + 1) % args.log_every == 0:
            logging.info(
                "Fast-converted %d episodes / %d frames so far into %s",
                total_episodes,
                total_frames,
                args.output_dir,
            )

    flush_data_chunk(data_buffer, output_dir=args.output_dir, chunk_index=current_chunk_index, hf_features=hf_features)
    flush_episode_chunk(episodes_buffer, output_dir=args.output_dir, chunk_index=current_chunk_index)

    tasks_df = pd.DataFrame({"task_index": list(task_to_index.values())}, index=list(task_to_index.keys()))
    write_tasks(tasks_df, args.output_dir)

    stats = {
        "observation.image": make_image_placeholder_stats(total_frames),
        "observation.state": make_zero_vector_stats(args.dummy_dim, total_frames),
        "action": make_zero_vector_stats(args.dummy_dim, total_frames),
    }
    write_stats(stats, args.output_dir)

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_to_index)
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_info(info, args.output_dir)

    logging.info(
        "Finished fast conversion: %d episodes, %d frames, %d unique task strings -> %s",
        total_episodes,
        total_frames,
        len(task_to_index),
        args.output_dir,
    )
    logging.info("Video materialization summary: %s", dict(video_mode_counts))
    logging.info(
        "Stats path for wsa_base_pretrain.sh should later be written to outputs/norm_stats/%s/delta/stats.json",
        args.robot_type,
    )


def convert_reencode(args: argparse.Namespace, pairs: list[tuple[Path, Path, str]], repo_id: str) -> None:
    first_h, first_w = get_video_shape(pairs[0][1])
    features = build_features(first_h, first_w, args.dummy_dim)
    effective_fps = args.fps / args.frame_stride
    if abs(effective_fps - round(effective_fps)) > 1e-6:
        raise ValueError(
            f"fps/frame_stride must be an integer for LeRobot metadata, got {args.fps}/{args.frame_stride}"
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(effective_fps)),
        features=features,
        root=args.output_dir,
        robot_type=args.robot_type,
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    dummy_state = np.zeros((args.dummy_dim,), dtype=np.float32)
    total_frames = 0
    total_episodes = 0
    converted_tasks: set[str] = set()
    try:
        for idx, (h5_path, mp4_path, task_rel) in enumerate(
            iter_progress(pairs, desc="Converting EgoDex episodes"), start=1
        ):
            frames_written, task_name = convert_episode_reencode(
                dataset,
                h5_path,
                mp4_path,
                task_rel,
                frame_stride=args.frame_stride,
                max_frames_per_episode=args.max_frames_per_episode,
                dummy_state=dummy_state,
            )
            total_frames += frames_written
            total_episodes += 1
            converted_tasks.add(task_name)

            if idx % args.log_every == 0:
                logging.info(
                    "Re-encoded %d episodes / %d frames so far into %s",
                    total_episodes,
                    total_frames,
                    args.output_dir,
                )
    except Exception:
        dataset.finalize()
        raise

    dataset.finalize()
    logging.info(
        "Finished re-encode conversion: %d episodes, %d frames, %d unique task strings -> %s",
        total_episodes,
        total_frames,
        len(converted_tasks),
        args.output_dir,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be >= 1")
    if args.dummy_dim <= 0:
        raise ValueError("--dummy-dim must be >= 1")
    if args.robot_type == "egodex_v" and args.dummy_dim != 2:
        raise ValueError("--dummy-dim must be 2 for robot_type=egodex_v so state/action stay vector-shaped.")
    if args.batch_encoding_size <= 0:
        raise ValueError("--batch-encoding-size must be >= 1")
    if args.image_writer_processes < 0:
        raise ValueError("--image-writer-processes must be >= 0")
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be >= 0")
    if args.episodes_per_chunk <= 0:
        raise ValueError("--episodes-per-chunk must be >= 1")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")


def convert_grouped_by_task_dir(args: argparse.Namespace, pairs: list[tuple[Path, Path, str]]) -> None:
    grouped_pairs: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for h5_path, mp4_path, task_rel in pairs:
        grouped_pairs[task_rel].append((h5_path, mp4_path, task_rel))

    args.output_dir.mkdir(parents=True, exist_ok=False)

    logging.info(
        "Converting EgoDex into %d task-directory repos under %s",
        len(grouped_pairs),
        args.output_dir,
    )

    total_episodes = 0
    for group_index, task_rel in enumerate(sorted(grouped_pairs.keys()), start=1):
        group_output_dir = args.output_dir / Path(task_rel)
        group_repo_id = task_rel_to_repo_id(task_rel)
        group_args = copy.copy(args)
        group_args.output_dir = group_output_dir
        group_pairs = grouped_pairs[task_rel]

        logging.info(
            "[%d/%d] Converting task repo %s (%d episodes) -> %s",
            group_index,
            len(grouped_pairs),
            task_rel,
            len(group_pairs),
            group_output_dir,
        )

        if args.conversion_mode == "fast":
            convert_fast(group_args, group_pairs, group_repo_id)
        else:
            convert_reencode(group_args, group_pairs, group_repo_id)
        total_episodes += len(group_pairs)

    logging.info(
        "Finished grouped conversion: %d repos, %d total episodes -> %s",
        len(grouped_pairs),
        total_episodes,
        args.output_dir,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_args(args)

    pairs = discover_episode_pairs(
        args.input_root,
        task_regex=args.task_regex,
        max_episodes=args.max_episodes,
    )
    if not pairs:
        raise RuntimeError(f"No paired EgoDex episodes found under {args.input_root}")

    repo_id = args.repo_id or args.output_dir.name
    logging.info(
        "Discovered %d paired EgoDex episodes under %s; conversion_mode=%s",
        len(pairs),
        args.input_root,
        args.conversion_mode,
    )

    if args.repo_layout == "task_dir":
        convert_grouped_by_task_dir(args, pairs)
        return

    if args.conversion_mode == "fast":
        convert_fast(args, pairs, repo_id)
    else:
        convert_reencode(args, pairs, repo_id)


if __name__ == "__main__":
    main()
