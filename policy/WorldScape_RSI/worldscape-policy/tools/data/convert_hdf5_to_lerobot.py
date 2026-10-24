#!/usr/bin/env python3
"""Materialize raw AgileX-style HDF5 episodes into LeRobot v2 layout.

Unlike ``convert_hdf5_to_native_meta.py``, this script writes parquet tabular
data and MP4 videos under ``data/`` and ``videos/``, then emits the full
``meta/`` bundle expected by ``NativeLeRobotDataset``.

Usage:
  python tools/data/convert_hdf5_to_lerobot.py \\
      --dataset-path /path/to/raw_hdf5 \\
      --output-path /path/to/lerobot_dataset \\
      --embodiment agilex

  python tools/data/convert_hdf5_to_lerobot.py \\
      --dataset-path /path/to/raw_hdf5 \\
      --output-path /path/to/lerobot_dataset \\
      --task-map /path/to/task_id_to_prompt.json \\
      --max-episodes 10 \\
      --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

_DATA = Path(__file__).resolve().parent
_REPO = _DATA.parent.parent
_SRC = _REPO / "src"
for path in (_DATA, _SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    DEFAULT_AGILEX_STATE_MAP,
    DEFAULT_HDF5_VIDEO_MAP,
    agilex_eef_arrays,
    array_stats,
    build_embodiment_json,
    build_lerobot_info,
    build_modality,
    encode_video,
    episode_length,
    load_task_map,
    parse_hdf5_video_map,
    parse_lerobot_video_map,
    parse_mapping,
    read_fps,
    read_hdf5_camera_frames,
    read_optional_bool_series,
    read_optional_text_series,
    scan_hdf5,
    task_id_from_hdf5,
    validate_embodiment,
    write_json,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def convert_episode(
    *,
    path: Path,
    episode_index: int,
    output_path: Path,
    chunk_size: int,
    fps: float,
    hdf5_video_map: OrderedDict[str, str],
    lerobot_video_map: OrderedDict[str, str],
    state_column: str,
    action_column: str,
    task_key: str,
    task_index: int,
    task_text: str,
    end_pose_key: str,
    qpos_key: str,
    arm_order: str,
    align_to_first_frame: bool,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        state, action = agilex_eef_arrays(
            handle,
            end_pose_key=end_pose_key,
            qpos_key=qpos_key,
            arm_order=arm_order,
            align_to_first_frame=align_to_first_frame,
            state_column=state_column,
            action_column=action_column,
        )
        length = min(len(state), len(action))
        state = state[:length]
        action = action[:length]
        is_exec = read_optional_bool_series(handle, length)
        if is_exec is None:
            is_exec = np.ones(length, dtype=bool)
        event_instruction = read_optional_text_series(handle, "event_instruction", length)
        subtask = read_optional_text_series(handle, "subtask", length)
        high_level = read_optional_text_series(handle, "language", length)
        if high_level is None:
            high_level = np.full(length, task_text, dtype=object)

        chunk_index = episode_index // chunk_size
        data_dir = output_path / f"data/chunk-{chunk_index:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        for short_name, hdf5_source in hdf5_video_map.items():
            video_key = lerobot_video_map[short_name]
            frames = read_hdf5_camera_frames(handle, hdf5_source)
            frames = frames[:length]
            video_path = (
                output_path
                / f"videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            )
            encode_video(frames, video_path, fps)

    timestamps = np.arange(length, dtype=np.float32) / float(fps)
    frame = pd.DataFrame(
        {
            state_column: [row.astype(np.float32) for row in state],
            action_column: [row.astype(np.float32) for row in action],
            "timestamp": timestamps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
            "is_exec": is_exec.astype(bool),
            task_key: np.full(length, task_text, dtype=object),
        }
    )
    if event_instruction is not None:
        frame["event_instruction"] = event_instruction
    elif subtask is not None:
        frame["event_instruction"] = subtask
    if high_level is not None:
        frame["annotation.language.language_instruction"] = high_level

    parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
    frame.to_parquet(parquet_path, index=False)
    return {
        "episode_index": episode_index,
        "tasks": [task_text],
        "task_index": task_index,
        "length": length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--embodiment", default="agilex")
    parser.add_argument("--embodiment-tag", dest="embodiment", help=argparse.SUPPRESS)
    parser.add_argument("--glob", default="**/episode.hdf5")
    parser.add_argument("--sources-list", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--state-keys", default=None)
    parser.add_argument("--action-keys", default=None)
    parser.add_argument("--hdf5-video-keys", default=None, help="HDF5 camera source mapping")
    parser.add_argument("--lerobot-video-keys", default=None, help="LeRobot video feature mapping")
    parser.add_argument("--state-column", default="observation.eef6d")
    parser.add_argument("--action-column", default="action.eef6d")
    parser.add_argument("--end-pose-key", default="observations.end_pose")
    parser.add_argument("--qpos-key", default="observations.qpos")
    parser.add_argument("--arm-order", choices=("left_first", "right_first"), default="left_first")
    parser.add_argument("--align-to-first-frame", action="store_true")
    parser.add_argument("--task-key", default="annotation.language.language_instruction")
    parser.add_argument("--task-map", type=Path, default=None)
    parser.add_argument("--default-task", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    output_path = args.output_path.resolve()
    embodiment = validate_embodiment(args.embodiment)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if output_path.exists():
        if not args.force:
            raise SystemExit(f"Output path already exists: {output_path}. Use --force to overwrite.")
        import shutil

        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    state_map = parse_mapping(args.state_keys, DEFAULT_AGILEX_STATE_MAP)
    action_map = parse_mapping(args.action_keys, DEFAULT_AGILEX_STATE_MAP)
    hdf5_video_map = parse_hdf5_video_map(args.hdf5_video_keys)
    lerobot_video_map = parse_lerobot_video_map(args.lerobot_video_keys)
    if set(hdf5_video_map) != set(lerobot_video_map):
        raise SystemExit("HDF5 and LeRobot video key names must match one-to-one")

    h5_paths = scan_hdf5(dataset_path, args.glob, args.sources_list, args.max_episodes)
    log.info("found %d HDF5 episodes", len(h5_paths))

    prompt_map = load_task_map(dataset_root, args.task_map)
    task_to_idx: OrderedDict[str, int] = OrderedDict()
    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    fps_values: list[float] = []

    def add_task(prompt: str) -> int:
        if prompt not in task_to_idx:
            task_to_idx[prompt] = len(task_to_idx)
            tasks.append({"task_index": task_to_idx[prompt], "task": prompt})
        return task_to_idx[prompt]

    for prompt in prompt_map.values():
        add_task(prompt)

    for episode_index, path in enumerate(tqdm(h5_paths, desc="Converting episodes")):
        with h5py.File(path, "r") as handle:
            fps_values.append(read_fps(handle, args.fps))
            source_task_id = task_id_from_hdf5(handle, path)
        prompt = prompt_map.get(source_task_id) or args.default_task or source_task_id
        task_index = add_task(prompt)
        episode_meta = convert_episode(
            path=path,
            episode_index=episode_index,
            output_path=output_path,
            chunk_size=args.chunk_size,
            fps=fps_values[-1],
            hdf5_video_map=hdf5_video_map,
            lerobot_video_map=lerobot_video_map,
            state_column=args.state_column,
            action_column=args.action_column,
            task_key=args.task_key,
            task_index=task_index,
            task_text=prompt,
            end_pose_key=args.end_pose_key,
            qpos_key=args.qpos_key,
            arm_order=args.arm_order,
            align_to_first_frame=args.align_to_first_frame,
        )
        episodes.append(episode_meta)

    fps_value = fps_values[0] if fps_values else args.fps
    modality = build_modality(
        state_map=state_map,
        action_map=action_map,
        video_map=lerobot_video_map,
        state_column=args.state_column,
        action_column=args.action_column,
        task_key=args.task_key,
    )
    info = build_lerobot_info(
        embodiment=embodiment,
        episodes=episodes,
        video_map=lerobot_video_map,
        first_h5=h5_paths[0],
        task_count=len(tasks),
        state_map=state_map,
        action_map=action_map,
        state_column=args.state_column,
        action_column=args.action_column,
        task_key=args.task_key,
        fps=fps_value,
        chunk_size=args.chunk_size,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    meta_dir = output_path / "meta"
    write_jsonl(meta_dir / "tasks.jsonl", tasks, force=True)
    write_jsonl(meta_dir / "episodes.jsonl", episodes, force=True)
    write_json(meta_dir / "modality.json", modality, force=True)
    write_json(meta_dir / "embodiment.json", build_embodiment_json(embodiment), force=True)
    write_json(meta_dir / "info.json", info, force=True)

    if not args.skip_stats:
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for path in tqdm(h5_paths, desc="Computing stats"):
            with h5py.File(path, "r") as handle:
                state, action = agilex_eef_arrays(
                    handle,
                    end_pose_key=args.end_pose_key,
                    qpos_key=args.qpos_key,
                    arm_order=args.arm_order,
                    align_to_first_frame=args.align_to_first_frame,
                    state_column=args.state_column,
                    action_column=args.action_column,
                )
            states.append(state)
            actions.append(action)
        stats = {
            args.state_column: array_stats(np.concatenate(states, axis=0)),
            args.action_column: array_stats(np.concatenate(actions, axis=0)),
        }
        write_json(meta_dir / "stats.json", stats, force=True)

    print("\n" + "=" * 64)
    print("HDF5 -> LeRobot conversion complete")
    print(f"  input: {dataset_root}")
    print(f"  output: {output_path}")
    print(f"  episodes: {len(episodes)}")
    print(f"  frames: {info['total_frames']}")
    print(f"  embodiment: {embodiment}")
    print("=" * 64)
    print("\nNext steps:")
    print("  python tools/data/convert_lerobot_to_native_meta.py --dataset-path", output_path)
    print("  Use worldscape_lerobot_* datasets with dataset_kwargs.data_root pointing here.")


if __name__ == "__main__":
    main()
