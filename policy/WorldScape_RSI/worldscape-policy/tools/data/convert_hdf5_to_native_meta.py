#!/usr/bin/env python3
"""Generate native HDF5 dataset metadata for WorldScape Policy training.

This mirrors DreamZero's ``convert_hdf5_to_gear.py``: it scans raw episode HDF5
files and writes LeRobot-style ``meta/`` files without materializing parquet or
MP4. The output is consumed by ``NativeHDF5Dataset`` and registered datasets such
as ``worldscape_hdf5_text``, ``worldscape_hdf5_goal``, and ``worldscape_hdf5_demo``.

Usage:
  python tools/data/convert_hdf5_to_native_meta.py \\
      --dataset-path /path/to/raw_hdf5 \\
      --embodiment agilex

  python tools/data/convert_hdf5_to_native_meta.py \\
      --dataset-path /path/to/raw_hdf5 \\
      --output-path /path/to/raw_hdf5 \\
      --embodiment agilex \\
      --task-map /path/to/task_id_to_prompt.json \\
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
from tqdm import tqdm

_DATA = Path(__file__).resolve().parent
_REPO = _DATA.parent.parent
_SRC = _REPO / "src"
for path in (_DATA, _SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    DEFAULT_AGILEX_STATE_MAP,
    agilex_eef_arrays,
    array_stats,
    build_embodiment_json,
    build_hdf5_info,
    build_modality,
    episode_length,
    load_task_map,
    parse_hdf5_video_map,
    parse_mapping,
    read_fps,
    relative_path,
    scan_hdf5,
    task_id_from_hdf5,
    validate_embodiment,
    write_json,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--embodiment", default="agilex")
    parser.add_argument("--embodiment-tag", dest="embodiment", help=argparse.SUPPRESS)
    parser.add_argument("--glob", default="**/episode.hdf5")
    parser.add_argument("--sources-list", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--state-keys", default=None)
    parser.add_argument("--action-keys", default=None)
    parser.add_argument("--video-keys", default=None)
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
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    output_path = (args.output_path or args.dataset_path).resolve()
    embodiment = validate_embodiment(args.embodiment)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    dataset_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    state_map = parse_mapping(args.state_keys, DEFAULT_AGILEX_STATE_MAP)
    action_map = parse_mapping(args.action_keys, DEFAULT_AGILEX_STATE_MAP)
    video_map = parse_hdf5_video_map(args.video_keys)
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

    for episode_index, path in enumerate(tqdm(h5_paths, desc="Scanning episodes")):
        with h5py.File(path, "r") as handle:
            length = episode_length(handle, args.end_pose_key)
            fps_values.append(read_fps(handle, args.fps))
            source_task_id = task_id_from_hdf5(handle, path)
        prompt = prompt_map.get(source_task_id) or args.default_task or source_task_id
        task_index = add_task(prompt)
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [prompt],
                "task_index": task_index,
                "length": length,
                "path": relative_path(path, dataset_root, args.absolute_paths),
                "source_task_id": source_task_id,
            }
        )

    fps_value = fps_values[0] if fps_values else args.fps
    if any(abs(value - fps_value) > 1e-6 for value in fps_values):
        log.warning("different fps values found; info.json uses first fps=%s", fps_value)

    modality = build_modality(
        state_map=state_map,
        action_map=action_map,
        video_map=video_map,
        state_column=args.state_column,
        action_column=args.action_column,
        task_key=args.task_key,
    )
    info = build_hdf5_info(
        embodiment=embodiment,
        dataset_root=dataset_root,
        episodes=episodes,
        video_map=video_map,
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
    write_jsonl(meta_dir / "tasks.jsonl", tasks, args.force)
    write_jsonl(meta_dir / "episodes.jsonl", episodes, args.force)
    write_json(meta_dir / "modality.json", modality, args.force)
    write_json(meta_dir / "embodiment.json", build_embodiment_json(embodiment), args.force)
    write_json(meta_dir / "info.json", info, args.force)

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
        write_json(meta_dir / "stats.json", stats, args.force)
    else:
        log.info("skip stats.json")

    print("\n" + "=" * 64)
    print("HDF5 native metadata generated")
    print(f"  raw dataset: {dataset_root}")
    print(f"  output meta: {meta_dir}")
    print(f"  episodes: {len(episodes)}")
    print(f"  frames: {info['total_frames']}")
    print(f"  tasks: {len(tasks)}")
    print(f"  embodiment: {embodiment}")
    print(f"  state keys: {list(state_map.keys())}")
    print(f"  action keys: {list(action_map.keys())}")
    print(f"  video keys: {list(video_map.keys())}")
    print("=" * 64)
    print("\nNext steps:")
    print("  Use worldscape_hdf5_text / goal / demo in configs/posttrain/agilex.yaml")
    print("  Point dataset_kwargs.data_root at the dataset root above.")


if __name__ == "__main__":
    main()
