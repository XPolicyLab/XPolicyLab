#!/usr/bin/env python3
"""Augment a LeRobot v2 dataset with native WorldScape Policy metadata.

This mirrors DreamZero's ``convert_lerobot_to_gear.py``. It scans existing
parquet and MP4 files, then writes or refreshes ``meta/modality.json``,
``meta/embodiment.json``, ``meta/stats.json``, ``meta/tasks.jsonl``, and
``meta/episodes.jsonl`` for ``NativeLeRobotDataset`` and datasets such as
``worldscape_lerobot_text``, ``worldscape_lerobot_goal``, and
``worldscape_lerobot_demo``.

Usage:
  python tools/data/convert_lerobot_to_native_meta.py \\
      --dataset-path /path/to/lerobot_dataset \\
      --embodiment agilex

  python tools/data/convert_lerobot_to_native_meta.py \\
      --dataset-path /path/to/lerobot_dataset \\
      --state-keys '{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],'
                     '"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}' \\
      --action-keys '{"left_pos":[0,3],"left_rot6d":[3,9],"left_gripper":[9,10],'
                     '"right_pos":[10,13],"right_rot6d":[13,19],"right_gripper":[19,20]}' \\
      --relative-action-keys left_pos left_rot6d right_pos right_rot6d \\
      --force
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

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
    array_stats,
    build_embodiment_json,
    parse_mapping,
    validate_embodiment,
    write_json,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found at {info_path}")
    return json.loads(info_path.read_text())


def get_parquet_paths(dataset_path: Path, info: dict) -> list[Path]:
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    paths: list[Path] = []
    for episode_index in range(total_episodes):
        chunk_index = episode_index // chunks_size
        path = dataset_path / pattern.format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        if path.exists():
            paths.append(path)
    return sorted(paths)


def detect_features(info: dict) -> dict:
    features = info.get("features", {})
    state_keys = [
        key
        for key in features
        if key.startswith("observation.state") or key.startswith("observation.eef6d")
    ]
    action_keys = [
        key for key in features if key == "action" or key.startswith("action.")
    ]
    video_keys = [key for key, meta in features.items() if meta.get("dtype") == "video"]
    if not video_keys:
        video_keys = [key for key in features if key.startswith("observation.images")]
    annotation_keys = [key for key in features if key.startswith("annotation")]
    return {
        "state": state_keys,
        "action": action_keys,
        "video": video_keys,
        "annotation": annotation_keys,
        "features": features,
    }


def build_modality_json(
    detected: dict,
    state_mapping: dict[str, list[int]] | None,
    action_mapping: dict[str, list[int]] | None,
    task_key: str | None,
) -> dict:
    features = detected["features"]
    modality: dict = {"state": {}, "action": {}, "video": {}, "annotation": {}}

    state_col = detected["state"][0] if detected["state"] else None
    if state_col and state_mapping:
        for name, (start, end) in state_mapping.items():
            modality["state"][name] = {
                "original_key": state_col,
                "start": start,
                "end": end,
                "rotation_type": None,
                "absolute": True,
                "dtype": features[state_col].get("dtype", "float32"),
                "range": None,
            }
    elif state_col:
        shape = features[state_col].get("shape", [1])
        dim = shape[0] if isinstance(shape, list) else shape
        modality["state"]["state"] = {
            "original_key": state_col,
            "start": 0,
            "end": dim,
            "rotation_type": None,
            "absolute": True,
            "dtype": features[state_col].get("dtype", "float32"),
            "range": None,
        }

    action_col = detected["action"][0] if detected["action"] else None
    if action_col and action_mapping:
        for name, (start, end) in action_mapping.items():
            modality["action"][name] = {
                "original_key": action_col,
                "start": start,
                "end": end,
                "rotation_type": None,
                "absolute": True,
                "dtype": features[action_col].get("dtype", "float32"),
                "range": None,
            }
    elif action_col:
        shape = features[action_col].get("shape", [1])
        dim = shape[0] if isinstance(shape, list) else shape
        modality["action"]["action"] = {
            "original_key": action_col,
            "start": 0,
            "end": dim,
            "rotation_type": None,
            "absolute": True,
            "dtype": features[action_col].get("dtype", "float32"),
            "range": None,
        }

    for video_key in detected["video"]:
        short_name = video_key.replace("observation.images.", "")
        modality["video"][short_name] = {"original_key": video_key}

    if task_key:
        short = task_key.replace("annotation.", "")
        modality["annotation"][short] = {"original_key": task_key}
    else:
        for annotation_key in detected["annotation"]:
            short = annotation_key.replace("annotation.", "")
            modality["annotation"][short] = {"original_key": annotation_key}
        if not modality["annotation"]:
            modality["annotation"]["language.language_instruction"] = {}

    return modality


def compute_stats(parquet_paths: list[Path], columns: list[str]) -> dict:
    stats: dict = {}
    for col in columns:
        chunks: list[np.ndarray] = []
        for parquet_path in tqdm(parquet_paths, desc=f"Stats [{col}]"):
            frame = pd.read_parquet(parquet_path)
            if col not in frame.columns:
                continue
            arr = np.stack(frame[col].values)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            chunks.append(arr)
        if chunks:
            stats[col] = array_stats(np.concatenate(chunks, axis=0))
    return stats


def compute_relative_stats(
    parquet_paths: list[Path],
    modality: dict,
    relative_action_keys: list[str],
    action_horizon: int = 24,
) -> dict:
    stats: dict = {}
    for rel_key in relative_action_keys:
        if rel_key not in modality["action"]:
            log.warning("relative action key %r not found in action modality, skipping", rel_key)
            continue
        if rel_key not in modality["state"]:
            log.warning(
                "relative action key %r has no matching state key; skipping relative stats",
                rel_key,
            )
            continue

        action_meta = modality["action"][rel_key]
        state_meta = modality["state"][rel_key]
        all_relative: list[np.ndarray] = []
        for parquet_path in tqdm(parquet_paths, desc=f"Relative stats [{rel_key}]"):
            frame = pd.read_parquet(parquet_path)
            action_col = action_meta["original_key"]
            state_col = state_meta["original_key"]
            if action_col not in frame.columns or state_col not in frame.columns:
                continue
            action_data = np.stack(frame[action_col].values).astype(np.float64)
            state_data = np.stack(frame[state_col].values).astype(np.float64)
            if action_data.ndim == 1:
                action_data = action_data.reshape(-1, 1)
            if state_data.ndim == 1:
                state_data = state_data.reshape(-1, 1)
            action_slice = action_data[:, action_meta["start"] : action_meta["end"]]
            state_slice = state_data[:, state_meta["start"] : state_meta["end"]]
            traj_len = len(frame)
            usable = traj_len - action_horizon
            for index in range(max(usable, 0)):
                reference = state_slice[index]
                chunk_end = min(index + action_horizon, traj_len)
                relative = action_slice[index:chunk_end] - reference
                all_relative.extend(relative)

        if not all_relative:
            log.warning("no relative actions computed for %r", rel_key)
            continue
        data = np.asarray(all_relative)
        stats[rel_key] = {
            "max": np.max(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def build_tasks(parquet_paths: list[Path], task_key: str | None) -> list[dict]:
    if task_key is None:
        return [{"task_index": 0, "task": ""}]
    task_set: dict[str, int] = {}
    for parquet_path in tqdm(parquet_paths, desc="Extracting tasks"):
        frame = pd.read_parquet(parquet_path)
        if task_key not in frame.columns:
            continue
        for value in frame[task_key].unique():
            text = str(value) if not isinstance(value, str) else value
            if text not in task_set:
                task_set[text] = len(task_set)
    if not task_set:
        return [{"task_index": 0, "task": ""}]
    return [
        {"task_index": index, "task": text}
        for text, index in sorted(task_set.items(), key=lambda item: item[1])
    ]


def build_episodes(
    parquet_paths: list[Path],
    task_key: str | None,
    tasks: list[dict],
) -> list[dict]:
    task_text_to_idx = {task["task"]: task["task_index"] for task in tasks}
    episodes: list[dict] = []
    for episode_index, parquet_path in enumerate(tqdm(parquet_paths, desc="Building episodes")):
        frame = pd.read_parquet(parquet_path)
        episode_tasks: list[str] = []
        if task_key and task_key in frame.columns:
            for value in frame[task_key].unique():
                text = str(value) if not isinstance(value, str) else value
                if text and text in task_text_to_idx:
                    episode_tasks.append(text)
        if not episode_tasks:
            episode_tasks = [""]
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": episode_tasks,
                "length": len(frame),
            }
        )
    return episodes


def validate_dataset(dataset_path: Path, info: dict, modality: dict) -> list[str]:
    warnings: list[str] = []
    for subdir in ("data", "videos", "meta"):
        if not (dataset_path / subdir).exists():
            warnings.append(f"Missing directory: {subdir}/")
    if not modality["video"]:
        warnings.append("No video features detected")
    if not modality["state"]:
        warnings.append("No state modality keys defined")
    if not modality["action"]:
        warnings.append("No action modality keys defined")
    if int(info.get("total_episodes", 0)) == 0:
        warnings.append("total_episodes is 0 in info.json")
    if info.get("fps") is None:
        warnings.append("fps not set in info.json")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--embodiment", default="agilex")
    parser.add_argument("--embodiment-tag", dest="embodiment", help=argparse.SUPPRESS)
    parser.add_argument("--state-keys", default=None)
    parser.add_argument("--action-keys", default=None)
    parser.add_argument("--relative-action-keys", nargs="*", default=None)
    parser.add_argument("--task-key", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    embodiment = validate_embodiment(args.embodiment)

    if args.output_path:
        output_path = args.output_path.resolve()
        if output_path != dataset_path:
            log.info("Copying dataset to %s", output_path)
            if output_path.exists():
                if not args.force:
                    raise SystemExit("Output path already exists. Use --force to overwrite.")
                shutil.rmtree(output_path)
            shutil.copytree(dataset_path, output_path)
            dataset_path = output_path
    else:
        output_path = dataset_path

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = load_info(dataset_path)
    detected = detect_features(info)

    log.info("Dataset: %s", dataset_path.name)
    log.info("  Episodes: %d", info.get("total_episodes", 0))
    log.info("  FPS: %s", info.get("fps", "not set"))
    log.info("  State columns: %s", detected["state"])
    log.info("  Action columns: %s", detected["action"])
    log.info("  Video features: %d camera(s)", len(detected["video"]))
    log.info("  Annotation columns: %s", detected["annotation"])

    if args.fps is not None:
        info["fps"] = args.fps
        write_json(output_path / "meta" / "info.json", info, force=True)
        log.info("  Overriding FPS to %s", args.fps)

    state_mapping = parse_mapping(args.state_keys, DEFAULT_AGILEX_STATE_MAP) if args.state_keys else None
    action_mapping = parse_mapping(args.action_keys, DEFAULT_AGILEX_STATE_MAP) if args.action_keys else None

    task_key = args.task_key
    if task_key is None and detected["annotation"]:
        for candidate in (
            "annotation.task",
            "annotation.language.language_instruction",
            "annotation.subtask",
        ):
            if candidate in detected["annotation"]:
                task_key = candidate
                break
        if task_key is None:
            task_key = detected["annotation"][0]
        log.info("  Auto-detected task key: %s", task_key)

    modality = build_modality_json(detected, state_mapping, action_mapping, task_key)
    write_json(meta_dir / "modality.json", modality, args.force)
    write_json(meta_dir / "embodiment.json", build_embodiment_json(embodiment), args.force)

    parquet_paths = get_parquet_paths(output_path, info)
    if not parquet_paths:
        raise SystemExit("No parquet files found. Check dataset structure.")
    log.info("  Found %d parquet files", len(parquet_paths))

    numeric_cols = detected["state"] + detected["action"]
    if "timestamp" in info.get("features", {}):
        numeric_cols.append("timestamp")

    stats_path = meta_dir / "stats.json"
    if stats_path.exists() and not args.force:
        log.info("  stats.json already exists, skipping")
    else:
        stats = compute_stats(parquet_paths, numeric_cols)
        write_json(stats_path, stats, force=True)

    rel_stats_path = meta_dir / "relative_stats_dreamzero.json"
    if args.relative_action_keys:
        if rel_stats_path.exists() and not args.force:
            log.info("  relative_stats_dreamzero.json already exists, skipping")
        else:
            rel_stats = compute_relative_stats(
                parquet_paths,
                modality,
                list(args.relative_action_keys),
                action_horizon=args.action_horizon,
            )
            if rel_stats:
                write_json(rel_stats_path, rel_stats, force=True)
            else:
                log.warning("  No relative stats computed")
    else:
        log.info("  Skipping relative stats (no --relative-action-keys provided)")

    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists() and not args.force:
        log.info("  tasks.jsonl already exists, skipping")
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    else:
        tasks = build_tasks(parquet_paths, task_key)
        write_jsonl(tasks_path, tasks, args.force)

    episodes_path = meta_dir / "episodes.jsonl"
    if episodes_path.exists() and not args.force:
        log.info("  episodes.jsonl already exists, skipping")
    else:
        episodes = build_episodes(parquet_paths, task_key, tasks)
        write_jsonl(episodes_path, episodes, args.force)

    warnings = validate_dataset(output_path, info, modality)
    if warnings:
        log.warning("Validation warnings:")
        for warning in warnings:
            log.warning("  - %s", warning)
    else:
        log.info("Validation passed")

    print("\n" + "=" * 60)
    print("LeRobot native metadata generated")
    print(f"  Output: {output_path}")
    print(f"  Embodiment: {embodiment}")
    print(f"  State keys: {list(modality['state'].keys())}")
    print(f"  Action keys: {list(modality['action'].keys())}")
    print(f"  Video keys: {list(modality['video'].keys())}")
    print(f"  Task key: {task_key or '(none)'}")
    print("=" * 60)
    print("\nNext steps:")
    print("  Use worldscape_lerobot_text / goal / demo in configs/posttrain/")
    print("  Point dataset_kwargs.data_root at the dataset root above.")


if __name__ == "__main__":
    main()
