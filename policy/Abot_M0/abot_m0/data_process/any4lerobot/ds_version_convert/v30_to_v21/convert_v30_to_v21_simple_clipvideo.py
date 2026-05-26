#!/usr/bin/env python
"""
将 LeRobot 数据集从 v3.0 格式转换回 v2.1 格式的简化脚本。

Usage:
    python convert_v30_to_v21_simple.py \
        --input-path /path/to/v3.0/dataset \
        --output-path /path/to/v2.1/dataset
"""

import argparse
import json
import logging
import math
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import jsonlines
import numpy as np
import pyarrow.parquet as pq
import tqdm
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    load_info,
    load_tasks,
    serialize_dict,
    unflatten_dict,
    write_info,
)
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

LEGACY_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MIN_VIDEO_DURATION = 1e-6
LEGACY_STATS_KEYS = ("mean", "std", "min", "max", "count")
COMPLETION_MARKER_FILE = "[].conversion_completed.json"


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pyarrow values into standard Python types for JSON dumps."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    """Load the consolidated metadata rows stored in ``meta/episodes``."""
    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def get_video_keys(root: Path) -> list[str]:
    """Extract video feature keys from dataset info."""
    info = load_info(root)
    features = info["features"]
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    return video_keys


def convert_tasks(root: Path, new_root: Path) -> None:
    """Convert tasks parquet to legacy JSONL."""
    logging.info("Converting tasks parquet to legacy JSONL")
    tasks = load_tasks(root)
    tasks = tasks.sort_values("task_index")

    out_path = new_root / LEGACY_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        for task, row in tasks.iterrows():
            writer.write(
                {
                    "task_index": int(row["task_index"]),
                    "task": _to_serializable(task),
                }
            )


def convert_info(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    """Convert info.json metadata to v2.1 schema."""
    info = load_info(root)
    logging.info("Converting info.json metadata to v2.1 schema")

    total_episodes = info.get("total_episodes") or len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    info["codebase_version"] = V21

    # Restore legacy layout templates.
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None

    # Remove v3-specific sizing hints which do not exist in v2.1.
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Restore per-feature metadata: camera entries already contain their own fps.
    for key, ft in info["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)

    write_info(info, new_root)


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group episode records by their data file location."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (
            int(record["data/chunk_index"]),
            int(record["data/file_index"]),
        )
        grouped[key].append(record)
    return grouped


def convert_data(root: Path, new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    """Convert consolidated parquet files back to per-episode files."""
    logging.info("Converting consolidated parquet files back to per-episode files")
    grouped = _group_episodes_by_data_file(episode_records)

    for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc="convert data files"):
        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start

            if length <= 0:
                raise ValueError(
                    "Invalid episode length computed during data conversion: "
                    f"episode_index={episode_index}, length={length}"
                )

            episode_table = table.slice(start, length)

            dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
            dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=dest_chunk,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group episode records by their video file location."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _validate_video_paths(src: Path, dst: Path) -> None:
    """Validate source and destination paths to prevent security issues."""
    src = Path(src)
    dst = Path(dst)

    try:
        src_resolved = src.resolve()
        dst_resolved = dst.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid path provided: {exc}") from exc

    if not src_resolved.exists():
        raise FileNotFoundError(f"Source video file does not exist: {src_resolved}")

    if not src_resolved.is_file():
        raise ValueError(f"Source path is not a regular file: {src_resolved}")

    valid_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if src_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Source file does not have a valid video extension: {src_resolved}")

    if dst_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Destination file does not have a valid video extension: {dst_resolved}")


def _extract_video_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
) -> None:
    """Extract a video segment using ffmpeg."""
    _validate_video_paths(src, dst)
    if "oxeauge" in str(src):
        end = end-0.4
    if not (0 <= start <= 86400):
        raise ValueError(f"Invalid start time: {start}")
    if not (0 <= end <= 86400):
        raise ValueError(f"Invalid end time: {end}")
    if start >= end:
        raise ValueError(f"Start time {start} must be less than end time {end}")

    duration = max(end - start, MIN_VIDEO_DURATION)

    if duration > 3600:
        raise ValueError(f"Video segment duration too long: {duration} seconds")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(dst),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while processing video '{src}' -> '{dst}'") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found; it is required for video conversion") from exc
    except subprocess.CalledProcessError as exc:
        error_msg = f"ffmpeg failed while splitting video '{src}' into '{dst}'"
        if exc.stderr:
            error_msg += f". Error: {exc.stderr.strip()}"
        raise RuntimeError(error_msg) from exc


def convert_videos(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str]) -> None:
    """Convert concatenated MP4 files back to per-episode videos."""
    if len(video_keys) == 0:
        logging.info("No video features detected; skipping video conversion")
        return

    logging.info("Converting concatenated MP4 files back to per-episode videos")

    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if len(grouped) == 0:
            logging.info("No video metadata found for key '%s'; skipping", video_key)
            continue

        for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc=f"convert videos ({video_key})"):
            src_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not src_path.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src_path}")

            records = sorted(records, key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"]))

            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])

                dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )

                _extract_video_segment(src_path, dest_path, start=start, end=end)


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    """Reconstruct legacy episodes and episodes_stats JSONL files."""
    logging.info("Reconstructing legacy episodes and episodes_stats JSONL files")

    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    def _filter_stats(stats: dict[str, Any]) -> dict[str, Any]:
        """Remove v3-only statistics keys so output matches the v2.1 schema."""
        filtered: dict[str, Any] = {}
        for feature, values in stats.items():
            if not isinstance(values, dict):
                continue
            keep = {k: v for k, v in values.items() if k in LEGACY_STATS_KEYS}
            if keep:
                filtered[feature] = keep
        return filtered

    with (
        jsonlines.open(episodes_path, mode="w") as episodes_writer,
        jsonlines.open(stats_path, mode="w") as stats_writer,
    ):
        for record in sorted(episode_records, key=lambda rec: int(rec["episode_index"])):
            legacy_episode = {
                key: value
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }

            serializable_episode = {key: _to_serializable(value) for key, value in legacy_episode.items()}
            episodes_writer.write(serializable_episode)

            stats_flat = {key: record[key] for key in record if key.startswith("stats/")}
            stats_nested = unflatten_dict(stats_flat).get("stats", {})
            stats_serialized = serialize_dict(_filter_stats(stats_nested))
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": stats_serialized,
                }
            )


def copy_ancillary_directories(root: Path, new_root: Path) -> None:
    """Copy additional directories like images if they exist."""
    for subdir in ["images"]:
        source = root / subdir
        if source.exists():
            dest = new_root / subdir
            logging.info(f"Copying {subdir} directory from {source} to {dest}")
            shutil.copytree(source, dest, dirs_exist_ok=True)


def validate_input_dataset(input_path: Path) -> None:
    """Validate that the input dataset is v3.0 format."""
    info = load_info(input_path)

    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V30:
        raise ValueError(
            f"Input dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script is specifically for converting v3.0 datasets to v2.1."
        )


def check_completion_marker(output_path: Path) -> bool:
    """Check if conversion completion marker file exists."""
    marker_path = output_path / COMPLETION_MARKER_FILE
    return marker_path.exists()


def create_completion_marker(output_path: Path, input_path: Path) -> None:
    """Create a marker file to indicate conversion is completed."""
    marker_path = output_path / COMPLETION_MARKER_FILE
    marker_data = {
        "conversion_completed": True,
        "completed_at": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "converter_version": "v30_to_v21_simple",
    }
    
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Created completion marker file: {marker_path}")


def convert_dataset(input_path: Path, output_path: Path, force: bool = False) -> None:
    """Main conversion function."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Check if conversion already completed
    if not force and check_completion_marker(output_path):
        logging.info(
            f"Conversion marker found at {output_path / COMPLETION_MARKER_FILE}. "
            f"Skipping conversion. Use --force to re-convert."
        )
        return

    # Validate input dataset version
    validate_input_dataset(input_path)

    # Create output directory
    if output_path.exists():
        logging.warning(f"Output path already exists: {output_path}. It will be removed.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting dataset from {input_path} to {output_path}")

    # Load episode records
    episode_records = load_episode_records(input_path)
    logging.info(f"Loaded {len(episode_records)} episode records")

    # Get video keys
    video_keys = get_video_keys(input_path)
    logging.info(f"Found {len(video_keys)} video keys: {video_keys}")

    # Convert components
    convert_info(input_path, output_path, episode_records, video_keys)
    convert_tasks(input_path, output_path)
    # convert_data(input_path, output_path, episode_records)
    convert_videos(input_path, output_path, episode_records, video_keys)
    convert_episodes_metadata(output_path, episode_records)
    copy_ancillary_directories(input_path, output_path)

    # Create completion marker
    create_completion_marker(output_path, input_path)

    logging.info(f"Conversion completed successfully! Output saved to {output_path}")


def main():
    init_logging()
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset from v3.0 format to v2.1 format"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="bridge_train_15000_20000_augmented",
        help="Task ID (e.g., task_351). If provided, will be used to construct input and output paths.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("/mnt/xlab-nas-2/vla_dataset/oxeauge/data/oxe-auge"),
        help="Path to the v3.0 format dataset folder",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/mnt/xlab-nas-2/vla_dataset/oxeauge/data/oxe-auge-v21"),
        help="Path to save the converted v2.1 format dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if completion marker exists",
    )

    args = parser.parse_args()

    # Default base paths
    input_base_path = args.input_path
    output_base_path = args.output_path

    input_path = input_base_path / args.task_id
    output_path = output_base_path / args.task_id


    convert_dataset(input_path, output_path, force=args.force)


if __name__ == "__main__":
    main()

