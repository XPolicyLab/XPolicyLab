#!/usr/bin/env python
"""Check a local LeRobot v3 dataset while conversion may still be running.

The checker focuses on the failure modes that make partial datasets unsafe:
unreadable parquet footers, mismatched episode metadata, missing referenced
data/video chunks, and video durations that are shorter than the episode spans.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class ParquetStatus:
    path: Path
    rows: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.rows is not None


@dataclass
class VideoStatus:
    path: Path
    duration_s: float | None = None
    fps: float | None = None
    frames: int | None = None
    counted_frames: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a local LeRobot v3 dataset and report the safe readable prefix."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Dataset directory containing meta/info.json, data/, videos/.",
    )
    parser.add_argument(
        "--expect-robot-type",
        default=None,
        help="Optional expected raw robot_type, for example arx5 or ARX5.",
    )
    parser.add_argument(
        "--check-videos",
        action="store_true",
        help="Open referenced mp4 files with PyAV and check fps/duration.",
    )
    parser.add_argument(
        "--count-video-frames",
        action="store_true",
        help="Decode every referenced video to count frames. Slow but strict.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Read data parquet index columns and verify per-episode row counts.",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=float,
        default=2.0,
        help="Allowed duration mismatch in frames. Default: 2.",
    )
    parser.add_argument(
        "--sample-load",
        type=int,
        default=0,
        help="Instantiate LeRobotDataset and read this many evenly spaced samples.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable JSON summary.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return exit code 0 when a non-empty safe prefix exists, even if the live dataset is partial.",
    )
    parser.add_argument(
        "--make-safe-view",
        type=Path,
        default=None,
        help="Create a stable LeRobot view containing only the safe readable prefix.",
    )
    parser.add_argument(
        "--overwrite-view",
        action="store_true",
        help="Remove an existing --make-safe-view directory before writing it.",
    )
    parser.add_argument(
        "--view-file-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to materialize data/video files in --make-safe-view. Default: hardlink.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_parquet_status(path: Path) -> ParquetStatus:
    if not path.exists():
        return ParquetStatus(path=path, error="missing")
    try:
        return ParquetStatus(path=path, rows=pq.read_metadata(path).num_rows)
    except Exception as exc:  # parquet footer is often invalid while conversion is running.
        return ParquetStatus(path=path, error=f"{type(exc).__name__}: {exc}")


def sorted_nested_parquets(root: Path, subdir: str) -> list[Path]:
    base = root / subdir
    if not base.exists():
        return []
    return sorted(base.glob("*/*.parquet"))


def format_data_path(info: dict[str, Any], row: pd.Series) -> str:
    return info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )


def format_video_path(info: dict[str, Any], row: pd.Series, video_key: str) -> str:
    return info["video_path"].format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in (info.get("features") or {}).items()
        if feature.get("dtype") == "video"
    ]


def check_schema(info: dict[str, Any], report: Report, expect_robot_type: str | None) -> str | None:
    codebase_version = str(info.get("codebase_version", ""))
    if codebase_version != "v3.0":
        report.error(f"meta/info.json codebase_version is {codebase_version!r}, expected 'v3.0'.")

    robot_type = str(info.get("robot_type", ""))
    if expect_robot_type is not None and robot_type != expect_robot_type:
        report.error(f"robot_type is {robot_type!r}, expected {expect_robot_type!r}.")

    try:
        from lerobot.transforms.constants import (
            get_feature_mapping,
            get_image_mapping,
            get_mask_mapping,
            infer_embodiment_variant,
        )

        features = info.get("features", {})
        resolved = infer_embodiment_variant(robot_type, features)
        get_feature_mapping(robot_type, features)
        get_image_mapping(robot_type, features)
        get_mask_mapping(robot_type, features)
        return str(resolved)
    except Exception as exc:
        report.error(f"WSABase transform mapping check failed: {type(exc).__name__}: {exc}")
        return None


def load_episodes(root: Path, report: Report) -> tuple[pd.DataFrame, list[ParquetStatus]]:
    episode_files = sorted_nested_parquets(root, "meta/episodes")
    statuses: list[ParquetStatus] = []
    frames: list[pd.DataFrame] = []
    if not episode_files:
        report.error("No episode metadata parquet found under meta/episodes/*/*.parquet.")
        return pd.DataFrame(), statuses

    for path in episode_files:
        status = read_parquet_status(path)
        statuses.append(status)
        if not status.ok:
            report.error(f"Unreadable episode parquet: {rel(path, root)} ({status.error})")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            report.error(f"Failed to read episode parquet rows: {rel(path, root)} ({exc})")
            continue
        df["_episode_source"] = rel(path, root)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), statuses

    episodes = pd.concat(frames, ignore_index=True)
    if "episode_index" in episodes:
        episodes = episodes.sort_values("episode_index", kind="stable").reset_index(drop=True)
    return episodes, statuses


def validate_required_columns(
    episodes: pd.DataFrame,
    video_keys: list[str],
    report: Report,
) -> bool:
    required = {
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    }
    for key in video_keys:
        required.update(
            {
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
                f"videos/{key}/to_timestamp",
            }
        )

    missing = sorted(required - set(episodes.columns))
    if missing:
        report.error(f"Episode metadata missing required columns: {missing}")
        return False
    return True


def collect_safe_prefix(
    root: Path,
    info: dict[str, Any],
    episodes: pd.DataFrame,
    video_keys: list[str],
    report: Report,
    tolerance_frames: float,
) -> tuple[int, int, dict[str, int], dict[str, int]]:
    fps = float(info.get("fps") or 0)
    if fps <= 0:
        report.error(f"Invalid fps in info.json: {info.get('fps')!r}")
        return 0, 0, {}, {}

    data_status_cache: dict[str, ParquetStatus] = {}
    expected_rows_by_data: dict[str, int] = defaultdict(int)
    expected_rows_by_video: dict[str, int] = defaultdict(int)
    safe_episodes = 0
    safe_frames = 0
    duration_tol_s = tolerance_frames / fps

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])

        if ep_idx != safe_episodes:
            report.warn(
                f"Safe prefix stopped before episode {safe_episodes}: next metadata row has episode_index={ep_idx}."
            )
            break
        if length <= 0:
            report.warn(f"Safe prefix stopped at episode {ep_idx}: non-positive length={length}.")
            break
        if from_idx != safe_frames or to_idx != safe_frames + length:
            report.warn(
                f"Safe prefix stopped at episode {ep_idx}: expected interval "
                f"{safe_frames}:{safe_frames + length}, got {from_idx}:{to_idx}."
            )
            break

        data_rel = format_data_path(info, row)
        if data_rel not in data_status_cache:
            data_status_cache[data_rel] = read_parquet_status(root / data_rel)
        data_status = data_status_cache[data_rel]
        if not data_status.ok:
            report.warn(
                f"Safe prefix stopped at episode {ep_idx}: data parquet {data_rel} is not readable "
                f"({data_status.error})."
            )
            break

        next_expected_data_rows = expected_rows_by_data[data_rel] + length
        if data_status.rows is not None and data_status.rows < next_expected_data_rows:
            report.warn(
                f"Safe prefix stopped at episode {ep_idx}: data parquet {data_rel} has "
                f"{data_status.rows} rows, needs at least {next_expected_data_rows}."
            )
            break

        video_ok = True
        for key in video_keys:
            video_rel = format_video_path(info, row, key)
            if not (root / video_rel).exists():
                report.warn(f"Safe prefix stopped at episode {ep_idx}: missing video {video_rel}.")
                video_ok = False
                break
            span = float(row[f"videos/{key}/to_timestamp"]) - float(row[f"videos/{key}/from_timestamp"])
            expected_span = length / fps
            if not math.isfinite(span) or abs(span - expected_span) > duration_tol_s:
                report.warn(
                    f"Safe prefix stopped at episode {ep_idx}: video metadata span for {key} is "
                    f"{span:.6f}s, expected about {expected_span:.6f}s."
                )
                video_ok = False
                break
        if not video_ok:
            break

        expected_rows_by_data[data_rel] = next_expected_data_rows
        for key in video_keys:
            video_rel = format_video_path(info, row, key)
            expected_rows_by_video[video_rel] += length
        safe_episodes += 1
        safe_frames += length

    return safe_episodes, safe_frames, dict(expected_rows_by_data), dict(expected_rows_by_video)


def check_all_parquets(
    root: Path,
    report: Report,
) -> tuple[dict[str, ParquetStatus], list[ParquetStatus]]:
    data_statuses = {rel(path, root): read_parquet_status(path) for path in sorted_nested_parquets(root, "data")}
    unreadable = [status for status in data_statuses.values() if not status.ok]
    for status in unreadable:
        report.error(f"Unreadable data parquet: {rel(status.path, root)} ({status.error})")
    if not data_statuses:
        report.error("No data parquet found under data/*/*.parquet.")
    return data_statuses, unreadable


def check_data_deep(
    root: Path,
    info: dict[str, Any],
    episodes: pd.DataFrame,
    safe_episodes: int,
    expected_rows_by_data: dict[str, int],
    report: Report,
    fps: float,
    tolerance_frames: float,
) -> None:
    columns = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
    tolerance_s = tolerance_frames / fps
    safe_rows = episodes.iloc[:safe_episodes]
    data_frames: dict[str, pd.DataFrame] = {}

    for data_rel, expected_rows in sorted(expected_rows_by_data.items()):
        try:
            df = pd.read_parquet(root / data_rel, columns=columns)
            data_frames[data_rel] = df
        except Exception as exc:
            report.error(f"Deep check failed reading {data_rel}: {exc}")
            continue

        if len(df) < expected_rows:
            report.error(f"{data_rel} has {len(df)} rows, expected at least {expected_rows}.")
        elif len(df) > expected_rows:
            report.warn(
                f"{data_rel} has {len(df)} rows but safe metadata references {expected_rows}; "
                "this is expected if conversion has flushed data ahead of metadata."
            )

    for _, row in safe_rows.iterrows():
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        from_idx = int(row["dataset_from_index"])
        data_rel = format_data_path(info, row)
        df = data_frames.get(data_rel)
        if df is None:
            continue

        ep_df = df[df["episode_index"] == ep_idx]
        if len(ep_df) != length:
            report.error(f"Episode {ep_idx} has {len(ep_df)} data rows in {data_rel}, expected {length}.")
            continue

        expected_index = list(range(from_idx, from_idx + length))
        if ep_df["index"].tolist() != expected_index:
            report.error(f"Episode {ep_idx} global index column is not contiguous at {from_idx}:{from_idx + length}.")

        expected_frame_index = list(range(length))
        if ep_df["frame_index"].tolist() != expected_frame_index:
            report.error(f"Episode {ep_idx} frame_index column is not 0..{length - 1}.")

        expected_ts = ep_df["frame_index"].astype(float) / fps
        max_ts_err = (ep_df["timestamp"].astype(float) - expected_ts).abs().max()
        if pd.notna(max_ts_err) and float(max_ts_err) > tolerance_s:
            report.error(
                f"Episode {ep_idx} timestamp mismatch: max error {float(max_ts_err):.6f}s "
                f"(tolerance {tolerance_s:.6f}s)."
            )


def inspect_video(path: Path, count_frames: bool) -> VideoStatus:
    if not path.exists():
        return VideoStatus(path=path, error="missing")
    try:
        import av
    except Exception as exc:
        return VideoStatus(path=path, error=f"PyAV import failed: {exc}")

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps_rate = stream.average_rate or stream.base_rate
            fps = float(fps_rate) if fps_rate is not None else None
            if stream.duration is not None:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration_s = float(container.duration / av.time_base)
            else:
                duration_s = None
            frames = int(stream.frames) if stream.frames else None
            counted_frames = None
            if count_frames:
                counted_frames = sum(1 for _ in container.decode(video=0))
            return VideoStatus(
                path=path,
                duration_s=duration_s,
                fps=fps,
                frames=frames,
                counted_frames=counted_frames,
            )
    except Exception as exc:
        return VideoStatus(path=path, error=f"{type(exc).__name__}: {exc}")


def check_videos(
    root: Path,
    info: dict[str, Any],
    expected_rows_by_video: dict[str, int],
    report: Report,
    tolerance_frames: float,
    count_frames: bool,
) -> dict[str, VideoStatus]:
    fps = float(info["fps"])
    tolerance_s = tolerance_frames / fps
    statuses: dict[str, VideoStatus] = {}
    for video_rel, expected_frames in sorted(expected_rows_by_video.items()):
        status = inspect_video(root / video_rel, count_frames)
        statuses[video_rel] = status
        if not status.ok:
            report.error(f"Video unreadable: {video_rel} ({status.error})")
            continue
        if status.fps is not None and abs(status.fps - fps) > 0.01:
            report.error(f"Video {video_rel} fps={status.fps:.3f}, expected {fps:.3f}.")
        expected_duration = expected_frames / fps
        if status.duration_s is not None and status.duration_s + tolerance_s < expected_duration:
            report.error(
                f"Video {video_rel} duration={status.duration_s:.6f}s is shorter than "
                f"safe episode span {expected_duration:.6f}s."
            )
        frame_count = status.counted_frames if status.counted_frames is not None else status.frames
        if frame_count is not None and frame_count < expected_frames:
            report.error(f"Video {video_rel} has {frame_count} frames, expected at least {expected_frames}.")
        if frame_count is not None and frame_count > expected_frames:
            report.warn(
                f"Video {video_rel} has {frame_count} frames while safe metadata references "
                f"{expected_frames}; this is fine for a live/partial conversion if timestamps still fit."
            )
    return statuses


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_safe_view(
    root: Path,
    dst: Path,
    info: dict[str, Any],
    episodes: pd.DataFrame,
    safe_episodes: int,
    safe_frames: int,
    expected_rows_by_data: dict[str, int],
    expected_rows_by_video: dict[str, int],
    mode: str,
    overwrite: bool,
    report: Report,
) -> None:
    if safe_episodes <= 0:
        report.error("Cannot create safe view because safe_episodes=0.")
        return

    dst = dst.expanduser().resolve()
    if dst.exists():
        if not overwrite:
            report.error(f"Safe view already exists: {dst}. Use --overwrite-view to replace it.")
            return
        shutil.rmtree(dst)

    (dst / "meta").mkdir(parents=True, exist_ok=True)

    safe_info = json.loads(json.dumps(info))
    safe_info["total_episodes"] = int(safe_episodes)
    safe_info["total_frames"] = int(safe_frames)
    safe_info["splits"] = {"train": f"0:{safe_episodes}"}
    (dst / "meta" / "info.json").write_text(
        json.dumps(safe_info, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    tasks_src = root / "meta" / "tasks.parquet"
    if tasks_src.exists():
        materialize_file(tasks_src, dst / "meta" / "tasks.parquet", mode)

    stats_src = root / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, dst / "meta" / "stats.json")
        report.warn(
            "Copied meta/stats.json into the safe view as-is. If conversion was live, stats may include "
            "frames beyond the safe prefix; external stats are safer for training."
        )

    readme_src = root / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, dst / "README.md")

    safe_episode_rows = episodes.iloc[:safe_episodes].copy()
    helper_cols = [col for col in safe_episode_rows.columns if col.startswith("_")]
    if helper_cols:
        safe_episode_rows = safe_episode_rows.drop(columns=helper_cols)
    if "meta/episodes/chunk_index" in safe_episode_rows.columns:
        safe_episode_rows["meta/episodes/chunk_index"] = 0
    if "meta/episodes/file_index" in safe_episode_rows.columns:
        safe_episode_rows["meta/episodes/file_index"] = 0

    episodes_dst = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_dst.parent.mkdir(parents=True, exist_ok=True)
    safe_episode_rows.to_parquet(episodes_dst, index=False)

    for data_rel in sorted(expected_rows_by_data):
        materialize_file(root / data_rel, dst / data_rel, mode)
    for video_rel in sorted(expected_rows_by_video):
        materialize_file(root / video_rel, dst / video_rel, mode)


def sample_load_dataset(root: Path, n: int, report: Report) -> None:
    if n <= 0:
        return
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(str(root))
        if len(dataset) == 0:
            report.error("LeRobotDataset loaded but has length 0.")
            return
        if n == 1:
            indices = [0]
        else:
            indices = sorted({round(i * (len(dataset) - 1) / (n - 1)) for i in range(n)})
        for idx in indices:
            _ = dataset[int(idx)]
    except Exception as exc:
        report.error(f"LeRobotDataset sample load failed: {type(exc).__name__}: {exc}")


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    report = Report()

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: missing {info_path}", file=sys.stderr)
        return 1

    info = load_json(info_path)
    video_keys = get_video_keys(info)
    if video_keys and not info.get("video_path"):
        report.error("info.json contains video features but video_path is missing/null.")
    resolved_robot_type = check_schema(info, report, args.expect_robot_type)

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        report.error("Missing meta/tasks.parquet.")
    else:
        task_status = read_parquet_status(tasks_path)
        if not task_status.ok:
            report.error(f"Unreadable meta/tasks.parquet ({task_status.error})")
        elif int(info.get("total_tasks", 0)) != task_status.rows:
            report.warn(
                f"info.total_tasks={info.get('total_tasks')} but meta/tasks.parquet rows={task_status.rows}."
            )

    episodes, episode_statuses = load_episodes(root, report)
    columns_ok = validate_required_columns(episodes, video_keys, report) if not episodes.empty else False

    data_statuses, unreadable_data = check_all_parquets(root, report)

    safe_episodes = 0
    safe_frames = 0
    expected_rows_by_data: dict[str, int] = {}
    expected_rows_by_video: dict[str, int] = {}
    if columns_ok and (not video_keys or info.get("video_path")):
        safe_episodes, safe_frames, expected_rows_by_data, expected_rows_by_video = collect_safe_prefix(
            root,
            info,
            episodes,
            video_keys,
            report,
            args.tolerance_frames,
        )

    if int(info.get("total_episodes", -1)) != len(episodes):
        report.warn(
            f"info.total_episodes={info.get('total_episodes')} but readable episode metadata rows={len(episodes)}."
        )
    if int(info.get("total_frames", -1)) != safe_frames:
        report.warn(f"info.total_frames={info.get('total_frames')} but safe readable prefix has {safe_frames}.")

    total_readable_data_rows = sum(status.rows or 0 for status in data_statuses.values() if status.ok)
    if total_readable_data_rows < safe_frames:
        report.error(f"Readable data parquet rows={total_readable_data_rows}, less than safe frames={safe_frames}.")

    if args.deep and columns_ok and safe_episodes > 0:
        check_data_deep(
            root,
            info,
            episodes,
            safe_episodes,
            expected_rows_by_data,
            report,
            float(info["fps"]),
            args.tolerance_frames,
        )

    video_statuses: dict[str, VideoStatus] = {}
    if args.check_videos or args.count_video_frames:
        video_statuses = check_videos(
            root,
            info,
            expected_rows_by_video,
            report,
            args.tolerance_frames,
            args.count_video_frames,
        )

    safe_view_path: str | None = None
    if args.make_safe_view is not None:
        write_safe_view(
            root,
            args.make_safe_view,
            info,
            episodes,
            safe_episodes,
            safe_frames,
            expected_rows_by_data,
            expected_rows_by_video,
            args.view_file_mode,
            args.overwrite_view,
            report,
        )
        safe_view_path = str(args.make_safe_view.expanduser().resolve())

    sample_load_dataset(root, args.sample_load, report)

    full_info_match = (
        safe_episodes == int(info.get("total_episodes", -1))
        and safe_frames == int(info.get("total_frames", -1))
        and not unreadable_data
        and all(status.ok for status in episode_statuses)
    )
    ok = full_info_match and not report.errors
    partial_ok = safe_episodes > 0 and args.allow_partial

    summary = {
        "dataset": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "resolved_robot_type": resolved_robot_type,
        "fps": info.get("fps"),
        "video_keys": video_keys,
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
        "readable_episode_rows": int(len(episodes)),
        "safe_episodes": int(safe_episodes),
        "safe_frames": int(safe_frames),
        "data_parquets": len(data_statuses),
        "unreadable_data_parquets": len(unreadable_data),
        "checked_video_files": len(video_statuses),
        "safe_view": safe_view_path,
        "errors": report.errors,
        "warnings": report.warnings,
        "result": "OK" if ok else ("PARTIAL" if safe_episodes > 0 else "BAD"),
    }

    print("LeRobot v3 integrity check")
    print(f"dataset: {summary['dataset']}")
    print(
        f"schema: codebase={summary['codebase_version']} robot_type={summary['robot_type']} "
        f"resolved={summary['resolved_robot_type']} fps={summary['fps']}"
    )
    print(f"video_keys: {', '.join(video_keys) if video_keys else '<none>'}")
    print(
        f"episodes: info={summary['info_total_episodes']} readable_meta={summary['readable_episode_rows']} "
        f"safe={summary['safe_episodes']}"
    )
    print(
        f"frames: info={summary['info_total_frames']} safe={summary['safe_frames']} "
        f"readable_data_rows={total_readable_data_rows}"
    )
    print(
        f"parquets: data={summary['data_parquets']} unreadable_data={summary['unreadable_data_parquets']} "
        f"checked_videos={summary['checked_video_files']}"
    )
    if safe_view_path is not None:
        print(f"safe_view: {safe_view_path}")

    if report.warnings:
        print("\nWarnings:")
        for message in report.warnings:
            print(f"  - {message}")
    if report.errors:
        print("\nErrors:")
        for message in report.errors:
            print(f"  - {message}")

    print(f"\nRESULT: {summary['result']}")
    if ok:
        print("This dataset is internally consistent and should be readable as-is.")
    elif safe_episodes > 0:
        print(
            "A safe prefix exists, but the live directory is not fully consistent. "
            "For training, use a stable snapshot/view of only the safe prefix."
        )
    else:
        print("No safe readable prefix was found.")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if ok or partial_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""
python tools/check_lerobot_v3_integrity.py \
  --dataset /inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2/clean_table \
  --deep \
  --sample-load 32 
"""
