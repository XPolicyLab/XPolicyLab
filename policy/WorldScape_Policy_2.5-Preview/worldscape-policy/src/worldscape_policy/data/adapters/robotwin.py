from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def discover_lerobot_roots(data_root: str | Path) -> tuple[Path, ...]:
    """Resolve either one LeRobot repository or a parent of clean repositories."""
    root = Path(data_root)
    if (root / "meta" / "info.json").is_file():
        return (root,)
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot data root does not exist: {root}")
    children = tuple(
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    )
    if not children:
        raise FileNotFoundError(
            f"LeRobot dataset needs meta/info.json or child repositories: {root}"
        )
    return children


def load_episode_map(path: str | Path | None) -> dict[str, dict[str, int]] | None:
    """Load the clean RoboTwin local-to-full episode mapping."""
    if path is None:
        return None
    map_path = Path(path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
    if not isinstance(datasets, Mapping):
        raise ValueError(f"RoboTwin episode map needs a 'datasets' object: {map_path}")
    result: dict[str, dict[str, int]] = {}
    for dataset_name, mapping in datasets.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(
                f"RoboTwin episode map entry {dataset_name!r} must be an object"
            )
        result[str(dataset_name)] = {
            str(local_index): int(source_index)
            for local_index, source_index in mapping.items()
        }
    return result


def apply_robotwin_labels(
    arrays: dict[str, np.ndarray],
    *,
    dataset_root: Path,
    episode_index: int,
    trajectory_length: int,
    subtask_label_root: Path | None,
    episode_map: dict[str, dict[str, int]] | None,
    strict: bool,
) -> None:
    """Attach frame-level AutoLabeler text to native builder input arrays."""
    if subtask_label_root is None:
        return
    source_index = episode_index
    if episode_map is not None:
        dataset_mapping = episode_map.get(dataset_root.name)
        if dataset_mapping is None:
            raise KeyError(
                f"RoboTwin episode map has no dataset {dataset_root.name!r}"
            )
        local_key = str(episode_index)
        if local_key not in dataset_mapping:
            raise KeyError(
                f"RoboTwin episode map has no episode "
                f"{dataset_root.name}/episode_{episode_index:06d}"
            )
        source_index = dataset_mapping[local_key]
    label_path = (
        subtask_label_root
        / "data"
        / f"chunk-{source_index // 1000:03d}"
        / f"episode_{source_index:06d}"
        / "v0001.json"
    )
    if not label_path.is_file():
        if strict:
            raise FileNotFoundError(
                f"RoboTwin subtask label is missing for episode "
                f"{episode_index}: {label_path}"
            )
        return

    try:
        label: Any = json.loads(label_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        if strict:
            raise ValueError(f"Invalid RoboTwin subtask label: {label_path}") from exc
        return
    high_level = (
        str(label.get("high_level", "")).strip()
        if isinstance(label, Mapping)
        else ""
    )
    segments = label.get("segments") if isinstance(label, Mapping) else None
    if not high_level or not isinstance(segments, list) or not segments:
        if strict:
            raise ValueError(f"Invalid RoboTwin subtask label: {label_path}")
        return

    event_values = np.full(trajectory_length, high_level, dtype=object)
    planning_values = np.full(
        trajectory_length,
        _format_planning_label(high_level, high_level),
        dtype=object,
    )
    covered = np.zeros(trajectory_length, dtype=np.bool_)
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        step_text = str(segment.get("step_text", "")).strip()
        if not step_text:
            continue
        try:
            start = max(0, int(segment.get("start_idx", 0)))
            end = min(
                trajectory_length,
                int(segment.get("end_idx", trajectory_length - 1)) + 1,
            )
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        event_values[start:end] = step_text
        planning_values[start:end] = _format_planning_label(high_level, step_text)
        covered[start:end] = True
    if strict and not np.all(covered):
        missing = np.flatnonzero(~covered)
        raise ValueError(
            f"RoboTwin subtask label {label_path} leaves "
            f"{missing.size}/{trajectory_length} frames uncovered "
            f"(first missing frame: {int(missing[0])})"
        )

    arrays["high_level_instruction"] = np.full(
        trajectory_length, high_level, dtype=object
    )
    arrays["event_instruction"] = event_values
    arrays["planning_labels_text"] = planning_values
    arrays["task_id"] = np.asarray(high_level, dtype=object)


def _format_planning_label(high_level: str, subtask: str) -> str:
    return (
        f"task: {high_level}, sub_task: {subtask}, "
        "embodiment_tag: robotwin"
    )


__all__ = [
    "apply_robotwin_labels",
    "discover_lerobot_roots",
    "load_episode_map",
]
