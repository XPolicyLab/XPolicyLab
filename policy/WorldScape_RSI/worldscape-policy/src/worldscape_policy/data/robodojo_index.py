"""Explicit, versioned RoboDojo temporal index snapshots (no pickle)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from worldscape_policy.data.robodojo_chunks import (
    task_chunk_counts,
    chunk_policy_options,
    task_chunk_limit,
)


INDEX_VERSION = 1


def index_options(dataset: Any) -> dict[str, Any]:
    return {
        "data_roots": [str(root.expanduser().resolve()) for root in dataset.roots],
        "file_pattern": dataset._index_file_pattern,
        "max_episodes": dataset._index_max_episodes,
        "temporal_packing": dataset.temporal_packing,
        "temporal_anchor_index": dataset.temporal_anchor_index,
        "max_chunk_size": dataset.temporal_max_chunk_size,
        "respect_trajectory_segments": dataset.respect_trajectory_segments,
        "action_horizon": dataset.action_horizon,
        "max_event_chunks": dataset.max_event_chunks,
        "step_stride": dataset.step_stride,
        **chunk_policy_options(dataset.chunk_policy, dataset.action_horizon),
    }


def read_index(path: str | Path, *, verify_files: bool = False) -> dict[str, Any]:
    """Read a frozen dataset snapshot; never silently rebuild an invalid index.

    By default the caller promises the source dataset is unchanged. Optional
    verification stats every source file but does not open its HDF5 contents.
    """
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"].item()))
        index = archive["index"]
        progress = archive["sampling_progress"]
        lengths = archive["lengths"]
    if (
        manifest.get("version") != INDEX_VERSION
        or manifest.get("kind") != "robodojo_joint"
    ):
        raise ValueError("Unsupported RoboDojo index format")
    paths = manifest["paths"]
    episodes = manifest["episodes"]
    options = manifest["options"]
    episode_lengths = np.asarray(manifest["episode_lengths"], dtype=np.int64)
    if (
        not paths
        or len(paths) != len(episodes)
        or len(set(paths)) != len(paths)
        or episode_lengths.shape != (len(paths),)
        or np.any(episode_lengths <= 0)
        or not options["temporal_packing"]
        or options["action_horizon"] not in {24, 48}
    ):
        raise ValueError("Invalid RoboDojo index manifest")
    if (
        index.dtype != np.int64
        or index.ndim != 2
        or index.shape[1] != 2
        or not len(index)
        or lengths.shape != (len(index), 3)
        or lengths.dtype != np.int64
        or progress.shape != (len(index),)
        or not np.isfinite(progress).all()
        or np.any((progress < 0) | (progress > 1))
        or np.any(index[:, 0] < 0)
        or np.any(index[:, 0] >= len(paths))
        or np.any(index[:, 1] < 0)
        or np.any(np.diff(index[:, 0]) < 0)
    ):
        raise ValueError("Invalid RoboDojo index arrays")
    chunk_limits = options["max_chunk_size"]
    if "chunk_policy" in options:
        horizon = options["action_horizon"]
        memory_chunks, _ = task_chunk_counts(horizon)
        if (
            options["chunk_policy"] != chunk_policy_options("task_aware", horizon)["chunk_policy"]
            or options["max_chunk_size"] != memory_chunks
        ):
            raise ValueError("Invalid RoboDojo index chunk policy")
        episode_limits = np.asarray([task_chunk_limit(Path(path), horizon) for path in paths])
        chunk_limits = episode_limits[index[:, 0]]
    if (
        np.any(index[:, 1] + options["action_horizon"] >= episode_lengths[index[:, 0]])
        or np.any(lengths[:, 1] < 1)
        or np.any(lengths[:, 1] > chunk_limits)
        or np.any(lengths[:, 0] != options["action_horizon"] * lengths[:, 1])
        or np.any(lengths[:, 2] != 8 * lengths[:, 1] + 1)
    ):
        raise ValueError("Invalid RoboDojo anchor/length geometry")
    if verify_files:
        for record in manifest["source_files"]:
            actual = Path(record["path"]).stat()
            if (
                actual.st_size != record["size"]
                or actual.st_mtime_ns != record["mtime_ns"]
            ):
                raise ValueError(f"RoboDojo index source changed: {record['path']}")
    return {
        "manifest": manifest,
        "index": index,
        "sampling_progress": progress,
        "lengths": lengths,
    }
