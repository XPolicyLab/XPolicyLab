"""Deterministic episode-level sample selection for RoboTwin validation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROBOTWIN_EPISODE_RANDOM_VALIDATION_CONTRACT = (
    "robotwin_episode_random_complete_chunk_v1"
)


def robotwin_validation_sample_seed(base_seed: int, selection_id: int) -> int:
    """Return the stable diffusion-noise seed for one manifest selection."""

    return int(base_seed) + int(selection_id) * 1_000_003


def is_robotwin_episode_dataset(dataset: Any) -> bool:
    """Return whether a dataset exposes the raw RoboTwin episode index."""

    module = type(dataset).__module__
    return (
        module.startswith("prismwam.datasets.robotwin_")
        and hasattr(dataset, "episodes")
        and hasattr(dataset, "cumulative_sizes")
    )


def assert_robotwin_validation_dataset(dataset: Any) -> None:
    """Fail closed unless a RoboTwin validation dataset follows the contract."""

    phases = tuple(str(phase) for phase in getattr(dataset, "phases", ()))
    if phases != ("demo_clean_val",):
        raise ValueError(
            "RoboTwin episode validation requires phases=[demo_clean_val], "
            f"got {phases}."
        )
    if int(getattr(dataset, "sample_stride", 0)) != 1:
        raise ValueError("RoboTwin episode validation requires sample_stride=1.")
    if not bool(getattr(dataset, "drop_incomplete_windows", False)):
        raise ValueError(
            "RoboTwin episode validation requires drop_incomplete_windows=true."
        )


def _complete_chunk_count(dataset: Any, episode: dict[str, Any]) -> int:
    """Return the number of indexed anchors whose video/action chunk is complete."""

    length = int(episode["length"])
    sample_stride = int(dataset.sample_stride)
    max_start = length - int(dataset.window_span)

    action_horizon = getattr(dataset, "action_horizon", None)
    if action_horizon is not None:
        action_start_offset = int(getattr(dataset, "action_start_offset", 0))
        max_start = min(
            max_start,
            length - action_start_offset - int(action_horizon),
        )

    if max_start < 0:
        return 0
    indexed_count = int(episode["num_samples"])
    complete_count = max_start // sample_stride + 1
    return min(indexed_count, complete_count)


def build_robotwin_episode_validation_manifest(
    dataset: Any,
    *,
    seed: int = 42,
    expected_tasks: int = 50,
    expected_episodes_per_task: int = 10,
) -> dict[str, Any]:
    """Choose one complete chunk from every indexed episode.

    Episodes are traversed in the dataset's stable index order and sampled by a
    dedicated RNG.  The returned dataset indices can therefore be reused by
    every checkpoint in a validation sweep.
    """

    if int(expected_tasks) <= 0 or int(expected_episodes_per_task) <= 0:
        raise ValueError("Expected task and episode counts must be positive.")
    if int(getattr(dataset, "sample_stride", 0)) <= 0:
        raise ValueError("Validation dataset must expose a positive sample_stride.")
    if not hasattr(dataset, "episodes") or not hasattr(dataset, "cumulative_sizes"):
        raise TypeError("Validation dataset must expose episodes and cumulative_sizes.")

    episodes = list(dataset.episodes)
    cumulative_sizes = list(dataset.cumulative_sizes)
    if len(episodes) != len(cumulative_sizes):
        raise ValueError(
            "Episode metadata and cumulative sizes have different lengths."
        )

    counts = Counter(str(episode["task"]) for episode in episodes)
    if len(counts) != int(expected_tasks):
        raise ValueError(
            f"Expected {expected_tasks} validation tasks, found {len(counts)}."
        )
    bad_counts = {
        task: count
        for task, count in sorted(counts.items())
        if count != int(expected_episodes_per_task)
    }
    if bad_counts:
        raise ValueError(
            "Validation must index exactly "
            f"{expected_episodes_per_task} episodes per task; mismatches={bad_counts}."
        )

    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    previous_cumulative = 0
    for selection_id, (episode, cumulative) in enumerate(
        zip(episodes, cumulative_sizes)
    ):
        complete_count = _complete_chunk_count(dataset, episode)
        if complete_count <= 0:
            raise ValueError(
                "Validation episode has no complete chunk: "
                f"{episode['path']} (length={episode['length']})."
            )
        local_index = rng.randrange(complete_count)
        dataset_index = previous_cumulative + local_index
        start_index = local_index * int(dataset.sample_stride)
        rows.append(
            {
                "selection_id": int(selection_id),
                "dataset_index": int(dataset_index),
                "task": str(episode["task"]),
                "phase": str(episode["phase"]),
                "episode_index": int(episode["episode_index"]),
                "episode_path": str(Path(episode["path"]).resolve()),
                "episode_length": int(episode["length"]),
                "start_index": int(start_index),
                "eligible_complete_chunks": int(complete_count),
            }
        )
        previous_cumulative = int(cumulative)

    payload: dict[str, Any] = {
        "contract": ROBOTWIN_EPISODE_RANDOM_VALIDATION_CONTRACT,
        "seed": int(seed),
        "task_count": int(expected_tasks),
        "episodes_per_task": int(expected_episodes_per_task),
        "sample_count": len(rows),
        "sample_stride": int(dataset.sample_stride),
        "samples": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_robotwin_episode_validation_manifest(
    dataset: Any,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Verify that a saved manifest still addresses the same raw samples."""

    if manifest.get("contract") != ROBOTWIN_EPISODE_RANDOM_VALIDATION_CONTRACT:
        raise ValueError(
            f"Unsupported validation manifest: {manifest.get('contract')!r}."
        )
    expected = build_robotwin_episode_validation_manifest(
        dataset,
        seed=int(manifest["seed"]),
        expected_tasks=int(manifest["task_count"]),
        expected_episodes_per_task=int(manifest["episodes_per_task"]),
    )
    if expected != manifest:
        raise ValueError(
            "Validation manifest no longer matches the dataset index; regenerate it "
            "instead of silently changing checkpoint inputs."
        )
    return list(expected["samples"])


def build_robodojo_episode_validation_manifest(dataset, *, seed=42):
    """Use every validation episode with one fixed complete chunk per episode."""
    if getattr(dataset, "episode_split", None) != "val" or not dataset.split_metadata:
        raise ValueError("RoboDojo episode validation requires the val partition")
    if dataset.sample_stride != 1 or not dataset.drop_incomplete_windows:
        raise ValueError(
            "RoboDojo validation requires sample_stride=1 and complete windows"
        )
    payload = build_robotwin_episode_validation_manifest(
        dataset,
        seed=seed,
        expected_tasks=dataset.split_metadata["expected_tasks"],
        expected_episodes_per_task=10,
    )
    payload.pop("manifest_sha256")
    payload["contract"] = "robodojo_episode_random_complete_chunk_v1"
    payload["split_id"] = dataset.split_metadata["split_id"]
    from prismwam.robodojo_split import fingerprint

    payload["manifest_sha256"] = fingerprint(payload)
    return payload
