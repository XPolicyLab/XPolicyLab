"""Fixed per-task 90/10 RoboDojo episode split: statistics on 90%, validation on 10%."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

SPLIT_VERSION = "robodojo_task_episode_90_10_v1"


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_index(source: dict, seed: int = 42, expected_tasks: int = 35) -> dict:
    groups = defaultdict(list)
    paths = set()
    for episode in source["episodes"]:
        path = episode["relative_path"]
        if path in paths or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Duplicate or unsafe episode path")
        paths.add(path)
        groups[(episode["task"], episode["embodiment"])].append(episode)
    if len(groups) != expected_tasks or any(len(v) != 100 for v in groups.values()):
        raise ValueError(
            f"Expected {expected_tasks} task/embodiment groups with 100 episodes each"
        )
    partitions = {"train": [], "val": []}
    for group, episodes in sorted(groups.items()):
        episodes = sorted(
            episodes, key=lambda e: (e["episode_index"], e["relative_path"])
        )
        if len({e["episode_index"] for e in episodes}) != 100:
            raise ValueError("Duplicate episode IDs within task")
        rng = random.Random(f"{SPLIT_VERSION}:{seed}:{group[0]}:{group[1]}")
        held_out = set(rng.sample(range(100), 10))
        for i, episode in enumerate(episodes):
            partitions["val" if i in held_out else "train"].append(episode)
    hashes = {name: fingerprint(episodes) for name, episodes in partitions.items()}
    identity = dict(
        version=SPLIT_VERSION,
        seed=seed,
        source_manifest_fingerprint=source["manifest_fingerprint"],
        partition_fingerprints=hashes,
        expected_tasks=expected_tasks,
    )
    identity["split_id"] = fingerprint(identity)
    return {
        name: dict(
            version=source["version"],
            root_dir=source["root_dir"],
            episodes=episodes,
            num_episodes=len(episodes),
            num_frames=sum(e["length"] for e in episodes),
            manifest_fingerprint=hashes[name],
            split={**identity, "role": name},
        )
        for name, episodes in partitions.items()
    }


def validate_partition(payload: dict, role: str) -> dict:
    split = payload.get("split", {})
    if split.get("version") != SPLIT_VERSION or split.get("role") != role:
        raise ValueError(f"Expected the RoboDojo {role} partition")
    identity = {k: v for k, v in split.items() if k not in ("role", "split_id")}
    if fingerprint(identity) != split.get("split_id"):
        raise ValueError("Invalid split identity")
    actual = fingerprint(payload["episodes"])
    if (
        actual != payload["manifest_fingerprint"]
        or actual != split["partition_fingerprints"][role]
    ):
        raise ValueError("Episode partition fingerprint mismatch")
    counts = Counter((e["task"], e["embodiment"]) for e in payload["episodes"])
    if len(counts) != split["expected_tasks"] or set(counts.values()) != {
        90 if role == "train" else 10
    }:
        raise ValueError("Invalid per-task episode counts")
    return split


def validate_validation_pair(train, val) -> None:
    """Training uses every episode; validation is the fixed 10-per-task partition."""
    if (
        getattr(train, "episode_split", None) != "full"
        or getattr(val, "episode_split", None) != "val"
    ):
        raise ValueError("RoboDojo trains on the full index and validates on val.json")
    for dataset, expected in ((train, 100), (val, 10)):
        counts = Counter(e["task"] for e in dataset.episodes)
        if len(counts) != 35 or set(counts.values()) != {expected}:
            raise ValueError(
                "Expected all 35 tasks, 100 training / 10 validation episodes per task"
            )
        paths = {str(e["path"].resolve()) for e in dataset.episodes}
        if len(paths) != len(dataset.episodes):
            raise ValueError("Duplicate RoboDojo episodes")
    train_paths = {str(e["path"].resolve()) for e in train.episodes}
    val_paths = {str(e["path"].resolve()) for e in val.episodes}
    if not val_paths <= train_paths:
        raise ValueError("Validation episodes must come from the training index")
    if train.action_stats_cache_path.resolve() != val.action_stats_cache_path.resolve():
        raise ValueError("Training and validation must share action statistics")
