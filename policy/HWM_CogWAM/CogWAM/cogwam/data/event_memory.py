"""Event-driven semantic-memory labels and boundary sampling.

The helpers in this module are deliberately independent from the model.  They
turn RoboDojo's cumulative ``complete_text`` annotation into a persistent
semantic-memory state and a delta-only UPDATE target.  Nothing is enabled
unless ``text_annotations.event_memory.enabled`` is set by a dataset recipe.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler

KEEP_DECISION = "KEEP"
UPDATE_DECISION = "UPDATE"


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def text_value(value: Any) -> str:
    """Read one parquet text cell without silently inventing a label."""

    if value is None:
        return ""
    try:
        array = np.asarray(value)
        if bool(array.ndim == 0 and array.dtype.kind == "f" and np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_n1(value: Any) -> str:
    """Audit-approved semantic comparison: case/space/final-punctuation neutral."""

    normalized = re.sub(r"\s+", " ", text_value(value)).strip().casefold()
    return normalized.rstrip(" .;,:!?")


def normalize_item_key(value: Any) -> str:
    """Stable N2-style key used only for memory-item set membership."""

    return re.sub(r"[^a-z0-9]+", " ", text_value(value).casefold()).strip()


def finished_task_items(value: Any, *, empty_value: str = "None.") -> list[str]:
    """Split RoboDojo's cumulative completed-task string conservatively."""

    normalized = re.sub(r"\s+", " ", text_value(value)).strip(" \t.;")
    empty_markers = {
        re.sub(r"\s+", " ", str(empty_value)).strip(" \t.;").casefold(),
        "none",
        "nothing",
        "n/a",
    }
    if not normalized or normalized.casefold() in empty_markers:
        return []
    pieces = re.split(r"(?:\r?\n|\s*[;|]\s*|(?<=[.!?])\s+)", text_value(value))
    return [
        re.sub(r"\s+", " ", piece).strip(" \t.;")
        for piece in pieces
        if re.sub(r"\s+", " ", piece).strip(" \t.;")
    ]


def format_finished_task_items(
    items: Sequence[str], *, empty_value: str = "None."
) -> str:
    cleaned = [text_value(item).strip(" \t.;") for item in items]
    cleaned = [item for item in cleaned if item]
    return empty_value if not cleaned else ". ".join(cleaned) + "."


def memory_delta(
    previous: Any,
    current: Any,
    *,
    empty_value: str = "None.",
) -> str:
    """Return the ordered cumulative-list difference, preserving current text."""

    previous_keys = {
        normalize_item_key(item)
        for item in finished_task_items(previous, empty_value=empty_value)
        if normalize_item_key(item)
    }
    added: list[str] = []
    seen = set(previous_keys)
    for item in finished_task_items(current, empty_value=empty_value):
        key = normalize_item_key(item)
        if key and key not in seen:
            seen.add(key)
            added.append(item)
    return format_finished_task_items(added, empty_value=empty_value)


def append_memory_delta(
    previous: Any,
    delta: Any,
    *,
    empty_value: str = "None.",
) -> str:
    """Client-side monotonic append with normalized de-duplication."""

    merged: list[str] = []
    seen = set()
    for value in (previous, delta):
        for item in finished_task_items(value, empty_value=empty_value):
            key = normalize_item_key(item)
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    return format_finished_task_items(merged, empty_value=empty_value)


def event_memory_from_trajectory(
    trajectory,
    base_index: int,
    text_config,
) -> dict[str, Any]:
    """Build one episode-local cached state and KEEP/UPDATE training label."""

    event_cfg = _cfg_get(text_config, "event_memory", {})
    if not bool(_cfg_get(event_cfg, "enabled", False)):
        return {}
    if not bool(_cfg_get(text_config, "enabled", False)):
        raise ValueError(
            "text_annotations.event_memory.enabled=true requires "
            "text_annotations.enabled=true"
        )

    offset = int(_cfg_get(event_cfg, "semantic_offset", -10))
    if offset >= 0:
        raise ValueError(f"event_memory.semantic_offset must be negative, got {offset}")
    normalization = str(_cfg_get(event_cfg, "normalization", "n1")).lower()
    if normalization != "n1":
        raise ValueError(
            "event_memory currently implements the audited N1 comparison only, "
            f"got normalization={normalization!r}"
        )
    fields = dict(_cfg_get(text_config, "fields", {}))
    subtask_source = str(fields.get("subtask_text", "subtask_text"))
    complete_source = str(fields.get("completed_subtask_text", "complete_text"))
    empty_memory = str(_cfg_get(event_cfg, "empty_memory", "None.")).strip()
    empty_subtask = str(
        _cfg_get(event_cfg, "empty_cached_subtask", "None.")
    ).strip()
    if not empty_memory or not empty_subtask:
        raise ValueError("event_memory empty sentinels must be non-empty")

    base_index = int(base_index)
    current_row = trajectory.iloc[base_index]
    current_subtask = text_value(current_row.get(subtask_source, ""))
    current_complete = text_value(current_row.get(complete_source, ""))
    if not current_subtask or not current_complete:
        raise ValueError(
            "Event-memory labels require non-empty current subtask/complete text: "
            f"base_index={base_index}, subtask={current_subtask!r}, "
            f"complete={current_complete!r}"
        )

    cache_index = base_index + offset
    cache_valid = cache_index >= 0
    if cache_valid:
        cached_row = trajectory.iloc[cache_index]
        cached_subtask = text_value(cached_row.get(subtask_source, ""))
        semantic_memory = text_value(cached_row.get(complete_source, ""))
        if not cached_subtask or not semantic_memory:
            raise ValueError(
                "Event-memory cache row contains an empty annotation: "
                f"base_index={base_index}, cache_index={cache_index}"
            )
    else:
        cached_subtask = empty_subtask
        semantic_memory = empty_memory

    changed = (
        not cache_valid
        or normalize_n1(current_subtask) != normalize_n1(cached_subtask)
        or normalize_n1(current_complete) != normalize_n1(semantic_memory)
    )
    decision = UPDATE_DECISION if changed else KEEP_DECISION
    delta = (
        memory_delta(semantic_memory, current_complete, empty_value=empty_memory)
        if cache_valid and changed
        else empty_memory
    )
    return {
        str(_cfg_get(event_cfg, "semantic_memory_field", "semantic_memory")): semantic_memory,
        str(
            _cfg_get(
                event_cfg,
                "cached_subtask_field",
                "cached_current_subtask",
            )
        ): cached_subtask,
        str(_cfg_get(event_cfg, "decision_field", "semantic_decision")): decision,
        str(_cfg_get(event_cfg, "memory_add_field", "memory_add")): delta,
        str(_cfg_get(event_cfg, "cache_valid_field", "semantic_cache_valid")): bool(
            cache_valid
        ),
    }


@dataclass(frozen=True)
class SemanticIndexPools:
    update: np.ndarray
    hard_keep: np.ndarray
    random_keep: np.ndarray
    phase0_total: int
    fingerprint: str


def semantic_index_fingerprint(dataset_path: Path, text_config) -> str:
    """Fingerprint metadata and every behaviorally relevant index setting."""

    event_cfg = _cfg_get(text_config, "event_memory", {})
    digest = hashlib.sha256()
    for relative in ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"):
        path = Path(dataset_path) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Semantic-index metadata is missing: {path}")
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    settings = {
        "format": "event_memory_phase_index_v1",
        "semantic_offset": int(_cfg_get(event_cfg, "semantic_offset", -10)),
        "replan_interval": int(_cfg_get(event_cfg, "replan_interval", 10)),
        "replan_phase": int(_cfg_get(event_cfg, "replan_phase", 0)),
        "normalization": str(_cfg_get(event_cfg, "normalization", "n1")),
        "empty_memory": str(_cfg_get(event_cfg, "empty_memory", "None.")),
        "empty_cached_subtask": str(
            _cfg_get(event_cfg, "empty_cached_subtask", "None.")
        ),
        "fields": dict(_cfg_get(text_config, "fields", {})),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _load_semantic_index(path: Path, expected_fingerprint: str) -> SemanticIndexPools:
    with np.load(path, allow_pickle=False) as payload:
        fingerprint = str(payload["fingerprint"].item())
        if fingerprint != expected_fingerprint:
            raise ValueError(
                "Semantic-index fingerprint mismatch: "
                f"actual={fingerprint}, expected={expected_fingerprint}"
            )
        return SemanticIndexPools(
            update=np.asarray(payload["update"], dtype=np.int64),
            hard_keep=np.asarray(payload["hard_keep"], dtype=np.int64),
            random_keep=np.asarray(payload["random_keep"], dtype=np.int64),
            phase0_total=int(payload["phase0_total"].item()),
            fingerprint=fingerprint,
        )


def _validate_semantic_index_ratio(pools: SemanticIndexPools, event_cfg) -> None:
    expected_ratio = float(
        _cfg_get(event_cfg, "expected_phase_update_ratio", 0.064)
    )
    tolerance = float(
        _cfg_get(event_cfg, "expected_phase_update_tolerance", 0.005)
    )
    actual_ratio = len(pools.update) / max(pools.phase0_total, 1)
    if abs(actual_ratio - expected_ratio) > tolerance:
        raise ValueError(
            "Semantic UPDATE ratio drifted outside the audited contract: "
            f"actual={actual_ratio:.6f}, "
            f"expected={expected_ratio:.6f}±{tolerance:.6f}"
        )


def build_or_load_semantic_index(
    dataset,
    text_config,
    cache_path: Path,
) -> SemanticIndexPools:
    """Build the phase-aligned boundary pools once on rank zero."""

    event_cfg = _cfg_get(text_config, "event_memory", {})
    fingerprint = semantic_index_fingerprint(dataset.dataset_path, text_config)
    cache_path = Path(cache_path)
    is_distributed = dist.is_available() and dist.is_initialized()
    # Without an initialized process group this is necessarily a standalone
    # build; a stale launcher RANK environment variable must not suppress it.
    rank = dist.get_rank() if is_distributed else 0

    if cache_path.is_file():
        try:
            pools = _load_semantic_index(cache_path, fingerprint)
            if is_distributed:
                dist.barrier()
            _validate_semantic_index_ratio(pools, event_cfg)
            return pools
        except (ValueError, KeyError, OSError):
            if rank != 0:
                if is_distributed:
                    dist.barrier()
                pools = _load_semantic_index(cache_path, fingerprint)
                _validate_semantic_index_ratio(pools, event_cfg)
                return pools

    if rank == 0:
        offset = abs(int(_cfg_get(event_cfg, "semantic_offset", -10)))
        interval = int(_cfg_get(event_cfg, "replan_interval", 10))
        phase = int(_cfg_get(event_cfg, "replan_phase", 0))
        if offset != interval:
            raise ValueError(
                "Event-memory phase sampler requires abs(semantic_offset) == "
                f"replan_interval, got {offset} and {interval}"
            )
        if interval <= 0 or not 0 <= phase < interval:
            raise ValueError(
                f"Invalid replan interval/phase: interval={interval}, phase={phase}"
            )

        update: list[int] = []
        keep: list[int] = []
        update_by_episode: dict[int, set[int]] = {}
        keep_by_episode: dict[int, set[int]] = {}
        starts = np.cumsum(
            np.concatenate(
                [np.asarray([0], dtype=np.int64), np.asarray(dataset.trajectory_lengths[:-1], dtype=np.int64)]
            )
        )
        fields = dict(_cfg_get(text_config, "fields", {}))
        text_columns = list(
            dict.fromkeys(
                [
                    str(fields.get("subtask_text", "subtask_text")),
                    str(fields.get("completed_subtask_text", "complete_text")),
                ]
            )
        )

        def read_text_trajectory(trajectory_id: int):
            # Semantic indexing should not read action/state arrays.  RoboDojo
            # is LeRobot v2.1, so its episode parquet can be addressed through
            # the same public path contract used by get_trajectory_data.
            try:
                import pandas as pd

                chunk_index = dataset.get_episode_chunk(trajectory_id)
                parquet_path = dataset.dataset_path / dataset.data_path_pattern.format(
                    episode_chunk=chunk_index,
                    episode_index=trajectory_id,
                )
                return pd.read_parquet(parquet_path, columns=text_columns)
            except (AttributeError, ImportError, KeyError, OSError, ValueError):
                # Retain compatibility with non-v2 wrappers used by unit/smoke
                # tests; the event recipe itself always takes the fast path.
                return dataset.get_trajectory_data(trajectory_id)

        phase0_total = 0
        for position, trajectory_id in enumerate(dataset.trajectory_ids):
            trajectory_id = int(trajectory_id)
            trajectory = read_text_trajectory(trajectory_id)
            length = int(dataset.trajectory_lengths[position])
            if len(trajectory) != length:
                raise ValueError(
                    f"Episode length mismatch for {trajectory_id}: parquet={len(trajectory)}, meta={length}"
                )
            episode_updates: set[int] = set()
            episode_keeps: set[int] = set()
            for base_index in range(phase, length, interval):
                global_index = int(starts[position]) + int(base_index)
                label = event_memory_from_trajectory(
                    trajectory,
                    base_index,
                    text_config,
                )[str(_cfg_get(event_cfg, "decision_field", "semantic_decision"))]
                phase0_total += 1
                if label == UPDATE_DECISION:
                    update.append(global_index)
                    episode_updates.add(base_index)
                elif label == KEEP_DECISION:
                    keep.append(global_index)
                    episode_keeps.add(base_index)
                else:
                    raise ValueError(f"Unknown semantic decision: {label!r}")
            update_by_episode[trajectory_id] = episode_updates
            keep_by_episode[trajectory_id] = episode_keeps

        hard_global: set[int] = set()
        for position, trajectory_id in enumerate(dataset.trajectory_ids):
            trajectory_id = int(trajectory_id)
            episode_keeps = keep_by_episode[trajectory_id]
            start = int(starts[position])
            for update_index in update_by_episode[trajectory_id]:
                for neighbor in (update_index - interval, update_index + interval):
                    if neighbor in episode_keeps:
                        hard_global.add(start + neighbor)
        keep_set = set(keep)
        random_keep = sorted(keep_set - hard_global)
        if not update or not hard_global or not random_keep:
            raise RuntimeError(
                "Semantic boundary pools must all be non-empty: "
                f"update={len(update)}, hard_keep={len(hard_global)}, "
                f"random_keep={len(random_keep)}"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                update=np.asarray(sorted(update), dtype=np.int64),
                hard_keep=np.asarray(sorted(hard_global), dtype=np.int64),
                random_keep=np.asarray(random_keep, dtype=np.int64),
                phase0_total=np.asarray(phase0_total, dtype=np.int64),
                fingerprint=np.asarray(fingerprint),
            )
        os.replace(tmp_path, cache_path)

    if is_distributed:
        dist.barrier()
    pools = _load_semantic_index(cache_path, fingerprint)
    _validate_semantic_index_ratio(pools, event_cfg)
    return pools


class SemanticBoundaryBatchSampler(Sampler[list[int]]):
    """Deterministic 2 UPDATE / 2 hard KEEP / 2 random KEEP batches."""

    def __init__(
        self,
        pools: SemanticIndexPools,
        *,
        seed: int,
        update_per_batch: int = 2,
        hard_keep_per_batch: int = 2,
        random_keep_per_batch: int = 2,
    ) -> None:
        self.pools = pools
        self.seed = int(seed)
        self.update_per_batch = int(update_per_batch)
        self.hard_keep_per_batch = int(hard_keep_per_batch)
        self.random_keep_per_batch = int(random_keep_per_batch)
        self.batch_size = (
            self.update_per_batch
            + self.hard_keep_per_batch
            + self.random_keep_per_batch
        )
        if (
            self.update_per_batch <= 0
            or self.hard_keep_per_batch <= 0
            or self.random_keep_per_batch <= 0
        ):
            raise ValueError("Every semantic batch category must be positive")
        self.epoch = 0
        self._num_batches = max(1, math.ceil(pools.phase0_total / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": int(self.epoch)}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.epoch = int(state_dict.get("epoch", 0))

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 0xE7E17])
        )
        def balanced_draw(pool: np.ndarray, count: int) -> np.ndarray:
            needed = self._num_batches * count
            pieces: list[np.ndarray] = []
            collected = 0
            while collected < needed:
                shuffled = rng.permutation(pool)
                take = min(len(shuffled), needed - collected)
                pieces.append(shuffled[:take])
                collected += take
            return np.concatenate(pieces).astype(np.int64, copy=False)

        updates = balanced_draw(self.pools.update, self.update_per_batch)
        hard_keeps = balanced_draw(
            self.pools.hard_keep, self.hard_keep_per_batch
        )
        random_keeps = balanced_draw(
            self.pools.random_keep, self.random_keep_per_batch
        )
        for batch_index in range(self._num_batches):
            update_start = batch_index * self.update_per_batch
            hard_start = batch_index * self.hard_keep_per_batch
            random_start = batch_index * self.random_keep_per_batch
            batch = np.concatenate(
                [
                    updates[
                        update_start : update_start + self.update_per_batch
                    ],
                    hard_keeps[
                        hard_start : hard_start + self.hard_keep_per_batch
                    ],
                    random_keeps[
                        random_start : random_start
                        + self.random_keep_per_batch
                    ],
                ]
            ).astype(np.int64)
            rng.shuffle(batch)
            yield batch.tolist()


__all__ = [
    "KEEP_DECISION",
    "UPDATE_DECISION",
    "SemanticBoundaryBatchSampler",
    "SemanticIndexPools",
    "append_memory_delta",
    "build_or_load_semantic_index",
    "event_memory_from_trajectory",
    "finished_task_items",
    "format_finished_task_items",
    "memory_delta",
    "normalize_item_key",
    "normalize_n1",
]
