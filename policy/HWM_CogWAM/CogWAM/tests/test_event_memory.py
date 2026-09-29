from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cogwam.data.event_memory import (
    KEEP_DECISION,
    UPDATE_DECISION,
    SemanticBoundaryBatchSampler,
    SemanticIndexPools,
    append_memory_delta,
    build_or_load_semantic_index,
    event_memory_from_trajectory,
    memory_delta,
)

TEXT_CONFIG = {
    "enabled": True,
    "fields": {
        "subtask_text": "subtask_text",
        "completed_subtask_text": "complete_text",
    },
    "event_memory": {
        "enabled": True,
        "semantic_offset": -10,
        "empty_memory": "None.",
        "empty_cached_subtask": "None.",
    },
}


def test_event_labels_use_episode_local_t_minus_10_state_and_delta() -> None:
    trajectory = pd.DataFrame(
        {
            "subtask_text": ["Open the drawer."] * 12
            + ["Pick up the cup."] * 18,
            "complete_text": ["None."] * 12
            + ["Open the drawer."] * 18,
        }
    )

    initial = event_memory_from_trajectory(trajectory, 0, TEXT_CONFIG)
    assert initial == {
        "semantic_memory": "None.",
        "cached_current_subtask": "None.",
        "semantic_decision": UPDATE_DECISION,
        "memory_add": "None.",
        "semantic_cache_valid": False,
    }

    keep = event_memory_from_trajectory(trajectory, 10, TEXT_CONFIG)
    assert keep["semantic_decision"] == KEEP_DECISION
    assert keep["cached_current_subtask"] == "Open the drawer."
    assert keep["semantic_memory"] == "None."

    update = event_memory_from_trajectory(trajectory, 20, TEXT_CONFIG)
    assert update["semantic_decision"] == UPDATE_DECISION
    assert update["cached_current_subtask"] == "Open the drawer."
    assert update["memory_add"] == "Open the drawer."


def test_memory_delta_and_client_append_are_monotonic_and_deduplicated() -> None:
    previous = "Open the drawer."
    current = "Open the drawer. Pick up the cup."
    assert memory_delta(previous, current) == "Pick up the cup."
    assert append_memory_delta(previous, "Pick up the cup.") == current
    assert append_memory_delta(current, "pick up the cup") == current


def test_semantic_sampler_emits_exact_two_two_two_mix() -> None:
    pools = SemanticIndexPools(
        update=np.asarray([1, 2, 3]),
        hard_keep=np.asarray([10, 11, 12]),
        random_keep=np.asarray([20, 21, 22]),
        phase0_total=12,
        fingerprint="test",
    )
    sampler = SemanticBoundaryBatchSampler(pools, seed=42)
    for batch in sampler:
        assert len(batch) == 6
        assert sum(index in {1, 2, 3} for index in batch) == 2
        assert sum(index in {10, 11, 12} for index in batch) == 2
        assert sum(index in {20, 21, 22} for index in batch) == 2

    sampler.set_epoch(7)
    restored = SemanticBoundaryBatchSampler(pools, seed=42)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == list(sampler)


def test_semantic_index_uses_phase0_global_frames_and_boundary_neighbors(
    tmp_path,
) -> None:
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "meta"
    metadata.mkdir(parents=True)
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        (metadata / name).write_text(name, encoding="utf-8")

    transition = pd.DataFrame(
        {
            "subtask_text": ["A"] * 15 + ["B"] * 25,
            "complete_text": ["None."] * 15 + ["A."] * 25,
        }
    )
    stable = pd.DataFrame(
        {
            "subtask_text": ["C"] * 40,
            "complete_text": ["None."] * 40,
        }
    )

    class _Dataset:
        dataset_path = dataset_root
        trajectory_ids = np.asarray([7, 9], dtype=np.int64)
        trajectory_lengths = np.asarray([40, 40], dtype=np.int64)

        @staticmethod
        def get_trajectory_data(trajectory_id):
            return transition if int(trajectory_id) == 7 else stable

    text_config = {
        **TEXT_CONFIG,
        "event_memory": {
            **TEXT_CONFIG["event_memory"],
            "replan_interval": 10,
            "replan_phase": 0,
            "expected_phase_update_ratio": 3 / 8,
            "expected_phase_update_tolerance": 0.0,
        },
    }
    pools = build_or_load_semantic_index(
        _Dataset(), text_config, tmp_path / "semantic_index_v1.npz"
    )

    assert pools.phase0_total == 8
    assert pools.update.tolist() == [0, 20, 40]
    assert pools.hard_keep.tolist() == [10, 30, 50]
    assert pools.random_keep.tolist() == [60, 70]

    drifted = {
        **text_config,
        "event_memory": {
            **text_config["event_memory"],
            "expected_phase_update_ratio": 0.0,
        },
    }
    with pytest.raises(ValueError, match="UPDATE ratio drifted"):
        build_or_load_semantic_index(
            _Dataset(), drifted, tmp_path / "semantic_index_v1.npz"
        )
