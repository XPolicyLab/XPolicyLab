from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from worldscape_policy.checkpoint.weights_io import (
    MODEL_FILENAME,
    MODEL_INDEX_FILENAME,
    checkpoint_weight_files,
    checkpoint_weights_sha256,
    load_checkpoint_state_dict,
    save_checkpoint_state_dict,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "first": torch.arange(16, dtype=torch.float32),
        "second": torch.arange(16, 32, dtype=torch.float32),
        "third": torch.ones(2, dtype=torch.float32),
    }


def test_saves_hugging_face_shards_and_round_trips(tmp_path) -> None:
    written = save_checkpoint_state_dict(_state(), tmp_path, max_shard_size=80)

    assert written[0].name == MODEL_INDEX_FILENAME
    index = json.loads((tmp_path / MODEL_INDEX_FILENAME).read_text())
    assert index["metadata"]["total_size"] == 136
    assert set(index["weight_map"]) == set(_state())
    assert {path.name for path in written[1:]} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    loaded = load_checkpoint_state_dict(tmp_path)
    for key, expected in _state().items():
        torch.testing.assert_close(loaded[key], expected)


def test_small_checkpoint_and_legacy_single_file_are_supported(tmp_path) -> None:
    written = save_checkpoint_state_dict(_state(), tmp_path, max_shard_size="1GB")
    assert [path.name for path in written] == [MODEL_FILENAME]
    assert not (tmp_path / MODEL_INDEX_FILENAME).exists()

    legacy = tmp_path / "legacy.safetensors"
    save_file({"value": torch.tensor([3.0])}, legacy)
    assert load_checkpoint_state_dict(legacy)["value"].item() == 3.0


def test_new_save_removes_stale_single_and_sharded_files(tmp_path) -> None:
    save_checkpoint_state_dict(_state(), tmp_path, max_shard_size=80)
    assert (tmp_path / MODEL_INDEX_FILENAME).is_file()

    save_checkpoint_state_dict(_state(), tmp_path, max_shard_size="1GB")
    assert (tmp_path / MODEL_FILENAME).is_file()
    assert not (tmp_path / MODEL_INDEX_FILENAME).exists()
    assert not list(tmp_path.glob("model-*-of-*.safetensors"))


def test_missing_shard_and_mismatched_index_fail_closed(tmp_path) -> None:
    save_checkpoint_state_dict(_state(), tmp_path, max_shard_size=80)
    shard = checkpoint_weight_files(tmp_path)[1]
    shard.unlink()
    with pytest.raises(FileNotFoundError, match="shard is missing"):
        load_checkpoint_state_dict(tmp_path)

    save_checkpoint_state_dict(_state(), tmp_path, max_shard_size=80)
    index_path = tmp_path / MODEL_INDEX_FILENAME
    index = json.loads(index_path.read_text())
    index["weight_map"]["first"] = index["weight_map"]["third"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index does not match"):
        load_checkpoint_state_dict(tmp_path)


def test_aggregate_hash_covers_index_and_every_shard(tmp_path) -> None:
    save_checkpoint_state_dict(_state(), tmp_path, max_shard_size=80)
    original = checkpoint_weights_sha256(tmp_path)
    index_path = tmp_path / MODEL_INDEX_FILENAME
    index = json.loads(index_path.read_text())
    index["metadata"]["test"] = True
    index_path.write_text(json.dumps(index))

    assert checkpoint_weights_sha256(tmp_path) != original
