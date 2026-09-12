from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tests.helpers.fixture_model_config import fixture_native_model_config
from worldscape_policy.checkpoint import (
    CheckpointTransformArtifact,
    ConversionReport,
    ConversionResult,
    EmbodimentTransform,
    TransformField,
    load_native_checkpoint,
    save_native_checkpoint,
    source_checkpoint_fingerprint,
)
from worldscape_policy.checkpoint.weights_io import checkpoint_weights_sha256


def _save_fixture_native_checkpoint(tmp_path, *, tokenizer_source="org/test-tokenizer"):
    source = nn.Linear(2, 3)
    result = ConversionResult(
        state_dict={key: value.detach() for key, value in source.state_dict().items()},
        report=ConversionReport(source_tensors=2, converted_tensors=2),
        key_mapping_version="fixture_v1",
    )
    save_native_checkpoint(
        result,
        tmp_path,
        model_variant="fixture",
        source_checkpoint_hash="sha256:fixture",
        source_files=[
            {"path": "source.safetensors", "size_bytes": 1, "sha256": "fixture"}
        ],
        model_config=fixture_native_model_config(),
        generation_config={
            "schema_version": "1",
            "num_dit_steps": 8,
            "num_inference_steps": 16,
            "dynamic_cache_schedule": False,
            "kv_cache_fifo": False,
            "cfg_scale": 5.0,
            "sigma_shift": 5.0,
        },
        normalization={
            "schema_version": "1",
            "visual": {
                "input_range": "zero_one",
                "model_range": "minus_one_one",
            },
        },
        transform_bundle=CheckpointTransformArtifact(
            image_input_range="zero_one",
            embodiments={
                "fixture": EmbodimentTransform(
                    embodiment_id=0,
                    max_state_dim=1,
                    max_action_dim=1,
                    state_fields=(
                        TransformField(
                            key="state.value",
                            size=1,
                            normalization=None,
                            statistics={},
                            per_horizon_statistics=None,
                            absolute=True,
                        ),
                    ),
                    action_fields=(
                        TransformField(
                            key="action.value",
                            size=1,
                            normalization=None,
                            statistics={},
                            per_horizon_statistics=None,
                            absolute=True,
                        ),
                    ),
                )
            },
            provenance={"source_format": "fixture"},
        ),
        provenance={"source": {"format": "fixture"}},
        tokenizer_source=tokenizer_source,
    )
    return source


def test_native_checkpoint_loader_requires_manifest_and_exact_keyspace(tmp_path):
    source = _save_fixture_native_checkpoint(tmp_path)
    target = nn.Linear(2, 3)
    manifest = load_native_checkpoint(target, tmp_path)

    assert manifest["format_version"] == "2"
    assert {
        "fixture.Qwen",
        "fixture.T5",
        "fixture.VAE",
        "fixture.Image",
        "fixture.CausalWanModel",
    }.issubset(manifest["dependencies"]["components"])
    torch.testing.assert_close(target.weight, source.weight)
    with pytest.raises(ValueError, match="validation failed"):
        load_native_checkpoint(nn.Linear(3, 3), tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "model.safetensors",
        "model_config.yaml",
        "generation_config.yaml",
        "normalization.json",
        "transform_bundle.json",
        "provenance.json",
        "tokenizer_reference.json",
    ],
)
def test_native_checkpoint_rejects_tampered_artifact_before_loading(
    tmp_path, relative_path
):
    _save_fixture_native_checkpoint(tmp_path)
    path = tmp_path / relative_path
    path.write_bytes(path.read_bytes() + b"\ntampered")

    with pytest.raises(ValueError, match="checksum does not match"):
        load_native_checkpoint(nn.Linear(2, 3), tmp_path)


def test_native_validation_hashes_model_weights_once(tmp_path, monkeypatch):
    import worldscape_policy.checkpoint.loader as checkpoint_loader

    _save_fixture_native_checkpoint(tmp_path)
    original = checkpoint_loader._sha256_file
    model_hash_calls = 0

    def counted(path):
        nonlocal model_hash_calls
        if path.name == "model.safetensors":
            model_hash_calls += 1
        return original(path)

    monkeypatch.setattr(checkpoint_loader, "_sha256_file", counted)
    checkpoint_loader.validate_native_checkpoint_artifacts(tmp_path)

    assert model_hash_calls == 1


def test_native_checkpoint_checksums_every_bundled_tokenizer_file(tmp_path):
    tokenizer = tmp_path / "source-tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"version":"1"}')
    (tokenizer / "special_tokens_map.json").write_text("{}")
    (tokenizer / "pytorch_model.bin").write_bytes(b"unused model weights")
    output = tmp_path / "checkpoint"
    output.mkdir()
    _save_fixture_native_checkpoint(output, tokenizer_source=tokenizer)
    assert not (output / "tokenizer" / "pytorch_model.bin").exists()
    (output / "tokenizer" / "tokenizer.json").write_text('{"version":"tampered"}')

    with pytest.raises(ValueError, match="checksum does not match"):
        load_native_checkpoint(nn.Linear(2, 3), output)


def test_native_checkpoint_rejects_tampered_model_shard(tmp_path):
    from safetensors.torch import load_file

    _save_fixture_native_checkpoint(tmp_path)
    tensors = load_file(str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors").unlink()
    save_file({"weight": tensors["weight"]}, str(tmp_path / "model-00001.safetensors"))
    save_file({"bias": tensors["bias"]}, str(tmp_path / "model-00002.safetensors"))
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "model-00001.safetensors",
                    "bias": "model-00002.safetensors",
                }
            }
        )
    )
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    retained = [
        path for path in manifest["artifacts"] if path != "model.safetensors"
    ]
    model_paths = [
        "model.safetensors.index.json",
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]
    manifest["artifacts"] = sorted([*retained, *model_paths])
    records = [
        record
        for record in manifest["artifact_checksums"]
        if record["path"] != "model.safetensors"
    ]
    for relative in model_paths:
        payload = (tmp_path / relative).read_bytes()
        records.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest["artifact_checksums"] = sorted(records, key=lambda item: item["path"])
    manifest["model_sha256"] = checkpoint_weights_sha256(tmp_path)
    manifest_path.write_text(json.dumps(manifest))
    load_native_checkpoint(nn.Linear(2, 3), tmp_path)
    (tmp_path / "model-00002.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum does not match"):
        load_native_checkpoint(nn.Linear(2, 3), tmp_path)


def test_native_checkpoint_rejects_missing_and_unexpected_artifacts(tmp_path):
    _save_fixture_native_checkpoint(tmp_path)
    (tmp_path / "normalization.json").unlink()
    with pytest.raises(FileNotFoundError, match="normalization.json"):
        load_native_checkpoint(nn.Linear(2, 3), tmp_path)

    _save_fixture_native_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum does not match"):
        load_native_checkpoint(nn.Linear(2, 3), tmp_path)


def test_native_checkpoint_rejects_manifest_path_traversal(tmp_path):
    _save_fixture_native_checkpoint(tmp_path)
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0] = "../outside"
    manifest["artifact_checksums"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsafe path"):
        load_native_checkpoint(nn.Linear(2, 3), tmp_path)


def test_native_checkpoint_rejects_legacy_format_version(tmp_path):
    _save_fixture_native_checkpoint(tmp_path)
    manifest_path = tmp_path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "1"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsupported format_version"):
        load_native_checkpoint(nn.Linear(2, 3), tmp_path)


def test_source_checkpoint_fingerprint_is_deterministic_for_shards(tmp_path):
    save_file({"a": torch.ones(1)}, str(tmp_path / "part-2.safetensors"))
    save_file({"b": torch.zeros(1)}, str(tmp_path / "part-1.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "part-2.safetensors",
                    "b": "part-1.safetensors",
                }
            }
        )
    )

    first = source_checkpoint_fingerprint(tmp_path)
    second = source_checkpoint_fingerprint(tmp_path)

    assert first == second
    assert first[0].startswith("sha256:")
    assert [record["path"] for record in first[1]] == [
        "model.safetensors.index.json",
        "part-1.safetensors",
        "part-2.safetensors",
    ]
