"""Test explicit tokenizer cache and pinned public-download failure handling."""

from __future__ import annotations

from importlib import import_module

import hashlib
import io
import json

import pytest

tokenizer = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_tokenizer")
Pi05CheckpointError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_assets").Pi05CheckpointError


@pytest.fixture
def resource(tmp_path, monkeypatch):
    policy = tmp_path / "policy"
    (policy / "pi05").mkdir(parents=True)
    metadata = tokenizer.load_pi05_tokenizer_metadata()
    data = b"fixture" * (metadata["size"] // 7) + b"x" * (metadata["size"] % 7)
    metadata["sha256"] = hashlib.sha256(data).hexdigest()
    path = policy / "pi05/tokenizer.json"
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        tokenizer,
        "TOKENIZER_MANIFEST_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(tokenizer, "TOKENIZER_SHA256", metadata["sha256"])
    return policy, metadata, data


def no_network(*_args, **_kwargs):
    raise AssertionError("Offline tokenizer setup must not access the network")


def test_official_tokenizer_identity():
    record = tokenizer.load_pi05_tokenizer_metadata()
    assert record["size"] == 4_264_023
    assert (
        record["sha256"]
        == "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
    )
    assert (
        record["url"]
        == "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
    )


def test_explicit_cache_is_verified_and_copied_to_openpi_cache_layout(
    resource, tmp_path, monkeypatch
):
    policy, metadata, data = resource
    cache, data_home = tmp_path / "tokenizer.cache", tmp_path / "openpi-cache"
    cache.write_bytes(data)
    monkeypatch.setattr(tokenizer, "build_opener", no_network)
    receipt = tokenizer.install_pi05_tokenizer(
        data_home, cache_file=cache, offline=True, policy_dir=policy
    )
    assert receipt["bytes"] == metadata["size"]
    assert (data_home / metadata["cache_relative_path"]).read_bytes() == data
    assert tokenizer.validate_pi05_tokenizer(data_home, policy_dir=policy) == receipt
    assert (
        tokenizer.install_pi05_tokenizer(data_home, offline=True, policy_dir=policy)
        == receipt
    )


def test_missing_and_bad_cache_fail_without_author_fallback(
    resource, tmp_path, monkeypatch
):
    policy, _, _ = resource
    monkeypatch.setattr(tokenizer, "build_opener", no_network)
    with pytest.raises(Pi05CheckpointError, match="unavailable offline"):
        tokenizer.install_pi05_tokenizer(
            tmp_path / "openpi-cache", offline=True, policy_dir=policy
        )
    cache = tmp_path / "corrupt.cache"
    cache.write_bytes(b"corrupt")
    with pytest.raises(Pi05CheckpointError, match="SHA-256 mismatch"):
        tokenizer.install_pi05_tokenizer(
            tmp_path / "openpi-cache", cache_file=cache, policy_dir=policy
        )
    assert cache.read_bytes() == b"corrupt"


class Response(io.BytesIO):
    status = 200


@pytest.mark.parametrize("kind", ["valid", "wrong_hash", "truncated", "oversized"])
def test_public_download_verifies_identity_and_never_sends_credentials(
    resource, tmp_path, monkeypatch, kind
):
    policy, metadata, data = resource
    payload = {
        "valid": data,
        "wrong_hash": b"x" * len(data),
        "truncated": data[:-1],
        "oversized": data + b"x",
    }[kind]

    class Client:
        def open(self, request, timeout):
            assert request.full_url == metadata["url"]
            assert request.get_header("Authorization") is None
            assert timeout == 60
            return Response(payload)

    monkeypatch.setattr(tokenizer, "build_opener", lambda *_args: Client())
    monkeypatch.setenv("HF_TOKEN", "fixture-do-not-transmit")
    monkeypatch.setenv("CODEX_API_KEY", "fixture-do-not-transmit")
    data_home = tmp_path / "openpi-cache"
    if kind == "valid":
        assert (
            tokenizer.install_pi05_tokenizer(data_home, policy_dir=policy)["sha256"]
            == metadata["sha256"]
        )
    else:
        with pytest.raises(Pi05CheckpointError):
            tokenizer.install_pi05_tokenizer(data_home, policy_dir=policy)
        assert not (data_home / metadata["cache_relative_path"]).exists()
    assert not list((data_home / "big_vision").glob(".pi05-tokenizer-*"))


def test_tokenizer_manifest_cannot_redirect_download(resource, tmp_path, monkeypatch):
    policy, _, _ = resource
    monkeypatch.setattr(tokenizer, "build_opener", no_network)
    path = policy / "pi05/tokenizer.json"
    document = json.loads(path.read_text())
    document["url"] = "https://untrusted.example/tokenizer"
    path.write_text(json.dumps(document))
    with pytest.raises(Pi05CheckpointError, match="manifest changed"):
        tokenizer.install_pi05_tokenizer(tmp_path / "openpi-cache", policy_dir=policy)
