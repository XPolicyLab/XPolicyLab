"""Exercise checkpoint identity and real file installation with small fixtures."""

from __future__ import annotations

from importlib import import_module

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

assets = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_assets")


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    policy = tmp_path / "policy"
    (policy / "pi05").mkdir(parents=True)
    manifest = assets.load_pi05_metadata()
    payload = {
        "params/manifest.ocdbt": b"fixture-orbax-manifest",
        "assets/arx_x5_sim/norm_stats.json": b'{"norm_stats":{}}',
    }
    manifest["files"] = [
        {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payload.items()
    ]
    manifest["inference_files"] = len(payload)
    manifest["inference_bytes"] = sum(map(len, payload.values()))

    def repin():
        path = policy / "pi05/checkpoint.json"
        path.write_text(json.dumps(manifest))
        monkeypatch.setattr(
            assets,
            "CHECKPOINT_MANIFEST_SHA256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    repin()
    return policy, manifest, payload, repin


def write_payload(root: Path, payload):
    for name, data in payload.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def no_network(*_args, **_kwargs):
    raise AssertionError("Checkpoint verification must not access the network")


def test_official_manifest_is_complete_and_has_no_optimizer_state():
    manifest = assets.load_pi05_metadata()
    assert manifest["revision"] == "35efbc7dedfdbeeb6e95fb749bd885d73d483e41"
    assert manifest["inference_files"] == 18
    assert manifest["inference_bytes"] == 12_440_992_402
    assert all("train_state" not in record["path"] for record in manifest["files"])
    assert {record["path"] for record in manifest["files"]} >= {
        "_CHECKPOINT_METADATA",
        "params/_METADATA",
        "assets/arx_x5_sim/norm_stats.json",
    }


def test_explicit_cache_installs_and_verifies_every_file_offline(
    checkpoint, tmp_path, monkeypatch
):
    policy, manifest, payload, _ = checkpoint
    cache, destination = tmp_path / "cache", tmp_path / "installed"
    write_payload(cache, payload)
    monkeypatch.setattr(assets, "build_opener", no_network)
    receipt = assets.install_pi05_checkpoint(
        destination, cache_dir=cache, offline=True, policy_dir=policy
    )
    assert receipt["verified_files"] == len(payload)
    assert receipt["verified_bytes"] == manifest["inference_bytes"]
    assert not receipt["model_load_verified"]
    assert assets.validate_pi05_checkpoint(destination, policy_dir=policy) == receipt
    assert (
        assets.install_pi05_checkpoint(destination, offline=True, policy_dir=policy)
        == receipt
    )
    for name, data in payload.items():
        assert (destination / name).read_bytes() == data


def test_corrupt_explicit_cache_is_preserved_and_rejected_before_download(
    checkpoint, tmp_path, monkeypatch
):
    policy, _, payload, _ = checkpoint
    cache, destination = tmp_path / "cache", tmp_path / "installed"
    write_payload(cache, payload)
    changed = cache / next(iter(payload))
    changed.write_bytes(b"corrupt")
    monkeypatch.setattr(assets, "build_opener", no_network)
    with pytest.raises(assets.Pi05CheckpointError, match="identity mismatch"):
        assets.install_pi05_checkpoint(destination, cache_dir=cache, policy_dir=policy)
    assert changed.read_bytes() == b"corrupt"
    assert not destination.exists()


def test_corrupt_installed_checkpoint_is_preserved(checkpoint, tmp_path, monkeypatch):
    policy, _, payload, _ = checkpoint
    destination = tmp_path / "installed"
    write_payload(destination, payload)
    changed = destination / next(iter(payload))
    changed.write_bytes(b"x" * len(payload[next(iter(payload))]))
    monkeypatch.setattr(assets, "build_opener", no_network)
    with pytest.raises(assets.Pi05CheckpointError, match="identity mismatch"):
        assets.install_pi05_checkpoint(destination, policy_dir=policy)
    assert changed.read_bytes().startswith(b"x")


def test_missing_offline_checkpoint_fails_without_author_fallback(
    checkpoint, tmp_path, monkeypatch
):
    policy, _, _, _ = checkpoint
    monkeypatch.setattr(assets, "build_opener", no_network)
    with pytest.raises(assets.Pi05CheckpointError, match="unavailable offline"):
        assets.install_pi05_checkpoint(
            tmp_path / "installed", offline=True, policy_dir=policy
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escaped",
        "/absolute",
        "params/../escaped",
        ".",
        "params//bad",
        "train_state/data",
    ],
)
def test_manifest_paths_cannot_escape_or_include_training_state(
    checkpoint, tmp_path, unsafe
):
    policy, manifest, _, repin = checkpoint
    manifest["files"][0]["path"] = unsafe
    repin()
    with pytest.raises(assets.Pi05CheckpointError, match="Unsafe"):
        assets.install_pi05_checkpoint(
            tmp_path / "installed", offline=True, policy_dir=policy
        )
    assert not (tmp_path / "escaped").exists()


def test_changed_manifest_is_rejected_before_cache_access(checkpoint, tmp_path):
    policy, _, _, _ = checkpoint
    with (policy / "pi05/checkpoint.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(assets.Pi05CheckpointError, match="manifest changed"):
        assets.validate_pi05_checkpoint(tmp_path / "absent", policy_dir=policy)


@pytest.mark.parametrize("link_at", ["root", "parent", "file", "partial"])
def test_checkpoint_symlinks_are_rejected(checkpoint, tmp_path, link_at):
    policy, _, payload, _ = checkpoint
    root, outside = tmp_path / "installed", tmp_path / "outside"
    outside.mkdir()
    if link_at == "root":
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir()
        if link_at == "parent":
            (root / "params").symlink_to(outside, target_is_directory=True)
        else:
            path = root / next(iter(payload))
            path.parent.mkdir()
            if link_at == "partial":
                path = path.with_name(path.name + ".partial")
            path.symlink_to(outside / "unchanged")
    with pytest.raises(assets.Pi05CheckpointError, match="symlink"):
        assets.install_pi05_checkpoint(root, offline=True, policy_dir=policy)
    assert not (outside / "unchanged").exists()


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}


def record_for(data):
    return {
        "path": "params/data",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_download_resumes_only_exact_range_and_never_sends_credentials(
    tmp_path, monkeypatch
):
    data = b"0123456789"
    partial = tmp_path / "download.partial"
    partial.write_bytes(data[:4])
    requests = []

    class Client:
        def open(self, request, timeout):
            requests.append(request)
            assert timeout == 60
            return Response(data[4:], 206, {"Content-Range": "bytes 4-9/10"})

    monkeypatch.setattr(assets, "build_opener", lambda *_args: Client())
    monkeypatch.setenv("HF_TOKEN", "fixture-do-not-transmit")
    monkeypatch.setenv("CODEX_API_KEY", "fixture-do-not-transmit")
    assets._download(record_for(data), partial)
    assert partial.read_bytes() == data
    assert requests[0].get_header("Range") == "bytes=4-"
    assert requests[0].get_header("Authorization") is None
    assert "/resolve/35efbc7dedfdbeeb6e95fb749bd885d73d483e41/" in requests[0].full_url


@pytest.mark.parametrize(
    "mode", ["bad_range", "truncated", "oversized", "hash_mismatch"]
)
def test_download_rejects_incomplete_or_wrong_pinned_bytes(tmp_path, monkeypatch, mode):
    data = b"0123456789"
    partial = tmp_path / "download.partial"
    if mode == "bad_range":
        partial.write_bytes(data[:4])
        body, status, headers = data[4:], 206, {"Content-Range": "bytes 3-8/10"}
    else:
        body = {
            "truncated": data[:-1],
            "oversized": data + b"x",
            "hash_mismatch": b"x" * len(data),
        }[mode]
        status, headers = 200, {}

    class Client:
        def open(self, *_args, **_kwargs):
            return Response(body, status, headers)

    monkeypatch.setattr(assets, "build_opener", lambda *_args: Client())
    with pytest.raises(assets.Pi05CheckpointError):
        assets._download(record_for(data), partial)
    assert not (tmp_path / "data").exists()
    if mode == "hash_mismatch":
        assert not partial.exists()


def test_http_errors_do_not_expose_signed_urls(tmp_path, monkeypatch):
    class Client:
        def open(self, *_args, **_kwargs):
            raise HTTPError(
                "https://cdn.example/secret?Signature=fixture-secret",
                403,
                "secret",
                {},
                None,
            )

    monkeypatch.setattr(assets, "build_opener", lambda *_args: Client())
    with pytest.raises(assets.Pi05CheckpointError, match="HTTP 403") as caught:
        assets._download(record_for(b"fixture"), tmp_path / "partial")
    assert "fixture-secret" not in str(caught.value)
    assert "cdn.example" not in str(caught.value)


def test_https_redirect_cannot_downgrade():
    handler = assets._HTTPSRedirect()
    with pytest.raises(assets.Pi05CheckpointError, match="HTTPS"):
        handler.redirect_request(
            Request("https://huggingface.co/file"),
            None,
            302,
            "",
            {},
            "http://example.com/file",
        )
