"""Check the source snapshot that can be installed without author workspaces."""

from __future__ import annotations

from importlib import import_module

import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

Pi05CheckpointError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_assets").Pi05CheckpointError
POLICY_DIR = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_source").POLICY_DIR
load_openpi_client = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_source").load_openpi_client
validate_pi05_source = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_source").validate_pi05_source


@pytest.fixture
def isolated_client(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in tuple(sys.modules):
        if name == "openpi_client" or name.startswith("openpi_client."):
            monkeypatch.delitem(sys.modules, name)
    yield
    for name in tuple(sys.modules):
        if name == "openpi_client" or name.startswith("openpi_client."):
            sys.modules.pop(name)


@pytest.fixture
def policy(tmp_path):
    root = tmp_path / "policy"
    (root / "vendor").mkdir(parents=True)
    shutil.copy2(
        POLICY_DIR / "vendor/pi05-source.json", root / "vendor/pi05-source.json"
    )
    shutil.copytree(
        POLICY_DIR / "vendor/pi05",
        root / "vendor/pi05",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return root


def test_shipped_sources_include_exact_runtime_overrides_and_server():
    receipt = validate_pi05_source()
    assert receipt["verified_files"] == 197
    assert receipt["verified_bytes"] == 1_878_844
    assert (
        receipt["xpolicylab_source_commit"]
        == "34d6135d609d46a10ccb81da24d5ddad0b4c1f7a"
    )
    assert receipt["openpi_upstream_commit_recorded"] is False


def test_agent_imports_real_codec_and_roundtrips_public_vla_payload(isolated_client):
    previous = set(sys.modules)
    codec = load_openpi_client()
    expected = POLICY_DIR / "vendor/pi05/openpi/packages/openpi-client/src"
    assert Path(codec.__file__).resolve() == expected / "openpi_client/msgpack_numpy.py"
    payload = {
        "state": np.arange(14, dtype=np.float32),
        "images": {"cam_high": np.full((3, 8, 9), [240], dtype=np.uint8)},
        "prompt": "Synthetic public-observation codec check.",
    }
    packed = codec.Packer().pack(payload)
    assert isinstance(packed, bytes)
    restored = codec.unpackb(packed)
    assert restored["prompt"] == payload["prompt"]
    for actual, original in (
        (restored["state"], payload["state"]),
        (restored["images"]["cam_high"], payload["images"]["cam_high"]),
    ):
        np.testing.assert_array_equal(actual, original)
        assert actual.dtype == original.dtype
    assert not any(
        name == "jax" or name.startswith(("jax.", "openpi."))
        for name in set(sys.modules) - previous
    ), "The Agent codec must not import the VLA model runtime"


@pytest.mark.parametrize("name", ["openpi_client", "openpi_client.msgpack_numpy"])
def test_agent_rejects_preloaded_foreign_client_source(
    isolated_client, monkeypatch, name
):
    foreign = ModuleType(name)
    foreign.__file__ = "/synthetic/untrusted/openpi_client.py"
    monkeypatch.setitem(sys.modules, name, foreign)
    with pytest.raises(Pi05CheckpointError, match="different OpenPI client source"):
        load_openpi_client()


@pytest.mark.parametrize("change", ["bytes", "extra", "missing", "symlink"])
def test_source_changes_are_rejected(policy, change):
    source = policy / "vendor/pi05/server/openpi_batch_server.py"
    if change == "bytes":
        source.write_bytes(source.read_bytes() + b"\n")
    elif change == "extra":
        source.with_name("unreviewed.py").write_text("# extra")
    elif change == "missing":
        source.unlink()
    else:
        source.unlink()
        source.symlink_to(POLICY_DIR / "vendor/pi05/server/openpi_batch_server.py")
    with pytest.raises(
        Pi05CheckpointError, match="identity mismatch|inventory differs|symlinks"
    ):
        validate_pi05_source(policy)


def test_source_manifest_tampering_fails_before_import(policy):
    with (policy / "vendor/pi05-source.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(Pi05CheckpointError, match="manifest changed"):
        validate_pi05_source(policy)
