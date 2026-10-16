"""Offline checks of prerequisite failure, artifact integrity and delegation."""

from __future__ import annotations

from importlib import import_module

import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

bootstrap = import_module("XPolicyLab.policy.SenseNova-RoboRSI.bootstrap")


def test_invalid_tool_recipe_stops_before_build_or_download(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bootstrap.py"])
    monkeypatch.setattr(bootstrap, "POLICY", tmp_path)
    monkeypatch.setattr(
        bootstrap,
        "check_input",
        lambda _: (_ for _ in ()).throw(ValueError("tool recipe invalid")),
    )
    monkeypatch.setattr(
        bootstrap, "fetch", lambda *a: pytest.fail("download before recipe check")
    )
    monkeypatch.setattr(
        bootstrap,
        "build_python",
        lambda *a: pytest.fail("build before recipe check"),
    )
    with pytest.raises(ValueError, match="tool recipe invalid"):
        bootstrap.main()
    assert list(tmp_path.iterdir()) == []


def test_check_has_no_network_build_or_environment_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", "--check"])
    monkeypatch.setattr(bootstrap, "POLICY", tmp_path)
    monkeypatch.setattr(
        bootstrap, "check_input", lambda _: {"status": "build_recipe_verified"}
    )
    monkeypatch.setattr(
        bootstrap, "build_python", lambda *a: pytest.fail("check built Python")
    )
    assert bootstrap.main() == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("actual", [b"short", b"expected+extra", b"modified"])
def test_fetch_rejects_truncated_extra_or_changed_bytes(tmp_path, monkeypatch, actual):
    expected = b"expected"
    pin = (
        "https://example.invalid/fixed",
        hashlib.sha256(expected).hexdigest(),
        len(expected),
    )
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: io.BytesIO(actual))
    with pytest.raises(ValueError):
        bootstrap.fetch(tmp_path, "fixed.tar.xz", pin)
    assert list(tmp_path.iterdir()) == []


def test_verified_cache_is_reused_without_network(tmp_path, monkeypatch):
    data = b"fixed public bytes"
    path = tmp_path / "fixed"
    path.write_bytes(data)
    monkeypatch.setattr(bootstrap, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert (
        bootstrap.fetch(
            tmp_path,
            path.name,
            (
                "https://example.invalid/fixed",
                hashlib.sha256(data).hexdigest(),
                len(data),
            ),
        )
        == path
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cached SHA"):
        bootstrap.fetch(
            tmp_path,
            path.name,
            (
                "https://example.invalid/fixed",
                hashlib.sha256(data).hexdigest(),
                len(data),
            ),
        )


def test_uv_reuses_only_exact_authenticated_executable(tmp_path, monkeypatch):
    archive = tmp_path / "uv.whl"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("uv-0.8.9.data/scripts/uv", b"public native fixture")
    monkeypatch.setattr(bootstrap, "fetch", lambda *a: archive)
    binary = bootstrap.install_uv(tmp_path)
    assert binary.stat().st_mode & 0o111
    assert bootstrap.install_uv(tmp_path) == binary
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its pinned wheel"):
        bootstrap.install_uv(tmp_path)


@pytest.mark.parametrize("installed", [False, True])
def test_installer_uses_selected_host_and_owned_local_scratch(
    tmp_path, monkeypatch, installed
):
    policy = tmp_path / "policy"
    policy.mkdir()
    uv = tmp_path / "uv/bin/uv"
    uv.parent.mkdir(parents=True)
    host = tmp_path / "python3.10"
    monkeypatch.setattr(sys, "argv", ["bootstrap.py"])
    monkeypatch.setattr(bootstrap, "POLICY", policy)
    monkeypatch.setenv("ROBORSI_BOOTSTRAP_ROOT", str(tmp_path / "bootstrap"))
    monkeypatch.setenv("ROBORSI_PYTHON", str(host))
    monkeypatch.setenv("CODEX_API_KEY", "test-only-key")
    monkeypatch.setattr(
        bootstrap,
        "check_input",
        lambda _: {"status": "installed_payload_verified"} if installed else {},
    )

    monkeypatch.setattr(bootstrap, "host_works", lambda p: p == str(host))
    monkeypatch.setattr(bootstrap, "install_uv", lambda _: uv)
    scratch = []

    def install(command, *, env):
        assert command == ["bash", str(policy / "install.sh")]
        assert env["ROBORSI_PYTHON"] == str(host)
        assert env["PATH"].split(os.pathsep)[0] == str(uv.parent)
        assert "CODEX_API_KEY" not in env
        directory = Path(env["TMPDIR"])
        assert directory.parent == Path("/tmp")
        assert directory.stat().st_uid == os.getuid()
        scratch.append(directory)
        return 0

    monkeypatch.setattr(bootstrap.subprocess, "call", install)
    assert bootstrap.main() == 0
    assert not scratch[0].exists()


def test_missing_build_headers_stop_before_source_download(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "host_works", lambda _: False)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "fixture-compiler")
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *a, **k: type("Result", (), {"returncode": 1})(),
    )
    monkeypatch.setattr(
        bootstrap,
        "fetch",
        lambda *a: pytest.fail("source downloaded before header check"),
    )
    with pytest.raises(ValueError, match="development headers"):
        bootstrap.build_python(tmp_path, 1)
    assert list(tmp_path.iterdir()) == []
