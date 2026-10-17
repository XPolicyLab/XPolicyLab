"""Public input validation and safe extraction for the independent tool Python."""

from __future__ import annotations

from importlib import import_module

import hashlib
import io
import shutil
import tarfile
from pathlib import Path

import pytest

installer = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tool_install")


@pytest.mark.parametrize(
    "root", ["/", "/tmp", "/var/tmp", "/opt", "/workspace/tools", "protected-alias"]
)
def test_tool_base_root_cannot_be_hidden_by_sandbox_mounts(tmp_path, root):
    if root == "protected-alias":
        alias = tmp_path / "alias"
        alias.symlink_to("/home", target_is_directory=True)
        root = str(alias / "tools")
    with pytest.raises(ValueError):
        installer._default_root(tmp_path / "policy", selected=root)


def test_existing_shared_directory_permissions_are_preserved(tmp_path):
    shared = tmp_path / "existing"
    shared.mkdir(mode=0o755)
    before = shared.stat().st_mode
    with pytest.raises(ValueError, match="private to its installing user"):
        installer._private_directory(shared)
    assert shared.stat().st_mode == before


def _archive(path, name, *, link=None):
    with tarfile.open(path, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if link is not None:
            member.type = tarfile.SYMTYPE
            member.linkname = link
            stream.addfile(member)
        else:
            member.size = 6
            member.mode = 0o777
            stream.addfile(member, io.BytesIO(b"public"))


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "python/../escape", "another/file"]
)
def test_python_archive_paths_cannot_escape(tmp_path, name):
    archive = tmp_path / "python.tar.gz"
    _archive(archive, name)
    root = tmp_path / "base"
    root.mkdir()
    with pytest.raises(ValueError):
        installer._python_payload(archive, root)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("link", ["/usr/bin/python3", "../../../escape"])
def test_python_archive_links_cannot_escape(tmp_path, link):
    archive = tmp_path / "python.tar.gz"
    _archive(archive, "python/bin/python", link=link)
    root = tmp_path / "base"
    root.mkdir()
    with pytest.raises(ValueError, match="link escapes"):
        installer._python_payload(archive, root)


def test_public_python_archive_is_verified_before_use(tmp_path):
    archive = tmp_path / "python-3.11.13.tar.gz"
    archive.write_bytes(b"public fixture")
    with pytest.raises(ValueError, match="Cached tool Python archive SHA-256"):
        installer._python_archive({"sha256": "0" * 64}, tmp_path)
    pin = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert installer._python_archive({"sha256": pin}, tmp_path) == archive


def test_public_inputs_need_no_private_environment_archive(monkeypatch):
    monkeypatch.delenv("ROBORSI_TOOL_ARCHIVE", raising=False)
    monkeypatch.delenv("ROBORSI_TOOL_ARCHIVE_URL", raising=False)
    result = installer.check_input()
    assert result["status"] == "public_build_inputs_verified"
    assert result["distribution_count"] == 77


def test_openpi_source_is_checked_before_downloading(tmp_path, monkeypatch):
    production = Path(installer.__file__).parent
    policy = tmp_path / "policy"
    shutil.copytree(
        production / "tools",
        policy / "tools",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(production / "submission.json", policy / "submission.json")
    with pytest.raises(OSError):
        installer.check_input(policy)
    assert not (policy / "runtime").exists()


def test_build_children_do_not_inherit_model_key_or_python_overrides(monkeypatch):
    for name in ("CODEX_API_KEY", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        monkeypatch.setenv(name, "test-value")
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    captured = {}
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *args, **kwargs: captured.update(kwargs)
    )
    installer._run(["public-build-command"])
    assert all(
        name not in captured["env"]
        for name in ("CODEX_API_KEY", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    )
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
