"""Exercise native wheel extraction and release identity at the install boundary."""

from __future__ import annotations

from importlib import import_module

import hashlib
import json
import stat
import zipfile

import pytest

unpack_cli_archive = import_module("XPolicyLab.policy.SenseNova-RoboRSI.cli_artifact").unpack_cli_archive


def artifact(tmp_path, extra=None, manifest_version="0.160.1"):
    path = tmp_path / "native-cli.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "codex_cli_bin/codex-package.json",
            json.dumps(
                {
                    "version": manifest_version,
                    "target": "x86_64-unknown-linux-musl",
                    "entrypoint": "bin/codex",
                }
            ),
        )
        info = zipfile.ZipInfo("codex_cli_bin/bin/codex")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, b"\x7fELF fixture")
        archive.writestr("wrapper.dist-info/METADATA", "Not a CLI identity")
        if extra:
            archive.writestr(*extra)
    return path, {
        "cli": {
            "archive_format": "wheel",
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": "0.160.1",
            "target": "x86_64-unknown-linux-musl",
        }
    }


def test_extracts_executable_native_payload_without_installing_python_metadata(
    tmp_path,
):
    archive, metadata = artifact(tmp_path)
    root = unpack_cli_archive(metadata, archive, tmp_path / "output")
    assert (root / "bin/codex").read_bytes() == b"\x7fELF fixture"
    assert (root / "bin/codex").stat().st_mode & 0o111
    assert not (root / "wrapper.dist-info").exists()


def test_wrong_artifact_hash_fails_before_creating_destination(tmp_path):
    archive, metadata = artifact(tmp_path)
    metadata["cli"]["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        unpack_cli_archive(metadata, archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_authenticated_wheel_cannot_escape_native_payload_root(tmp_path):
    archive, metadata = artifact(tmp_path, ("codex_cli_bin/../escaped", b"unsafe"))
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_cli_archive(metadata, archive, tmp_path / "output")
    assert not (tmp_path / "escaped").exists()


def test_authenticated_wheel_cannot_install_symlinks(tmp_path):
    link = zipfile.ZipInfo("codex_cli_bin/link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive, metadata = artifact(tmp_path, (link, b"/private"))
    with pytest.raises(ValueError, match="regular files"):
        unpack_cli_archive(metadata, archive, tmp_path / "output")


def test_wrapper_version_does_not_replace_native_package_identity(tmp_path):
    archive, metadata = artifact(tmp_path, manifest_version="9.9.9")
    with pytest.raises(ValueError, match="package identity"):
        unpack_cli_archive(metadata, archive, tmp_path / "output")
