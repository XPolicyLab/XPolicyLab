"""Failed real ICL copies remain rejected and retain bounded integrity evidence."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from roborsi.components.icl import build_package
from roborsi.components.icl import build, workspace


def sealed_package(tmp_path, files=None):
    source = tmp_path / "material"
    source.mkdir()
    for name, data in (files or {"reference.txt": b"frozen public evidence"}).items():
        (source / name).write_bytes(data)
    return build_package(source, tmp_path / "package", package_id="offline-evidence")


def failure_details(error):
    prefix = "source package changed during materialization; "
    assert str(error.value).startswith(prefix)
    return json.loads(str(error.value)[len(prefix):])


def test_normal_materialization_keeps_independent_bytes_without_diagnostic_reads(
    tmp_path, monkeypatch
):
    package = sealed_package(tmp_path)

    def unexpected_diagnostics(*_args):
        pytest.fail("A successful copy must not reread payloads for diagnostics")

    monkeypatch.setattr(workspace, "_copy_mismatch_details", unexpected_diagnostics)
    result = workspace.materialize_package(package, tmp_path / "episode/icl")
    original = package.root / "payload/reference.txt"
    copied = result.root / "reference.txt"
    assert copied.read_bytes() == original.read_bytes()
    assert result.source_manifest_sha256 == package.manifest_sha256
    copied.write_text("private episode notes")
    assert original.read_bytes() == b"frozen public evidence"


def test_same_size_corrupt_copy_is_rejected_with_source_and_target_hashes(tmp_path, monkeypatch):
    package = sealed_package(tmp_path)
    original_copy = build.shutil.copyfile

    def corrupt_copy(source, target, **kwargs):
        result = original_copy(source, target, **kwargs)
        Path(target).write_bytes(b"frozen public evidencf")
        return result

    monkeypatch.setattr(build.shutil, "copyfile", corrupt_copy)
    destination = tmp_path / "episode/icl"
    with pytest.raises(ValueError) as error:
        workspace.materialize_package(package, destination)
    details = failure_details(error)
    assert {key: details[key] for key in ("missing", "extra", "changed", "omitted")} == {
        "missing": 0, "extra": 0, "changed": 1, "omitted": 0,
    }
    source_sha = hashlib.sha256(b"frozen public evidence").hexdigest()
    target_sha = hashlib.sha256(b"frozen public evidencf").hexdigest()
    assert details["details"] == [{
        "kind": "changed", "path": "reference.txt",
        "expected_sha256": source_sha, "copied_sha256": target_sha,
        "source_current": {"bytes": 22, "sha256": source_sha},
        "target_current": {"bytes": 22, "sha256": target_sha},
    }]
    assert not destination.exists()
    assert not list(destination.parent.glob(".icl-*"))
    assert (package.root / "payload/reference.txt").read_bytes() == b"frozen public evidence"


def test_inventory_failures_keep_missing_extra_and_unreadable_evidence(
    tmp_path, monkeypatch
):
    package = sealed_package(tmp_path)
    original_copy = workspace._copy_tree

    def change_inventory(source, target):
        copied = original_copy(source, target)
        (target / "reference.txt").unlink()
        copied.pop("reference.txt")
        (target / "extra.txt").write_bytes(b"unexpected")
        copied["extra.txt"] = hashlib.sha256(b"unexpected").hexdigest()
        return copied

    monkeypatch.setattr(workspace, "_copy_tree", change_inventory)
    with pytest.raises(ValueError) as error:
        workspace.materialize_package(package, tmp_path / "episode/icl")
    details = failure_details(error)
    assert (details["missing"], details["extra"], details["changed"]) == (1, 1, 0)
    missing, extra = details["details"]
    assert (missing["kind"], missing["path"], missing["copied_sha256"]) == (
        "missing", "reference.txt", None,
    )
    assert missing["target_current"] == {"error": "FileNotFoundError"}
    assert (extra["kind"], extra["path"], extra["expected_sha256"]) == (
        "extra", "extra.txt", None,
    )
    assert extra["source_current"] == {"error": "FileNotFoundError"}


def test_diagnostic_details_are_limited_to_twenty_files(tmp_path, monkeypatch):
    package = sealed_package(
        tmp_path, {f"reference-{index:02}.txt": b"evidence" for index in range(25)}
    )
    original_copy = build.shutil.copyfile

    def corrupt_copy(source, target, **kwargs):
        result = original_copy(source, target, **kwargs)
        Path(target).write_bytes(b"changed!")
        return result

    monkeypatch.setattr(build.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError) as error:
        workspace.materialize_package(package, tmp_path / "episode/icl")
    details = failure_details(error)
    assert details["changed"] == 25
    assert details["omitted"] == 5
    assert len(details["details"]) == 20
    assert [row["path"] for row in details["details"]] == [
        f"reference-{index:02}.txt" for index in range(20)
    ]


def intercept_target_reads(monkeypatch, responses):
    """Simulate early EOF on the new copy while its real on-disk size is full."""
    original_open = Path.open
    reads = []

    def selected_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "rb" and path.parent.name == "workspace":
            reads.append(path)
            response = responses[min(len(reads) - 1, len(responses) - 1)]
            if response is not None:
                return io.BytesIO(response)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", selected_open)
    return reads


def test_transient_empty_and_partial_reads_reopen_and_hash_complete_target(
    tmp_path, monkeypatch
):
    package = sealed_package(tmp_path)
    reads = intercept_target_reads(monkeypatch, [b"", b"frozen", None])
    delays = []
    monkeypatch.setattr(build.time, "sleep", delays.append)
    result = workspace.materialize_package(package, tmp_path / "episode/icl")
    assert len(reads) == 3
    assert delays == [0.05, 0.05]
    assert (result.root / "reference.txt").read_bytes() == b"frozen public evidence"


def test_persistent_premature_eof_rejects_even_when_target_stat_size_is_full(
    tmp_path, monkeypatch
):
    package = sealed_package(tmp_path)
    reads = intercept_target_reads(monkeypatch, [b""])
    delays = []
    monkeypatch.setattr(build.time, "sleep", delays.append)
    destination = tmp_path / "episode/icl"
    with pytest.raises(ValueError, match="expected_bytes=22, read_bytes=0"):
        workspace.materialize_package(package, destination)
    assert len(reads) == 3
    assert delays == [0.05, 0.05]
    assert not destination.exists()
    assert not list(destination.parent.glob(".icl-*"))


def test_legitimate_zero_byte_payload_is_valid_without_retry(tmp_path, monkeypatch):
    package = sealed_package(tmp_path, {"empty.txt": b""})
    reads = intercept_target_reads(monkeypatch, [None])

    def unexpected_retry(_delay):
        pytest.fail("A valid empty file does not need retry")

    monkeypatch.setattr(build.time, "sleep", unexpected_retry)
    result = workspace.materialize_package(package, tmp_path / "episode/icl")
    assert len(reads) == 1
    assert (result.root / "empty.txt").read_bytes() == b""


def test_oversized_copy_rejects_immediately_without_retry(tmp_path, monkeypatch):
    package = sealed_package(tmp_path)
    original_copy = build.shutil.copyfile

    def oversized_copy(source, target, **kwargs):
        result = original_copy(source, target, **kwargs)
        Path(target).write_bytes(b"frozen public evidence plus unexpected bytes")
        return result

    def unexpected_retry(_delay):
        pytest.fail("An oversized copy must fail immediately")

    monkeypatch.setattr(build.shutil, "copyfile", oversized_copy)
    monkeypatch.setattr(build.time, "sleep", unexpected_retry)
    with pytest.raises(ValueError, match="after 1 attempts"):
        workspace.materialize_package(package, tmp_path / "episode/icl")


def test_retry_rejects_target_replaced_by_symlink(tmp_path, monkeypatch):
    package = sealed_package(tmp_path)
    reads = intercept_target_reads(monkeypatch, [b""])

    def replace_target(_delay):
        reads[-1].unlink()
        reads[-1].symlink_to(package.root / "payload/reference.txt")

    monkeypatch.setattr(build.time, "sleep", replace_target)
    with pytest.raises(ValueError, match="cannot traverse links"):
        workspace.materialize_package(package, tmp_path / "episode/icl")
    assert len(reads) == 1
    assert (package.root / "payload/reference.txt").read_bytes() == b"frozen public evidence"
