"""Check actual offline ICL installation and failure paths with sealed data."""

from __future__ import annotations

from importlib import import_module

import hashlib
import io
import json
import shutil
import tarfile

import pytest

install_icl = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_assets").install_icl
load_icl_metadata = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_assets").load_icl_metadata
validate_icl_install = import_module("XPolicyLab.policy.SenseNova-RoboRSI.icl_assets").validate_icl_install
FIXTURE_BUNDLE_ID = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").FIXTURE_BUNDLE_ID
write_icl_fixture = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").write_icl_fixture


@pytest.fixture
def policy(tmp_path):
    directory = tmp_path / "policy"
    directory.mkdir()
    (directory / "submission.json").write_text(
        json.dumps({"source": {"commit": "cfa591575172b638a8436fa95db28e9469785c9c"}})
    )
    write_icl_fixture(directory, "stack_bowls")
    return directory


def package_path(policy):
    return policy / "runtime/icl/packages" / FIXTURE_BUNDLE_ID


def archive_path(policy):
    return policy / "assets/icl" / f"{FIXTURE_BUNDLE_ID}.tar.gz"


def repin_artifact(policy):
    archive = archive_path(policy)
    lock_path = policy / "icl/artifacts.json"
    lock = json.loads(lock_path.read_text())
    record = lock["artifacts"][FIXTURE_BUNDLE_ID]
    record["archive_size"] = archive.stat().st_size
    record["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock))
    submission_path = policy / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["icl"]["artifacts_sha256"] = hashlib.sha256(
        lock_path.read_bytes()
    ).hexdigest()
    submission_path.write_text(json.dumps(submission))


def test_installed_complete_package_passes_full_hash_verification(policy):
    catalog, artifacts = load_icl_metadata(policy)
    assert set(catalog["bundles"]) == set(artifacts["artifacts"]) == {FIXTURE_BUNDLE_ID}
    result = validate_icl_install(policy, full=True)
    assert result["bundles"] == result["mapped_tasks"] == 1
    assert result["check"] == "all payload hashes"


def test_clean_install_extracts_pinned_archive_and_is_idempotent(policy):
    expected = (package_path(policy) / "payload/START_HERE.md").read_bytes()
    shutil.rmtree(package_path(policy))
    result = install_icl(policy)
    assert result["payload_integrity"] == "every file verified during installation"
    assert (package_path(policy) / "payload/START_HERE.md").read_bytes() == expected
    assert install_icl(policy) == result


def test_missing_unpublished_archive_does_not_fall_back_to_author_path(policy):
    shutil.rmtree(package_path(policy))
    archive_path(policy).unlink()
    with pytest.raises(ValueError, match="fixed download URL is not published"):
        install_icl(policy)
    assert not package_path(policy).exists()


def test_changed_archive_is_rejected_before_extraction(policy):
    shutil.rmtree(package_path(policy))
    with archive_path(policy).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive identity mismatch"):
        install_icl(policy)
    assert not package_path(policy).exists()


def test_changed_installed_data_is_preserved_and_rejected(policy):
    changed = package_path(policy) / "payload/START_HERE.md"
    changed.write_text("modified experience")
    with pytest.raises(ValueError, match="file inventory differs"):
        install_icl(policy)
    assert changed.read_text() == "modified experience"


@pytest.mark.parametrize("unsafe", ["traversal", "symlink"])
def test_verified_archive_cannot_escape_or_link_outside_package(
    policy, tmp_path, unsafe
):
    archive = archive_path(policy)
    with tarfile.open(archive, "r:gz") as source:
        members = [
            (item, source.extractfile(item).read()) for item in source if item.isfile()
        ]
    with tarfile.open(archive, "w:gz") as destination:
        for item, data in members:
            destination.addfile(item, io.BytesIO(data))
        item = tarfile.TarInfo(
            "../escaped" if unsafe == "traversal" else "payload/linked"
        )
        if unsafe == "symlink":
            item.type = tarfile.SYMTYPE
            item.linkname = str(tmp_path / "outside")
        destination.addfile(item)
    repin_artifact(policy)
    shutil.rmtree(package_path(policy))
    with pytest.raises(ValueError, match="unsafe path|unexpected entry or link"):
        install_icl(policy)
    assert not (policy / "runtime/icl/packages/escaped").exists()
    assert not package_path(policy).exists()


def test_committed_lock_tampering_is_rejected(policy):
    with (policy / "icl/catalog.json").open("a") as stream:
        stream.write("\n ")
    with pytest.raises(ValueError, match="catalog is missing or changed"):
        load_icl_metadata(policy)


def test_explicit_archive_cache_still_requires_the_pinned_bytes(
    policy, tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy2(archive_path(policy), cache / archive_path(policy).name)
    archive_path(policy).unlink()
    shutil.rmtree(package_path(policy))
    monkeypatch.setenv("ROBORSI_ICL_ARCHIVE_DIR", str(cache))
    assert install_icl(policy)["bundles"] == 1
