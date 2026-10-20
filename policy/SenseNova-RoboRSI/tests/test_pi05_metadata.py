"""157-package metadata and source identity stay independent during installation."""

from __future__ import annotations

from importlib import import_module

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

metadata = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_metadata")
runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_runtime")
Pi05CheckpointError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_assets").Pi05CheckpointError

P = Path(__file__).resolve().parents[1]


def copy_policy_metadata(path):
    shutil.copytree(P / "pi05", path / "pi05")
    return path


def write_distributions(site, versions):
    site.mkdir(parents=True, exist_ok=True)
    for name, version in versions.items():
        directory = site / f"{name.replace('-', '_')}-{version}.dist-info"
        directory.mkdir(exist_ok=True)
        (directory / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")


def complete_site(site):
    versions = metadata.expected_runtime_distributions(P)
    write_distributions(site, {n: v for n, v in versions.items() if n != "lerobot"})
    metadata.install_lerobot_metadata(P, site)
    return versions


def test_original_metadata_and_complete_157_package_identity_are_pinned():
    original = metadata.validate_lerobot_metadata(P)
    assert len(metadata.expected_runtime_distributions(P)) == 157
    assert original["runtime_distributions"]["lerobot"] == "0.1.0"
    assert "wheel" not in original["runtime_distributions"]
    assert original["files"]["METADATA"]["sha256"] == (
        "157350f8dc4c5c5901dd65018dbe36a1e15580a4d5d74e246cbd90cf11946ce5"
    )
    rows = list(
        csv.reader(io.StringIO((P / "pi05/lerobot-metadata/RECORD").read_text()))
    )
    source = [r for r in rows if r[0].startswith("lerobot/")]
    assert len(source) == 62
    assert sum(bool(r[1]) for r in source) == 34


def test_metadata_install_preserves_original_RECORD_and_all100_source_files(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    runtime._apply_overrides(P, site)
    before = {
        p.relative_to(site): p.read_bytes()
        for p in (site / "lerobot").rglob("*")
        if p.is_file()
    }
    assert len(before) == 100
    metadata.install_lerobot_metadata(P, site)
    metadata.install_lerobot_metadata(P, site)
    assert before == {
        p.relative_to(site): p.read_bytes()
        for p in (site / "lerobot").rglob("*")
        if p.is_file()
    }
    runtime._check_overrides(P, site)
    assert (site / "lerobot-0.1.0.dist-info/RECORD").read_bytes() == (
        P / "pi05/lerobot-metadata/RECORD"
    ).read_bytes()
    assert (site / "lerobot-0.1.0.dist-info").stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("mutation", ["changed", "extra", "symlink"])
def test_committed_metadata_tampering_is_rejected_before_copy(tmp_path, mutation):
    policy = copy_policy_metadata(tmp_path / "policy")
    source = policy / "pi05/lerobot-metadata"
    if mutation == "changed":
        (source / "RECORD").write_text("fabricated record")
    elif mutation == "extra":
        (source / "extra").write_text("unexpected file")
    else:
        record = source / "RECORD"
        record.unlink()
        record.symlink_to(P / "pi05/lerobot-metadata/RECORD")
    site = tmp_path / "site"
    site.mkdir()
    with pytest.raises(Pi05CheckpointError, match="metadata"):
        metadata.install_lerobot_metadata(policy, site)
    assert not list(site.iterdir())


def test_existing_foreign_or_linked_metadata_is_not_overwritten(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    target = site / "lerobot-0.1.0.dist-info"
    target.symlink_to(P / "pi05/lerobot-metadata", target_is_directory=True)
    with pytest.raises(Pi05CheckpointError, match="linked"):
        metadata.install_lerobot_metadata(P, site)
    target.unlink()
    target.mkdir()
    (target / "RECORD").write_text("foreign metadata")
    with pytest.raises(Pi05CheckpointError, match="inventory"):
        metadata.install_lerobot_metadata(P, site)
    assert (target / "RECORD").read_text() == "foreign metadata"


def test_runtime_lock_drift_cannot_silently_redefine_157_packages(tmp_path):
    policy = copy_policy_metadata(tmp_path / "policy")
    lock = policy / "pi05/requirements.lock"
    lock.write_text(lock.read_text().replace("numpy==1.26.4", "numpy==2.5.4"))
    with pytest.raises(Pi05CheckpointError, match="157-package"):
        metadata.expected_runtime_distributions(policy)


def test_full_distribution_inventory_requires_exact_set_without_imports(tmp_path):
    expected = complete_site(tmp_path / "site")
    result = metadata.check_runtime_distributions(P, tmp_path / "site")
    assert result["count"] == 157 and result["versions"] == expected
    assert (
        result["lerobot_metadata_manifest_sha256"] == metadata.METADATA_MANIFEST_SHA256
    )
    assert "does not attest" in result["historical_RECORD"]


def test_nested_vendor_metadata_does_not_redefine_runtime_distribution_scope(
    tmp_path, monkeypatch
):
    site = tmp_path / "site"
    complete_site(site)
    vendor = tmp_path / "nested-setuptools-vendor"
    write_distributions(vendor, {"wheel": "0.43.0"})
    monkeypatch.syspath_prepend(str(vendor))
    assert metadata.check_runtime_distributions(P, site)["count"] == 157


@pytest.mark.parametrize(
    "mutation", ["wheel", "unknown", "missing", "version", "duplicate", "record"]
)
def test_runtime_inventory_rejects_extras_missing_duplicates_or_changed_metadata(
    tmp_path, mutation
):
    site = tmp_path / "site"
    complete_site(site)
    if mutation in {"wheel", "unknown"}:
        write_distributions(site, {mutation: "0.48.0"})
    elif mutation == "missing":
        shutil.rmtree(site / "numpy-1.26.4.dist-info")
    elif mutation == "version":
        (site / "numpy-1.26.4.dist-info/METADATA").write_text(
            "Name: numpy\nVersion: 2.5.4\n"
        )
    elif mutation == "duplicate":
        write_distributions(site, {"NumPy": "1.26.4"})
    else:
        (site / "lerobot-0.1.0.dist-info/RECORD").write_text("rewritten record")
    with pytest.raises(Pi05CheckpointError, match="metadata|inventory"):
        metadata.check_runtime_distributions(P, site)


def test_installer_keeps_wheel_build_only_and_receipts_exact_157_packages(
    tmp_path, monkeypatch
):
    policy = copy_policy_metadata(tmp_path / "policy")
    shutil.copytree(P / "vendor/pi05", policy / "vendor/pi05")
    shutil.copyfile(P / "vendor/pi05-source.json", policy / "vendor/pi05-source.json")
    base = policy / "runtime/pi05-python/bin/python3.11"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"fixture interpreter; never executed")
    identity = {
        "sha256": "a" * 64,
        "version": "3.11.17",
        "executable": "bin/python3.11",
    }
    (base.parent.parent / ".roborsi-python.json").write_text(
        json.dumps(
            {
                "archive_sha256": identity["sha256"],
                "executable_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
            }
        )
    )
    python = policy / "runtime/pi05-env/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base)
    site = python.parent.parent / "lib/python3.11/site-packages"
    calls = []

    def run(argv, **_kwargs):
        argv = list(map(str, argv))
        calls.append(argv)
        if "uninstall" in argv:
            assert argv[-1] == "wheel"
            shutil.rmtree(site / "wheel-0.48.0.dist-info")
        elif "-r" in argv:
            if argv[-1].endswith("pi05-bootstrap.lock"):
                write_distributions(site, {"wheel": "0.48.0"})
            else:
                versions = metadata.expected_runtime_distributions(policy)
                write_distributions(
                    site, {n: v for n, v in versions.items() if n != "lerobot"}
                )

    monkeypatch.setattr(runtime, "_python_metadata", lambda _p: identity)
    monkeypatch.setattr(runtime, "_uv", lambda _p: "fixture-uv")
    monkeypatch.setattr(runtime, "_run", run)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(stdout="3.11.17\n"),
    )
    monkeypatch.setattr(
        runtime, "_cuda_toolchain", lambda *_a, **_kw: {"fixture": True}
    )
    monkeypatch.setattr(runtime, "install_pi05_checkpoint", lambda *_a, **_kw: None)
    monkeypatch.setattr(runtime, "install_pi05_tokenizer", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        runtime,
        "check_pi05_runtime",
        lambda *_a, **_kw: json.loads(
            (policy / "runtime/pi05-installed.json").read_text()
        ),
    )
    result = runtime.install_pi05_runtime(
        policy, checkpoint_dir=tmp_path / "checkpoint"
    )
    assert len(calls) == 3 and calls[-1][-1] == "wheel"
    assert result["runtime_distributions"]["count"] == 157
    assert "wheel" not in result["runtime_distributions"]["versions"]
    assert result["runtime_distributions"]["versions"]["lerobot"] == "0.1.0"
    assert (
        result["lerobot_metadata_manifest_sha256"] == metadata.METADATA_MANIFEST_SHA256
    )
    runtime._check_overrides(policy, site)
