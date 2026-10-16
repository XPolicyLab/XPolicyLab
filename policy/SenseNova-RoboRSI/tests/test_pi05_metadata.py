"""B157 provenance stays distinct from the consistent inference environment."""

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
    write_distributions(site, versions)
    return versions


def test_original_metadata_and_complete_B157_identity_are_pinned():
    original = metadata.validate_lerobot_metadata(P)
    expected = metadata.expected_runtime_distributions(P)
    assert len(expected) == 142
    assert set(original["runtime_distributions"]) - set(expected) == (
        metadata.EXCLUDED_RUNTIME_DISTRIBUTIONS
    )
    assert all(original["runtime_distributions"][n] == v for n, v in expected.items())
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


def test_source_overlay_preserves_all100_files_without_legacy_distribution(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    runtime._apply_overrides(P, site)
    before = {
        p.relative_to(site): p.read_bytes()
        for p in (site / "lerobot").rglob("*")
        if p.is_file()
    }
    assert len(before) == 100
    metadata.validate_lerobot_metadata(P)
    assert before == {
        p.relative_to(site): p.read_bytes()
        for p in (site / "lerobot").rglob("*")
        if p.is_file()
    }
    runtime._check_overrides(P, site)
    assert not (site / "lerobot-0.1.0.dist-info").exists()


@pytest.mark.parametrize("mutation", ["changed", "extra", "symlink"])
def test_committed_reference_tampering_is_rejected(tmp_path, mutation):
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
        metadata.expected_runtime_distributions(policy)
    assert not list(site.iterdir())


@pytest.mark.parametrize("linked", [False, True])
def test_legacy_distribution_metadata_cannot_reenter_runtime(tmp_path, linked):
    site = tmp_path / "site"
    complete_site(site)
    target = site / "lerobot-0.1.0.dist-info"
    if linked:
        target.symlink_to(P / "pi05/lerobot-metadata", target_is_directory=True)
    else:
        write_distributions(site, {"lerobot": "0.1.0"})
    with pytest.raises(Pi05CheckpointError, match="inventory"):
        metadata.check_runtime_distributions(P, site)


def test_runtime_lock_drift_cannot_silently_redefine_B157(tmp_path):
    policy = copy_policy_metadata(tmp_path / "policy")
    lock = policy / "pi05/requirements.lock"
    lock.write_text(lock.read_text().replace("numpy==1.26.4", "numpy==2.5.4"))
    with pytest.raises(Pi05CheckpointError, match="B157"):
        metadata.expected_runtime_distributions(policy)


def test_full_distribution_inventory_requires_exact_set_without_imports(tmp_path):
    expected = complete_site(tmp_path / "site")
    result = metadata.check_runtime_distributions(P, tmp_path / "site")
    assert result["count"] == 142 and result["versions"] == expected
    assert (
        result["lerobot_metadata_manifest_sha256"] == metadata.METADATA_MANIFEST_SHA256
    )
    assert "does not attest" in result["historical_RECORD"]
    assert result["reference_distribution_count"] == 157
    assert not result["lerobot_metadata_installed"]
    assert set(result["excluded_distributions"]) == metadata.EXCLUDED_RUNTIME_DISTRIBUTIONS


def test_nested_vendor_metadata_does_not_redefine_runtime_distribution_scope(
    tmp_path, monkeypatch
):
    site = tmp_path / "site"
    complete_site(site)
    vendor = tmp_path / "nested-setuptools-vendor"
    write_distributions(vendor, {"wheel": "0.43.0"})
    monkeypatch.syspath_prepend(str(vendor))
    assert metadata.check_runtime_distributions(P, site)["count"] == 142


@pytest.mark.parametrize(
    "mutation", ["wheel", "unknown", "missing", "version", "duplicate", "cloud"]
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
        write_distributions(site, {"gcsfs": "2025.3.0"})
    with pytest.raises(Pi05CheckpointError, match="metadata|inventory"):
        metadata.check_runtime_distributions(P, site)


def test_installer_removes_legacy_cloud_distributions_and_preserves_source(
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
            assert metadata.EXCLUDED_RUNTIME_DISTRIBUTIONS <= set(argv)
            shutil.rmtree(site / "wheel-0.48.0.dist-info")
            for name in metadata.EXCLUDED_RUNTIME_DISTRIBUTIONS:
                for directory in site.glob(f"{name.replace('-', '_')}-*.dist-info"):
                    shutil.rmtree(directory)
        elif "-r" in argv:
            if argv[-1].endswith("pi05-bootstrap.lock"):
                write_distributions(site, {"wheel": "0.48.0"})
                original = metadata.validate_lerobot_metadata(policy)["runtime_distributions"]
                write_distributions(
                    site,
                    {n: original[n] for n in metadata.EXCLUDED_RUNTIME_DISTRIBUTIONS},
                )
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
    assert len(calls) == 4 and calls[2][-1] == "wheel"
    assert calls[-1][1:3] == ["pip", "check"]
    assert result["runtime_distributions"]["count"] == 142
    assert "wheel" not in result["runtime_distributions"]["versions"]
    assert not metadata.EXCLUDED_RUNTIME_DISTRIBUTIONS & (
        result["runtime_distributions"]["versions"].keys()
    )
    assert not (site / "lerobot-0.1.0.dist-info").exists()
    assert (
        result["lerobot_metadata_manifest_sha256"] == metadata.METADATA_MANIFEST_SHA256
    )
    runtime._check_overrides(policy, site)
