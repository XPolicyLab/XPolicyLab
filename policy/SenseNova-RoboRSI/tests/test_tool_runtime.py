"""Reject changed public build inputs and changed installed tool packages."""

from __future__ import annotations

from importlib import import_module

import json
import os
import shutil
from pathlib import Path

import pytest

tools = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tool_runtime")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def installed_tools(tmp_path):
    policy = tmp_path / "policy"
    production = Path(tools.__file__).parent
    shutil.copytree(
        production / "tools",
        policy / "tools",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    metadata = json.loads((policy / "tools/runtime.json").read_text())
    metadata["paths"].update(runtime="runtime/agent-venv", base="runtime/tools-base")
    roots = {name: policy / metadata["paths"][name] for name in ("runtime", "base")}
    python = roots["base"] / "bin/python3.11"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test ELF identity, never executed")
    metadata["python"]["binary_sha256"] = tools._sha(python)
    (roots["runtime"] / "bin").mkdir(parents=True)
    (roots["runtime"] / "bin/python").symlink_to(python)
    site = roots["runtime"] / "lib/python3.11/site-packages"
    for row in metadata["distributions"]:
        path = site / (
            row["name"].replace("-", "_") + "-" + row["version"] + ".dist-info/METADATA"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Name: {row['name']}\nVersion: {row['version']}\n")
    pins = json.loads((policy / "tools/openpi-client-files.json").read_text())
    source = production / "vendor/pi05/openpi/packages/openpi-client/src/openpi_client"
    for name in pins:
        target = site / "openpi_client" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    extra = site / "numpy/payload.bin"
    extra.parent.mkdir()
    extra.write_bytes(b"scientific payload")
    (policy / "runtime").chmod(0o700)
    for root in roots.values():
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o644)
    _write(policy / "tools/runtime.json", metadata)
    _write(
        policy / "submission.json",
        {
            "tool_runtime": {
                "manifest_path": "tools/runtime.json",
                "manifest_sha256": tools._sha(policy / "tools/runtime.json"),
            }
        },
    )
    inventory = policy / metadata["paths"]["inventory"]
    _write(inventory, tools._inventory(roots))
    _write(
        policy / metadata["paths"]["receipt"],
        {
            "schema_version": 2,
            "metadata_sha256": tools._sha(policy / "tools/runtime.json"),
            "python_version": "3.11.13",
            "installation_mode": "public-source-build",
            "inventory_sha256": tools._sha(inventory),
        },
    )
    return policy, metadata, site


def test_resolver_keeps_independent_venv_spelling(installed_tools):
    policy, metadata, _ = installed_tools
    assert (
        tools.resolve_tool_python(policy)
        == policy / metadata["paths"]["runtime"] / "bin/python"
    )
    receipt = tools.validate_tool_install(policy, full=True)
    assert receipt["distribution_count"] == 77
    assert receipt["full_payload_verified"] is True
    assert receipt["source_payloads_verified"] is True


def test_public_recipe_keeps_runtime_versions_without_development_packages():
    metadata = tools.load_tool_metadata(Path(tools.__file__).parent)
    packages = tools._packages(metadata["distributions"])
    assert len(packages) == 77
    assert {
        "heart-runtime", "pytest", "pytest-timeout", "iniconfig", "pluggy", "pygments"
    }.isdisjoint(packages)
    assert packages["openpi-client"] == "0.1.0"
    assert packages["openai-codex-cli-bin"] == "0.159.3"
    assert metadata["python_version"] == "3.11.13"
    assert "archive" not in metadata


def test_build_resource_tampering_fails_before_execution(installed_tools):
    policy, _, _ = installed_tools
    (policy / "tools/requirements.lock").write_text("numpy==2\n")
    with pytest.raises(tools.ToolRuntimeError, match="build resource changed"):
        tools.load_tool_metadata(policy)


@pytest.mark.parametrize("change", ["missing", "version", "additional"])
def test_distribution_identity_is_exact(installed_tools, change):
    policy, _, site = installed_tools
    target = next(site.glob("*.dist-info/METADATA"))
    if change == "missing":
        target.unlink()
    elif change == "version":
        target.write_text("Name: numpy\nVersion: 0.0.0\n")
    else:
        target = site / "unexpected-1.dist-info/METADATA"
        target.parent.mkdir()
        target.write_text("Name: unexpected\nVersion: 1\n")
    with pytest.raises(tools.ToolRuntimeError):
        tools.validate_tool_install(policy)


def test_original_openpi_payload_changes_fail_at_startup(installed_tools):
    policy, _, site = installed_tools
    path = site / "openpi_client/msgpack_numpy.py"
    value = path.read_bytes()
    path.write_bytes(b"X" + value[1:])
    with pytest.raises(tools.ToolRuntimeError, match="SHA-256 differs"):
        tools.validate_tool_install(policy)


def test_full_check_catches_same_size_numerical_payload_change(installed_tools):
    policy, _, site = installed_tools
    path = site / "numpy/payload.bin"
    path.write_bytes(b"X" * path.stat().st_size)
    with pytest.raises(tools.ToolRuntimeError, match="SHA-256 differs"):
        tools.validate_tool_install(policy, full=True)


def test_interpreter_cannot_link_to_host_python(installed_tools):
    policy, metadata, _ = installed_tools
    python = policy / metadata["paths"]["runtime"] / "bin/python"
    python.unlink()
    python.symlink_to("/usr/bin/python3")
    with pytest.raises(tools.ToolRuntimeError, match="link changed or escapes"):
        tools.resolve_tool_python(policy)


def test_receipt_cannot_select_another_build(installed_tools):
    policy, metadata, _ = installed_tools
    receipt = policy / metadata["paths"]["receipt"]
    value = json.loads(receipt.read_text())
    value["metadata_sha256"] = "0" * 64
    _write(receipt, value)
    with pytest.raises(tools.ToolRuntimeError, match="receipt is missing or invalid"):
        tools.resolve_tool_python(policy)


def test_generated_bytecode_does_not_change_source_inventory(installed_tools):
    policy, _, site = installed_tools
    bytecode = site / "openpi_client/__pycache__/generated.cpython-311.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(os.urandom(16))
    assert tools.validate_tool_install(policy)["distribution_count"] == 77


def test_explicit_root_cannot_silently_reuse_another_install(
    installed_tools, monkeypatch, tmp_path
):
    policy, _, _ = installed_tools
    monkeypatch.setenv("ROBORSI_TOOL_ROOT", str(tmp_path / "another-tools"))
    with pytest.raises(tools.ToolRuntimeError, match="differs from the installed"):
        tools.resolve_tool_python(policy)
