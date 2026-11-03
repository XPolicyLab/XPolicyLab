"""Exercise real transport primitives with the target Python version simulated."""

from __future__ import annotations

from importlib import import_module

import builtins
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

capabilities = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_capabilities")
tool_runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tool_runtime")


@pytest.fixture(autouse=True)
def target_python_sys(monkeypatch):
    """Keep tests on the available interpreter; mock only the probe's sys view."""

    class TargetPython:
        version_info = (3, 10, 12, "final", 0)

        def __getattr__(self, name):
            return getattr(sys, name)

    selected = TargetPython()
    monkeypatch.setattr(capabilities, "sys", selected)
    return selected


def tool_stdout(policy, *, base=None):
    packages = json.loads(
        (Path(capabilities.__file__).parent / "tools/runtime.json").read_text()
    )["distributions"]
    return json.dumps(
        {
            "marker": "ROBORSI_ORIGINAL_TOOL_CPU_OK",
            "python": "3.11.13",
            "prefix": "/runtime",
            "base_prefix": str(base or policy / "runtime/tools-base"),
            "distributions": packages,
            "stdlib_verified": True,
            "numpy_verified": True,
            "pillow_verified": True,
            "scipy_verified": True,
            "strategy_tools_verified": True,
        }
    ).encode()


def captured_descriptors(monkeypatch):
    original = os.memfd_create
    created = []

    def create(name, flags):
        assert name == "roborsi-runtime-probe"
        assert flags == os.MFD_CLOEXEC
        descriptor = original(name, flags)
        created.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "memfd_create", create)
    return created


def assert_closed(descriptors):
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


def test_real_memfd_roundtrip_handles_partial_writes_and_closes(monkeypatch):
    created = captured_descriptors(monkeypatch)
    original_write = os.write
    writes = []

    def partial_write(descriptor, data):
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        written = original_write(descriptor, data[:3])
        writes.append(written)
        return written

    monkeypatch.setattr(os, "write", partial_write)
    receipt = capabilities.check_python_capabilities()
    assert receipt["python"] == "3.10.12"
    assert receipt["memfd_create_verified"] is True
    assert receipt["write_read_verified"] is True
    assert receipt["fd_cloexec_verified"] is True
    assert receipt["descriptor_closed"] is True
    assert receipt["child_process_started"] is False
    assert sum(writes) == receipt["probe_bytes"]
    assert len(writes) > 1
    assert_closed(created)


@pytest.mark.parametrize("feature", ["memfd_create", "MFD_CLOEXEC"])
def test_missing_python_api_fails_with_required_interpreter(monkeypatch, feature):
    monkeypatch.delattr(os, feature)
    with pytest.raises(
        capabilities.RuntimeCapabilityError, match="Linux Python 3.10.12"
    ):
        capabilities.check_python_capabilities()


@pytest.mark.parametrize("version", [(3, 10, 11), (3, 10, 13), (3, 11, 13), (3, 12, 3)])
def test_wrong_host_version_fails_before_any_syscall(
    monkeypatch, target_python_sys, version
):
    monkeypatch.setattr(target_python_sys, "version_info", (*version, "final", 0))
    monkeypatch.setattr(
        os,
        "memfd_create",
        lambda *_args: pytest.fail("Wrong host version must fail before memfd"),
    )
    with pytest.raises(
        capabilities.RuntimeCapabilityError, match="Linux Python 3.10.12"
    ):
        capabilities.check_python_capabilities()


def test_kernel_denied_memfd_is_a_static_error_without_raw_values(monkeypatch):
    def denied(*_args):
        raise OSError(errno.EPERM, "SECRET_FROM_RAW_SYSTEM_ERROR")

    monkeypatch.setattr(os, "memfd_create", denied)
    with pytest.raises(
        capabilities.RuntimeCapabilityError, match="memfd_create"
    ) as caught:
        capabilities.check_python_capabilities()
    assert "SECRET_FROM_RAW_SYSTEM_ERROR" not in str(caught.value)


@pytest.mark.parametrize("stage", ["write", "lseek", "read", "fcntl"])
def test_probe_syscall_failure_closes_its_owned_descriptor(monkeypatch, stage):
    created = captured_descriptors(monkeypatch)

    def failed(*_args):
        raise OSError(errno.EIO, "SECRET_FROM_RAW_SYSTEM_ERROR")

    module = fcntl if stage == "fcntl" else os
    monkeypatch.setattr(module, stage, failed)
    with pytest.raises(capabilities.RuntimeCapabilityError) as caught:
        capabilities.check_python_capabilities()
    assert "SECRET_FROM_RAW_SYSTEM_ERROR" not in str(caught.value)
    assert_closed(created)


@pytest.mark.parametrize(
    "failure", ["missing-cloexec", "zero-write", "wrong-offset", "different-bytes"]
)
def test_probe_rejects_nonfunctional_api_and_closes(monkeypatch, failure):
    created = captured_descriptors(monkeypatch)
    if failure == "missing-cloexec":
        monkeypatch.setattr(fcntl, "fcntl", lambda *_args: 0)
    elif failure == "zero-write":
        monkeypatch.setattr(os, "write", lambda *_args: 0)
    elif failure == "wrong-offset":
        monkeypatch.setattr(os, "lseek", lambda *_args: 1)
    else:
        monkeypatch.setattr(os, "read", lambda *_args: b"wrong public bytes")
    with pytest.raises(capabilities.RuntimeCapabilityError):
        capabilities.check_python_capabilities()
    assert_closed(created)


def test_python_probe_reads_no_environment_or_api_key(monkeypatch):
    class NoEnvironment:
        def denied(self, *_args, **_kwargs):
            raise AssertionError("Capability checks must not read environment values")

        __getitem__ = __contains__ = __iter__ = get = keys = items = values = denied

    with monkeypatch.context() as isolated:
        isolated.setattr(os, "environ", NoEnvironment())
        isolated.setattr(os, "getenv", NoEnvironment().denied)
        receipt = capabilities.check_python_capabilities()
    assert receipt["descriptor_closed"] is True


def test_close_failure_is_not_reported_as_a_success(monkeypatch):
    created = captured_descriptors(monkeypatch)
    original_close = os.close

    def close_then_error(descriptor):
        original_close(descriptor)
        raise OSError(errno.EIO, "PRIVATE_CLOSE_ERROR")

    with monkeypatch.context() as isolated:
        isolated.setattr(os, "close", close_then_error)
        with pytest.raises(
            capabilities.RuntimeCapabilityError, match="could not close"
        ) as caught:
            capabilities.check_python_capabilities()
    assert "PRIVATE_CLOSE_ERROR" not in str(caught.value)
    assert_closed(created)


def test_cli_imports_only_stdlib_and_emits_public_json(
    monkeypatch, capsys, target_python_sys
):
    original_import = builtins.__import__

    def standard_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"roborsi", "yaml", "numpy", "torch", "jax"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", standard_import)
    path = Path(capabilities.__file__)
    spec = importlib.util.spec_from_file_location("standalone_capability_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "sys", target_python_sys)
    assert module.main(["--python-only"]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert json.loads(captured.out)["memfd_create_verified"] is True


def test_cli_feature_failure_is_nonzero_without_traceback(monkeypatch, capsys):
    monkeypatch.delattr(os, "memfd_create")
    assert capabilities.main(["--python-only"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "Linux Python 3.10.12" in captured.err
    assert "Traceback" not in captured.err


@pytest.fixture
def sandbox_policy(tmp_path, monkeypatch):
    """Use the real sealed transport source with a private fixture binary pin."""
    write_source_manifest = import_module("XPolicyLab.policy.SenseNova-RoboRSI.preflight").write_source_manifest

    production = Path(capabilities.__file__).parent
    policy = tmp_path / "policy"
    (policy / "vendor").mkdir(parents=True)
    for name in ("roborsi_source", "roborsi-source.json", "robodojo_adaptive"):
        source = production / "vendor" / name
        target = policy / "vendor" / name
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copyfile(source, target)
    installed = policy / "runtime/RoboRSI"
    shutil.copytree(policy / "vendor/roborsi_source", installed)
    binary = policy / "runtime/codex/codex-resources/bwrap"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"private offline unit-test bubblewrap fixture")
    metadata = json.loads((production / "submission.json").read_text())
    metadata["cli"]["companion_sha256"]["codex-resources/bwrap"] = hashlib.sha256(
        binary.read_bytes()
    ).hexdigest()
    cli_root = policy / "runtime/codex"
    (cli_root / "bin").mkdir()
    for relative in ("bin/codex", "bin/codex-code-mode-host"):
        path = cli_root / relative
        path.write_bytes(b"\x7fELF private offline fixture " + relative.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative == "bin/codex":
            metadata["cli"]["sha256"] = digest
        else:
            metadata["cli"]["companion_sha256"][relative] = digest
    (cli_root / "codex-package.json").write_text(
        json.dumps(
            {
                "version": metadata["cli"]["version"],
                "target": metadata["cli"]["target"],
                "entrypoint": "bin/codex",
            }
        )
    )
    (policy / "submission.json").write_text(json.dumps(metadata))
    write_source_manifest(
        metadata,
        policy,
        installed,
        artifact_sha256=metadata["source"]["manifest_sha256"],
        origin="committed_bundle",
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    tool_base = policy / "runtime/tools-base"
    (tool_base / "bin").mkdir(parents=True)
    (tool_base / "bin/python3.11").write_bytes(
        b"public tool interpreter fixture; not executed"
    )
    tool_python = policy / "runtime/agent-venv/bin/python"
    tool_python.parent.mkdir(parents=True)
    # The venv becomes /runtime in the original sandbox; its separate base
    # remains bound at this absolute path, so a relative cross-root link fails.
    tool_python.symlink_to(tool_base / "bin/python3.11")
    packages = json.loads((production / "tools/runtime.json").read_text())[
        "distributions"
    ]
    monkeypatch.setattr(
        tool_runtime, "resolve_tool_python", lambda _policy: tool_python
    )
    monkeypatch.setattr(
        tool_runtime, "load_tool_metadata", lambda _policy: {"distributions": packages}
    )
    loose_reference_names = {
        "adapter",
        "trial_adapter",
        "eef_kinematics",
        "waypoints",
        "hybrid_adapter",
        "hybrid_host",
    }

    def reference_module(name):
        return (
            name == "robots"
            or name.startswith("robots.")
            or name == "policylab"
            or name.startswith("policylab.")
            or name in loose_reference_names
        )

    original_modules = {
        name: module for name, module in sys.modules.items() if reference_module(name)
    }
    for name in original_modules:
        del sys.modules[name]
    try:
        yield policy
    finally:
        for name in tuple(sys.modules):
            if reference_module(name):
                del sys.modules[name]
        sys.modules.update(original_modules)


def test_sandbox_check_uses_original_memfd_args_and_private_child_environment(
    sandbox_policy, monkeypatch
):
    descriptors = []
    private_homes = []
    monkeypatch.setenv("CODEX_API_KEY", "TEST_SECRET_NOT_FORWARDED")
    monkeypatch.setenv("HTTPS_PROXY", "https://TEST_SECRET_NOT_FORWARDED.invalid")

    def marker(command, **kwargs):
        assert command[1] == "--args"
        descriptor = int(command[2])
        descriptors.append(descriptor)
        assert descriptor in kwargs["pass_fds"]
        options = os.read(descriptor, 32768).split(b"\0")
        assert b"--unshare-user" in options and b"--unshare-pid" in options
        assert b"--proc" not in options and b"--dev" in options
        assert [
            b"--dir",
            b"/proc",
            b"--dir",
            b"/proc/self",
            b"--symlink",
            b"/codex",
            b"/proc/self/exe",
        ] == options[options.index(b"--dir") : options.index(b"--dev")]
        assert b"--ro-bind" in options and b"--clearenv" in options
        assert b"HTTPS_PROXY" not in options
        key_index = options.index(b"CODEX_API_KEY")
        assert options[key_index - 1] == b"--setenv" and options[key_index + 1] == b""
        assert b"TEST_SECRET_NOT_FORWARDED" not in options
        assert command[3:8] == ["--", "/runtime/bin/python", "-I", "-B", "-c"]
        assert "from scipy.spatial.transform import Rotation" in command[8]
        assert kwargs["timeout"] == 90
        assert set(kwargs["env"]) == {"PATH", "LANG", "HOME"}
        home = Path(kwargs["env"]["HOME"])
        assert home.is_dir()
        private_homes.append(home)
        return subprocess.CompletedProcess(
            command, 0, tool_stdout(sandbox_policy), b"IGNORED_PRIVATE_STDERR"
        )

    monkeypatch.setattr(subprocess, "run", marker)
    receipt = capabilities.check_sandbox_capabilities(sandbox_policy)
    assert receipt["original_transport_verified"] is True
    assert receipt["memfd_arguments_verified"] is True
    assert receipt["rootless_namespaces_verified"] is True
    assert receipt["marker_verified"] is True
    assert receipt["adaptive_sandbox_verified"] is True
    assert receipt["proc_layout_verified"] is True
    assert "proc_dev_ro_bind_verified" not in receipt
    assert receipt["model_or_api_used"] is False
    assert receipt["tool_distribution_count"] == 82
    assert receipt["tools"]["python"] == "3.11.13"
    assert "heart_runtime_verified" not in receipt["tools"]
    assert "TEST_SECRET" not in json.dumps(receipt)
    assert "IGNORED_PRIVATE_STDERR" not in json.dumps(receipt)
    assert private_homes and all(not path.exists() for path in private_homes)
    assert_closed(descriptors)


@pytest.mark.parametrize(
    "failure", ["namespace-denied", "timeout", "launch", "wrong-marker"]
)
def test_sandbox_check_fails_without_raw_child_output_or_fallback(
    sandbox_policy, monkeypatch, failure
):
    descriptors = []
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        descriptors.append(int(command[2]))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 90, output=b"PRIVATE_OUTPUT", stderr=b"PRIVATE_ERROR"
            )
        if failure == "launch":
            raise OSError(errno.EPERM, "PRIVATE_ERROR")
        return subprocess.CompletedProcess(
            command,
            1 if failure == "namespace-denied" else 0,
            b"PRIVATE_OUTPUT",
            b"PRIVATE_ERROR",
        )

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(capabilities.RuntimeCapabilityError) as caught:
        capabilities.check_sandbox_capabilities(sandbox_policy)
    assert "PRIVATE_" not in str(caught.value)
    assert len(calls) == 1
    assert_closed(descriptors)


@pytest.mark.parametrize("target", ["binary", "transport", "adaptive"])
def test_sandbox_identity_changes_fail_before_any_child_launch(
    sandbox_policy, monkeypatch, target
):
    changed = (
        sandbox_policy
        / {
            "binary": "runtime/codex/codex-resources/bwrap",
            "transport": "runtime/RoboRSI/robots/robodojo/agents/icl_pi05/bwrap_transport.py",
            "adaptive": "vendor/robodojo_adaptive/source/local_runtime.py",
        }[target]
    )
    changed.write_bytes(changed.read_bytes() + b"\nchanged\n")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "Changed identity must not launch a child"
        ),
    )
    with pytest.raises(capabilities.RuntimeCapabilityError):
        capabilities.check_sandbox_capabilities(sandbox_policy)


def test_cli_policy_dir_runs_full_installed_check(monkeypatch, capsys, tmp_path):
    seen = []

    def checked(policy):
        seen.append(policy)
        return {"marker_verified": True}

    monkeypatch.setattr(capabilities, "check_sandbox_capabilities", checked)
    assert capabilities.main(["--policy-dir", str(tmp_path)]) == 0
    assert seen == [tmp_path]
    assert json.loads(capsys.readouterr().out) == {"marker_verified": True}


def test_cached_foreign_transport_is_rejected_before_launch(
    sandbox_policy, monkeypatch
):
    name = "robots.robodojo.agents.icl_pi05.bwrap_transport"
    foreign = ModuleType(name)
    foreign.__file__ = "/synthetic/other-source/bwrap_transport.py"
    monkeypatch.setitem(sys.modules, name, foreign)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "Foreign transport must not launch a child"
        ),
    )
    with pytest.raises(
        capabilities.RuntimeCapabilityError, match="different sandbox transport"
    ):
        capabilities.check_sandbox_capabilities(sandbox_policy)


def test_adaptive_probe_uses_launcher_tmp_python_base_bind_order(
    sandbox_policy, tmp_path, monkeypatch
):
    base = tmp_path / "standalone-python"
    (base / "bin").mkdir(parents=True)
    python = base / "bin/python3.11"
    python.write_bytes(b"private interpreter path fixture; never executed")
    tool_python = sandbox_policy / "runtime/agent-venv/bin/python"
    tool_python.unlink()
    tool_python.symlink_to(python)

    def marker(command, **kwargs):
        options = os.read(int(command[2]), 32768).split(b"\0")[:-1]
        triple = [b"--ro-bind", os.fsencode(base), os.fsencode(base)]
        indices = [
            index
            for index in range(len(options) - 2)
            if options[index : index + 3] == triple
        ]
        assert len(indices) == 1
        assert indices[0] > options.index(b"--tmpfs")
        assert options[-3:] == triple
        assert b"--proc" not in options
        return subprocess.CompletedProcess(
            command, 0, tool_stdout(sandbox_policy, base=base), b""
        )

    monkeypatch.setattr(subprocess, "run", marker)
    assert (
        capabilities.check_sandbox_capabilities(sandbox_policy)[
            "adaptive_sandbox_verified"
        ]
        is True
    )


def test_tool_venv_link_matches_original_builder_runtime_and_base_mounts(
    sandbox_policy, tmp_path
):
    base = sandbox_policy / "runtime/tools-base"
    expected = str(base / "bin/python3.11")
    assert Path(expected).is_absolute()
    assert expected == str(base / "bin/python3.11")
    python = sandbox_policy / "runtime/agent-venv/bin/python"
    python.unlink()
    python.symlink_to("python3.11")
    python.with_name("python3.11").symlink_to(expected)
    bwrap = sandbox_policy / "runtime/codex/codex-resources/bwrap"
    args, _ = capabilities._adaptive_sandbox_arguments(
        sandbox_policy, bwrap, tmp_path / "probe", python
    )
    boundary = args.index("--")
    triples = [args[index : index + 3] for index in range(boundary - 2)]
    assert ["--ro-bind", str(python.parent.parent), "/runtime"] in triples
    assert ["--ro-bind", str(base), str(base)] in triples
    # Resolve the selected link after the venv has been remapped to /runtime.
    sandbox_target = Path(os.path.normpath(str(Path("/runtime/bin") / expected)))
    assert sandbox_target == base / "bin/python3.11"
    assert sandbox_target.is_relative_to(base)
    # The old host-valid relative spelling instead resolved to an unbound root.
    old = Path(os.path.normpath("/runtime/bin/../../tools-base/bin/python3.11"))
    assert old == Path("/tools-base/bin/python3.11")
    assert not old.is_relative_to(base)


def test_full_probe_requires_selected_tools_before_importing_launcher(
    monkeypatch, tmp_path
):
    def missing(_policy):
        raise tool_runtime.ToolRuntimeError("Separate original tools are missing")

    monkeypatch.setattr(tool_runtime, "resolve_tool_python", missing)
    monkeypatch.setattr(
        capabilities,
        "_verified_sandbox_transport",
        lambda *_: pytest.fail("Missing tools must fail before the launcher"),
    )
    with pytest.raises(tool_runtime.ToolRuntimeError, match="Separate original tools"):
        capabilities.check_sandbox_capabilities(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("python", "3.12.3"),
        ("prefix", "/host"),
        ("scipy_verified", False),
        ("strategy_tools_verified", False),
        ("distributions", []),
    ],
)
def test_sandbox_marker_cannot_hide_wrong_tool_identity(
    sandbox_policy, monkeypatch, field, value
):
    payload = json.loads(tool_stdout(sandbox_policy))
    payload[field] = value

    def incorrect(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, json.dumps(payload).encode(), b""
        )

    monkeypatch.setattr(subprocess, "run", incorrect)
    with pytest.raises(capabilities.RuntimeCapabilityError, match="complete 82"):
        capabilities.check_sandbox_capabilities(sandbox_policy)


def test_tool_import_failure_reports_only_a_fixed_public_stage(
    sandbox_policy, monkeypatch
):
    def failed(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            b'{"error_stage":"scientific CPU imports"}',
            b"PRIVATE_CHILD_ERROR",
        )

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(
        capabilities.RuntimeCapabilityError, match="scientific CPU imports"
    ) as caught:
        capabilities.check_sandbox_capabilities(sandbox_policy)
    assert "PRIVATE_CHILD_ERROR" not in str(caught.value)


def test_cli_missing_tools_is_nonzero_without_unfiltered_traceback(
    monkeypatch, capsys, tmp_path
):
    def missing(_policy):
        raise tool_runtime.ToolRuntimeError("Original tools are missing")

    monkeypatch.setattr(tool_runtime, "resolve_tool_python", missing)
    assert capabilities.main(["--policy-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "Original tools are missing" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
