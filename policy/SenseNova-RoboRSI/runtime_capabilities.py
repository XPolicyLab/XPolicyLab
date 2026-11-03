"""Check the main Python's required Linux transport capabilities offline."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_PROBE = b"RoboRSI public anonymous-file capability probe\n"
_SANDBOX_MARKER = b"ROBORSI_SANDBOX_CAPABILITY_OK\n"
_TOOL_TIMEOUT = 90
_TOOL_PROBE = r"""
import json
import sys
from pathlib import Path

stage = "python identity"
try:
    if sys.version_info[:3] != (3, 11, 13) or sys.prefix != "/runtime":
        raise ValueError()
    base = Path(sys.argv[1])
    if Path(sys.base_prefix) != base:
        raise ValueError()
    stage = "standard-library CPU imports"
    import bz2
    import ctypes
    import decimal
    import encodings
    import hashlib
    import importlib.metadata
    import lzma
    import sqlite3
    import ssl
    public = b"RoboRSI public original-tool CPU probe"
    assert bz2.decompress(bz2.compress(public)) == public
    assert lzma.decompress(lzma.compress(public)) == public
    assert decimal.Decimal("0.1") + decimal.Decimal("0.2") == decimal.Decimal("0.3")
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert len(hashlib.sha256(public).digest()) == 32
    ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with sqlite3.connect(":memory:") as db:
        assert db.execute("select 1").fetchone() == (1,)
    for module in (encodings, ssl, sqlite3, ctypes, decimal):
        assert Path(module.__file__).resolve().is_relative_to(base)
    stage = "scientific CPU imports"
    import numpy
    import scipy
    from scipy.spatial.transform import Rotation
    import PIL
    from PIL import Image
    assert numpy.array([1, 2, 3]).sum() == 6
    assert numpy.allclose(Rotation.identity().as_matrix(), numpy.eye(3))
    assert Image.new("RGB", (2, 2), (1, 2, 3)).getpixel((0, 0)) == (1, 2, 3)
    stage = "original strategy tool imports"
    import openpi_client.msgpack_numpy
    import openpi_client.websocket_client_policy
    sys.path.insert(0, "/tools")
    import robot
    assert callable(robot.observe) and callable(robot.vla_propose)
    assert Path(robot.__file__) == Path("/tools/robot.py")
    for module in (numpy, scipy, PIL, openpi_client.msgpack_numpy):
        assert Path(module.__file__).is_relative_to("/runtime/lib/python3.11/site-packages")
    stage = "distribution metadata"
    distributions = sorted(
        [[dist.metadata["Name"], dist.version] for dist in importlib.metadata.distributions()]
    )
    print(json.dumps({"marker": "ROBORSI_ORIGINAL_TOOL_CPU_OK", "python": "3.11.13",
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "distributions": distributions, "stdlib_verified": True,
        "numpy_verified": True, "pillow_verified": True, "scipy_verified": True,
        "strategy_tools_verified": True}))
except BaseException:
    print(json.dumps({"error_stage": stage}))
    raise SystemExit(1) from None
"""
_REQUIREMENT = (
    "Use Linux Python 3.10.12 built with os.memfd_create and MFD_CLOEXEC support, "
    "with memfd creation permitted by the kernel and container. "
    "The original sandbox transport has no fallback."
)


class RuntimeCapabilityError(ValueError):
    """The selected Python cannot run the unchanged sandbox transport."""


def check_python_capabilities() -> dict[str, object]:
    """Exercise the real memfd syscall, byte transport and close-on-exec flag.

    Only the standard library is required. The probe uses public constant bytes,
    never reads environment variables, creates no persistent file, and does not
    launch a child process, sandbox, model, simulator or network operation.
    """
    if sys.platform != "linux" or sys.version_info[:3] != (3, 10, 12):
        raise RuntimeCapabilityError(_REQUIREMENT)
    create = getattr(os, "memfd_create", None)
    flags = getattr(os, "MFD_CLOEXEC", None)
    if not callable(create) or not isinstance(flags, int):
        raise RuntimeCapabilityError(
            "The selected Python lacks os.memfd_create or MFD_CLOEXEC. " + _REQUIREMENT
        )
    try:
        import fcntl
    except ImportError:
        raise RuntimeCapabilityError(
            "The selected Python lacks the standard fcntl module. " + _REQUIREMENT
        ) from None

    descriptor = None
    stage = "memfd_create"
    try:
        descriptor = create("roborsi-runtime-probe", flags)
        stage = "FD_CLOEXEC verification"
        if not fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
            raise RuntimeCapabilityError(
                "The memfd probe did not preserve FD_CLOEXEC. " + _REQUIREMENT
            )
        stage = "memfd write"
        remaining = memoryview(_PROBE)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise RuntimeCapabilityError(
                    "The memfd probe could not write its public bytes. " + _REQUIREMENT
                )
            remaining = remaining[written:]
        stage = "memfd rewind"
        if os.lseek(descriptor, 0, os.SEEK_SET) != 0:
            raise RuntimeCapabilityError(
                "The memfd probe could not rewind its public bytes. " + _REQUIREMENT
            )
        stage = "memfd read"
        if os.read(descriptor, len(_PROBE) + 1) != _PROBE:
            raise RuntimeCapabilityError(
                "The memfd probe read different public bytes. " + _REQUIREMENT
            )
    except OSError:
        # Syscall messages may contain caller-controlled values. Report only
        # the fixed operation name and required capability, never raw errors.
        raise RuntimeCapabilityError(
            f"The Python capability probe failed during {stage}. " + _REQUIREMENT
        ) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise RuntimeCapabilityError(
                    "The memfd probe could not close its file descriptor. "
                    + _REQUIREMENT
                ) from None
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "platform": "linux",
        "check": "real anonymous memfd write, rewind, read and close",
        "memfd_create_verified": True,
        "fd_cloexec_verified": True,
        "write_read_verified": True,
        "descriptor_closed": True,
        "probe_bytes": len(_PROBE),
        "child_process_started": False,
    }


def _verified_sandbox_transport(policy_dir: Path):
    """Import the original installed transport only after its source checks."""
    try:
        from .preflight import (
            load_submission,
            sha256_file,
            validate_companions,
            validate_source_manifest,
        )
    except ImportError:
        from preflight import (
            load_submission,
            sha256_file,
            validate_companions,
            validate_source_manifest,
        )

    metadata = load_submission(policy_dir)
    validate_source_manifest(metadata, policy_dir)
    binary = policy_dir / "runtime/codex/codex-resources/bwrap"
    expected = metadata["cli"]["companion_sha256"]["codex-resources/bwrap"]
    if binary.is_symlink() or not binary.is_file() or sha256_file(binary) != expected:
        raise RuntimeCapabilityError(
            "The bundled bubblewrap differs from its pinned companion SHA-256."
        )
    cli_root = policy_dir / "runtime/codex"
    validate_companions(metadata, cli_root)
    cli = cli_root / "bin/codex"
    if not cli.is_file() or sha256_file(cli) != metadata["cli"]["sha256"]:
        raise RuntimeCapabilityError("The native CLI differs from its pinned SHA-256.")
    source = (policy_dir / "runtime/RoboRSI").resolve(strict=True)
    name = "robots.robodojo.agents.icl_pi05.bwrap_transport"
    original_path = source / "robots/robodojo/agents/icl_pi05/bwrap_transport.py"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != original_path:
        raise RuntimeCapabilityError(
            "A different sandbox transport source is already imported."
        )
    return module.BubblewrapArgumentTransport, binary, expected


def _adaptive_sandbox_arguments(
    policy_dir: Path, binary: Path, root: Path, agent_python: Path | None = None
):
    """Use the exact verified adaptive builder and existing launcher path fixes."""
    try:
        from .hybrid_backend import load_reference_source
        from .hybrid_launcher import HybridCodexCliPlanner, _sandbox_builder
    except ImportError:
        from hybrid_backend import load_reference_source
        from hybrid_launcher import HybridCodexCliPlanner, _sandbox_builder

    metadata = json.loads((policy_dir / "submission.json").read_text())
    profile = metadata["hybrid"]
    reference = load_reference_source(
        policy_dir / profile["source_path"],
        policy_dir / profile["manifest_path"],
        profile["manifest_sha256"],
    )
    if agent_python is None:
        try:
            from .tool_runtime import resolve_tool_python
        except ImportError:
            from tool_runtime import resolve_tool_python
        agent_python = resolve_tool_python(policy_dir)
    for name in (
        "run/bridge",
        "run/tools",
        "run/workspace",
        "run/observations",
        "training",
        "codex-home",
    ):
        (root / name).mkdir(parents=True)
    (root / "run/tools/robot.py").write_bytes(
        (reference.source_dir / "code/policylab/robot.py").read_bytes()
    )
    # Only the pure argument builder is used; no planner, API or CLI is started.
    planner = object.__new__(HybridCodexCliPlanner)
    planner.reference = reference
    planner.bubblewrap = binary
    planner._sandbox_code = _sandbox_builder(reference)
    planner.config = SimpleNamespace(env_vars=(), env={})
    arguments = planner._sandbox(
        {
            "agent_python": str(agent_python),
            "codex_binary": str(policy_dir / "runtime/codex/bin/codex"),
            "training_set": str(root / "training"),
        },
        root / "run",
        root / "codex-home",
        ["/bin/sh", "-c", "printf 'ROBORSI_SANDBOX_CAPABILITY_OK\\n'"],
        {"CODEX_API_KEY": ""},
    )
    return arguments, profile["manifest_sha256"]


def check_sandbox_capabilities(policy_dir: Path) -> dict[str, object]:
    """Run the independent tool Python through the original memfd transport.

    Run after source/dependency/CLI installation and before large model downloads.
    This verifies required namespace and mount operations; it does not execute
    Codex, a model, a GPU operation or a network request. Child environments are
    explicit public constants; caller keys, proxies and HOME are never forwarded.
    """
    python_receipt = check_python_capabilities()
    try:
        from .tool_runtime import _packages, load_tool_metadata, resolve_tool_python
    except ImportError:
        from tool_runtime import _packages, load_tool_metadata, resolve_tool_python

    # Resolve before importing the launcher: neither host nor VLA is a fallback.
    tool_python = resolve_tool_python(policy_dir)
    tool_metadata = load_tool_metadata(policy_dir)
    tool_base = tool_python.resolve(strict=True).parent.parent
    try:
        policy_dir = Path(policy_dir).resolve(strict=True)
        transport_class, binary, expected = _verified_sandbox_transport(policy_dir)
    except RuntimeCapabilityError:
        raise
    except (OSError, ValueError, ImportError, AttributeError, TypeError, KeyError):
        raise RuntimeCapabilityError(
            "The pinned original sandbox transport or its installed source identity is unavailable."
        ) from None
    transport = transport_class(subprocess.run, binary, expected)
    try:
        with tempfile.TemporaryDirectory(prefix="roborsi-sandbox-check-") as temporary:
            arguments, adaptive_manifest = _adaptive_sandbox_arguments(
                policy_dir, binary, Path(temporary).resolve(strict=True), tool_python
            )
            boundary = arguments.index("--")
            arguments[boundary + 1 :] = [
                "/runtime/bin/python",
                "-I",
                "-B",
                "-c",
                _TOOL_PROBE,
                str(tool_base),
            ]
            result = transport(
                arguments,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_TOOL_TIMEOUT,
                check=False,
                cwd=temporary,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": temporary},
            )
    except subprocess.TimeoutExpired:
        raise RuntimeCapabilityError(
            "The original tool sandbox CPU probe timed out after 90 seconds."
        ) from None
    except (OSError, ValueError, ImportError, AttributeError, TypeError, KeyError):
        raise RuntimeCapabilityError(
            "The original memfd sandbox argument transport could not launch the bundled bubblewrap."
        ) from None
    if result.returncode != 0:
        try:
            stage = json.loads(result.stdout).get("error_stage")
        except (ValueError, AttributeError):
            stage = None
        if stage in {
            "python identity",
            "standard-library CPU imports",
            "scientific CPU imports",
            "original strategy tool imports",
            "distribution metadata",
        }:
            raise RuntimeCapabilityError(
                f"The original tool sandbox CPU probe failed during {stage}. "
                "Its Python 3.11.13 payload and original dependency identities must be preserved."
            )
        raise RuntimeCapabilityError(
            "The bundled sandbox capability probe failed: required user/PID/IPC/UTS namespaces or read-only mounts are unavailable "
            f"(return code {result.returncode}). No unsandboxed fallback is permitted."
        )
    try:
        tool_receipt = json.loads(result.stdout)
        if (
            not isinstance(tool_receipt, dict)
            or tool_receipt.get("marker") != "ROBORSI_ORIGINAL_TOOL_CPU_OK"
            or tool_receipt.get("python") != "3.11.13"
            or tool_receipt.get("prefix") != "/runtime"
            or tool_receipt.get("base_prefix") != str(tool_base)
            or _packages(tool_receipt["distributions"])
            != _packages(tool_metadata["distributions"])
            or any(
                tool_receipt.get(name) is not True
                for name in (
                    "stdlib_verified",
                    "numpy_verified",
                    "pillow_verified",
                    "scipy_verified",
                    "strategy_tools_verified",
                )
            )
        ):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise RuntimeCapabilityError(
            "The tool sandbox did not verify Python 3.11.13, its complete "
            f"{len(tool_metadata['distributions'])} declared distributions and CPU imports."
        ) from None
    return {
        **python_receipt,
        "check": "pinned original memfd transport and independent original tool Python CPU probe",
        "bwrap_sha256": expected,
        "original_transport_verified": True,
        "adaptive_manifest_sha256": adaptive_manifest,
        "adaptive_sandbox_verified": True,
        "memfd_arguments_verified": True,
        "rootless_namespaces_verified": True,
        "proc_layout_verified": True,
        "dev_ro_bind_verified": True,
        "marker_verified": True,
        "tools": {
            key: value for key, value in tool_receipt.items() if key != "distributions"
        },
        "tool_distribution_count": len(tool_receipt["distributions"]),
        "child_process_started": True,
        "model_or_api_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--python-only",
        action="store_true",
        help="check the selected Python before installing dependencies or assets",
    )
    mode.add_argument(
        "--policy-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="check the installed original sandbox in this policy",
    )
    args = parser.parse_args(argv)
    try:
        receipt = (
            check_python_capabilities()
            if args.python_only
            else check_sandbox_capabilities(args.policy_dir)
        )
    except (ValueError, OSError, ImportError) as error:
        print(f"Runtime capability check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
