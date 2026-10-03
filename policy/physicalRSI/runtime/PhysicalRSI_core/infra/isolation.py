"""Linux chroot/seccomp execution of Python with a read-only stdlib runtime.

Requires a privileged trusted controller. No fallback to unrestricted exec.
Only explicitly supplied source and input enter the jail. A single precreated
result file is writable; native robot RPC and third-party packages are not yet
part of this backend.
"""

import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from .batch import ProcessJob, run_batch
from .storage import atomic_json, digest, file_digest, read_json


def build_runtime(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    executable = Path(sys.executable).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()

    def copy(source):
        target = root / str(source).lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    copy(executable)
    shutil.copytree(
        stdlib,
        root / str(stdlib).lstrip("/"),
        ignore=shutil.ignore_patterns(
            "site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter"
        ),
    )
    binaries = [executable, *stdlib.joinpath("lib-dynload").glob("*.so")]
    dependencies = set()
    for binary in binaries:
        result = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=True, timeout=15
        )
        if "not found" in result.stdout:
            raise RuntimeError("Runtime dynamic dependency missing")
        dependencies.update(
            re.findall(r"(?:=>\s+|^\s*)(/[^\s]+)", result.stdout, re.MULTILINE)
        )
    for name in sorted(dependencies):
        copy(Path(name))
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Isolation runtime must contain concrete files")
        if path.is_file():
            path.chmod(0o555)
            files[str(path.relative_to(root))] = file_digest(path)
    manifest = dict(
        schema="physicalrsi.isolation-runtime/v1",
        executable=str(executable),
        files=files,
    )
    atomic_json(root / "manifest.json", manifest)
    return manifest


def execute(
    source,
    inputs,
    *,
    runtime,
    output,
    timeout_s=5,
    memory_bytes=256 * 1024 * 1024,
    output_bytes=1024 * 1024,
    handlers=None,
    max_calls=100,
    cancelled=None,
):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(
            "Linux chroot privilege is required; execution was not started"
        )
    if (
        not math.isfinite(timeout_s)
        or timeout_s <= 0
        or memory_bytes <= 0
        or output_bytes <= 0
    ):
        raise ValueError("Positive finite isolation limits required")
    if handlers is not None and (
        type(max_calls) is not int
        or max_calls < 1
        or not isinstance(handlers, dict)
        or any(not isinstance(k, str) or not callable(v) for k, v in handlers.items())
    ):
        raise ValueError(
            "Explicit primitive handlers and positive call budget required"
        )
    runtime = Path(runtime).resolve()
    manifest = read_json(runtime / "manifest.json")
    if manifest.get("schema") != "physicalrsi.isolation-runtime/v1":
        raise ValueError("Unsupported isolation runtime schema")
    for name in manifest["files"]:
        if (
            not Path(name).parts
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).parts[0]
            in {"policy.py", "input.json", "response.json", "output"}
        ):
            raise ValueError("Isolation runtime path must stay relative")
    executable = manifest["executable"]
    if not executable.startswith("/") or executable[1:] not in manifest["files"]:
        raise ValueError("Isolation executable must be pinned in runtime")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        root / "request.json",
        dict(
            schema="physicalrsi.isolated-execution/v1",
            source_sha256=digest(source),
            input_sha256=digest(inputs),
            runtime_sha256=digest(manifest),
            capabilities=sorted(handlers or {}),
            max_calls=max_calls,
            limits=dict(
                timeout_s=timeout_s,
                memory_bytes=memory_bytes,
                output_bytes=output_bytes,
            ),
        ),
    )
    jail = root / "root"
    jail.mkdir()
    for name, sha in manifest["files"].items():
        path = runtime / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(runtime)
            or file_digest(path) != sha
        ):
            raise ValueError("Isolation runtime changed or escaped its root")
        target = jail / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
    (jail / "policy.py").write_text(source)
    atomic_json(jail / "input.json", inputs)
    atomic_json(jail / "response.json", {})
    result_dir = jail / "output"
    result_dir.mkdir()
    result_path = result_dir / "result.json"
    result_path.touch()
    for path in jail.rglob("*"):
        path.chmod(
            0o555
            if path.is_dir()
            else 0o444
            if path.name in {"policy.py", "input.json", "result.json", "response.json"}
            else 0o555
        )
    os.chown(result_path, 65534, 65534)
    result_path.chmod(0o600)
    jail.chmod(0o555)
    job = ProcessJob(
        "isolated",
        (
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("isolation_entry.py")),
            str(jail),
            manifest["executable"],
            str(math.ceil(timeout_s)),
            str(memory_bytes),
            str(output_bytes),
        ),
        root,
        timeout_s=timeout_s,
    )
    if handlers is not None:
        from .isolation_bridge import run

        result = run(
            job,
            root / "processes" / "isolated",
            result_path,
            jail / "response.json",
            handlers,
            max_calls=max_calls,
            message_bytes=output_bytes,
            cancelled=cancelled,
        )
        atomic_json(root / "result.json", result)
        return result
    process = run_batch([job], root / "processes", workers=1, cancelled=cancelled)[0]
    if process["state"] != "completed" or process["returncode"] != 0:
        raise RuntimeError("Isolated program failed; inspect retained process evidence")
    result = read_json(result_path)
    atomic_json(root / "result.json", result)
    return result
