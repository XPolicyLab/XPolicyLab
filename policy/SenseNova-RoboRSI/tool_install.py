"""Build the separate Python 3.11.13 tool environment from fixed public sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from .tool_runtime import (
        _inventory,
        _relative,
        _sha,
        load_tool_metadata,
        validate_tool_install,
    )
except ImportError:
    from tool_runtime import (
        _inventory,
        _relative,
        _sha,
        load_tool_metadata,
        validate_tool_install,
    )

POLICY_DIR = Path(__file__).resolve().parent


def _default_root(policy: Path, *, selected: str | None = None) -> Path:
    """Keep the Python base outside mounts hidden by the original sandbox."""
    configured = (
        selected if selected is not None else os.environ.get("ROBORSI_TOOL_ROOT")
    )
    if configured:
        root = Path(configured).expanduser().absolute()
    else:
        identity = hashlib.sha256(os.fsencode(str(policy.resolve()))).hexdigest()[:16]
        root = Path("/tmp") / f"sensenova-roborsi-tools-{os.getuid()}-{identity}"
    protected = (
        "/runtime",
        "/workspace",
        "/training",
        "/tools",
        "/bridge",
        "/observations",
        "/home",
        "/proc",
        "/dev",
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc",
    )
    if (
        not root.is_absolute()
        or ".." in root.parts
        or len(root.parts) < 3
        or root in {Path("/tmp"), Path("/var/tmp")}
        or any(
            root.is_relative_to(Path(item)) or Path(item).is_relative_to(root)
            for item in protected
        )
    ):
        raise ValueError(
            "ROBORSI_TOOL_ROOT must be a private directory outside reserved sandbox mounts"
        )
    resolved = root.resolve()
    if any(resolved.is_relative_to(Path(item)) for item in protected):
        raise ValueError("Tool root resolves below a reserved sandbox mount")
    return resolved


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("Tool runtime directories must not be symlinks")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError(
            "Tool runtime directory must be private to its installing user"
        )


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    environment = dict(os.environ)
    environment.pop("CODEX_API_KEY", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _python_archive(spec: dict, root: Path) -> Path:
    archive = root / "python-3.11.13.tar.gz"
    if not archive.exists():
        descriptor, temporary_name = tempfile.mkstemp(prefix=".python-", dir=root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                request = Request(
                    spec["url"], headers={"User-Agent": "SenseNova-RoboRSI-installer"}
                )
                with urlopen(request, timeout=120) as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if _sha(temporary) != spec["sha256"]:
                raise ValueError(
                    "Tool Python archive SHA-256 differs from its fixed release"
                )
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if archive.is_symlink() or _sha(archive) != spec["sha256"]:
        raise ValueError("Cached tool Python archive SHA-256 differs")
    return archive


def _python_payload(archive: Path, destination: Path) -> None:
    """Extract only safe regular files/directories, then contained symlinks."""
    links = []
    names = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = _relative(member.name)
            if name.parts[0] != "python" or member.name in names:
                raise ValueError("Unexpected or duplicate public Python archive path")
            names.add(member.name)
            relative = Path(*name.parts[1:])
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                link = Path(member.linkname)
                if link.is_absolute() or not (
                    target.parent / link
                ).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Public Python archive link escapes its root")
                links.append((target, member.linkname))
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise ValueError("Unsupported public Python archive entry")
    for target, value in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(value)


def check_input(policy_dir: Path = POLICY_DIR) -> dict:
    """Validate committed sources before any download; no private archive input."""
    policy = Path(policy_dir).resolve(strict=True)
    metadata = load_tool_metadata(policy)
    pins = json.loads((policy / "tools/openpi-client-files.json").read_text())
    source = policy / "vendor/pi05/openpi/packages/openpi-client/src/openpi_client"
    for name, digest in pins.items():
        path = source / _relative(name)
        if path.is_symlink() or _sha(path) != digest:
            raise ValueError("The original vendored OpenPI client source changed")
    return {
        "status": "public_build_inputs_verified",
        "python_version": metadata["python_version"],
        "distribution_count": 82,
    }


def _wheel(package: Path, build_python: Path, output: Path) -> Path:
    _run(
        [
            str(build_python),
            "-c",
            "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
            str(output),
        ],
        cwd=package,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one wheel from the fixed source package")
    return wheels[0]


def install_tools(policy_dir: Path = POLICY_DIR) -> dict:
    """Install hash-locked public wheels plus the original OpenPI client source."""
    policy = Path(policy_dir).resolve(strict=True)
    check_input(policy)
    metadata = load_tool_metadata(policy)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise ValueError("The fixed model-tool Python requires Linux x86_64")
    receipt_path = policy / metadata["paths"]["receipt"]
    if receipt_path.exists() or receipt_path.is_symlink():
        return validate_tool_install(policy, full=True)
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required by the policy installer")
    root = _default_root(policy)
    _private_directory(root)
    base = root / "tools-base"
    environment = root / "agent-venv"
    if (
        base.exists()
        or base.is_symlink()
        or environment.exists()
        or environment.is_symlink()
    ):
        raise ValueError(
            "Inspect the existing unfinished tool build; select a fresh ROBORSI_TOOL_ROOT"
        )
    runtime = policy / "runtime"
    if runtime.is_symlink():
        raise ValueError("The policy runtime directory must be real")
    runtime.mkdir(exist_ok=True)
    alias = (policy / metadata["paths"]["runtime"]).parent
    if alias.exists() or alias.is_symlink():
        raise ValueError("Inspect the existing tool deployment before installation")
    archive = _python_archive(metadata["python"], root)
    _python_payload(archive, base)
    if _sha(base / "bin/python3.11") != metadata["python"]["binary_sha256"]:
        raise ValueError("Tool Python executable SHA-256 differs")
    _run(
        [str(base / "bin/python3.11"), "-m", "venv", "--without-pip", str(environment)]
    )
    build = root / "build-venv"
    _run([str(base / "bin/python3.11"), "-m", "venv", "--without-pip", str(build)])
    uv_command = [
        uv,
        "--cache-dir",
        os.environ.get("UV_CACHE_DIR", str(root / "uv-cache")),
    ]
    for target, lock in ((environment, "requirements.lock"), (build, "build.lock")):
        _run(
            [
                *uv_command,
                "pip",
                "sync",
                "--link-mode",
                "copy",
                "--python",
                str(target / "bin/python"),
                "--require-hashes",
                str(policy / "tools" / lock),
            ]
        )
    with tempfile.TemporaryDirectory(prefix=".tool-build-", dir=root) as directory:
        temporary = Path(directory)
        openpi = temporary / "openpi"
        openpi.mkdir()
        source = policy / "vendor/pi05/openpi/packages/openpi-client/src/openpi_client"
        pins = json.loads((policy / "tools/openpi-client-files.json").read_text())
        for name, digest in pins.items():
            path = source / _relative(name)
            if path.is_symlink() or _sha(path) != digest:
                raise ValueError("The original vendored OpenPI client source changed")
            target = openpi / "src/openpi_client" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copy2(
            policy / "tools/openpi-client.pyproject.toml", openpi / "pyproject.toml"
        )
        shutil.copy2(policy / "vendor/pi05/openpi/LICENSE", openpi / "LICENSE")
        output = temporary / "wheels"
        output.mkdir()
        wheel = _wheel(openpi, build / "bin/python", output)
        _run(
            [
                *uv_command,
                "pip",
                "install",
                "--link-mode",
                "copy",
                "--python",
                str(environment / "bin/python"),
                "--no-deps",
                str(wheel),
            ]
        )
    _run([*uv_command, "pip", "check", "--python", str(environment / "bin/python")])
    _run(
        [
            str(environment / "bin/python"),
            "-I",
            "-B",
            "-c",
            "import openpi_client,numpy,scipy,PIL",
        ]
    )
    for payload in (base, environment):
        for directory, names, files in os.walk(payload, followlinks=False):
            Path(directory).chmod(0o755)
            for name in files:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    alias.symlink_to(root, target_is_directory=True)
    inventory_path = policy / metadata["paths"]["inventory"]
    if inventory_path.exists() or inventory_path.is_symlink():
        raise ValueError(
            "Inspect the existing model-tool inventory before installation"
        )
    inventory_path.write_text(
        json.dumps(
            _inventory({"runtime": environment, "base": base}), separators=(",", ":")
        )
        + "\n"
    )
    receipt = {
        "schema_version": 2,
        "metadata_sha256": _sha(policy / "tools/runtime.json"),
        "inventory_sha256": _sha(inventory_path),
        "python_version": "3.11.13",
        "installation_mode": "public-source-build",
        "source_payloads_verified": True,
        "b_alignment": "82 original distribution versions; unused historical heart-runtime editable is omitted",
    }
    with receipt_path.open("x") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")
    return validate_tool_install(policy, full=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--check-input", action="store_true")
    args = parser.parse_args()
    try:
        operation = check_input if args.check_input else install_tools
        result = operation(args.policy_dir)
    except (
        ValueError,
        OSError,
        KeyError,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Model-tool installation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
