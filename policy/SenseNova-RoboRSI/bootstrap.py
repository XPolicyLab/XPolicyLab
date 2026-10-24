"""Build the policy prerequisites, then run the pinned installer."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

try:
    from .tool_install import check_input
except ImportError:
    from tool_install import check_input

POLICY = Path(__file__).resolve().parent
# Python.org's 3.10.12 release/Sigstore digest; uv's published PyPI wheel digest.
PYTHON = (
    "https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tar.xz",
    "afb74bf19130e7a47d10312c8f5e784f24e0527981eab68e20546cfb865830b8",
    19654836,
)
UV = (
    "https://files.pythonhosted.org/packages/ae/a4/2db5939a3a993a06bca0a42e2120b4385bf1a4ff54242780701759252052/uv-0.8.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "011d2b2d4781555f7f7d29d2f0d6b2638fc60eeff479406ed570052664589e6a",
    19259174,
)
HOST_PROBE = """
import sys,os,fcntl,ssl,sqlite3,bz2,lzma,ctypes,_decimal,tarfile,venv,zlib
assert sys.version_info[:3]==(3,10,12)
assert callable(tarfile.data_filter)
f=os.memfd_create('roborsi-bootstrap',os.MFD_CLOEXEC)
try:
 assert fcntl.fcntl(f,fcntl.F_GETFD)&fcntl.FD_CLOEXEC
 assert os.write(f,b'RoboRSI')==7
 os.lseek(f,0,os.SEEK_SET)
 assert os.read(f,7)==b'RoboRSI'
finally:os.close(f)
"""


def fetch(cache: Path, name: str, pin: tuple[str, str, int]) -> Path:
    path = cache / name
    if path.is_symlink():
        raise ValueError(f"Linked bootstrap cache is not accepted: {name}")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(pin[0], timeout=60) as source:
                    remaining = pin[2]
                    while remaining:
                        block = source.read(min(1048576, remaining))
                        if not block:
                            raise ValueError(f"Truncated download: {name}")
                        stream.write(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise ValueError(f"Download exceeds fixed size: {name}")
                stream.flush()
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != pin[1]:
                    raise ValueError(f"Download SHA-256 differs: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    if (
        path.stat().st_size != pin[2]
        or hashlib.sha256(path.read_bytes()).hexdigest() != pin[1]
    ):
        raise ValueError(f"Cached SHA-256 differs: {name}")
    return path


def host_works(python: str | Path) -> bool:
    try:
        return (
            subprocess.run(
                [str(python), "-I", "-B", "-c", HOST_PROBE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except OSError:
        return False


def run_build(command: list[str], source: Path) -> None:
    try:
        subprocess.run(
            command,
            cwd=source,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        print(error.stderr[-4000:], file=sys.stderr)
        raise


def build_python(root: Path, jobs: int) -> Path:
    destination = root / "python"
    python = destination / "bin/python3.10"
    if host_works(python):
        return python
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Existing bootstrap Python is incompatible: {destination}")
    if not all(shutil.which(name) for name in ("cc", "make")):
        raise ValueError(
            "Building Python requires cc/make and development libraries; see INSTALL.md"
        )
    headers = "\n".join(
        f"#include <{name}>"
        for name in (
            "ffi.h",
            "openssl/ssl.h",
            "sqlite3.h",
            "zlib.h",
            "bzlib.h",
            "lzma.h",
        )
    )
    checked = subprocess.run(
        [
            "cc",
            *shlex.split(os.environ.get("CPPFLAGS", "")),
            "-x",
            "c",
            "-fsyntax-only",
            "-",
        ],
        input=headers,
        text=True,
        capture_output=True,
    )
    if checked.returncode:
        raise ValueError(
            "Python build development headers are missing; see INSTALL.md prerequisites"
        )
    archive = fetch(root / "downloads", "Python-3.10.12.tar.xz", PYTHON)
    print("Building Python 3.10.12 from the fixed official source...", flush=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=root) as work:
        work = Path(work)
        with tarfile.open(archive, "r:xz") as source:
            source.extractall(work, filter="data")
        source_root = work / "Python-3.10.12"
        run_build(
            [
                str(source_root / "configure"),
                "--quiet",
                f"--prefix={destination}",
                "--with-ensurepip=no",
            ],
            source_root,
        )
        run_build(["make", "-s", f"-j{jobs}"], source_root)
        staging = work / "install"
        run_build(["make", "-s", f"DESTDIR={staging}", "install"], source_root)
        (staging / destination.relative_to("/")).rename(destination)
    if not host_works(python):
        raise ValueError(
            "Built Python lacks required memfd/standard-library support; see INSTALL.md build dependencies"
        )
    return python


def install_uv(root: Path) -> Path:
    executable = root / "uv/bin/uv"
    archive = fetch(root / "downloads", "uv-0.8.9.whl", UV)
    with zipfile.ZipFile(archive) as package:
        members = [
            name
            for name in package.namelist()
            if name in {"uv/uv", "uv-0.8.9.data/scripts/uv"}
        ]
        if len(members) != 1:
            raise ValueError("The pinned uv wheel has an unexpected executable")
        data = package.read(members[0])
        if executable.exists() or executable.is_symlink():
            if executable.is_symlink() or executable.read_bytes() != data:
                raise ValueError("Existing bootstrap uv differs from its pinned wheel")
        else:
            executable.parent.mkdir(parents=True, exist_ok=True)
            with executable.open("xb") as output:
                output.write(data)
    executable.chmod(0o755)
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="check inputs without downloads or builds"
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    os.environ.pop("CODEX_API_KEY", None)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise ValueError("This installer requires Linux x86_64")
    if args.jobs < 1 or not hasattr(tarfile, "data_filter"):
        raise ValueError(
            "Use a seed Python with tarfile extraction filters and --jobs >= 1 (the official simulator Python is suitable)"
        )
    # Verify the committed build recipe before compiling or downloading dependencies.
    check_input(POLICY)
    if args.check:
        print(
            "The pinned tool build recipe is ready. Installation builds/reuses Python 3.10.12 and uv 0.8.9; see INSTALL.md."
        )
        return 0
    root = (
        Path(
            os.environ.get("ROBORSI_BOOTSTRAP_ROOT", str(POLICY / "runtime/bootstrap"))
        )
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    selected = os.environ.get("ROBORSI_PYTHON")
    if selected and not host_works(selected):
        raise ValueError(
            "ROBORSI_PYTHON must pass exact Python 3.10.12/memfd/stdlib checks"
        )
    candidate = selected or shutil.which("python3.10")
    python = (
        candidate
        if candidate and host_works(candidate)
        else build_python(root, args.jobs)
    )
    uv = install_uv(root)
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        ROBORSI_PYTHON=str(python),
        PATH=str(uv.parent) + os.pathsep + environment.get("PATH", "/usr/bin:/bin"),
    )
    # The original bubblewrap checks need an accessible local temporary directory.
    with tempfile.TemporaryDirectory(prefix="roborsi-install-", dir="/tmp") as scratch:
        environment["TMPDIR"] = scratch
        return subprocess.call(["bash", str(POLICY / "install.sh")], env=environment)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Installation prerequisite failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
