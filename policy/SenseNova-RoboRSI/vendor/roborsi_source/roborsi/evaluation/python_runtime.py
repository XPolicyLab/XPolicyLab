# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Describe a base Python using only its standard library (also a script)."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import sysconfig
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash a file without loading large binaries into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def describe_python() -> dict[str, str | None]:
    """Fingerprint the interpreter, standard library and Python shared library.

    Installed third-party packages and bytecode caches are excluded here;
    the environment bundle owns third-party dependencies separately.
    """
    root = Path(sysconfig.get_path("stdlib")).resolve()
    digest = hashlib.sha256()

    def visit(directory: Path, ancestors: frozenset[Path]) -> None:
        resolved = directory.resolve()
        if resolved in ancestors:
            raise ValueError("cyclic directory link in Python standard library")
        ancestors = ancestors | {resolved}
        for path in sorted(directory.iterdir()):
            if path.name in {"site-packages", "dist-packages", "__pycache__"}:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_dir():
                visit(path, ancestors)
            elif path.is_file():
                entry = [path.relative_to(root).as_posix(), file_sha256(path)]
                digest.update(json.dumps(entry, separators=(",", ":")).encode())

    visit(root, frozenset())
    library = Path(sysconfig.get_config_var("LIBDIR") or "") / (
        sysconfig.get_config_var("LDLIBRARY") or "missing"
    )
    return {
        "version": sys.version,
        "implementation": sys.implementation.name,
        "platform": sysconfig.get_platform(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "libc": " ".join(platform.libc_ver()),
        "executable_sha256": file_sha256(Path(sys.executable).resolve()),
        "stdlib_sha256": digest.hexdigest(),
        "shared_library_sha256": file_sha256(library) if library.is_file() else None,
    }


if __name__ == "__main__":
    print(json.dumps(describe_python(), sort_keys=True))
