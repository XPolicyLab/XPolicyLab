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

"""Collect declared Benchmark build inputs independently of runtime sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def build_inputs(directory: Path) -> tuple[Path, ...]:
    """Expand build-inputs.json paths/globs inside one Benchmark directory.

    Declared directories are traversed recursively. Missing inputs, symlinks
    and reserved export paths fail before any destination is created.
    """
    manifest = directory / "build-inputs.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "inputs"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["inputs"], list)
    ):
        raise ValueError(
            "build-inputs.json requires schema_version: 1 and an inputs list"
        )
    files = set()

    def collect(path: Path) -> None:
        relative = path.relative_to(directory)
        if relative.parts and relative.parts[0] in {
            "benchmark-runtime",
            "source.json",
            "build.json",
        }:
            raise ValueError(f"reserved build input path: {relative}")
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"build input cannot be a symlink: {relative}")
        if path.is_dir():
            children = sorted(path.iterdir())
            if not children:
                raise ValueError(f"build input directory is empty: {relative}")
            for child in children:
                collect(child)
        elif path.is_file():
            # These names can be represented literally in generated dockerignore rules.
            if any(c in relative.as_posix() for c in "\n\r\\*?[]!#"):
                raise ValueError(f"unsupported build input filename: {relative}")
            files.add(path)
        else:
            raise ValueError(f"build input is not a regular file: {relative}")

    for pattern in [
        "Dockerfile",
        "Dockerfile.dockerignore",
        "build-inputs.json",
        *data["inputs"],
    ]:
        if (
            not isinstance(pattern, str)
            or not pattern.strip()
            or PurePosixPath(pattern).is_absolute()
            or ".." in PurePosixPath(pattern).parts
            or any(c in pattern for c in "\\\n\r")
        ):
            raise ValueError(
                "build input patterns must stay inside the Benchmark directory"
            )
        # Refuse linked directory traversal even when glob would otherwise omit the link.
        for part in PurePosixPath(pattern).parents:
            if (directory / part).is_symlink():
                raise ValueError("build input cannot traverse a symlink")
        matches = [directory] if pattern == "." else sorted(directory.glob(pattern))
        if not matches:
            raise ValueError(f"build input pattern matched no files: {pattern}")
        for path in matches:
            collect(path)
    return tuple(sorted(files))


def file_hashes(root: Path, paths: tuple[Path, ...]) -> dict[str, dict[str, str | int]]:
    """Record deterministic relative names and SHA-256 values."""
    return {
        p.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "mode": p.stat().st_mode & 0o777,
        }
        for p in paths
    }


def include_inputs(rules: str, names: tuple[str, ...]) -> str:
    """Keep declared build inputs visible to Docker's existing deny-all recipe."""
    included = {"benchmark-runtime", "benchmark-runtime/**", "source.json"}
    for name in names:
        path = PurePosixPath(name)
        included.add(name)
        included.update(str(p) for p in path.parents if str(p) != ".")
    return (
        rules.rstrip()
        + "\n# Declared RoboRSI build inputs\n"
        + "".join(f"!{name}\n" for name in sorted(included))
    )
