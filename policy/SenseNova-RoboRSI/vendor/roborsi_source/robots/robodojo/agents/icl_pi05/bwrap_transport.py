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

"""Transport sandbox options and authenticated model-base path relocations."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from .identity import SECRET_NAMES

_RESERVED = tuple(
    Path(path)
    for path in (
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
)


def model_base_relocation(
    agent_python: Path, identity: dict[str, Any]
) -> tuple[Path, Path] | None:
    """Select the unique model-base mapping from an already verified receipt."""
    if not identity.get("path_relocations"):
        return None
    base = agent_python.resolve(strict=True).parent.parent
    matches = []
    for mapping in identity.get("path_relocations", []):
        if set(mapping) != {"original", "deployment"}:
            raise ValueError("Malformed verified Python path relocation")
        original, deployment = mapping["original"], mapping["deployment"]
        for value in (original, deployment):
            if (
                not isinstance(value, str)
                or "\x00" in value
                or not Path(value).is_absolute()
                or ".." in Path(value).parts
                or Path(value).as_posix() != value
            ):
                raise ValueError("Noncanonical verified Python path relocation")
        if Path(deployment) == base:
            matches.append((Path(original), base))
    if len(matches) > 1:
        raise ValueError("Ambiguous verified model-base relocation")
    if not matches:
        if identity.get("path_relocations"):
            raise ValueError(
                "Relocated environment lacks its verified model-base mapping"
            )
        return None
    return matches[0]


def _relocated_options(
    options: list[bytes], relocation: tuple[Path, Path] | None
) -> list[bytes]:
    if relocation is None:
        return options
    original, deployed = relocation
    for path in (original, deployed):
        if (
            not path.is_absolute()
            or ".." in path.parts
            or "\x00" in str(path)
            or path == Path("/tmp")
            or any(part.lower() in SECRET_NAMES for part in path.parts)
            or any(
                path.is_relative_to(root) or root.is_relative_to(path)
                for root in _RESERVED
            )
        ):
            raise ValueError("Unsafe verified model-base relocation")
    if original == deployed or not deployed.is_dir() or deployed.is_symlink():
        raise ValueError("Model-base relocation requires distinct materialized roots")
    triple = [b"--ro-bind", os.fsencode(deployed), os.fsencode(deployed)]
    indices = [i for i in range(len(options) - 2) if options[i : i + 3] == triple]
    if len(indices) != 1:
        raise ValueError(
            "Original sandbox must bind the verified model base exactly once"
        )
    index = indices[0]
    # The original launcher mounts /tmp after its base bind. Keep relocated
    # interpreter links usable and restore the unchanged pyvenv.cfg home.
    return (
        options[:index]
        + options[index + 3 :]
        + triple
        + [b"--ro-bind", os.fsencode(deployed), os.fsencode(original)]
    )


class BubblewrapArgumentTransport:
    """Hide original sandbox option values from the host process argument list.

    Only the exact configured bubblewrap is transformed. Optional verified
    model-base relocation changes bind order and adds its original read-only
    path alias; all other options and model argv stay original. No fallback or
    retry is added.
    """

    def __init__(
        self,
        spawn: Callable[..., Any],
        executable: Path,
        expected_sha256: str | None,
        *,
        base_relocation: tuple[Path, Path] | None = None,
    ) -> None:
        self._spawn = spawn
        self._executable = executable
        self._sha256 = expected_sha256
        self._base_relocation = base_relocation

    def __call__(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        if (
            not isinstance(args, (list, tuple))
            or not args
            or os.fspath(args[0]) != os.fspath(self._executable)
        ):
            return self._spawn(args, *positional, **kwargs)
        # Do not allow an executable override or shell to bypass exact identity.
        if kwargs.get("executable") is not None or kwargs.get("shell", False):
            raise ValueError("Unsupported verified sandbox process options")
        digest = hashlib.sha256()
        with self._executable.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if self._sha256 is None or digest.hexdigest() != self._sha256:
            raise ValueError("Sandbox executable differs from original reference")
        boundary = next(
            (i for i, arg in enumerate(args[1:], 1) if os.fsencode(arg) == b"--"),
            None,
        )
        if boundary is None or boundary == len(args) - 1:
            raise ValueError("Original sandbox command requires a model boundary")
        options = [os.fsencode(arg) for arg in args[1:boundary]]
        if any(b"\0" in option for option in options):
            raise ValueError("Invalid sandbox option encoding")
        options = _relocated_options(options, self._base_relocation)
        payload = b"\0".join(options) + b"\0"
        fd = os.memfd_create("icl-sandbox-options", os.MFD_CLOEXEC)
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("Incomplete anonymous sandbox option write")
                remaining = remaining[written:]
            os.lseek(fd, 0, os.SEEK_SET)
            options_kwargs = dict(kwargs)
            options_kwargs["pass_fds"] = tuple(kwargs.get("pass_fds", ())) + (fd,)
            command = [args[0], "--args", str(fd), *args[boundary:]]
            return self._spawn(command, *positional, **options_kwargs)
        finally:
            os.close(fd)
