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

"""Shared Docker commands, pinned images, mounts and private environment files."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from roborsi.evaluation.deadline import CaseDeadline

CommandRunner = Callable[[list[str], float], str]


def validate_image_reference(reference: str) -> None:
    """Require an immutable repository digest suitable for Docker CLI arguments."""
    if not isinstance(reference, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", reference
    ):
        raise ValueError("Docker image must be pinned by repository@sha256:digest")


class DockerCommandError(RuntimeError):
    """A Docker CLI command failed without exposing shell execution."""


def _docker_command(
    args: list[str],
    timeout_s: float,
    *,
    abort_check: Callable[[], None] | None = None,
) -> str:
    try:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except OSError as exc:
        raise DockerCommandError(f"Docker command failed: {exc}") from exc
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            if abort_check is not None:
                abort_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout_s)
            try:
                stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException as exc:
        if process.poll() is None:
            process.kill()
        process.communicate()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise DockerCommandError(
                f"Docker command timed out after {timeout_s:.1f}s"
            ) from exc
        raise
    if process.returncode:
        detail = stderr.strip() or stdout.strip()
        raise DockerCommandError(f"Docker command failed: {detail}")
    return stdout.strip()


@contextmanager
def environment_file(values: Mapping[str, str]) -> Iterator[Path]:
    """Keep environment values out of command arguments and remove them after run."""
    for name, value in values.items():
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or not isinstance(value, str)
            or any(c in value for c in "\r\n\0")
        ):
            raise ValueError(
                "Docker environment needs valid names and single-line values"
            )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="roborsi-env-"
    ) as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")
        stream.flush()
        yield Path(stream.name)


def remove_container(command_runner: CommandRunner, container_id: str) -> None:
    """Treat removal failure as success only after confirming absence."""
    try:
        command_runner(["docker", "rm", "--force", container_id], 10.0)
    except DockerCommandError:
        remaining = command_runner(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"id={container_id}"
                if re.fullmatch(r"[0-9a-f]{12,64}", container_id)
                else f"name=^{container_id}$",
                "--format",
                "{{.ID}}",
            ],
            5.0,
        )
        if remaining.strip():
            raise


def remove_network(command_runner: CommandRunner, network: str) -> None:
    """Remove an owned network, checking absence if Docker reports a failure."""
    try:
        command_runner(["docker", "network", "rm", network], 10.0)
    except DockerCommandError:
        remaining = command_runner(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                f"name=^{network}$",
                "--format",
                "{{.ID}}",
            ],
            5.0,
        )
        if remaining.strip():
            raise


class DockerRuntime:
    """Shared launch mechanics for separately owned Agent and Benchmark containers."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner = _docker_command,
        container_user: str | None = None,
    ) -> None:
        self._command_runner = command_runner
        self._container_user = (
            f"{os.getuid()}:{os.getgid()}" if container_user is None else container_user
        )
        if not re.fullmatch(r"[0-9]+:[0-9]+", self._container_user):
            raise ValueError("container_user must be numeric UID:GID")

    def _run_command(
        self, args: list[str], timeout_s: float, deadline: CaseDeadline | None
    ) -> str:
        if deadline is not None:
            timeout_s = deadline.remaining(timeout_s)
        output = self._call_command(args, timeout_s, deadline)
        if deadline is not None:
            deadline.check()
        return output

    def _call_command(
        self, args: list[str], timeout_s: float, deadline: CaseDeadline | None
    ) -> str:
        if self._command_runner is _docker_command and deadline is not None:
            return _docker_command(args, timeout_s, abort_check=deadline.check)
        return self._command_runner(args, timeout_s)

    def ensure_reference(
        self, reference: str, *, deadline: CaseDeadline | None = None
    ) -> str:
        """Pull by digest if needed, then verify a matching local RepoDigest."""
        validate_image_reference(reference)
        self._run_command(
            ["docker", "version", "--format", "{{.Server.Version}}"], 5.0, deadline
        )
        try:
            image = self._inspect_image(reference, deadline=deadline)
        except DockerCommandError:
            self._run_command(["docker", "pull", reference], 1800.0, deadline)
            image = self._inspect_image(reference, deadline=deadline)
        expected = reference.rpartition("@")[2]
        if not any(
            isinstance(item, str) and item.endswith(f"@{expected}")
            for item in image["RepoDigests"]
        ):
            raise ValueError(f"Docker image digest mismatch: {reference}")
        return image["Id"]

    def _inspect_image(
        self, reference: str, *, deadline: CaseDeadline | None = None
    ) -> dict:
        raw = self._run_command(
            [
                "docker",
                "image",
                "inspect",
                reference,
                "--format",
                "{{json .}}",
            ],
            10.0,
            deadline,
        )
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Docker image inspect returned invalid JSON") from exc
        if (
            not isinstance(values, dict)
            or not isinstance(values.get("RepoDigests"), list)
            or not isinstance(values.get("Id"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", values["Id"])
        ):
            raise ValueError("Docker image inspect returned invalid identity")
        return values

    @staticmethod
    def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
        text = str(source)
        if any(character in text for character in (",", "\n", "\r")):
            raise ValueError(f"unsupported Docker bind source path: {source}")
        option = f"type=bind,src={text},dst={target}"
        return f"{option},readonly" if readonly else option
