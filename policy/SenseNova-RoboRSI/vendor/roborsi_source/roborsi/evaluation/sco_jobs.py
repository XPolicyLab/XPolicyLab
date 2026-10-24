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

"""Own SCO jobs without a Docker daemon or shared Benchmark output files."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.docker_common import validate_image_reference
from roborsi.evaluation.result import write_json_atomic

_TERMINAL = {"SUCCEEDED", "FAILED", "SUSPENDED", "DELETED", "CANCELLED", "STOPPED"}
_WORKER_TERMINAL = {"Succeeded", "Failed", "Deleted", "Completed"}
_JOB_ID = re.compile(r"[a-z][a-z0-9-]+\Z")


class ScoCommandError(RuntimeError):
    """A SCO operation failed; command arguments and credentials stay private."""


def _sco_command(args: list[str], timeout_s: float) -> str:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScoCommandError(f"SCO {args[3]} failed ({type(exc).__name__})") from None
    if completed.returncode:
        raise ScoCommandError(f"SCO {args[3]} exited with code {completed.returncode}")
    return completed.stdout.strip()


@dataclass(frozen=True)
class ScoConfig:
    """Explicit workspace, pool, resource and network settings for one worker."""

    workspace: str
    cluster: str
    worker_spec: str
    vpc_id: str
    zone: str
    mounts: tuple[str, ...] = ()
    executable: str = "sco"
    profile: str | None = None
    region: str | None = None
    priority: str = "NORMAL"
    quota_type: str = "reserved"
    poll_interval_s: float = 2.0
    startup_timeout_s: float = 900.0
    cleanup_timeout_s: float = 120.0
    command_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for field in (
            "workspace",
            "cluster",
            "worker_spec",
            "vpc_id",
            "zone",
            "executable",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value
                or any(c in value for c in "\r\n\0")
            ):
                raise ValueError(f"invalid SCO {field}")
        if self.priority not in {"NORMAL", "HIGH", "HIGHEST"}:
            raise ValueError("invalid SCO priority")
        if self.quota_type not in {"reserved", "spot"}:
            raise ValueError("invalid SCO quota type")
        for value in (
            self.poll_interval_s,
            self.startup_timeout_s,
            self.cleanup_timeout_s,
            self.command_timeout_s,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "SCO timeouts and poll interval must be finite and positive"
                )
        destinations = set()
        for mount in self.mounts:
            if not isinstance(mount, str):
                raise ValueError("SCO mounts must be strings")
            volume, sep, destination = mount.partition(":")
            if (
                not sep
                or not volume
                or not destination.startswith("/")
                or any(c in mount for c in ",\r\n\0")
                or ".." in volume.split("/")
                or ".." in destination.split("/")
                or destination in destinations
            ):
                raise ValueError(
                    "SCO mounts must be unique volume[/subdir]:/path values"
                )
            destinations.add(destination)

    @classmethod
    def from_file(cls, path: str | Path) -> ScoConfig:
        """Read a nonsecret JSON configuration; reject unknown fields."""
        document = json.loads(Path(path).read_text())
        if not isinstance(document, dict):
            raise ValueError("SCO configuration must be an object")
        if "mounts" in document:
            if not isinstance(document["mounts"], list):
                raise ValueError("SCO mounts must be an array")
            document["mounts"] = tuple(document["mounts"])
        try:
            return cls(**document)
        except TypeError as exc:
            raise ValueError("invalid SCO configuration fields") from exc


def parse_workers(output: str) -> list[dict[str, str]]:
    """Read SCO 2.0.2's worker table, rejecting unrecognized output."""
    rows = [
        line.split("|")[1:-1] for line in output.splitlines() if line.startswith("|")
    ]
    if not rows or [cell.strip() for cell in rows[0]] != [
        "WORKER_NAME",
        "RESOURCE",
        "HOST_IP",
        "POD_IP",
        "PHASE",
    ]:
        raise ScoCommandError("unrecognized SCO worker table")
    result = []
    for row in rows[1:]:
        if len(row) != 5:
            raise ScoCommandError("invalid SCO worker row")
        name, _, host, pod, phase = [cell.strip() for cell in row]
        if not name:
            continue  # Wrapped resource descriptions do not identify workers.
        if not _JOB_ID.fullmatch(name) or not phase:
            raise ScoCommandError("invalid SCO worker identity or phase")
        result.append({"name": name, "host_ip": host, "pod_ip": pod, "phase": phase})
    return result


class ScoJob:
    """Submit at most once, reconcile uncertain replies, and verify worker release.

    The receipt contains ownership and scheduling evidence, never environment
    values or a startup command. Keep it if submission or cleanup is uncertain.
    """

    def __init__(
        self,
        config: ScoConfig,
        receipt: Path,
        *,
        command_runner: Callable[[list[str], float], str] = _sco_command,
    ) -> None:
        self.config, self.receipt = config, Path(receipt)
        self._run = command_runner
        self.display_name = f"roborsi-{uuid.uuid4().hex}"
        self.job_id: str | None = None
        self.uid: str | None = None
        self._submitted = False
        self._closed = False
        self._record = {
            "workspace": config.workspace,
            "cluster": config.cluster,
            "display_name": self.display_name,
            "state": "planned",
        }

    @classmethod
    def from_receipt(cls, config: ScoConfig, receipt: Path, **kwargs) -> ScoJob:
        """Recover ownership for cleanup after a Runner process interruption."""
        record = json.loads(Path(receipt).read_text())
        if (
            record.get("workspace") != config.workspace
            or record.get("cluster") != config.cluster
            or not re.fullmatch(r"roborsi-[0-9a-f]{32}", record.get("display_name", ""))
            or (
                record.get("job_id") is not None
                and not _JOB_ID.fullmatch(record["job_id"])
            )
        ):
            raise ValueError("SCO receipt does not match the configured owner")
        job = cls(config, receipt, **kwargs)
        job._record = record
        job.display_name = record["display_name"]
        job.job_id, job.uid = record.get("job_id"), record.get("uid")
        job._submitted = record.get("state") != "planned"
        # Recheck actual state even when a previous process recorded cleanup.
        return job

    def _save(self, **fields) -> None:
        self._record.update(fields)
        write_json_atomic(self.receipt, self._record)

    def _call(self, operation: str, *args: str, timeout_s: float | None = None) -> str:
        command = [
            self.config.executable,
            "acp",
            "jobs",
            operation,
            *args,
            "--workspace-name",
            self.config.workspace,
        ]
        for flag in ("profile", "region"):
            value = getattr(self.config, flag)
            if value:
                command.extend([f"--{flag}", value])
        return self._run(
            command, self.config.command_timeout_s if timeout_s is None else timeout_s
        )

    def _describe(self, timeout_s: float | None = None) -> dict:
        if self.job_id is None:
            raise ScoCommandError("SCO job has no confirmed resource ID")
        document = json.loads(
            self._call("describe", self.job_id, "-o", "json", timeout_s=timeout_s)
        )
        if (
            document.get("name") != self.job_id
            or document.get("display_name") != self.display_name
            or document.get("workspace_name") != self.config.workspace
            or not document.get("uid")
            or (self.uid is not None and document["uid"] != self.uid)
        ):
            raise ScoCommandError("SCO job ownership does not match receipt")
        self.uid = document["uid"]
        # Disk failures must not prevent ownership checks and remote cleanup.
        self._record.update(job_id=self.job_id, uid=self.uid, state=document["state"])
        return document

    def _reconcile(self, timeout_s: float | None = None) -> bool:
        budget = CaseDeadline(
            self.config.command_timeout_s if timeout_s is None else timeout_s
        )
        output = self._call(
            "list",
            "--display-name",
            self.display_name,
            "-o",
            "json",
            "--page-size",
            "100",
            timeout_s=budget.remaining(),
        )
        # SCO 2.0.2 emits this text even when JSON output was requested.
        if output.strip() == "No jobs found":
            return False
        try:
            jobs = json.loads(output)
        except json.JSONDecodeError:
            raise ScoCommandError("invalid SCO job list response") from None
        if not isinstance(jobs, list):
            raise ScoCommandError("invalid SCO job list")
        matches = [job for job in jobs if job.get("display_name") == self.display_name]
        if len(matches) > 1:
            raise ScoCommandError("multiple SCO jobs match the submission receipt")
        if not matches:
            return False
        name = matches[0].get("name", "")
        if not _JOB_ID.fullmatch(name):
            raise ScoCommandError("invalid SCO resource ID")
        self.job_id = name
        self._describe(budget.remaining())
        return True

    def submit(
        self,
        image: str,
        command: str,
        *,
        token: str,
        deadline: CaseDeadline | None = None,
        startup_deadline: CaseDeadline | None = None,
    ) -> None:
        """Submit a pinned image with one worker and an ephemeral HTTP token."""
        if self._submitted:
            raise ScoCommandError("SCO submission cannot be repeated")
        if self.receipt.exists():
            raise FileExistsError(
                "SCO receipt already exists; recover its job before retrying"
            )
        validate_image_reference(image)
        if not command or "\0" in command:
            raise ValueError("SCO startup command is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", token):
            raise ValueError(
                "SCO Benchmark token must be URL-safe and at least 32 characters"
            )
        if deadline:
            deadline.check()
        if startup_deadline:
            startup_deadline.check()
        self._save(state="submitting", image=image)
        self._submitted = True
        args = [
            "--aec2-name",
            self.config.cluster,
            "--job-name",
            self.display_name,
            "--container-image-url",
            image,
            "--training-framework",
            "pytorch",
            "--worker-spec",
            self.config.worker_spec,
            "--worker-nodes",
            "1",
            "--priority",
            self.config.priority,
            "--quota-type",
            self.config.quota_type,
            "--vpc-id",
            self.config.vpc_id,
            "--az",
            self.config.zone,
            "--command",
            command,
            "--env",
            f"ROBORSI_BENCHMARK_TOKEN:{token}",
        ]
        if self.config.mounts:
            args.extend(["--storage-mount", ",".join(self.config.mounts)])

        def timeout():
            limit = self.config.command_timeout_s
            return min(
                deadline.remaining(limit) if deadline else limit,
                startup_deadline.remaining(limit) if startup_deadline else limit,
            )

        try:
            output = self._call("create", *args, timeout_s=timeout())
            match = re.search(
                r"\bjob ([a-z][a-z0-9-]+) submitted successfully\b", output
            )
            if match:
                self.job_id = match[1]
                self._describe(timeout())
            elif not self._reconcile(timeout()):
                raise ScoCommandError(
                    "SCO submission response has no confirmed job; retain receipt"
                )
            self._save()
        except BaseException as submission_error:
            # A lost create reply can still mean a live billed job. Reconcile and
            # stop it independently of the cancelled case's deadline.
            try:
                self.close()
            except BaseException as cleanup_error:
                raise submission_error from cleanup_error
            raise

    def describe(self, timeout_s: float | None = None) -> dict:
        """Return ownership-checked control-plane data without persisting secrets."""
        return self._describe(timeout_s)

    def workers(self, timeout_s: float | None = None) -> list[dict[str, str]]:
        if self.job_id is None:
            raise ScoCommandError("SCO job has no confirmed resource ID")
        return parse_workers(
            self._call("get-workers", self.job_id, timeout_s=timeout_s)
        )

    def wait_endpoint(
        self,
        port: int,
        *,
        deadline: CaseDeadline | None = None,
        startup_deadline: CaseDeadline | None = None,
    ) -> str:
        """Wait for one running worker; HTTP readiness belongs to the caller."""
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid Benchmark service port")
        startup = startup_deadline or CaseDeadline(self.config.startup_timeout_s)
        limit = self.config.command_timeout_s
        while True:
            timeout = min(
                startup.remaining(limit),
                deadline.remaining(limit) if deadline else limit,
            )
            description = self.describe(timeout)
            if description["state"] in _TERMINAL:
                raise ScoCommandError(
                    f"SCO job ended before service startup: {description['state']}"
                )
            workers = self.workers(
                min(
                    startup.remaining(limit),
                    deadline.remaining(limit) if deadline else limit,
                )
            )
            if len(workers) > 1:
                raise ScoCommandError("expected exactly one SCO worker")
            # The job summary can still say QUEUEING after its worker starts.
            # The caller checks HTTP readiness after discovering this address.
            if (
                description["state"] in {"RUNNING", "QUEUEING"}
                and workers
                and workers[0]["phase"] == "Running"
            ):
                address = ipaddress.ip_address(workers[0]["pod_ip"])
                host = f"[{address}]" if address.version == 6 else str(address)
                self._save(endpoint=f"http://{host}:{port}", worker=workers[0])
                return f"http://{host}:{port}"
            time.sleep(
                min(
                    self.config.poll_interval_s,
                    startup.remaining(),
                    deadline.remaining() if deadline else 30,
                )
            )

    def close(self) -> None:
        """Stop only this job and confirm its workers no longer consume resources."""
        if self._closed or not self._submitted:
            return
        cleanup = CaseDeadline(self.config.cleanup_timeout_s)
        control_plane_terminal = None
        workers_released = None
        try:
            while self.job_id is None:
                if self._reconcile(cleanup.remaining(self.config.command_timeout_s)):
                    break
                time.sleep(min(self.config.poll_interval_s, cleanup.remaining()))
            document = self.describe(cleanup.remaining(self.config.command_timeout_s))
            if document["state"] not in _TERMINAL | {"SUSPENDING"}:
                try:
                    self._call(
                        "stop",
                        self.job_id,
                        timeout_s=cleanup.remaining(self.config.command_timeout_s),
                    )
                except ScoCommandError:
                    pass  # Confirm state even if the stop acknowledgement was lost.
            while True:
                document = self.describe(
                    cleanup.remaining(self.config.command_timeout_s)
                )
                workers = self.workers(cleanup.remaining(self.config.command_timeout_s))
                control_plane_terminal = document["state"] in _TERMINAL
                workers_released = (
                    control_plane_terminal
                    and all(w["phase"] in _WORKER_TERMINAL for w in workers)
                ) or (bool(workers) and all(w["phase"] == "Deleted" for w in workers))
                # SCO can retain SUSPENDING after every worker is Deleted.
                # Confirm resource release without claiming a terminal job state.
                if workers_released and (
                    control_plane_terminal or document["state"] == "SUSPENDING"
                ):
                    self._save(
                        cleanup_verified=True,
                        workers_released=True,
                        control_plane_terminal=control_plane_terminal,
                        workers=workers,
                    )
                    self._closed = True
                    return
                time.sleep(min(self.config.poll_interval_s, cleanup.remaining()))
        except BaseException:
            self._save(
                cleanup_verified=False,
                workers_released=workers_released,
                control_plane_terminal=control_plane_terminal,
            )
            raise
