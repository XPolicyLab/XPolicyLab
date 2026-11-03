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

"""Launch a digest-pinned Benchmark Adapter container from a registry lock."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from collections.abc import Callable
from pathlib import Path

from roborsi.benchmarks.registry import BenchmarkLock
from roborsi.evaluation.benchmark_client import (
    BenchmarkClient,
    BenchmarkTransport,
)
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.evaluation.docker_common import (
    CommandRunner,
    DockerCommandError,
    DockerRuntime,
    _docker_command,
    environment_file,
    remove_container,
)
from roborsi.evaluation.messages import (
    BenchmarkRuntimeInfo,
    BenchmarkRuntimeRecord,
)
from roborsi.evaluation.services import AdapterClient
from roborsi.utils.daemon import pick_free_port
from roborsi.utils.rpc import RpcClient, wait_for_ready
from roborsi.utils.rpc.http_abort import interrupt_http_calls

_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")
ClientFactory = Callable[[str], RpcClient]


class AdapterContainer:
    """Own one container and its typed Adapter client."""

    def __init__(
        self,
        container_id: str,
        client: RpcClient,
        command_runner: CommandRunner,
        *,
        runtime_record: BenchmarkRuntimeRecord | None = None,
        verify_runtime: Callable[[], BenchmarkRuntimeRecord | None] | None = None,
    ) -> None:
        self.container_id = container_id
        self.client = (
            BenchmarkClient(client.endpoint, token=client.token)
            if isinstance(client, BenchmarkTransport)
            else AdapterClient(client)
        )
        self._rpc = client
        self._command_runner = command_runner
        self._closed = False
        self.runtime_record = runtime_record
        self._verify_runtime = verify_runtime

    def close(self) -> None:
        """Check the runtime, then remove it even when a check fails."""
        if self._closed:
            return
        self._closed = True
        checked = None
        try:
            if self._verify_runtime is not None:
                checked = self._verify_runtime()
        finally:
            self._remove()
        if checked is not None:
            self.runtime_record = checked.model_copy(
                update={"verified_after_close": True}
            )

    def _remove(self) -> None:
        try:
            self._rpc.call("shutdown", timeout_s=1.0)
        except Exception:
            pass
        finally:
            try:
                self._rpc.close()
            finally:
                try:
                    remove_container(self._command_runner, self.container_id)
                except DockerCommandError:
                    self._closed = False
                    raise

    def __enter__(self) -> AdapterContainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerExecutor(DockerRuntime):
    """Verify pinned images and host a Benchmark Adapter over loopback HTTP."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner = _docker_command,
        client_factory: ClientFactory = BenchmarkTransport,
        container_user: str | None = None,
    ) -> None:
        super().__init__(command_runner=command_runner, container_user=container_user)
        self._client_factory = client_factory

    def ensure_image(
        self, lock: BenchmarkLock, *, deadline: CaseDeadline | None = None
    ) -> str:
        """Pull and verify the manifest's pinned image."""
        return self.ensure_reference(lock.image_reference, deadline=deadline)

    def start_adapter(
        self,
        lock: BenchmarkLock,
        case_dir: str | Path,
        *,
        host_port: int | None = None,
        deadline: CaseDeadline | None = None,
    ) -> AdapterContainer:
        """Start a pinned Adapter and wait for its RPC health check."""
        if not isinstance(lock, BenchmarkLock):
            raise TypeError(
                "Docker executor requires a lock with verified local assets"
            )
        if deadline is not None:
            deadline.check()
        lock.verify()
        if deadline is not None:
            deadline.check()
        directory = Path(case_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"case directory is missing: {directory}")
        image_id = self.ensure_image(lock, deadline=deadline)
        lock.verify()
        if deadline is not None:
            deadline.check()
        manifest = lock.as_dict()
        token = secrets.token_urlsafe(32)
        port = pick_free_port() if host_port is None else host_port
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("host_port must be a valid TCP port")
        container_name = f"roborsi-{uuid.uuid4().hex}"
        command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--pull=never",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g",
            "--user",
            self._container_user,
            "--publish",
            f"127.0.0.1:{port}:{manifest['service_port']}",
            "--cpus",
            str(manifest["resources"]["cpus"]),
            "--memory",
            f"{manifest['resources']['memory_mb']}m",
        ]
        gpus = manifest["resources"]["gpus"]
        if gpus:
            command.extend(["--gpus", str(gpus)])
        asset_root = lock.manifest_path.parent.resolve()
        for asset in manifest["assets"]:
            source = (asset_root / asset["path"]).resolve()
            command.extend(
                [
                    "--mount",
                    self._mount(
                        source, f"/roborsi/assets/{asset['name']}", readonly=True
                    ),
                ]
            )
        run_timeout = deadline.remaining(120.0) if deadline is not None else 120.0
        try:
            environment = {"ROBORSI_BENCHMARK_TOKEN": token}
            with environment_file(environment) as env_path:
                command.extend(
                    [
                        "--env-file",
                        str(env_path),
                        lock.image_reference,
                        *manifest["adapter_command"],
                    ]
                )
                container_id = self._call_command(command, run_timeout, deadline)
        except Exception:
            try:
                self._command_runner(["docker", "rm", "--force", container_name], 10.0)
            except DockerCommandError:
                pass
            raise
        if not _CONTAINER_ID.fullmatch(container_id):
            try:
                self._command_runner(["docker", "rm", "--force", container_name], 10.0)
            except DockerCommandError:
                pass
            raise ValueError("Docker run did not return a container ID")
        rpc: RpcClient | None = None
        try:
            if deadline is not None:
                deadline.check()
            rpc = self._authenticated_client(port, token)
            wait_for_ready(
                rpc,
                timeout_s=manifest["resources"]["startup_timeout_s"],
                poll_interval_s=0.2,
                remaining_s=deadline.remaining if deadline is not None else None,
            )
            if deadline is not None:
                deadline.check()
            with interrupt_http_calls(deadline.check if deadline is not None else None):
                info = AdapterClient(rpc).runtime_info(
                    timeout_s=deadline.remaining(30.0) if deadline else 30.0
                )
            initial = self._runtime_record(
                lock, directory, container_id, image_id, port, info, deadline, token
            )
        except Exception:
            try:
                if rpc is not None:
                    rpc.close()
            finally:
                try:
                    self._command_runner(
                        ["docker", "rm", "--force", container_id], 10.0
                    )
                except DockerCommandError:
                    pass
            raise

        def verify_runtime() -> BenchmarkRuntimeRecord | None:
            if deadline is not None:
                try:
                    deadline.check()
                except (CaseCancelled, CaseTimeout):
                    return None
            probe = self._authenticated_client(port, token)
            try:
                with interrupt_http_calls(deadline.check if deadline else None):
                    current_info = AdapterClient(probe).runtime_info(
                        timeout_s=deadline.remaining(5.0) if deadline else 5.0
                    )
            finally:
                probe.close()
            current = self._runtime_record(
                lock,
                directory,
                container_id,
                image_id,
                port,
                current_info,
                deadline,
                token,
            )
            if current != initial:
                raise ValueError("Benchmark runtime changed during evaluation")
            return current

        return AdapterContainer(
            container_id,
            rpc,
            self._command_runner,
            runtime_record=initial,
            verify_runtime=verify_runtime,
        )

    def _authenticated_client(self, port: int, token: str) -> RpcClient:
        client = self._client_factory(f"http://127.0.0.1:{port}")
        if isinstance(client, BenchmarkTransport):
            client.token = token
        return client

    def _runtime_record(
        self,
        lock: BenchmarkLock,
        directory: Path,
        container_id: str,
        image_id: str,
        port: int,
        info: BenchmarkRuntimeInfo | None,
        deadline: CaseDeadline | None,
        token: str,
    ) -> BenchmarkRuntimeRecord:
        """Verify the actual image, immutable root, resources and every bind mount."""
        lock.verify()
        manifest = lock.as_dict()
        value = json.loads(
            self._run_command(
                ["docker", "inspect", container_id, "--format", "{{json .}}"],
                10.0,
                deadline,
            )
        )
        config, host = value["Config"], value["HostConfig"]
        if f"ROBORSI_BENCHMARK_TOKEN={token}" not in config.get("Env", []):
            raise ValueError("Benchmark authentication differs from requested runtime")
        if (
            value["Image"] != image_id
            or not value["State"]["Running"]
            or host["ReadonlyRootfs"] is not True
            or config["Cmd"] != manifest["adapter_command"]
            or config["User"] != self._container_user
            or host["NanoCpus"] != round(manifest["resources"]["cpus"] * 1_000_000_000)
            or host["Memory"] != manifest["resources"]["memory_mb"] * 1024 * 1024
        ):
            raise ValueError("Docker container differs from requested runtime")
        expected_ports = {
            f"{manifest['service_port']}/tcp": [
                {"HostIp": "127.0.0.1", "HostPort": str(port)}
            ]
        }
        if host["PortBindings"] != expected_ports:
            raise ValueError("Docker service port differs from requested binding")
        requests = host.get("DeviceRequests") or []
        gpus = manifest["resources"]["gpus"]
        if (not gpus and requests) or (
            gpus
            and (
                len(requests) != 1
                or requests[0].get("Count") != gpus
                or requests[0].get("DeviceIDs")
                or ["gpu"] not in requests[0].get("Capabilities", [])
            )
        ):
            raise ValueError("Docker GPU request differs from requested resources")
        if set(host.get("Tmpfs") or {}) != {"/tmp"}:
            raise ValueError("Docker temporary mounts differ from requested runtime")
        expected_mounts = {}
        for asset in manifest["assets"]:
            expected_mounts[f"/roborsi/assets/{asset['name']}"] = (
                str((lock.manifest_path.parent / asset["path"]).resolve()),
                False,
            )
        mounts = value["Mounts"]
        actual_mounts = {
            mount["Destination"]: (mount["Source"], mount["RW"])
            for mount in mounts
            if mount["Type"] == "bind"
        }
        if actual_mounts != expected_mounts or any(
            mount["Type"] != "bind" and mount["Destination"] != "/tmp"
            for mount in mounts
        ):
            raise ValueError("Docker mounts differ from checked case files and assets")
        if info is not None:
            for field in (
                "benchmark",
                "benchmark_version",
                "benchmark_commit",
                "adapter_version",
                "evaluation",
            ):
                if getattr(info, field) != manifest[field]:
                    raise ValueError(f"runtime {field} differs from Benchmark manifest")
        # Keep accidentally baked-in credentials out of results. The image digest
        # identifies the complete files; this hash detects environment overrides.
        environment_digest = hashlib.sha256(
            json.dumps(
                sorted(
                    item
                    for item in (config.get("Env") or [])
                    if not item.startswith("ROBORSI_BENCHMARK_TOKEN=")
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        configuration = {
            "image_id": image_id,
            "command": config["Cmd"],
            "entrypoint": config.get("Entrypoint"),
            "working_dir": config.get("WorkingDir"),
            "environment_sha256": environment_digest,
            "authentication": "per_container_bearer",
            "user": config["User"],
            **{
                key: host.get(key)
                for key in (
                    "NanoCpus",
                    "Memory",
                    "DeviceRequests",
                    "NetworkMode",
                    "ReadonlyRootfs",
                    "Tmpfs",
                    "CpusetCpus",
                    "CpusetMems",
                    "Runtime",
                )
            },
        }
        return BenchmarkRuntimeRecord(
            info=info,
            docker_image_digest=manifest["image_digest"],
            assets_version=manifest["assets_version"],
            assets=tuple(
                {k: v for k, v in a.items() if k != "path"} for a in manifest["assets"]
            ),
            configuration=configuration,
        )
