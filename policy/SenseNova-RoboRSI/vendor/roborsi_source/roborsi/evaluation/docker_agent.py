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

"""Run an Agent with access only to its own workspace and read-only case images."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.evaluation.docker_common import (
    CommandRunner,
    DockerRuntime,
    _docker_command,
    environment_file,
    remove_container,
    remove_network,
    validate_image_reference,
)
from roborsi.evaluation.messages import ContainerAgentRuntimeRecord
from roborsi.evaluation.services import AgentClient
from roborsi.utils.daemon import pick_free_port
from roborsi.utils.rpc import RpcClient, make_rpc_client, wait_for_ready

_FACTORY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")
_PORT = 8001


class AgentContainer:
    """Own the Agent client, container and its dedicated Docker network."""

    def __init__(
        self,
        container_id: str,
        network: str,
        rpc: RpcClient,
        command_runner: CommandRunner,
        verify_runtime: Callable[[], ContainerAgentRuntimeRecord | None],
    ) -> None:
        self.container_id, self.network = container_id, network
        self.client = AgentClient(rpc)
        self.runtime_record: ContainerAgentRuntimeRecord | None = None
        self._rpc, self._command_runner = rpc, command_runner
        self._verify_runtime = verify_runtime
        self._closed = False

    def close(self) -> None:
        """Inspect the runtime, then remove the container and network on every exit."""
        if self._closed:
            return
        checked = None
        try:
            checked = self._verify_runtime()
        finally:
            try:
                self._rpc.close()
            finally:
                try:
                    remove_container(self._command_runner, self.container_id)
                finally:
                    remove_network(self._command_runner, self.network)
                self._closed = True
        if checked is not None:
            self.runtime_record = checked.model_copy(
                update={"verified_after_close": True}
            )

    def __enter__(self) -> AgentContainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerAgentLauncher(DockerRuntime):
    """Launch a pinned Agent image without mounting Runner results or host code.

    The image must contain RoboRSI, the named factory and its Python dependencies.
    Only explicitly selected environment variables are passed from the host.
    The Docker daemon and host administrator remain trusted.
    """

    def __init__(
        self,
        image: str,
        factory: str,
        *,
        agent_config: str | None = None,
        agent_overrides: Sequence[str] = (),
        cpus: float = 1.0,
        memory_mb: int = 2048,
        pids_limit: int = 256,
        startup_timeout_s: float = 60.0,
        credential_env: Sequence[str] = (),
        command_runner: CommandRunner = _docker_command,
        client_factory: Callable[[str], RpcClient] = make_rpc_client,
        container_user: str | None = None,
    ) -> None:
        super().__init__(command_runner=command_runner, container_user=container_user)
        validate_image_reference(image)
        if not isinstance(factory, str) or not _FACTORY.fullmatch(factory):
            raise ValueError("Agent factory must be module:function")
        from roborsi.agents.composition import normalize_overrides
        from roborsi.agents.config import bundle_package

        if agent_config is not None:
            package = bundle_package(agent_config)
            if factory != f"{package}:make_agent":
                raise ValueError("Agent config must match its installed bundle factory")
        if agent_overrides and agent_config is None:
            raise ValueError("Agent overrides require agent_config")
        self.agent_config = agent_config
        self.agent_overrides = normalize_overrides(agent_overrides)
        if (
            not math.isfinite(cpus)
            or cpus <= 0
            or not math.isfinite(startup_timeout_s)
            or startup_timeout_s <= 0
        ):
            raise ValueError(
                "Agent CPU and startup timeout must be positive and finite"
            )
        if (
            type(memory_mb) is not int
            or memory_mb <= 0
            or type(pids_limit) is not int
            or pids_limit <= 0
        ):
            raise ValueError("Agent memory and PID limits must be positive integers")
        if isinstance(credential_env, str) or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name.startswith(("PYTHON", "LD_", "DYLD_", "ROBORSI_", "HEART_"))
            or name in {"PATH", "HOME", "TMPDIR"}
            for name in credential_env
        ):
            raise ValueError("credential_env must contain credential variable names")
        self.image, self.factory = image, factory
        self.cpus, self.memory_mb, self.pids_limit = cpus, memory_mb, pids_limit
        self.startup_timeout_s = startup_timeout_s
        self.credential_env = tuple(sorted(set(credential_env)))
        self._client_factory = client_factory

    @property
    def agent_version(self) -> str:
        """Version the deployed image and entry point together."""
        return hashlib.sha256(
            json.dumps(
                {
                    "image": self.image,
                    "factory": self.factory,
                    **(
                        {"overrides": list(self.agent_overrides)}
                        if self.agent_overrides
                        else {}
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def verify_version(self, requested: str) -> None:
        """Require the version derived from the image digest and factory name."""
        if requested != self.agent_version:
            raise ValueError(
                "declared Agent version differs from its image and factory"
            )

    def start(
        self,
        case_dir: str | Path,
        *,
        deadline: CaseDeadline | None = None,
        media_dir: str | Path | None = None,
    ) -> AgentContainer:
        """Start one restricted Agent and remove all resources on startup failure."""
        directory = Path(case_dir)
        media = Path(media_dir) if media_dir is not None else None
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Agent workspace must be an existing directory")
        if media is None or media.is_symlink() or not media.is_dir():
            raise ValueError("Docker Agent requires a case media directory")
        directory, media = directory.resolve(), media.resolve()
        if directory.is_relative_to(media) or media.is_relative_to(directory):
            raise ValueError("Agent workspace and media must be separate directories")
        image_id = self.ensure_reference(self.image, deadline=deadline)
        name, network = (
            f"roborsi-agent-{uuid.uuid4().hex}",
            f"roborsi-agent-net-{uuid.uuid4().hex}",
        )
        port = pick_free_port()
        command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--pull=never",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--network",
            network,
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--user",
            self._container_user,
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{self.memory_mb}m",
            "--workdir",
            "/roborsi/work",
            "--publish",
            f"127.0.0.1:{port}:{_PORT}",
            "--entrypoint",
            "python",
            "--mount",
            self._mount(directory, "/roborsi/work"),
            "--mount",
            self._mount(media, "/roborsi/media", readonly=True),
        ]
        service = [
            "-I",
            "-B",
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "agent",
            *(
                ["--agent-config", self.agent_config]
                if self.agent_config
                else ["--factory", self.factory]
            ),
            "--case-dir",
            "/roborsi/work",
            "--host",
            "0.0.0.0",
            "--port",
            str(_PORT),
        ]
        for override in self.agent_overrides:
            service.extend(["--agent-set", override])
        environment = {
            "HOME": "/roborsi/work",
            "TMPDIR": "/tmp",
            "ROBORSI_MEDIA_DIR": "/roborsi/media",
            "PYTHONDONTWRITEBYTECODE": "1",
            **{
                key: os.environ[key] for key in self.credential_env if key in os.environ
            },
        }
        rpc = None
        container_id = name
        try:
            self._run_command(
                ["docker", "network", "create", "--driver", "bridge", network],
                30.0,
                deadline,
            )
            with environment_file(environment) as env_path:
                command.extend(["--env-file", str(env_path), self.image, *service])
                container_id = self._call_command(
                    command, deadline.remaining(120) if deadline else 120, deadline
                )
            if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
                container_id = name
                raise ValueError("Docker run did not return an Agent container ID")
            if deadline:
                deadline.check()
            rpc = self._client_factory(f"http://127.0.0.1:{port}")
            wait_for_ready(
                rpc,
                timeout_s=self.startup_timeout_s,
                poll_interval_s=0.1,
                remaining_s=deadline.remaining if deadline else None,
            )
            initial = self._runtime_record(
                container_id,
                network,
                image_id,
                port,
                directory,
                media,
                service,
                deadline,
            )
        except BaseException:
            try:
                if rpc is not None:
                    rpc.close()
            finally:
                try:
                    remove_container(self._command_runner, container_id)
                finally:
                    remove_network(self._command_runner, network)
            raise

        def verify_runtime() -> ContainerAgentRuntimeRecord | None:
            if deadline:
                try:
                    deadline.check()
                except (CaseCancelled, CaseTimeout):
                    return None
            current = self._runtime_record(
                container_id,
                network,
                image_id,
                port,
                directory,
                media,
                service,
                deadline,
            )
            if current != initial:
                raise ValueError("Agent container changed during evaluation")
            return current

        return AgentContainer(
            container_id, network, rpc, self._command_runner, verify_runtime
        )

    def _runtime_record(
        self,
        container_id: str,
        network: str,
        image_id: str,
        port: int,
        directory: Path,
        media: Path,
        service: list[str],
        deadline: CaseDeadline | None,
    ) -> ContainerAgentRuntimeRecord:
        value = json.loads(
            self._run_command(
                ["docker", "inspect", container_id, "--format", "{{json .}}"],
                10.0,
                deadline,
            )
        )
        config, host = value["Config"], value["HostConfig"]
        if (
            value["Image"] != image_id
            or not value["State"]["Running"]
            or config["Cmd"] != service
            or config["Entrypoint"] != ["python"]
            or config["WorkingDir"] != "/roborsi/work"
            or config["User"] != self._container_user
            or host["ReadonlyRootfs"] is not True
            or host.get("Privileged")
            or host.get("CapAdd")
            or host.get("CapDrop") != ["ALL"]
            or "no-new-privileges" not in host.get("SecurityOpt", [])
            or host["PidsLimit"] != self.pids_limit
            or host["NetworkMode"] != network
            or host["NanoCpus"] != round(self.cpus * 1_000_000_000)
            or host["Memory"] != self.memory_mb * 1024 * 1024
            or host.get("PidMode") not in (None, "")
            or host.get("IpcMode") != "private"
            or host.get("Devices")
            or host.get("DeviceRequests")
        ):
            raise ValueError(
                "Agent container differs from requested isolation or resources"
            )
        if host["PortBindings"] != {
            f"{_PORT}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]
        }:
            raise ValueError("Agent port must be bound only to host loopback")
        if host.get("Tmpfs") != {"/tmp": "rw,nosuid,nodev,size=512m"}:
            raise ValueError("Agent temporary mounts differ")
        mounts = value["Mounts"]
        if {
            m["Destination"]: (m["Source"], m["RW"])
            for m in mounts
            if m["Type"] == "bind"
        } != {
            "/roborsi/work": (str(directory), True),
            "/roborsi/media": (str(media), False),
        } or any(m["Type"] != "bind" and m["Destination"] != "/tmp" for m in mounts):
            raise ValueError("Agent mounts expose unexpected host files")
        if set(value["NetworkSettings"]["Networks"]) != {network}:
            raise ValueError("Agent joined an unexpected network")
        details = json.loads(
            self._run_command(["docker", "network", "inspect", network], 10.0, deadline)
        )[0]
        if details["Driver"] != "bridge" or set(details.get("Containers") or {}) != {
            container_id
        }:
            raise ValueError("Agent network is not dedicated to this case")
        return ContainerAgentRuntimeRecord(
            agent_version=self.agent_version,
            docker_image_digest=self.image.rpartition("@")[2],
            factory=self.factory,
            configuration={
                "cpus": self.cpus,
                "memory_mb": self.memory_mb,
                "pids_limit": self.pids_limit,
                "user": self._container_user,
                "network": "dedicated_bridge",
                "read_only_root": True,
                "media": "read_only",
                "workspace": "case_only",
                "credential_env": list(self.credential_env),
                **(
                    {"agent_overrides": list(self.agent_overrides)}
                    if self.agent_overrides
                    else {}
                ),
            },
        )
