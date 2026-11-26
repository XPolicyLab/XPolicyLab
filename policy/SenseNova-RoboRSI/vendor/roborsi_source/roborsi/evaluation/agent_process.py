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

"""Start a Python Agent service independently of Benchmark dependencies."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.messages import AgentRuntimeRecord
from roborsi.evaluation.services import AgentClient
from roborsi.utils.daemon import ProcessDaemon, pick_free_port
from roborsi.utils.rpc import RpcClient, make_rpc_client, wait_for_ready


class AgentProcess:
    """Own the Agent RPC client and its case-scoped child process."""

    def __init__(
        self,
        daemon: ProcessDaemon,
        rpc: RpcClient,
        *,
        verify_runtime: Callable[[], AgentRuntimeRecord | None] | None = None,
    ) -> None:
        self.daemon = daemon
        self.client = AgentClient(rpc)
        self._rpc = rpc
        self._closed = False
        self._verify_runtime = verify_runtime
        self.runtime_record: AgentRuntimeRecord | None = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._rpc.call("shutdown", timeout_s=1.0)
        except Exception:
            pass
        finally:
            try:
                self.daemon.stop(timeout=5.0)
            finally:
                self._rpc.close()
                if self._verify_runtime is not None:
                    self.runtime_record = self._verify_runtime()

    def __enter__(self) -> AgentProcess:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class PythonAgentLauncher:
    """Run a trusted Agent factory in its own Python executable and process."""

    def __init__(
        self,
        factory: str | None = None,
        *,
        agent_config: str | Path | None = None,
        agent_overrides: Sequence[str] = (),
        python_executable: str = sys.executable,
        env_overrides: dict[str, str] | None = None,
        startup_timeout_s: float = 30.0,
        cwd: str | Path | None = None,
        isolated: bool = False,
        source_root: Path | None = None,
        verify_runtime: Callable[[], AgentRuntimeRecord | None] | None = None,
    ) -> None:
        if (factory is None) == (agent_config is None):
            raise ValueError("provide one of factory or agent_config")
        if factory is not None and ":" not in factory:
            raise ValueError("Agent factory must be module:function")
        if startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        self.factory = factory
        self.agent_config = None
        self._preset = None
        self.agent_overrides = ()
        if agent_overrides and agent_config is None:
            raise ValueError("Agent overrides require agent_config")
        if agent_config is not None:
            from roborsi.agents.composition import normalize_overrides
            from roborsi.agents.config import AgentConfig

            self.agent_config = (
                str(agent_config)
                if str(agent_config).startswith("bundle:")
                else str(Path(agent_config).absolute())
            )
            self.agent_overrides = normalize_overrides(agent_overrides)
            self._preset = AgentConfig.load(
                self.agent_config, overrides=self.agent_overrides
            )
            if source_root is not None:
                raise ValueError("Agent presets declare their own source_root")
        self.python_executable = python_executable
        self.env_overrides = env_overrides
        self.startup_timeout_s = startup_timeout_s
        self.cwd = str(cwd) if cwd is not None else None
        self.isolated = isolated
        self.source_root = source_root
        self.verify_runtime = verify_runtime

    @property
    def agent_version(self) -> str | None:
        """Return the selected preset identity; legacy factories keep caller labels."""
        return self._preset.fingerprint if self._preset is not None else None

    def verify_version(self, requested: str) -> None:
        """Reject changed preset inputs or a mismatched version before a case."""
        if self._preset is not None:
            if requested != self.agent_version:
                raise ValueError("declared Agent version differs from config inputs")
            self._preset.verify()

    def start(
        self,
        case_dir: str | Path,
        *,
        deadline: CaseDeadline | None = None,
        media_dir: str | Path | None = None,
    ) -> AgentProcess:
        """Start and health-check one Agent; clean up on failed startup."""
        if deadline is not None:
            deadline.check()
        if self._preset is not None:
            self._preset.verify()
        directory = Path(case_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"case directory is missing: {directory}")
        environment = dict(self.env_overrides or {})
        if media_dir is not None:
            media_directory = Path(media_dir)
            if media_directory.is_symlink() or not media_directory.is_dir():
                raise ValueError("Agent media directory is unavailable")
            environment["ROBORSI_MEDIA_DIR"] = str(media_directory.resolve())
        port = pick_free_port()
        command = [
            self.python_executable,
            *(["-I", "-B"] if self.isolated else []),
            "-m",
            "roborsi.evaluation.service_host",
            "--kind",
            "agent",
            *(
                ["--factory", self.factory]
                if self.factory is not None
                else [
                    "--agent-config",
                    self.agent_config,
                    "--config-sha256",
                    self._preset.fingerprint,
                ]
            ),
            "--case-dir",
            str(directory),
            "--port",
            str(port),
            "--parent-watch",
        ]
        if self.source_root is not None:
            command.extend(["--source-root", str(self.source_root)])
        for override in self.agent_overrides:
            command.extend(["--agent-set", override])
        daemon = ProcessDaemon(
            "roborsi-agent",
            command,
            env_overrides=environment,
            log_path=str(directory / "agent-service.log"),
            cwd=self.cwd,
            prepend_rlinf_path=False,
            inherit_environment=not self.isolated,
        )
        daemon.start()
        rpc: RpcClient | None = None
        try:
            rpc = make_rpc_client(f"http://127.0.0.1:{port}")
            wait_for_ready(
                rpc,
                timeout_s=self.startup_timeout_s,
                poll_interval_s=0.1,
                daemon=daemon,
                remaining_s=deadline.remaining if deadline is not None else None,
            )
            if deadline is not None:
                deadline.check()
        except Exception:
            try:
                daemon.stop(timeout=5.0)
            finally:
                if rpc is not None:
                    rpc.close()
            raise

        def verify_runtime() -> AgentRuntimeRecord | None:
            if self._preset is not None:
                self._preset.verify()
            return self.verify_runtime() if self.verify_runtime is not None else None

        return AgentProcess(daemon, rpc, verify_runtime=verify_runtime)
