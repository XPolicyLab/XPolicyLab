# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a pinned Codex CLI, retaining its own agent loop and raw event stream."""

from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roborsi.dashboard.events import TranscriptEvent, UsageEvent
from roborsi.evaluation.result import write_json_atomic
from roborsi.planner.base import PlannerFailure, PlannerResult, strip_mcp_prefix
from roborsi.planner.codex_cli_config import CodexCliConfig
from roborsi.planner.sdk_episode import SdkEpisode, prompt_parts

if TYPE_CHECKING:
    from pydantic_ai.messages import BinaryContent

    from roborsi.dashboard.events import DashboardEventSink
    from roborsi.dashboard.interaction import DashboardInteractionPort
    from roborsi.tools.toolkit import Toolkit

_CLEANUP_TIMEOUT_S = 30.0


def _process_audit_status() -> str:
    """Require procfs PIDs to belong to the caller's signal namespace."""
    if sys.platform != "linux":
        return "unavailable_non_linux"
    try:
        proc_pid = int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    except (OSError, ValueError):
        return "unavailable_procfs"
    return "available" if proc_pid == os.getpid() else "unavailable_pid_namespace"


def _linux_processes() -> dict[int, tuple[int, int, str, str]]:
    """Read parent/group and birth identity without assuming PID ownership."""
    # Some sandboxes expose host procfs while Popen/os.kill use inner PIDs.
    # Those numeric identities cannot establish ownership in this namespace.
    if _process_audit_status() != "available":
        return {}
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            processes[int(entry.name)] = (
                int(fields[1]),
                int(fields[2]),
                fields[0],
                fields[19],
            )
        except (OSError, ValueError, IndexError):
            continue
    return processes


def _owned_descendants(pid: int) -> dict[int, str]:
    processes = _linux_processes()
    owned = {pid}
    while True:
        descendants = {
            child
            for child, (parent, group, _, _) in processes.items()
            if parent in owned or group == pid
        }
        if descendants <= owned:
            break
        owned.update(descendants)
    return {child: processes[child][3] for child in owned if child in processes}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terminate_process(process: subprocess.Popen, deadline: float) -> None:
    """Stop the owned group and any live descendants that created new sessions."""
    descendants = _owned_descendants(process.pid) if sys.platform == "linux" else {}

    def signal_descendants(number):
        current = _linux_processes()
        for pid, birth in descendants.items():
            if pid in current and current[pid][3] == birth:
                try:
                    os.kill(pid, number)
                except ProcessLookupError:
                    pass

    if sys.platform == "linux":
        signal_descendants(signal.SIGTERM)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=max(0.001, min(1.0, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit first; still kill any grandchildren holding pipe FDs.
    if sys.platform == "linux":
        signal_descendants(signal.SIGKILL)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=max(0.001, deadline - time.monotonic()))
    while descendants:
        current = _linux_processes()
        alive = {
            pid: birth
            for pid, birth in descendants.items()
            if pid in current and current[pid][3] == birth and current[pid][2] != "Z"
        }
        if not alive:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Codex CLI processes survived cleanup: {sorted(alive)}")
        time.sleep(0.01)


def preflight_codex_cli(
    config: CodexCliConfig,
    *,
    repo_root: str | Path,
    timeout_s: float = 10.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check the exact executable and environment used by a subsequent run.

    This is an offline version check, not proof of model access. A script/shim
    hash identifies the entry point only; its underlying binary remains unknown.
    """
    cwd = Path(repo_root).resolve()
    if not cwd.is_dir():
        raise ValueError(f"Codex CLI working directory does not exist: {cwd}")
    child_env = dict(env) if env is not None else config.subprocess_env()
    entry = Path(config.executable).expanduser()
    explicit_path = any(
        separator in config.executable
        for separator in (os.sep, os.altsep)
        if separator is not None
    )
    if entry.is_absolute() or explicit_path:
        resolved = entry if entry.is_absolute() else cwd / entry
        executable = str(resolved.resolve())
    else:
        # Match child lookup: relative and empty PATH entries refer to its cwd.
        # An absent PATH keeps isolated environments from finding implicit tools.
        search_path = ""
        if "PATH" in child_env:
            search_path = os.pathsep.join(
                str((cwd / directory).resolve())
                for directory in child_env["PATH"].split(os.pathsep)
            )
        executable = shutil.which(config.executable, path=search_path)
    if executable is None or not Path(executable).is_file():
        raise ValueError(f"Codex CLI executable was not found: {config.executable}")
    path = Path(executable).resolve()
    if not os.access(path, os.X_OK):
        raise ValueError(f"Codex CLI entry is not executable: {path}")
    digest = _sha256(path)
    if config.expected_sha256 is not None and digest != config.expected_sha256:
        raise ValueError("Codex CLI executable SHA-256 does not match expected_sha256")
    process = subprocess.Popen(
        [str(path), "--version"],
        cwd=cwd,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=sys.platform == "linux",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Codex CLI version check timed out") from exc
    finally:
        _terminate_process(process, time.monotonic() + 2)
    version = stdout.decode("utf-8", errors="replace").strip()
    if process.returncode != 0:
        raise ValueError(
            f"Codex CLI version check exited with code {process.returncode}"
        )
    if version != config.expected_version:
        raise ValueError(
            f"Codex CLI version mismatch: expected {config.expected_version!r}, got {version!r}"
        )
    with path.open("rb") as stream:
        magic = stream.read(4)
    native = magic == b"\x7fELF" or magic[:2] == b"MZ"
    companions = []
    for name in ("codex-code-mode-host", "codex-code-mode-host.exe"):
        companion = path.parent / name
        if companion.is_file():
            companions.append({"path": str(companion), "sha256": _sha256(companion)})
    return {
        "executable": str(path),
        "version": version,
        "sha256": digest,
        "native_binary": {"path": str(path), "sha256": digest} if native else None,
        "binary_identity_status": "entry_native" if native else "entry_only",
        "companions": companions,
        "process_audit_status": _process_audit_status(),
    }


def _config_sources(cwd: Path, env: Mapping[str, str]) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    home = env.get("CODEX_HOME")
    if home:
        candidates.add(Path(home) / "config.toml")
    elif env.get("HOME"):
        candidates.add(Path(env["HOME"]) / ".codex" / "config.toml")
    for directory in (cwd, *cwd.parents):
        candidates.update(
            directory / name
            for name in ("AGENTS.md", "AGENTS.override.md", ".codex/config.toml")
        )
    for directory in (cwd / ".agents/skills", cwd / ".codex/skills"):
        if directory.is_dir():
            candidates.update(directory.rglob("SKILL.md"))
    sources = []
    for path in sorted(candidates):
        if path.is_file():
            try:
                sources.append(
                    {"path": str(path), "sha256": _sha256(path), "loaded": "unverified"}
                )
            except OSError:
                sources.append(
                    {
                        "path": str(path),
                        "loaded": "unverified",
                        "hash_status": "unreadable",
                    }
                )
    return sources


def _redacted(overrides: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: "[redacted]"
        if any(
            word in key.lower()
            for word in ("password", "secret", "authorization", "bearer", "api_key")
        )
        else value
        for key, value in overrides.items()
    }


def _toml(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _cli_arguments(args: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Place top-level-only options before the exec subcommand."""
    global_args, exec_args = [], []
    index = 0
    while index < len(args):
        arg = args[index]
        option, separator, _ = arg.partition("=")
        if option in {"--ask-for-approval", "-a", "--search"}:
            global_args.append(arg)
            if option != "--search" and not separator:
                index += 1
                global_args.append(args[index])
        else:
            exec_args.append(arg)
        index += 1
    return global_args, exec_args


class _Recorder:
    def __init__(self, episode: SdkEpisode | None, sink: DashboardEventSink | None):
        self.episode = episode
        self.sink = sink
        self.messages: list[dict[str, Any]] = []
        self.thread_id: str | None = None
        self.failure: PlannerFailure | None = None
        self.completed = False
        self.usage: dict[str, Any] = {}
        self._seen: set[str] = set()
        self.tool_calls = 0
        self._other_tool_calls = 0

    def accept(self, line: bytes) -> None:
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError("CLI event must be an object with a string type")
        except (ValueError, UnicodeError) as exc:
            self.fail(f"Invalid Codex CLI JSON event: {exc}", "protocol_error")
            return
        kind = event["type"]
        if kind == "thread.started":
            self.thread_id = event.get("thread_id")
        elif kind in ("turn.failed", "error"):
            error = event.get("error", event.get("message", "Codex CLI error"))
            self.fail(
                str(error), "turn_failed" if kind == "turn.failed" else "cli_error"
            )
        elif kind == "turn.completed":
            self.completed = True
            usage = event.get("usage")
            if isinstance(usage, dict):
                # One exec has one initial turn. Repeated cumulative snapshots
                # replace earlier counts; cached tokens are subsets of input.
                self.usage = usage
        elif kind == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                self.fail(
                    "Codex CLI item.completed lacks an item object", "protocol_error"
                )
                return
            roborsi_tool = (
                item.get("type") == "mcp_tool_call" and item.get("server") == "roborsi"
            )
            if roborsi_tool and self.episode is not None:
                name = item.get("tool")
                if isinstance(name, str):
                    self.episode.acknowledge(strip_mcp_prefix(name), item.get("result"))
            identifier = str(item.get("id", ""))
            if identifier and identifier in self._seen:
                return
            if identifier:
                self._seen.add(identifier)
            if item.get("type") in (
                "mcp_tool_call",
                "command_execution",
                "file_change",
                "web_search",
            ):
                self.tool_calls += 1
                if not roborsi_tool:
                    self._other_tool_calls += 1
            self.messages.append(event)
            if self.sink is not None and self.sink.enabled:
                text = item.get("text")
                if isinstance(text, str):
                    self.sink.emit(
                        TranscriptEvent({"role": "assistant", "content": text})
                    )

    def fail(self, message: str, reason: str) -> None:
        if self.failure is None:
            self.failure = PlannerFailure(message, kind="infrastructure", reason=reason)

    def stats(self) -> dict[str, Any]:
        def count(key):
            value = self.usage.get(key)
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                else None
            )

        inp, out = count("input_tokens"), count("output_tokens")
        status = (
            "complete"
            if inp is not None and out is not None
            else "partial"
            if self.usage
            else "unavailable"
        )
        return {
            "total_input_tokens": inp,
            "total_output_tokens": out,
            "total_cached_input_tokens": count("cached_input_tokens"),
            "total_cache_creation_input_tokens": count("cache_write_input_tokens"),
            "total_reasoning_output_tokens": count("reasoning_output_tokens"),
            "total_tokens": inp + out if inp is not None and out is not None else None,
            "usage_status": status,
            "usage_source": "codex_cli.turn.completed",
            "total_cost_usd": None,
            "requests": None,
            "turns_used": int(self.completed),
            # Dispatches cover RoboRSI even if CLI completion events are lost.
            # Other tools are visible only through their deduplicated CLI items.
            "tool_calls": self.episode.dispatches + self._other_tool_calls
            if self.episode is not None
            else self.tool_calls,
        }


class CodexCliExternalLauncher:
    """Expose an injected CLI launcher without taking over its controller.

    The injected function owns command construction, sandboxing, stdin and the
    original artifacts. Its controller owns polling, cancellation and recovery.
    This adapter does not read events, wait, retry or stop the returned process.
    ``identity_config`` is only an optional offline identity check: its runtime
    overrides and environment are not applied to the injected launch function.
    """

    def __init__(
        self,
        launch: Callable[..., subprocess.Popen],
        *,
        identity_config: CodexCliConfig | None = None,
        resolve_executable: Callable[[Mapping[str, Any]], str | Path] | None = None,
    ) -> None:
        """Keep the original launch implementation and optional binary pin.

        Args:
            launch: Function with the original six-argument launch contract.
            identity_config: Version and optional SHA pin for offline checks.
            resolve_executable: Resolve the actual payload used by ``launch``;
                for a sandboxed CLI, this is the host binary rather than its
                public mount path or npm shim.
        """
        if not callable(launch):
            raise TypeError("External CLI launcher must be callable")
        if resolve_executable is not None and identity_config is None:
            raise ValueError("An executable resolver requires identity_config")
        self._launch = launch
        self._identity_config = identity_config
        self._resolve_executable = resolve_executable

    def launch(
        self,
        cfg: Mapping[str, Any],
        run: Path,
        private_home: Path,
        segment: Path,
        prompt: str,
        session: str | None = None,
    ) -> subprocess.Popen:
        """Delegate one fresh or resumed segment and return its real process.

        The original objects and prompt are passed unchanged. Optional identity
        evidence is written separately before launch, so a recording failure
        cannot orphan a running process. No framework timeout is imposed.
        """
        if self._identity_config is not None:
            config = self._identity_config
            if self._resolve_executable is not None:
                config = replace(config, executable=str(self._resolve_executable(cfg)))
            identity = preflight_codex_cli(config, repo_root=run)
            # A version-check executable may rewrite itself. Verify its bytes
            # again before the original launcher gets control.
            if _sha256(Path(identity["executable"])) != identity["sha256"]:
                raise ValueError("Codex CLI executable changed during preflight")
            write_json_atomic(
                segment / "cli-backend.json",
                {
                    "schema_version": 1,
                    "execution_mode": "external_controller",
                    "identity": identity,
                    "identity_check_scope": "offline_host_executable",
                    "process_control": "original_controller",
                },
            )
        return self._launch(cfg, run, private_home, segment, prompt, session)


class CodexCliPlanner:
    """Own one non-interactive CLI process per solve and its local transports."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        repo_root: str | Path,
        model: str | None,
        timeout_s: float | None,
        dashboard_events: DashboardEventSink | None,
        reasoning_effort: str,
        config: CodexCliConfig,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.model = model
        self.timeout_s = timeout_s
        self.dashboard_events = dashboard_events
        self.reasoning_effort = reasoning_effort
        self.config = config
        self._preflight_cache: tuple[dict[str, Any], dict[str, str]] | None = None
        self._preflight_lock = threading.Lock()

    def preflight(self) -> dict[str, Any]:
        """Check identity before tool discovery and retain this launch environment.

        The next solve consumes this check once. Later solves perform a fresh
        check, and every launch verifies the entry hash again after tool setup.
        """
        env = self.config.subprocess_env()
        identity = preflight_codex_cli(self.config, repo_root=self.repo_root, env=env)
        with self._preflight_lock:
            self._preflight_cache = (identity, env)
        return json.loads(json.dumps(identity))

    def _call_dir(self) -> Path:
        root = self.output_dir / "codex-cli"
        root.mkdir(parents=True, exist_ok=True)
        counter = 1
        while True:
            path = root / f"call-{counter:04d}"
            try:
                path.mkdir(mode=0o700)
                return path
            except FileExistsError:
                counter += 1

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str | list[str | BinaryContent],
        toolkit: Toolkit | None,
        max_turns: int | None,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PlannerResult:
        """Run the CLI without changing its prompt, tool loop, or turn budget.

        ``max_turns`` must be None: CLI event counts are not model-turn budgets.
        An external-tool Agent passes no Toolkit and owns its file/socket bridge.
        """
        if max_turns is not None:
            raise ValueError("codex_cli requires max_turns=None")
        if input_queue is not None or dashboard_interaction is not None:
            raise ValueError("codex_cli does not support interactive steer")
        if (self.config.tools == "external") != (toolkit is None):
            raise ValueError(
                "codex_cli toolkit must be absent for external tools and present for MCP"
            )
        if sys.platform != "linux":
            raise ValueError(
                "codex_cli execution currently requires Linux process-group cleanup"
            )
        directory = self._call_dir()
        runtime: dict[str, Any] = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "expected_version": self.config.expected_version,
            "requested_executable": self.config.executable,
            "cwd": str(self.repo_root),
            "output_dir": str(directory),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_turns": None,
            "planner_timeout_s": self.timeout_s,
            "tools": self.config.tools,
            "inherit_env": self.config.inherit_env,
            "requested_config": _redacted(self.config.config_overrides),
            "resolved_config_status": "unverified",
            "configuration_identity_status": "incomplete",
            "cleanup": {"status": "not_started"},
        }
        write_json_atomic(directory / "runtime.json", runtime)
        episode = SdkEpisode(toolkit, cancel_event) if toolkit is not None else None
        recorder = _Recorder(episode, self.dashboard_events)
        process = None
        server = None
        capture = None
        reader_threads: list[threading.Thread] = []
        io_errors: list[str] = []
        events: queue.Queue[bytes | None] = queue.Queue()
        failure: PlannerFailure | None = None
        controlled_stop = False
        exit_code = None
        phase = "preflight"
        try:
            with self._preflight_lock:
                prepared = self._preflight_cache
                self._preflight_cache = None
            if prepared is None:
                env = self.config.subprocess_env()
                identity = preflight_codex_cli(
                    self.config, repo_root=self.repo_root, env=env
                )
            else:
                identity, env = prepared
            runtime["environment_names"] = sorted(env)
            runtime.update(identity)
            runtime["configuration_sources"] = _config_sources(self.repo_root, env)
            prompt = prompt_parts(system_prompt, user_message)
            text_parts = []
            images = []
            input_order = []
            for part in [prompt] if isinstance(prompt, str) else prompt:
                if isinstance(part, str):
                    text_parts.append(part)
                    input_order.append({"type": "text", "index": len(text_parts) - 1})
                else:
                    suffix = {
                        "image/png": "png",
                        "image/jpeg": "jpg",
                        "image/webp": "webp",
                        "image/gif": "gif",
                    }.get(part.media_type)
                    if suffix is None:
                        raise ValueError(
                            f"Unsupported CLI input image type: {part.media_type}"
                        )
                    from PIL import Image

                    with Image.open(io.BytesIO(part.data)) as media:
                        expected_format = "JPEG" if suffix == "jpg" else suffix.upper()
                        if media.format != expected_format:
                            raise ValueError(
                                "CLI image bytes do not match the declared media type"
                            )
                        media.verify()
                    image_path = directory / f"input-{len(images) + 1:04d}.{suffix}"
                    image_path.write_bytes(part.data)
                    images.append(image_path)
                    input_order.append(
                        {
                            "type": "image",
                            "path": str(image_path),
                            "sha256": _sha256(image_path),
                            "media_type": part.media_type,
                        }
                    )
            text = "\n\n".join(text_parts)
            (directory / "input.txt").write_bytes(text.encode("utf-8"))
            runtime["input_order"] = input_order
            if cancel_event is not None and cancel_event.is_set():
                raise PlannerFailure(
                    "Codex CLI cancelled", kind="cancelled", reason="closed"
                )
            phase = "startup"
            overrides = dict(self.config.config_overrides)
            overrides["model_reasoning_effort"] = self.reasoning_effort
            if self.config.provider_capture is not None:
                from roborsi.planner.provider_capture import ProviderCapture

                capture = ProviderCapture(
                    self.config.provider_capture,
                    directory / "provider-capture",
                    env=env,
                )
                runtime["provider_capture_requested"] = {
                    "model_provider": self.config.provider_capture.model_provider,
                    "upstream_base_url": self.config.provider_capture.upstream_base_url,
                    "request_encrypted_content": self.config.provider_capture.request_encrypted_content,
                }
                overrides[
                    f"model_providers.{self.config.provider_capture.model_provider}.base_url"
                ] = capture.start()
                overrides["model_provider"] = (
                    self.config.provider_capture.model_provider
                )
            if episode is not None:
                from roborsi.planner.utils.http_mcp_server import HttpMcpServer

                server = HttpMcpServer(episode)
                overrides["mcp_servers.roborsi.url"] = server.start()
                overrides["mcp_servers.roborsi.enabled"] = True
                schema = toolkit.get_tools_spec()
                runtime["tool_schema_sha256"] = hashlib.sha256(
                    json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
            if server is not None or capture is not None:
                for key in ("NO_PROXY", "no_proxy"):
                    env[key] = ",".join(
                        filter(None, [env.get(key), "127.0.0.1", "localhost"])
                    )
            global_args, exec_args = _cli_arguments(self.config.cli_args)
            argv = [
                runtime["executable"],
                *global_args,
                "exec",
                "--json",
                "--color",
                "never",
            ]
            if self.model is not None:
                argv.extend(["--model", self.model])
            argv.extend(exec_args)
            for key, value in overrides.items():
                argv.extend(["-c", f"{key}={_toml(value)}"])
            for image in images:
                argv.extend(["--image", str(image)])
            argv.append("-")
            # Configuration validation excludes literal credentials from argv.
            runtime["argv"] = argv
            runtime["launch_config"] = _redacted(overrides)
            runtime["environment_names"] = sorted(env)
            write_json_atomic(directory / "runtime.json", runtime)
            if _sha256(Path(runtime["executable"])) != runtime["sha256"]:
                raise ValueError("Codex CLI executable changed after its version check")
            process = subprocess.Popen(
                argv,
                cwd=self.repo_root,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            runtime["pid"] = process.pid

            def read_stdout():
                try:
                    with (directory / "events.jsonl").open("wb") as target:
                        for line in iter(process.stdout.readline, b""):
                            target.write(line)
                            target.flush()
                            events.put(line)
                except Exception as exc:
                    io_errors.append(f"CLI stdout recording failed: {exc}")
                finally:
                    events.put(None)

            def read_stderr():
                try:
                    with (directory / "stderr.log").open("wb") as target:
                        while chunk := process.stderr.read(65536):
                            target.write(chunk)
                            target.flush()
                except Exception as exc:
                    io_errors.append(f"CLI stderr recording failed: {exc}")

            def write_stdin():
                try:
                    process.stdin.write(text.encode("utf-8"))
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                except Exception as exc:
                    io_errors.append(f"CLI input write failed: {exc}")

            for target in (read_stdout, read_stderr, write_stdin):
                thread = threading.Thread(
                    target=target, daemon=True, name=f"codex-cli-{target.__name__}"
                )
                thread.start()
                reader_threads.append(thread)
            phase = "running"
            started = time.monotonic()
            while True:
                try:
                    line = events.get(timeout=0.02)
                    if line is None:
                        break
                    recorder.accept(line)
                except queue.Empty:
                    pass
                expired = bool(
                    self.timeout_s and time.monotonic() - started >= self.timeout_s
                )
                if recorder.failure is not None:
                    failure = recorder.failure
                    controlled_stop = True
                    break
                if io_errors:
                    raise OSError(io_errors[0])
                if episode is not None and episode.should_shutdown(
                    timeout_expired=expired
                ):
                    failure = episode.failure
                    controlled_stop = True
                    break
                if cancel_event is not None and cancel_event.is_set():
                    failure = PlannerFailure(
                        "Codex CLI cancelled", kind="cancelled", reason="closed"
                    )
                    controlled_stop = True
                    break
                if expired:
                    failure = PlannerFailure(
                        "Codex CLI timed out", kind="infrastructure", reason="timeout"
                    )
                    controlled_stop = True
                    break
        except PlannerFailure as exc:
            failure = exc
        except Exception as exc:
            failure = PlannerFailure(
                f"{type(exc).__name__}: {exc}",
                kind="configuration" if phase == "preflight" else "infrastructure",
                reason="preflight_failed" if phase == "preflight" else "backend_error",
            )
        finally:
            cleanup_started = time.monotonic()
            deadline = cleanup_started + _CLEANUP_TIMEOUT_S
            cleanup_errors = []
            toolkit_thread = None
            if episode is not None:
                episode.stopped.set()

                def close_toolkit():
                    try:
                        # SdkEpisode releases deferred waits before taking the
                        # toolkit dispatch lock. A noncooperative owner cannot
                        # prevent process and transport cleanup from starting.
                        episode.cancel_active_and_wait()
                    except Exception as exc:
                        cleanup_errors.append(f"tool cleanup: {exc}")

                toolkit_thread = threading.Thread(
                    target=close_toolkit, daemon=True, name="codex-cli-toolkit-cleanup"
                )
                toolkit_thread.start()
            if process is not None:
                try:
                    _terminate_process(process, deadline)
                except Exception as exc:
                    cleanup_errors.append(f"process cleanup: {exc}")
                exit_code = process.poll()
            for thread in reader_threads:
                thread.join(max(0, deadline - time.monotonic()))
                if thread.is_alive():
                    cleanup_errors.append(f"reader still running: {thread.name}")
            while not events.empty():
                line = events.get_nowait()
                if line is not None:
                    recorder.accept(line)
            if episode is not None:
                toolkit_thread.join(max(0, deadline - time.monotonic()))
                if toolkit_thread.is_alive():
                    cleanup_errors.append("toolkit cleanup exceeded deadline")
                episode.finalize()
            if server is not None:
                try:
                    server_thread = server._thread
                    server.stop(timeout_s=max(0, min(5, deadline - time.monotonic())))
                    if server_thread is not None and server_thread.is_alive():
                        cleanup_errors.append("MCP server cleanup exceeded deadline")
                except Exception as exc:
                    cleanup_errors.append(f"MCP cleanup: {exc}")
            if capture is not None:
                try:
                    capture.stop(timeout_s=max(0, min(5, deadline - time.monotonic())))
                    runtime["provider_capture"] = capture.summary
                except Exception as exc:
                    cleanup_errors.append(f"provider capture cleanup: {exc}")
            if process is not None and not any(
                thread.is_alive() for thread in reader_threads
            ):
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
            runtime["cleanup"] = {
                "status": "failed" if cleanup_errors else "complete",
                "errors": cleanup_errors,
                "elapsed_s": time.monotonic() - cleanup_started,
            }
            if cleanup_errors:
                failure = PlannerFailure(
                    "; ".join(cleanup_errors),
                    kind="infrastructure",
                    reason="cleanup_failed",
                )
            elif failure is None:
                failure = (
                    episode.failure if episode is not None else None
                ) or recorder.failure
            if failure is None and exit_code not in (None, 0) and not controlled_stop:
                failure = PlannerFailure(
                    f"Codex CLI exited with code {exit_code}",
                    kind="infrastructure",
                    reason="process_failed",
                )
            if failure is None and not controlled_stop and not recorder.completed:
                failure = PlannerFailure(
                    "Codex CLI stream ended without turn.completed",
                    kind="infrastructure",
                    reason="incomplete_stream",
                )
            if failure is None and io_errors:
                failure = PlannerFailure(
                    io_errors[0], kind="infrastructure", reason="recording_failed"
                )
            if (
                failure is None
                and capture is not None
                and runtime.get("provider_capture", {}).get("requests") == 0
            ):
                failure = PlannerFailure(
                    "Provider capture was enabled but observed no requests; "
                    "verify the selected custom provider and endpoint override",
                    kind="infrastructure",
                    reason="capture_not_observed",
                )
            stats = recorder.stats()
            if (
                self.dashboard_events is not None
                and self.dashboard_events.enabled
                and stats["usage_status"] == "complete"
            ):
                self.dashboard_events.emit(
                    UsageEvent(
                        stats["total_input_tokens"],
                        stats["total_output_tokens"],
                        stats["tool_calls"],
                    )
                )
            result = PlannerResult(
                finish_result=episode.finish_result if episode else None,
                messages=recorder.messages,
                stats=stats,
                error=str(failure) if failure else None,
                error_kind=failure.kind if failure else None,
                stop_reason=failure.reason if failure else "completed",
            )
            runtime.update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "thread_id": recorder.thread_id,
                    "exit_code": exit_code,
                    "controlled_stop": controlled_stop,
                }
            )
            write_json_atomic(directory / "runtime.json", runtime)
            write_json_atomic(
                directory / "result.json",
                {
                    "finish_result": result.finish_result,
                    "error": result.error,
                    "error_kind": result.error_kind,
                    "stop_reason": result.stop_reason,
                    "exit_code": exit_code,
                    "stats": stats,
                },
            )
        return result
