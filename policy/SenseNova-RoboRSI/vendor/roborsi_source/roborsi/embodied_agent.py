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

"""Benchmark-neutral entry point for RoboRSI planners and user-owned MCP tools."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import queue
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import BinaryContent

from roborsi.dashboard.events import NullDashboardEventSink
from roborsi.data_convert import TextDocument, convert_planner_input
from roborsi.llm import LLMConfig, LLMUsage
from roborsi.planner.base import PlannerFailure, PlannerResult, build_planner
from roborsi.runtime import RuntimeConfig
from roborsi.runtime.lifecycle import (
    activate_trace,
    close_toolkit_trace,
    create_trace,
)

if TYPE_CHECKING:
    from roborsi.evaluation.embodied_bridge import _EmbodiedToolConnector
    from roborsi.evaluation.messages import Action, Observation, StepResult, Task
    from roborsi.planner.codex_cli_config import CodexCliConfig
    from roborsi.planner.provider_capture import ProviderCaptureConfig
from roborsi.runtime.skills import load_skill
from roborsi.session import EnvState
from roborsi.tools.toolkit import ToolResult
from roborsi.utils.config import require_project_env
from roborsi.utils.logging import get_logger

logger = get_logger("embodied_agent")


@dataclass(frozen=True, slots=True)
class McpServer:
    """Connection to a user-owned Streamable HTTP or stdio MCP server.

    Exactly one of ``url`` and ``command`` must be set. A stdio server is
    started for each :meth:`EmbodiedAgent.run` or
    :meth:`EmbodiedAgent.start_episode` and stopped when it ends.
    """

    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    headers: Mapping[str, str] | None = None
    expose_unprefixed: bool = False
    tools: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("MCP server name must contain only letters, digits, or _")
        if bool(self.url) == bool(self.command):
            raise ValueError("set exactly one of url and command")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("MCP server URL must use http:// or https://")
        if self.tools is not None:
            if not isinstance(self.tools, (list, tuple)) or any(
                not isinstance(name, str) or not name.strip() for name in self.tools
            ):
                raise ValueError("MCP tools must be a sequence of names")
            if len(self.tools) != len(set(self.tools)):
                raise ValueError("MCP tool names must be unique")
            object.__setattr__(self, "tools", tuple(self.tools))


@dataclass(frozen=True, slots=True)
class LocalToolSpec:
    """Benchmark-owned tool schema, optionally executed in this process.

    Without a handler, the tool must be named in ``start_episode``'s
    ``deferred_tools`` so the benchmark can execute it on its environment thread.
    Set ``propagate_errors`` to stop the run on handler exceptions. Typed
    ``PlannerFailure`` exceptions always propagate; return ``ToolResult`` for
    recoverable input errors. With the API planner, ``advertise=False`` omits
    this schema from model requests while still accepting returned calls.
    Runtime tool selection can still disable execution of the tool.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult] | None = None
    propagate_errors: bool = False
    advertise: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not self.name.replace("_", "").isalnum()
            or self.name == "finish"
        ):
            raise ValueError(
                "local tool name must contain only letters, digits, or _ and cannot be finish"
            )
        if not isinstance(self.description, str):
            raise TypeError("local tool description must be text")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("local tool input_schema must be a mapping")
        if self.handler is not None and not callable(self.handler):
            raise TypeError("local tool handler must be callable")
        if not isinstance(self.propagate_errors, bool):
            raise TypeError("propagate_errors must be a boolean")
        if not isinstance(self.advertise, bool):
            raise TypeError("advertise must be a boolean")
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(self.input_schema)


@dataclass(slots=True)
class _RemoteTool:
    server: str
    name: str
    spec: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """A model tool call awaiting execution by the benchmark owner."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class _PendingResponse:
    call: PendingToolCall
    ready: threading.Event = field(default_factory=threading.Event)
    result: ToolResult | None = None
    error: Exception | None = None


class _ToolBridge:
    """Hand selected MCP calls to the thread that owns the environment."""

    def __init__(self) -> None:
        self.calls: queue.Queue[PendingToolCall | None] = queue.Queue()
        self._pending: dict[str, _PendingResponse] = {}
        self._lock = threading.Lock()
        self._closed = False
        self.cancelled = threading.Event()

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        call = PendingToolCall(uuid.uuid4().hex, name, dict(arguments))
        pending = _PendingResponse(call)
        with self._lock:
            if self._closed:
                raise PlannerFailure(
                    "embodied episode is closed", kind="cancelled", reason="closed"
                )
            self._pending[call.call_id] = pending
        self.calls.put(call)
        pending.ready.wait()
        with self._lock:
            self._pending.pop(call.call_id, None)
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise PlannerFailure(
                "embodied episode closed before tool completion",
                kind="cancelled",
                reason="closed",
            )
        return pending.result

    def respond(self, call: PendingToolCall, result: ToolResult) -> None:
        with self._lock:
            pending = self._pending.get(call.call_id)
            if pending is None or pending.call != call or pending.ready.is_set():
                raise ValueError("unknown or already completed embodied tool call")
            if result.name != call.name:
                raise ValueError("tool result name does not match pending call")
            pending.result = result
            pending.ready.set()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.cancelled.set()
            for pending in self._pending.values():
                pending.ready.set()
        self.calls.put(None)

    def fail(self, call: PendingToolCall, error: Exception) -> None:
        with self._lock:
            pending = self._pending.get(call.call_id)
            if pending is None or pending.call != call or pending.ready.is_set():
                raise ValueError("unknown or already completed embodied tool call")
            pending.error = error
            pending.ready.set()


class EmbodiedEpisode:
    """Run an agent while the benchmark executes selected tool calls."""

    def __init__(self, bridge: _ToolBridge, worker: threading.Thread) -> None:
        self._bridge = bridge
        self._worker = worker
        self._result: PlannerResult | None = None
        self._error: BaseException | None = None
        self._done = threading.Event()

    def next_call(self, timeout: float | None = None) -> PendingToolCall | None:
        """Wait for a tool call, or return ``None`` after the episode ends."""
        if self.closed:
            return None
        try:
            call = self._bridge.calls.get(timeout=timeout)
            return None if self.closed else call
        except queue.Empty:
            raise TimeoutError("no embodied tool call before timeout") from None

    def complete(self, call: PendingToolCall, result: ToolResult) -> None:
        """Return an executed tool result, including any camera content blocks."""
        self._bridge.respond(call, result)

    def fail(self, call: PendingToolCall, error: Exception) -> None:
        """Stop a pending tool with an error instead of model-visible feedback."""
        self._bridge.fail(call, error)

    def wait(self, timeout: float | None = None) -> PlannerResult:
        """Wait for the planner result, propagating startup failures."""
        if not self._done.wait(timeout):
            raise TimeoutError("embodied episode did not finish before timeout")
        if self._error is not None:
            raise RuntimeError("embodied episode failed") from self._error
        assert self._result is not None
        return self._result

    @property
    def closed(self) -> bool:
        """Whether new actions have been disabled (cleanup may still be running)."""
        return self._bridge.cancelled.is_set()

    @property
    def done(self) -> bool:
        """Whether the worker has released its tools and the agent's run lock."""
        return self._done.is_set()

    def close(self, timeout: float | None = 5) -> None:
        """Disable dispatch and cancel the planner, then wait for safe cleanup.

        Already admitted handlers must return before their connections close.
        Raises TimeoutError if cleanup is still running; callers can inspect
        ``done`` and call ``close`` again. ``timeout=None`` waits without a limit.
        """
        self._bridge.close()
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError("embodied episode cleanup is still running")

    def __enter__(self) -> EmbodiedEpisode:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _McpToolkit:
    """Adapt MCP discovery and calls to the planner's toolkit contract."""

    def __init__(
        self,
        servers: Sequence[McpServer],
        output_dir: Path,
        *,
        local_tools: Sequence[LocalToolSpec] = (),
        bridge: _ToolBridge | None = None,
        deferred_tools: Sequence[str] = (),
        include_finish: bool = True,
    ):
        self.state = EnvState(output_dir)
        self._servers = tuple(servers)
        self._local_tools = {tool.name: tool for tool in local_tools}
        if len(self._local_tools) != len(local_tools):
            raise ValueError("local tool names must be unique")
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, _RemoteTool] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._call_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._bridge = bridge
        self._deferred_tools = frozenset(deferred_tools)
        self._include_finish = include_finish

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for server in self._servers:
                    if server.url:
                        # MCP 1.23 accepted headers on the transport; newer
                        # 1.x releases configure them on an HTTP client.
                        transport_kwargs: dict[str, Any]
                        if (
                            "http_client"
                            in inspect.signature(streamable_http_client).parameters
                        ):
                            import httpx

                            http_client = await stack.enter_async_context(
                                httpx.AsyncClient(
                                    headers=dict(server.headers or {}),
                                    timeout=httpx.Timeout(30, read=300),
                                )
                            )
                            transport_kwargs = {"http_client": http_client}
                        else:
                            transport_kwargs = {"headers": dict(server.headers or {})}
                        read, write, _ = await stack.enter_async_context(
                            streamable_http_client(server.url, **transport_kwargs)
                        )
                    else:
                        params = StdioServerParameters(
                            command=server.command,
                            args=list(server.args),
                            env={**os.environ, **(server.env or {})},
                            cwd=server.cwd,
                        )
                        read, write = await stack.enter_async_context(
                            stdio_client(params)
                        )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    self._sessions[server.name] = session
                    discovered = (await session.list_tools()).tools
                    if server.tools is not None:
                        missing = set(server.tools) - {tool.name for tool in discovered}
                        if missing:
                            raise ValueError(
                                f"unknown MCP tools for {server.name}: {sorted(missing)}"
                            )
                    for tool in discovered:
                        if server.tools is not None and tool.name not in server.tools:
                            continue
                        public_name = (
                            tool.name
                            if server.expose_unprefixed
                            else f"{server.name}__{tool.name}"
                        )
                        if (
                            public_name in self._tools
                            or public_name in self._local_tools
                            or public_name == "finish"
                        ):
                            raise ValueError(f"duplicate MCP tool: {public_name}")
                        self._tools[public_name] = _RemoteTool(
                            server=server.name,
                            name=tool.name,
                            spec={
                                "name": public_name,
                                "description": tool.description or "",
                                "input_schema": tool.inputSchema,
                            },
                        )
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._startup_error = exc
            logger.error("MCP connection failed: %s", exc)
            self._ready.set()
        finally:
            self._loop = None

    def start(self) -> None:
        """Connect to every MCP server and discover its tools."""
        if self._thread is not None:
            raise RuntimeError("MCP toolkit is already started")
        if not self._servers:
            self._validate_local_tools()
            return
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._serve()), name="embodied-mcp", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("failed to start MCP tools") from self._startup_error
        try:
            self._validate_local_tools()
        except ValueError:
            self.close()
            raise

    def _validate_local_tools(self) -> None:
        unknown = self._deferred_tools.difference(
            self._tools.keys() | self._local_tools.keys()
        )
        if unknown:
            raise ValueError(f"unknown deferred tools: {sorted(unknown)}")
        missing = [
            name
            for name, tool in self._local_tools.items()
            if tool.handler is None and name not in self._deferred_tools
        ]
        if missing:
            raise ValueError(f"local tools need handlers or deferral: {missing}")

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return remote tool schemas and a local completion tool."""
        tools = [tool.spec for tool in self._tools.values()]
        tools.extend(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": copy.deepcopy(tool.input_schema),
                **({"advertise": False} if not tool.advertise else {}),
            }
            for tool in self._local_tools.values()
        )
        if not self._include_finish:
            return tools
        return tools + [
            {
                "name": "finish",
                "description": "Report your conclusion. Benchmark success is judged by the environment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure", "stuck"],
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                    "additionalProperties": False,
                },
            }
        ]

    async def _call(self, tool: _RemoteTool, arguments: dict[str, Any]) -> Any:
        return await self._sessions[tool.server].call_tool(tool.name, arguments)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Call one remote tool while preserving its MCP text and image blocks."""
        with self._dispatch_lock:
            if self._bridge is not None and self._bridge.cancelled.is_set():
                raise PlannerFailure(
                    "embodied episode is closed", kind="cancelled", reason="closed"
                )
            return self._execute_tool(name, input_dict)

    def _execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        if name == "finish" and self._include_finish:
            status = input_dict.get("status")
            summary = input_dict.get("summary")
            if status not in {"success", "failure", "stuck"} or not isinstance(
                summary, str
            ):
                return ToolResult(
                    name=name, result={"error": "invalid finish arguments"}
                )
            return ToolResult(
                name=name,
                result={"_finish": True, "status": status, "summary": summary},
            )
        if name in self._deferred_tools:
            assert self._bridge is not None
            return self._bridge.execute(name, input_dict)
        local = self._local_tools.get(name)
        if local is not None:
            assert local.handler is not None
            try:
                result = local.handler(input_dict)
            except Exception as exc:
                if local.propagate_errors or isinstance(exc, PlannerFailure):
                    raise
                return ToolResult(name=name, result={"error": str(exc)})
            if not isinstance(result, ToolResult) or result.name != name:
                raise ValueError(f"local tool {name} must return a matching ToolResult")
            return result
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        with self._call_lock:
            if self._loop is None:
                raise RuntimeError("MCP toolkit is closed")
            future = asyncio.run_coroutine_threadsafe(
                self._call(tool, input_dict), self._loop
            )
            try:
                response = future.result()
            except Exception as exc:
                return ToolResult(name=name, result={"error": str(exc)})
        blocks: list[dict[str, Any]] = []
        for item in response.content:
            if item.type == "text":
                blocks.append({"type": "text", "text": item.text})
            elif item.type == "image":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mimeType,
                            "data": item.data,
                        },
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "text",
                        "text": item.model_dump_json(by_alias=True),
                    }
                )
        structured = getattr(response, "structuredContent", None)
        if structured is None:
            text_blocks = [block["text"] for block in blocks if block["type"] == "text"]
            if len(text_blocks) == 1:
                try:
                    structured = json.loads(text_blocks[0])
                except json.JSONDecodeError:
                    structured = {"text": text_blocks[0]}
            else:
                structured = {"text": "\n".join(text_blocks)}
        result_dict = (
            structured if isinstance(structured, dict) else {"value": structured}
        )
        if response.isError:
            result_dict.setdefault("error", "MCP tool failed")
        if not any(block["type"] == "text" for block in blocks):
            blocks.insert(
                0, {"type": "text", "text": json.dumps(result_dict, ensure_ascii=False)}
            )
        result = ToolResult(
            name=name,
            result=result_dict,
        )
        result.is_finish = False
        result.content_blocks = blocks or result.content_blocks
        return result

    def cancel_active_and_wait(self) -> None:
        """Wait for an active physical action before releasing its transport."""
        with self._dispatch_lock:
            return None

    def cancel_pending(self) -> None:
        """Wake deferred calls before waiting for their dispatch locks."""
        if self._bridge is not None:
            self._bridge.close()

    def close(self) -> None:
        """Close owned MCP sessions after outstanding calls finish."""
        self.cancel_pending()
        self.cancel_active_and_wait()
        with self._call_lock:
            if self._loop is not None and self._stop is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
            if self._thread is not None:
                self._thread.join()
            self._thread = None
            self._loop = None


@dataclass(slots=True)
class EmbodiedAgent:
    """Run RoboRSI with benchmark instructions and local or MCP robot tools.

    Construct once per evaluation configuration, then call :meth:`run` for
    each episode. The benchmark owns reset, success checks, and scoring.
    ``runtime`` optionally configures isolated delegates for the API planner;
    it requires the ``runtime`` installation extra.
    """

    mcp_servers: Sequence[McpServer]
    output_dir: str | Path
    local_tools: Sequence[LocalToolSpec] = ()
    planner: str = "api"
    model: str | None = None
    base_url: str | None = None
    llm: LLMConfig | None = None
    max_turns: int | None = 100
    max_tokens: int | None = 8192
    planner_timeout_s: int | None = None
    reasoning_effort: str = "none"
    runtime: RuntimeConfig | None = None
    codex_cli: CodexCliConfig | None = None
    provider_capture: ProviderCaptureConfig | None = None
    working_dir: str | Path | None = None
    system_prompt: str = (
        "Use the current images and robot state to complete the task. "
        "Submit one roborsi_action using the provided native action schema, then "
        "use its observation feedback to choose the next action. "
        "The Benchmark determines when the episode ends and whether it succeeds."
    )
    action_timeout_s: float | None = 120.0
    evaluation_skills: Sequence[str | Path] = ()
    evaluation_skill_paths: Sequence[str | Path] = ()
    evaluation_memory: Sequence[TextDocument] = ()
    evaluation_initial_context: Sequence[str | BinaryContent] = ()
    _run_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _evaluation_connector: _EmbodiedToolConnector | None = field(
        default=None, init=False, repr=False
    )
    _evaluation_closed: bool = field(default=False, init=False, repr=False)

    def reset(self, task: Task) -> None:
        """Bind one evaluation case and its native action tool to this Agent.

        A service factory can return this Agent directly. The evaluation runtime
        supplies ``ROBORSI_MEDIA_DIR``; model and simulator dependencies stay separate.
        Local and MCP tools remain available alongside the native action tool.
        """
        from roborsi.evaluation.embodied_bridge import (
            _EmbodiedToolConnector,
            roborsi_action_tool,
        )

        if self._evaluation_connector is not None or self._evaluation_closed:
            raise ValueError("Embodied Agent owns only one evaluation case")
        schema = task.capabilities.get("action_schema")
        if not isinstance(schema, dict):
            raise ValueError("evaluation task needs an action_schema")
        tools = tuple(self.local_tools)
        if not any(tool.name == "roborsi_action" for tool in tools):
            tools += (roborsi_action_tool(schema),)
        connector = _EmbodiedToolConnector(
            replace(self, local_tools=tools),
            media_dir=require_project_env("ROBORSI_MEDIA_DIR"),
            system_prompt=self.system_prompt,
            output_dir=self.output_dir,
            action_timeout_s=self.action_timeout_s,
            cleanup_timeout_s=30.0 if self.planner == "codex_cli" else None,
            skills=self.evaluation_skills,
            skill_paths=self.evaluation_skill_paths,
            memory=self.evaluation_memory,
            initial_context=self.evaluation_initial_context,
        )
        connector.reset(task)
        self._evaluation_connector = connector

    def act(self, observation: Observation) -> Action:
        """Feed observation feedback and return the next native action tool call."""
        if self._evaluation_connector is None or self._evaluation_closed:
            raise ValueError("Embodied Agent has no active evaluation case")
        return self._evaluation_connector.act(observation)

    def observe_end(self, step: StepResult) -> None:
        """Deliver terminal observation feedback without requesting another action."""
        if self._evaluation_connector is None or self._evaluation_closed:
            raise ValueError("Embodied Agent has no active evaluation case")
        self._evaluation_connector.observe_end(step)

    def close(self) -> None:
        """Release the case's pending call and model/tool connections."""
        self._evaluation_closed = True
        if self._evaluation_connector is not None:
            self._evaluation_connector.close()

    def run(
        self,
        task: str,
        *,
        system_prompt: str,
        skills: Sequence[str | Path] = (),
        skill_paths: Sequence[str | Path] = (),
        memory: Sequence[TextDocument] = (),
        initial_context: Sequence[str | BinaryContent] = (),
        output_dir: str | Path | None = None,
        episode_ctx: dict[str, Any] | None = None,
    ) -> PlannerResult:
        """Run one episode, loading each supplied ``SKILL.md`` into context.

        Args:
            task: Episode instruction or benchmark task description.
            system_prompt: Benchmark and robot-specific rules.
            skills: Paths to skill Markdown files. Each file is read fresh for
                this episode, so task-specific files can change between runs.
            skill_paths: Task-specific skill directories indexed for on-demand
                read_skill calls; combined with runtime.skill_paths.
            memory: Selected, authorized memory excerpts with optional source
                identifiers. Appended to the task as reference context, not
                system instructions. No memory files are read automatically.
            initial_context: Text and images placed after the task in the first
                user message. A stable prefix can be reused by prompt caching.
            output_dir: Optional per-episode artifact directory. Defaults to
                the directory given to the agent constructor.
            episode_ctx: Runtime dependencies shared by the context engine and
                user tool closures. Create a separate dictionary per episode;
                its contents never become tool schema parameters.

        Returns:
            RoboRSI's planner result. Its ``finish_result`` is self-reported;
            callers must use the benchmark's own success signal for scoring.
        """
        return self._run(
            task,
            system_prompt=system_prompt,
            skills=skills,
            skill_paths=skill_paths,
            memory=memory,
            initial_context=initial_context,
            output_dir=output_dir,
            episode_ctx=episode_ctx,
        )

    def start_episode(
        self,
        task: str,
        *,
        system_prompt: str,
        deferred_tools: Sequence[str],
        skills: Sequence[str | Path] = (),
        skill_paths: Sequence[str | Path] = (),
        memory: Sequence[TextDocument] = (),
        initial_context: Sequence[str | BinaryContent] = (),
        output_dir: str | Path | None = None,
        include_finish: bool = False,
        episode_ctx: dict[str, Any] | None = None,
    ) -> EmbodiedEpisode:
        """Start one persistent run with benchmark-executed MCP tools."""
        if not deferred_tools:
            raise ValueError("start_episode requires at least one deferred tool")
        bridge = _ToolBridge()
        episode: EmbodiedEpisode

        def run_worker() -> None:
            try:
                episode._result = self._run(
                    task,
                    system_prompt=system_prompt,
                    skills=skills,
                    skill_paths=skill_paths,
                    memory=memory,
                    initial_context=initial_context,
                    output_dir=output_dir,
                    bridge=bridge,
                    deferred_tools=deferred_tools,
                    include_finish=include_finish,
                    episode_ctx=episode_ctx,
                )
            except BaseException as exc:
                logger.exception("Embodied episode worker failed")
                episode._error = exc
            finally:
                bridge.close()
                episode._done.set()

        worker = threading.Thread(
            target=run_worker, name="embodied-episode", daemon=True
        )
        episode = EmbodiedEpisode(bridge, worker)
        worker.start()
        return episode

    def _run(
        self,
        task: str,
        *,
        system_prompt: str,
        skills: Sequence[str | Path],
        skill_paths: Sequence[str | Path],
        initial_context: Sequence[str | BinaryContent],
        output_dir: str | Path | None,
        memory: Sequence[TextDocument] = (),
        bridge: _ToolBridge | None = None,
        deferred_tools: Sequence[str] = (),
        include_finish: bool = True,
        episode_ctx: dict[str, Any] | None = None,
    ) -> PlannerResult:
        if self.planner not in {"api", "claude_code", "codex", "codex_cli"}:
            raise ValueError(f"unsupported embodied planner: {self.planner}")
        if self.planner == "codex_cli":
            from roborsi.planner.codex_cli_config import CodexCliConfig

            if not isinstance(self.codex_cli, CodexCliConfig):
                raise ValueError("codex_cli planner requires a CodexCliConfig")
            if self.max_turns is not None:
                raise ValueError(
                    "codex_cli requires max_turns=None; CLI turns cannot be capped"
                )
            if self.max_tokens is not None:
                raise ValueError("codex_cli requires max_tokens=None")
            if self.codex_cli.tools == "external":
                raise ValueError(
                    "tools=external requires a custom Agent using CodexCliPlanner directly"
                )
        elif self.codex_cli is not None:
            raise ValueError("codex_cli settings require the codex_cli planner")
        if self.provider_capture is not None and self.planner not in {
            "codex",
            "codex_cli",
        }:
            raise ValueError("provider_capture requires codex or codex_cli")
        if self.action_timeout_s is None and self.planner != "codex_cli":
            raise ValueError("action_timeout_s=None requires the codex_cli planner")
        runtime = self.runtime
        if skill_paths:
            if self.planner != "api":
                raise ValueError("on-demand skill_paths require the api planner")
            runtime = replace(
                runtime or RuntimeConfig(),
                skill_paths=(*(runtime.skill_paths if runtime else ()), *skill_paths),
            )
        if runtime is not None and self.planner != "api":
            raise ValueError("runtime is supported only by the api planner")
        if runtime is not None:
            runtime.validate_resources()
            if runtime.llm is not None and (
                self.llm is not None
                or self.model is not None
                or self.base_url is not None
            ):
                raise ValueError("pass runtime.llm or llm/model/base_url, not both")
        if self.llm is not None and self.planner != "api":
            raise ValueError("llm is supported only by the api planner")
        if self.planner != "api" and any(
            not tool.advertise for tool in self.local_tools
        ):
            raise ValueError("unadvertised local tools require the api planner")
        if self.llm is not None and (
            self.model is not None or self.base_url is not None
        ):
            raise ValueError("pass either llm or model/base_url")
        if not self.mcp_servers and not self.local_tools:
            raise ValueError("at least one MCP server or local tool is required")
        names = [server.name for server in self.mcp_servers]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique")
        if not task.strip() or not system_prompt.strip():
            raise ValueError("task and system_prompt must be non-empty")
        if self.planner != "codex_cli" and (
            type(self.max_turns) is not int or self.max_turns < 1
        ):
            raise ValueError("max_turns must be positive")
        context = convert_planner_input(
            prompt=system_prompt,
            query=task,
            memory=memory,
            skills=[load_skill(path) for path in skills],
            initial_context=initial_context,
        )
        output = Path(output_dir if output_dir is not None else self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("EmbodiedAgent already has an active episode")
        try:
            toolkit = _McpToolkit(
                self.mcp_servers,
                output,
                local_tools=self.local_tools,
                bridge=bridge,
                deferred_tools=deferred_tools,
                include_finish=include_finish,
            )
            trace = None
            result = None
            try:
                initial = None
                if self.planner == "api":
                    from roborsi.runtime.context_engine import (
                        initialize_episode_context,
                    )

                    runtime, initial = asyncio.run(
                        initialize_episode_context(
                            runtime,
                            context,
                            episode_ctx if episode_ctx is not None else {},
                        )
                    )
                planner = build_planner(
                    self.planner,
                    output_dir=output,
                    recipe_tag="embodied_agent",
                    robot_name="embodied_agent",
                    base_url=self.base_url,
                    model=self.model,
                    llm_config=self.llm,
                    max_tokens=self.max_tokens,
                    planner_timeout_s=self.planner_timeout_s,
                    reasoning_effort=self.reasoning_effort,
                    dashboard_events=NullDashboardEventSink(),
                    runtime=runtime,
                    include_image_reader=False,
                    require_tool_call=bridge is not None,
                    working_dir=Path(self.working_dir).resolve()
                    if self.working_dir is not None
                    else output.resolve(),
                    **({"codex_cli": self.codex_cli} if self.codex_cli else {}),
                    **(
                        {"provider_capture": self.provider_capture}
                        if self.provider_capture
                        else {}
                    ),
                )
                if self.planner == "codex_cli":
                    planner.preflight()
                trace = create_trace(
                    output, runtime, NullDashboardEventSink(), toolkit=toolkit
                )
                toolkit.start()
                with activate_trace(trace):
                    result = planner.solve(
                        system_prompt=initial.system_prompt
                        if initial
                        else context.system_prompt,
                        user_message=context.user_message,
                        toolkit=toolkit,
                        max_turns=self.max_turns,
                        **({"initial_messages": initial.messages} if initial else {}),
                        **({"cancel_event": bridge.cancelled} if bridge else {}),
                    )
                result.stats["llm_usage"] = LLMUsage.from_planner_stats(
                    result.stats
                ).as_dict()
                if self.planner == "codex_cli":
                    usage = result.stats["llm_usage"]
                    usage["status"] = result.stats.get("usage_status", "unavailable")
                    if "usage_source" in result.stats:
                        usage["source"] = result.stats["usage_source"]
                    for key, counter in {
                        "input_tokens": "total_input_tokens",
                        "output_tokens": "total_output_tokens",
                        "cache_read_tokens": "total_cached_input_tokens",
                        "cache_write_tokens": "total_cache_creation_input_tokens",
                        "reasoning_output_tokens": "total_reasoning_output_tokens",
                    }.items():
                        if (
                            usage["status"] == "unavailable"
                            or result.stats.get(counter) is None
                        ):
                            usage[key] = None
                    usage["total_tokens"] = (
                        usage["input_tokens"] + usage["output_tokens"]
                        if usage["input_tokens"] is not None
                        and usage["output_tokens"] is not None
                        else None
                    )
                return result
            finally:
                close_toolkit_trace(
                    toolkit,
                    trace,
                    error=result.error if result is not None else None,
                    cancelled=bridge is not None and bridge.cancelled.is_set(),
                )
        finally:
            self._run_lock.release()
