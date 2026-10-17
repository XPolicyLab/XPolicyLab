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

"""Provider-independent tool-use agent loop built on pydantic-ai.

The loop wraps the agent's :class:`~roborsi.tools.toolkit.Toolkit` as
pydantic-ai function tools and drives :class:`pydantic_ai.Agent` runs,
streaming each turn so progress is logged in real time. Task completion is
signalled by the robot-provided ``finish`` tool, whose result carries ``_finish``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import queue
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelSettings,
    RunContext,
    Tool,
    ToolReturn,
)
from pydantic_ai.capabilities import AbstractCapability, Thinking
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.usage import RunUsage, UsageLimits

from roborsi.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from roborsi.dashboard.interaction import (
    DashboardInteractionPort,
    DashboardMessage,
)
from roborsi.dashboard.planner_control import DashboardPlannerControl
from roborsi.llm.client import OMITTED_HISTORY_IMAGE_TEXT, build_model_settings
from roborsi.planner.base import REASONING_EFFORTS, PlannerFailure, PlannerResult
from roborsi.planner.utils.user_input import QUIT_TOKENS
from roborsi.session import EnvState
from roborsi.tools.toolkit import Toolkit
from roborsi.utils.logging import get_logger

if TYPE_CHECKING:
    from roborsi.runtime import RuntimeConfig

logger = get_logger("api_loop")


@dataclasses.dataclass
class _MaximumOpenAIThinking(AbstractCapability[Any]):
    """Request the gateway's extended OpenAI ``max`` reasoning level."""

    id: str | None = "thinking"

    def get_model_settings(self) -> ModelSettings:
        # TokenHub exposes ``max`` beyond pydantic-ai's portable ``xhigh``
        # ThinkingLevel. Preserve the exact gateway contract.
        return ModelSettings(
            openai_reasoning_effort="max"  # type: ignore[typeddict-item]
        )


#: Console-log truncation limits (characters).
_TEXT_LOG_LIMIT = 500
_ARGS_LOG_LIMIT = 250
_TOOL_LOG_LIMIT = 350
#: Cap on cumulative decoded image bytes kept in the resent request history.
_MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: Always retain at least this many of the most recent images, even if a single
#: frame exceeds the byte budget, so the model never loses its current view.
_MIN_RECENT_IMAGES = 2


def _user_message_log(message: str | list[str | BinaryContent]) -> str:
    """Keep camera bytes out of the lightweight planner transcript."""
    if isinstance(message, str):
        return message
    return "\n".join(part if isinstance(part, str) else "[image]" for part in message)


def _record_stop(run: Any | None, *, status: str, reason: str) -> None:
    """Distinguish planner stop boundaries from SDK iterator cleanup cancellation."""
    from roborsi.runtime.trace import current_trace

    recorder = current_trace()
    if recorder is not None:
        if run is None:
            recorder.mark_run_stop(status=status, reason=reason)
        else:
            recorder.mark_agent_stop(run.run_id, status=status, reason=reason)


@dataclasses.dataclass
class _ToolVisibility(AbstractCapability):
    """Hide schemas after execution permissions are established by PrepareTools."""

    hidden: frozenset[str]

    async def before_model_request(
        self, ctx: RunContext, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        from roborsi.runtime.trace import current_trace

        parameters = request_context.model_request_parameters
        visible = [
            tool for tool in parameters.function_tools if tool.name not in self.hidden
        ]
        trace = current_trace()
        if trace is not None:
            trace.emit(
                "tool_visibility",
                {
                    "hidden_tools": sorted(self.hidden),
                    "tools": [
                        {"name": tool.name, "input_schema": tool.parameters_json_schema}
                        for tool in visible
                    ],
                },
            )
        return dataclasses.replace(
            request_context,
            model_request_parameters=dataclasses.replace(
                parameters, function_tools=visible
            ),
        )


class ApiAgentLoop:
    """Planner that runs the tool-calling loop via a pydantic-ai ``Agent``."""

    def __init__(
        self,
        model: Model,
        max_tokens: int | None = 8192,
        no_images: bool = False,
        reasoning_effort: str = "none",
        *,
        dashboard_events: DashboardEventSink,
        timeout_s: int | None = None,
        image_history_groups: int | None = None,
        runtime: RuntimeConfig | None = None,
        preserve_initial_image_count: int = 0,
        include_image_reader: bool = True,
        require_tool_call: bool = False,
    ):
        """Store the pydantic-ai model and the output-token cap."""
        self._model = model
        self._max_tokens = max_tokens
        self._dashboard_events = dashboard_events
        self._no_images = no_images
        self._timeout_s = timeout_s
        self._image_history_groups = image_history_groups
        self._runtime = runtime
        self._preserve_initial_image_count = preserve_initial_image_count
        self._include_image_reader = include_image_reader
        self._require_tool_call = require_tool_call
        if reasoning_effort not in (*REASONING_EFFORTS, "max"):
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        if reasoning_effort == "max" and model.system != "openai":
            raise ValueError("max reasoning effort requires an OpenAI-compatible model")
        self._reasoning_effort = reasoning_effort

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str | list[str | BinaryContent],
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
        initial_messages: list[ModelMessage] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PlannerResult:
        """Run the tool-calling loop until finish, normal stop, or budget."""
        if cancel_event is not None and (
            input_queue is not None or dashboard_interaction is not None
        ):
            raise ValueError("cancel_event requires a noninteractive run")
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        if not isinstance(user_message, str) and (
            input_queue is not None or dashboard_interaction is not None
        ):
            raise ValueError("multimodal initial context requires a noninteractive run")
        if dashboard_interaction is not None:
            if (
                self._runtime is not None
                and self._runtime.response_processor is not None
            ):
                raise ValueError("response_processor requires a non-Dashboard run")
            if initial_messages is not None:
                raise ValueError("custom initial messages require a non-Dashboard run")
            return asyncio.run(
                self._solve_dashboard(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    interaction=dashboard_interaction,
                )
            )
        solve = self._solve(
            system_prompt=system_prompt,
            user_message=user_message,
            toolkit=toolkit,
            max_turns=max_turns,
            input_queue=input_queue,
            initial_messages=initial_messages,
            cancel_event=cancel_event,
        )
        if input_queue is not None:
            return asyncio.run(solve)
        try:
            return asyncio.run(
                self._run_until_stopped(solve, cancel_event=cancel_event)
            )
        except asyncio.TimeoutError:
            toolkit.cancel_active_and_wait()
            logger.error("API planner timed out after %ss", self._timeout_s)
            return PlannerResult(
                finish_result=None,
                messages=[{"role": "user", "content": _user_message_log(user_message)}],
                stats={},
                error=f"API planner timed out after {self._timeout_s}s",
                error_kind="infrastructure",
                stop_reason="timeout",
            )

    async def _run_until_stopped(
        self, solve: Any, *, cancel_event: threading.Event | None
    ) -> PlannerResult:
        """Cancel an in-flight request without interrupting physical handlers."""
        task = asyncio.create_task(solve)

        async def watch_close() -> None:
            assert cancel_event is not None
            while not cancel_event.is_set():
                await asyncio.sleep(0.02)
            task.cancel()

        watcher = asyncio.create_task(watch_close()) if cancel_event else None
        try:
            return await asyncio.wait_for(task, timeout=self._timeout_s)
        finally:
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher

    async def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str | list[str | BinaryContent],
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        initial_messages: list[ModelMessage] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> PlannerResult:
        agent = self._build_agent(system_prompt, toolkit)

        interactive = input_queue is not None
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _user_message_log(user_message)}
        ]
        observer = _ApiRunObserver(
            dashboard_events=self._dashboard_events,
            messages=messages,
            max_turns=max_turns,
        )
        last_error: str | None = None
        usage: RunUsage | None = None
        quit_requested = False
        no_tool_repairs = 0
        error_kind: str | None = None
        stop_reason: str | None = None
        processor = self._runtime.response_processor if self._runtime else None

        def _inject_pending(run: Any) -> bool:
            """Drain queued user lines into the live run; True => end session.

            Each line is enqueued ``asap`` so it lands in the next model request
            (the next turn boundary). This runs on the event-loop thread, so
            mutating the run's pending-message queue here is race-free.
            """
            while True:
                try:
                    line = input_queue.get_nowait()  # type: ignore[union-attr]
                except queue.Empty:
                    return False
                if line is None:
                    return True
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return True
                if not line:
                    continue
                run.enqueue(line, priority="asap")
                messages.append({"role": "user", "content": line})
                logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))

        async def _await_next() -> str | None:
            """Block off-loop for the next user line between runs (None => end)."""
            logger.info("awaiting input — type a message to continue, /quit to end")
            while True:
                line = await asyncio.to_thread(input_queue.get)  # type: ignore[union-attr]
                if line is None:
                    return None
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return None
                if line:
                    logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))
                    return line

        seed = None if initial_messages is not None else user_message
        history: list[ModelMessage] | None = initial_messages
        active_run: Any | None = None
        active_run_counted = False
        try:
            while True:
                run_turns = 0
                ended_without_tool = False
                last_response_tool_calls = 0
                active_run = None
                active_run_counted = False
                retry_feedback = None
                # request_limit overrides pydantic-ai's default (50) so the
                # manual max_turns break below is what bounds each run.
                async with agent.iter(
                    seed,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=max_turns + 1),
                ) as run:
                    active_run = run
                    async for node in run:
                        if interactive and _inject_pending(run):
                            quit_requested = True
                            _record_stop(run, status="cancelled", reason="user_quit")
                            break
                        if Agent.is_call_tools_node(node):
                            run_turns += 1
                            observer.observe_response(
                                node.model_response,
                                run.usage,
                                log_turn=run_turns,
                            )

                            if processor is not None:
                                from roborsi.runtime.context_engine import (
                                    validate_context,
                                )
                                from roborsi.runtime.response import (
                                    ResponseDecision,
                                )

                                decision = processor(node.model_response)
                                if not isinstance(decision, ResponseDecision):
                                    raise TypeError(
                                        "response_processor must return ResponseDecision"
                                    )
                                # The SDK history references this response object. Update
                                # it in place so dispatch and replay use the same calls.
                                node.model_response.parts = decision.response.parts
                                messages[-1] = _serialize_response(node.model_response)
                                if decision.feedback is not None:
                                    validate_context(
                                        [], [node.model_response, decision.feedback]
                                    )
                                    retry_feedback = decision.feedback
                                    for part in retry_feedback.parts:
                                        if isinstance(part, ToolReturnPart):
                                            messages.append(
                                                {
                                                    "role": "tool",
                                                    "name": part.tool_name,
                                                    "tool_call_id": part.tool_call_id,
                                                    "content": part.content,
                                                }
                                            )
                                        elif isinstance(part, UserPromptPart):
                                            messages.append(
                                                {
                                                    "role": "user",
                                                    "content": part.content,
                                                }
                                            )
                                    history = [*run.all_messages(), retry_feedback]
                                    _record_stop(
                                        run, status="completed", reason="response_retry"
                                    )
                                    break

                            tool_calls_before = observer.tool_calls
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    observer.observe_tool(event, run.usage)
                            last_response_tool_calls = (
                                observer.tool_calls - tool_calls_before
                            )
                            if last_response_tool_calls:
                                no_tool_repairs = 0

                            if observer.finish_result is not None:
                                logger.info("FINISH called: %s", observer.finish_result)
                                _record_stop(run, status="completed", reason="finish")
                                stop_reason = "finish"
                                break
                            if observer.turns >= max_turns:
                                logger.info(
                                    "reached max_turns=%d. Stopping.", max_turns
                                )
                                _record_stop(run, status="limit", reason="max_turns")
                                stop_reason = "max_turns"
                                break
                        elif Agent.is_end_node(node):
                            ended_without_tool = (
                                last_response_tool_calls == 0
                                and observer.finish_result is None
                            )
                            if interactive:
                                logger.info(
                                    "model ended turn without a tool call "
                                    "— awaiting your input."
                                )
                            else:
                                logger.info(
                                    "model ended turn without a tool call. Stopping."
                                )
                            break

                    usage = run.usage if usage is None else usage + run.usage
                    active_run_counted = True
                    if retry_feedback is None and (
                        interactive or (self._require_tool_call and ended_without_tool)
                    ):
                        history = run.all_messages()

                if retry_feedback is not None:
                    if observer.turns >= max_turns:
                        stop_reason = "max_turns"
                        break
                    seed = None
                    continue

                if (
                    self._require_tool_call
                    and ended_without_tool
                    and observer.turns < max_turns
                    and not quit_requested
                ):
                    no_tool_repairs += 1
                    if no_tool_repairs >= 3:
                        last_error = "model kept failing to produce a tool call"
                        error_kind = "capability"
                        stop_reason = "invalid_output"
                        break
                    seed = "Reply with exactly one tool call from the provided tools."
                    messages.append({"role": "user", "content": seed})
                    continue

                # finish, quit, non-interactive, or the cumulative turn budget is
                # spent => end the whole session so max_turns is enforced across
                # every run, not per run.
                if (
                    observer.finish_result is not None
                    or quit_requested
                    or not interactive
                    or observer.turns >= max_turns
                ):
                    break
                nxt = await _await_next()
                if nxt is None:
                    _record_stop(None, status="cancelled", reason="user_quit")
                    break
                seed = nxt
                messages.append({"role": "user", "content": seed})
        except asyncio.CancelledError:
            if cancel_event is None or not cancel_event.is_set():
                raise
            _record_stop(active_run, status="cancelled", reason="closed")
            stop_reason = "closed"
        except UsageLimitExceeded as e:
            logger.info("usage limit reached: %s", e)
            stop_reason = "usage_limit"
        except Exception as e:  # noqa: BLE001 - surfaced via PlannerResult.error
            if (
                cancel_event is not None
                and cancel_event.is_set()
                and isinstance(e, PlannerFailure)
                and e.kind == "cancelled"
            ):
                _record_stop(active_run, status="cancelled", reason="closed")
                stop_reason = "closed"
            else:
                last_error = _api_error_text(e, no_images=self._no_images)
                error_kind, stop_reason = _failure_category(e)
                logger.error("agent run failed: %s", last_error)
        finally:
            # Preserve completed requests on failure or usage-limit exits without
            # replacing usage already accumulated across interactive runs.
            if active_run is not None and not active_run_counted:
                usage = active_run.usage if usage is None else usage + active_run.usage

        return PlannerResult(
            finish_result=observer.finish_result,
            messages=messages,
            stats=_build_stats(usage, observer.turns, observer.tool_calls),
            error=last_error,
            error_kind=error_kind,
            stop_reason=stop_reason or ("user_quit" if quit_requested else "model_end"),
        )

    async def _solve_dashboard(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort,
    ) -> PlannerResult:
        """Drive cancellable PydanticAI runs from complete history checkpoints."""
        agent = self._build_agent(system_prompt, toolkit)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        def emit_user(text: str, *, initial: bool = False) -> None:
            if not initial:
                messages.append({"role": "user", "content": text})
            self._dashboard_events.emit(
                TranscriptEvent(
                    {"type": "initial_prompt"}
                    if initial
                    else {"type": "user", "text": text}
                )
            )

        control = DashboardPlannerControl(
            interaction=interaction,
            cancel_active_and_wait=toolkit.cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=lambda: emit_user(user_message, initial=True),
            defer_message_ack=True,
        )
        observer = _ApiRunObserver(
            dashboard_events=self._dashboard_events,
            messages=messages,
            max_turns=max_turns,
        )
        session = _ApiDashboardSession(
            agent=agent,
            control=control,
            observer=observer,
            max_turns=max_turns,
            no_images=self._no_images,
        )
        error: str | None = None
        try:
            await asyncio.wait_for(
                session.run(user_message),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            error = f"API planner timed out after {self._timeout_s}s"
            logger.error("%s", error)
            control.end()
        except Exception as exc:
            error = _api_error_text(exc, no_images=self._no_images)
            logger.error("agent run failed: %s", error)
            control.end()
        finally:
            try:
                await control.cancel_active_toolkit()
            except Exception as exc:
                cleanup_error = (
                    f"API toolkit cancellation failed: {type(exc).__name__}: {exc}"
                )
                logger.warning(cleanup_error)
                error = error or cleanup_error
            await session.close()

        return PlannerResult(
            finish_result=observer.finish_result,
            messages=messages,
            stats={
                "backend": "api",
                **_build_stats(session.usage, observer.turns, observer.tool_calls),
            },
            error=error or session.error,
        )

    def _build_agent(self, system_prompt: str, toolkit: Toolkit) -> Agent:
        """Build an Agent for terminal or Dashboard execution."""
        from roborsi.runtime.factory import build_runtime_agent
        from roborsi.runtime.trace import RuntimeTraceCapability

        thinking_effort: str | bool = self._reasoning_effort
        thinking = (
            _MaximumOpenAIThinking()
            if thinking_effort == "max"
            else Thinking(
                effort=False if thinking_effort == "none" else thinking_effort
            )
        )
        hidden = frozenset(
            spec["name"]
            for spec in toolkit.get_tools_spec()
            if not spec.get("advertise", True)
        )
        return build_runtime_agent(
            model=self._model,
            system_prompt=system_prompt,
            tools=_build_tools(
                toolkit,
                no_images=self._no_images,
                include_image_reader=self._include_image_reader,
            ),
            max_tokens=self._max_tokens,
            runtime=self._context_runtime(),
            capabilities=[
                RuntimeTraceCapability(),
                thinking,
                *([_ToolVisibility(hidden)] if hidden else []),
            ],
        )

    def _context_runtime(self) -> RuntimeConfig:
        """Route legacy image settings through the built-in context engine."""
        from roborsi.runtime import RuntimeConfig
        from roborsi.runtime.context_engine import ContextPolicy

        # History configuration alone must not add delegation tools.
        runtime = self._runtime or RuntimeConfig()
        if (
            self._image_history_groups is None
            and not self._preserve_initial_image_count
        ):
            return runtime
        if (
            runtime.context_engine is not None
            or runtime.context_engine_factory is not None
        ):
            raise ValueError(
                "image history settings must be handled by the custom context_engine"
            )
        if runtime.context is not None and (
            runtime.context.image_history_groups is not None
            or runtime.context.preserve_initial_image_count
        ):
            raise ValueError(
                "configure image history in context or LLMConfig, not both"
            )
        base = runtime.context or ContextPolicy()
        return dataclasses.replace(
            runtime,
            context=dataclasses.replace(
                base,
                image_history_groups=self._image_history_groups,
                preserve_initial_image_count=self._preserve_initial_image_count,
            ),
        )


@dataclasses.dataclass
class _ApiRunObserver:
    """Record model/tool events shared by terminal and Dashboard runs."""

    dashboard_events: DashboardEventSink
    messages: list[dict[str, Any]]
    max_turns: int
    turns: int = 0
    tool_calls: int = 0
    finish_result: dict[str, Any] | None = None
    pending_finish: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def observe_response(
        self,
        response: ModelResponse,
        usage: RunUsage,
        *,
        log_turn: int | None = None,
    ) -> None:
        self.turns += 1
        message = _serialize_response(response)
        self.messages.append(message)
        _log_response(
            response,
            usage,
            self.turns if log_turn is None else log_turn,
            self.max_turns,
        )
        for block in message["content"]:
            if block["type"] == "text":
                payload = {"type": "text", "text": block["text"]}
            elif block["type"] == "thinking":
                payload = {"type": "thinking", "text": block["thinking"]}
            else:
                continue
            self.dashboard_events.emit(TranscriptEvent(payload))
        self.emit_usage(usage)

    def observe_tool(self, event: Any, usage: RunUsage) -> bool:
        completed = False
        if isinstance(event, FunctionToolCallEvent):
            self.tool_calls += 1
            part = event.part
            args = part.args_as_dict()
            self.dashboard_events.emit(
                TranscriptEvent(
                    {"type": "tool_call", "tool": part.tool_name, "args": args}
                )
            )
            if part.tool_name == "finish" and part.tool_call_id is not None:
                self.pending_finish[part.tool_call_id] = {"_finish": True, **args}
        elif isinstance(event, FunctionToolResultEvent):
            completed = True
            message = _serialize_tool_result(event)
            self.messages.append(message)
            _log_tool_result(message)
            part = event.part
            is_error = bool(getattr(part, "is_error", False))
            if not is_error:
                try:
                    terminal = json.loads(message["content"])
                except (TypeError, ValueError):
                    terminal = None
                if isinstance(terminal, dict) and terminal.get("_finish") is True:
                    self.finish_result = terminal
            pending = self.pending_finish.pop(message["tool_call_id"], None)
            if (
                pending is not None
                and not is_error
                and not (isinstance(terminal, dict) and terminal.get("error"))
                and "finish refused" not in message["content"]
            ):
                self.finish_result = (
                    pending if self.finish_result is None else self.finish_result
                )
            self.dashboard_events.emit(
                TranscriptEvent(
                    {
                        "type": "tool_result",
                        "tool": message.get("name") or "tool_result",
                        "result": {
                            "is_error": is_error,
                            "size": len(message["content"]),
                        },
                    }
                )
            )
        self.emit_usage(usage)
        return completed

    def emit_usage(self, usage: RunUsage) -> None:
        self.dashboard_events.emit(
            UsageEvent(
                inp=int(usage.input_tokens or 0),
                out=int(usage.output_tokens or 0),
                tool_calls=self.tool_calls,
            )
        )


class _ApiDashboardSession:
    """Own serial, independent PydanticAI runs for one Dashboard TaskRun."""

    def __init__(
        self,
        *,
        agent: Agent,
        control: DashboardPlannerControl,
        observer: _ApiRunObserver,
        max_turns: int,
        no_images: bool,
    ) -> None:
        self._agent = agent
        self._control = control
        self._max_turns = max_turns
        self._no_images = no_images
        self._observer = observer
        self._history: list[ModelMessage] = []
        self.usage = RunUsage()
        self._pending_prompts: deque[tuple[str | None, str]] = deque()
        self._run_task: asyncio.Task[Any] | None = None
        self._active_prompt = False
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        await self.submit(prompt)
        await self._control.start()
        await self._control.run(self)

    async def submit(self, text: str) -> int:
        """Queue Dashboard input as a new independent API run."""
        return self._queue_prompt(text)

    async def submit_dashboard_message(self, message: DashboardMessage) -> int:
        """Queue Dashboard input and defer acknowledgement until it starts."""
        return self._queue_prompt(message.text, message_id=message.message_id)

    def _queue_prompt(self, text: str, *, message_id: str | None = None) -> int:
        if self._closing:
            raise RuntimeError("API conversation is closed")
        if self.error is not None:
            raise RuntimeError(self.error)
        self._pending_prompts.append((message_id, text))
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._run_pending_prompts())
        return 1

    async def interrupt(self) -> int:
        run_task = self._run_task
        interrupted = int(self._active_prompt) + len(self._pending_prompts)
        discarded_message_ids = tuple(
            message_id
            for message_id, _ in self._pending_prompts
            if message_id is not None
        )
        self._pending_prompts.clear()
        for message_id in discarded_message_ids:
            self._control.message_discarded(message_id)
        if run_task is None or run_task.done():
            return interrupted
        run_task.cancel()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await run_task
        finally:
            if self._run_task is run_task:
                self._run_task = None
        return interrupted

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        await self.interrupt()

    async def _run_pending_prompts(self) -> None:
        task = asyncio.current_task()
        try:
            while self._pending_prompts and not self._closing:
                message_id, seed = self._pending_prompts.popleft()
                self._active_prompt = True
                try:
                    if message_id is not None:
                        self._control.message_started(message_id, seed)
                    if not await self._run_agent(seed):
                        self._pending_prompts.clear()
                        return
                    await self._control.complete(self)
                finally:
                    self._active_prompt = False
        finally:
            if self._run_task is task:
                self._run_task = None

    async def _run_agent(self, seed: str) -> bool:
        run_completed = False
        run: Any | None = None
        node: Any | None = None
        try:
            async with self._agent.iter(
                seed,
                message_history=list(self._history),
                usage=self.usage,
                usage_limits=UsageLimits(request_limit=self._max_turns + 1),
            ) as run:
                node = run.next_node
                while not Agent.is_end_node(node):
                    if Agent.is_call_tools_node(node):
                        await self._process_tool_node(run, node)
                    if (
                        self._observer.finish_result is not None
                        or self._observer.turns >= self._max_turns
                    ):
                        finished = self._observer.finish_result is not None
                        _record_stop(
                            run,
                            status="completed" if finished else "limit",
                            reason="finish" if finished else "max_turns",
                        )
                        self._control.end()
                        return False
                    if self._pending_prompts:
                        # Dashboard input accepted at this tool boundary starts
                        # a fresh run from the checkpoint captured below.
                        node = await run.next(node)
                        _record_stop(run, status="completed", reason="message_handoff")
                        break
                    node = await run.next(node)

                run_completed = True
        except Exception as exc:
            self.error = _api_error_text(exc, no_images=self._no_images)
            logger.error("agent run failed: %s", self.error)
            if not self._closing:
                self._control.end()
        finally:
            # Preserve interrupted tool results for PydanticAI to repair on the
            # next run. Older supported releases can leave a bare tool-call
            # response when cancellation wins before any tool returns; remove
            # only that unusable frontier.
            if run is not None:
                history = list(run.all_messages())
                if (
                    history
                    and isinstance(history[-1], ModelResponse)
                    and history[-1].tool_calls
                ):
                    if request := getattr(node, "request", None):
                        history.append(request)
                    else:
                        history.pop()
                self._history = history
        return run_completed and not self._closing

    async def _process_tool_node(self, run: Any, node: Any) -> None:
        self._observer.observe_response(node.model_response, run.usage)

        async with node.stream(run.ctx) as stream:
            async for event in stream:
                tool_completed = self._observer.observe_tool(event, run.usage)
                if (
                    tool_completed
                    and self._observer.finish_result is None
                    and self._observer.turns < self._max_turns
                ):
                    await self._control.tool_completed(self)


def _build_model_settings(model: Model, max_tokens: int | None) -> ModelSettings:
    """Build model settings, enabling prompt caching for Anthropic models."""
    return build_model_settings(model, max_tokens)


def _prune_history_images(
    messages: list[ModelMessage],
    *,
    max_groups: int | None = None,
    preserve_initial_image_count: int = 0,
) -> list[ModelMessage]:
    """Replace older image groups with text and bound recent image bytes."""
    # Every image in history, oldest -> newest: (msg_idx, part_idx, item_idx, nbytes).
    located: list[tuple[int, int, int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            if not isinstance(part, UserPromptPart) or not isinstance(
                part.content, list
            ):
                continue
            for ii, item in enumerate(part.content):
                if isinstance(item, BinaryContent) and item.media_type.startswith(
                    "image/"
                ):
                    located.append((mi, pi, ii, len(item.data)))

    if not located:
        return messages

    # Demonstration images form a stable prefix; the initial live observation
    # is dynamic and must age out like later observations.
    protected = {entry[:3] for entry in located[:preserve_initial_image_count]}
    dynamic = [entry for entry in located if entry[:3] not in protected]
    groups = list(dict.fromkeys((mi, pi) for mi, pi, _, _ in dynamic))
    recent_groups = set(groups[-max_groups:]) if max_groups is not None else set(groups)

    # Walk newest -> oldest, keeping images while under the byte budget.
    keep: set[tuple[int, int, int]] = set()
    total = 0
    for rank, (mi, pi, ii, nbytes) in enumerate(reversed(dynamic)):
        if (mi, pi) in recent_groups and (
            rank < _MIN_RECENT_IMAGES or total + nbytes <= _MAX_HISTORY_IMAGE_BYTES
        ):
            keep.add((mi, pi, ii))
            total += nbytes

    keep.update(protected)

    if len(keep) == len(located):
        return messages

    drop_items_by_part: dict[tuple[int, int], set[int]] = {}
    for mi, pi, ii, _ in located:
        if (mi, pi, ii) not in keep:
            drop_items_by_part.setdefault((mi, pi), set()).add(ii)

    new_messages = list(messages)
    for (mi, pi), drop_items in drop_items_by_part.items():
        message = new_messages[mi]
        part = message.parts[pi]
        new_content = [
            OMITTED_HISTORY_IMAGE_TEXT if ci in drop_items else item
            for ci, item in enumerate(part.content)
        ]
        new_parts = list(message.parts)
        new_parts[pi] = dataclasses.replace(part, content=new_content)
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)

    return new_messages


def _is_image_rejection(e: Exception) -> bool:
    """True when the provider says its model lacks image input support.

    Matches errors like ``400 {'code': 10007, 'msg': "Bad Request: [message
    type 'image_url' is not supported]"}`` from OpenAI-compatible endpoints
    serving text-only models.
    """
    if not isinstance(e, ModelHTTPError):
        return False
    if not 400 <= e.status_code < 500:
        return False
    message = str(e).lower()
    return any(
        phrase in message
        for phrase in (
            "image_url' is not supported",
            'image_url" is not supported',
            "image input is not supported",
            "image inputs are not supported",
            "images are not supported",
            "model does not support image",
            "model does not support vision",
        )
    )


def _api_error_text(error: Exception, *, no_images: bool) -> str:
    text = f"{type(error).__name__}: {error}"
    if not no_images and _is_image_rejection(error):
        text += (
            "\n\nThe model rejected image input — it is likely a text-only "
            "model (no vision support). Re-run with --no-images: RoboRSI will "
            "then keep every visual observation as a file-path text notice "
            "instead of sending image bytes."
        )
    return text


def _failure_category(error: Exception) -> tuple[str, str]:
    if isinstance(error, PlannerFailure):
        return error.kind, error.reason
    if isinstance(error, (ModelAPIError, ConnectionError, TimeoutError)):
        return "infrastructure", "request_failed"
    # Preserve unknown runtime failures; consumers must never treat an error as
    # a model's decision to give up merely because its category is unfamiliar.
    return "runtime", "exception"


def _build_tools(
    toolkit: Toolkit,
    *,
    no_images: bool = False,
    include_image_reader: bool = True,
) -> list[Tool]:
    """Build the API-only image reader plus pydantic-ai toolkit wrappers."""
    # sequential=True serializes a turn's tool calls so the toolkit's
    # single-operation lock never rejects an overlapping call.
    tools: list[Tool] = []
    finished = threading.Event()
    if include_image_reader:
        image_reader = _make_image_reader(toolkit.state, no_images=no_images)
        tools.append(Tool(image_reader, name="read_image", sequential=True))
    for spec in toolkit.get_tools_spec():
        name = spec["name"]
        tools.append(
            Tool.from_schema(
                function=_make_tool_function(
                    toolkit,
                    name,
                    no_images=no_images,
                    finished=finished,
                ),
                name=name,
                description=spec.get("description", ""),
                json_schema=spec.get("input_schema")
                or {"type": "object", "properties": {}},
                takes_ctx=False,
                sequential=True,
            )
        )
    return tools


def _make_image_reader(
    state: EnvState,
    *,
    no_images: bool,
) -> Callable[[str, int], ToolReturn | dict[str, str] | str]:
    if no_images:

        def read_image_tool(name: str, step: int = -1) -> str:
            return read_image_text_only(name, step, state=state)

        read_image_tool.__name__ = "read_image"
        read_image_tool.__doc__ = read_image_text_only.__doc__
        return read_image_tool

    def read_image_tool(name: str, step: int = -1) -> ToolReturn | dict[str, str]:
        return read_image(name, step, state=state)

    read_image_tool.__name__ = "read_image"
    read_image_tool.__doc__ = read_image.__doc__
    return read_image_tool


def read_image(
    name: str, step: int = -1, *, state: EnvState
) -> ToolReturn | dict[str, str]:
    """Read a step-scoped image artifact as visual input.

    Artifact failures are returned as structured tool errors so a bad
    model-supplied name or step does not abort the agent run.
    """
    try:
        resolved_step, path = _resolve_image_artifact(state, name, step)
        content = BinaryContent(
            data=state.load_bytes(name, step=resolved_step),
            media_type=_image_media_type(path),
        )
    except Exception as e:
        return {"error": str(e)}
    return ToolReturn(
        return_value={"artifact": name, "step": resolved_step},
        content=[content],
    )


def read_image_text_only(
    name: str, step: int = -1, *, state: EnvState
) -> str | dict[str, str]:
    """Acknowledge an image artifact without sending bytes to the model."""
    try:
        resolved_step, _ = _resolve_image_artifact(state, name, step)
    except Exception as e:
        return {"error": str(e)}
    return (
        f"Image artifact {name!r} exists at step {resolved_step}, but image "
        "input is disabled (--no-images, text-only model). Reason from textual "
        "state instead: view_env_state, back_project, and numeric tool results."
    )


def _resolve_image_artifact(
    state: EnvState,
    name: str,
    step: int,
) -> tuple[int, Path]:
    record = state.get(step)
    path = state.artifact_path(name, step=record.step_idx)
    if name not in record.artifacts or not path.is_file():
        raise FileNotFoundError(
            f"image artifact {name!r} is not available at step {step}"
        )
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"artifact {name!r} is not an image")
    return record.step_idx, path


def _image_media_type(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"


def _make_tool_function(
    toolkit: Toolkit,
    name: str,
    *,
    no_images: bool = False,
    finished: threading.Event | None = None,
):
    """Return a callable that dispatches one tool call to the toolkit."""

    def _call(**kwargs: Any) -> Any:
        if finished is not None and finished.is_set():
            # Return feedback for every emitted call, but never dispatch a new
            # action after native termination. This also keeps trace IDs paired.
            return json.dumps(
                {
                    "error": "episode already finished",
                    "code": "episode_finished",
                    "skipped": True,
                }
            )
        result = toolkit.execute_tool(name, kwargs)
        if (
            finished is not None
            and isinstance(result.result, dict)
            and result.result.get("_finish") is True
        ):
            finished.set()
        text, user_content = _content_blocks_to_pydantic(result.content_blocks)
        if result.is_finish:
            # Keep terminal JSON untruncated while camera bytes remain in the
            # content blocks. Raw image bytes are not JSON-serializable.
            text = json.dumps(
                {
                    key: value
                    for key, value in result.result.items()
                    if key
                    not in {
                        "_image_bytes",
                        "_image_cam_bytes",
                        "_image_nav_bytes",
                        "_image_wrist_bytes",
                    }
                },
                ensure_ascii=False,
                default=str,
            )
        if no_images:
            user_content = [
                item for item in user_content if not isinstance(item, BinaryContent)
            ]
        if user_content:
            return ToolReturn(return_value=text, content=user_content)
        return text

    _call.__name__ = name
    return _call


def _content_blocks_to_pydantic(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[str | BinaryContent]]:
    """Use the first text block as tool output; preserve later block order."""
    text: str | None = None
    user_content: list[str | BinaryContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            value = block.get("text", "")
            if text is None:
                text = value
            else:
                user_content.append(value)
        elif block_type == "image":
            source = block.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                user_content.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=source.get("media_type", "image/png"),
                    )
                )
    return text or "{}", user_content


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Render one assistant turn as a serialisable transcript message."""
    content: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            if part.content:
                content.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            if part.content:
                content.append({"type": "thinking", "thinking": part.content})
        elif isinstance(part, ToolCallPart):
            content.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": _tool_arguments_for_log(part),
                }
            )
    return {"role": "assistant", "content": content}


def _serialize_tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """Render one tool result as a serialisable transcript message (no images)."""
    part = event.part
    content = getattr(part, "content", None)
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    return {
        "role": "tool",
        "name": getattr(part, "tool_name", None),
        "tool_call_id": getattr(part, "tool_call_id", None),
        "content": content,
    }


def _tool_arguments_for_log(part: ToolCallPart) -> Any:
    """Logging must not preempt the policy's handling of malformed JSON."""
    try:
        return part.args_as_dict()
    except (ValueError, AssertionError):
        return part.args


def _build_stats(
    usage: RunUsage | None, turns: int, n_tool_calls: int
) -> dict[str, Any]:
    """Assemble the run stats dict from accumulated usage and counters."""
    stats: dict[str, Any] = {"turns_used": turns, "tool_calls": n_tool_calls}
    if usage is not None:
        stats.update(
            {
                "total_input_tokens": int(usage.input_tokens or 0),
                "total_output_tokens": int(usage.output_tokens or 0),
                "cache_read_tokens": int(usage.cache_read_tokens or 0),
                "cache_write_tokens": int(usage.cache_write_tokens or 0),
                "requests": int(usage.requests or 0),
            }
        )
        if usage.details.get("reasoning_tokens") is not None:
            stats["total_reasoning_output_tokens"] = int(
                usage.details["reasoning_tokens"]
            )
        if usage.cost is not None:
            stats["total_cost_usd"] = float(usage.cost)
    return stats


def _log_response(
    response: ModelResponse, usage: RunUsage, turn: int, max_turns: int
) -> None:
    """Log model text, thinking, tool calls, and cumulative usage for a turn."""
    logger.info("=== turn %d/%d ===", turn, max_turns)
    for part in response.parts:
        if isinstance(part, TextPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[model] %s", text)
        elif isinstance(part, ThinkingPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[think] %s", _clip(text, _TEXT_LOG_LIMIT))
        elif isinstance(part, ToolCallPart):
            args = json.dumps(_tool_arguments_for_log(part), default=str)
            logger.info("[tool>] %s(%s)", part.tool_name, _clip(args, _ARGS_LOG_LIMIT))
    logger.info(
        "[usage] in=%s out=%s cache_read=%s cache_write=%s requests=%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.requests,
    )


def _log_tool_result(message: dict[str, Any]) -> None:
    """Log a one-line summary of a tool result."""
    content = " ".join((message.get("content") or "").split())
    logger.info("[tool<] %s: %s", message.get("name"), _clip(content, _TOOL_LOG_LIMIT))


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an overflow marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(+%d)" % (len(text) - limit)
