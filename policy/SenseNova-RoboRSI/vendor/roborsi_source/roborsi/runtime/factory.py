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

"""Assemble PydanticAI agents from RoboRSI's existing models, context and tools."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, PrepareTools
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from roborsi.data_convert import convert_planner_input
from roborsi.llm.client import build_model_settings
from roborsi.llm.retry import RetryLoggingModel
from roborsi.runtime.config import RuntimeConfig, SubAgentConfig
from roborsi.runtime.context_engine import (
    ContextEngineCapability,
    _EpisodeContextEngine,
)
from roborsi.runtime.skills import SkillCatalog, load_skill

if TYPE_CHECKING:
    from pydantic_ai.capabilities import AgentCapability, WrapRunHandler
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.usage import UsageLimits

_CHILD_READERS = frozenset({"read_image", "read_text_file", "list_dir", "read_skill"})
_DYNAMIC_AGENT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _context_capabilities(
    capabilities: Sequence[AgentCapability],
    config: RuntimeConfig | SubAgentConfig,
    parent: RuntimeConfig | None = None,
) -> list[AgentCapability]:
    """Apply the selected context policy after any explicitly supplied processors."""
    selected = config
    if parent is not None and all(
        getattr(config, field) is None
        for field in ("context", "context_engine", "context_engine_factory")
    ):
        selected = parent
    policy = selected.context
    image_llm = config.llm if config.llm is not None else selected.llm
    if image_llm is not None:
        from roborsi.runtime.context_engine import ContextPolicy

        if (
            (
                image_llm.image_history_groups is not None
                or image_llm.preserve_initial_image_count
            )
            and selected.context_engine is None
            and selected.context_engine_factory is None
        ):
            policy = replace(
                policy or ContextPolicy(),
                image_history_groups=image_llm.image_history_groups,
                preserve_initial_image_count=image_llm.preserve_initial_image_count,
            )
    callback = selected.context_engine
    factory = selected.context_engine_factory
    if selected is parent and isinstance(callback, _EpisodeContextEngine):
        factory = callback.for_child
        callback = None
    engine = ContextEngineCapability(
        engine=callback,
        factory=factory,
        policy=policy,
    )
    result = list(capabilities)
    result.append(engine)
    return result


def _skill_tools(
    config: RuntimeConfig | SubAgentConfig,
    tools: Sequence[Tool],
    instructions: str,
) -> tuple[list[Tool], str]:
    catalog = SkillCatalog(config.skill_paths, max_bytes=config.skill_max_bytes)
    result = list(tools)
    if catalog.entries:
        if any(tool.name == "read_skill" for tool in result):
            raise ValueError("read_skill conflicts with a configured skill reader")
        result.append(Tool(catalog.read_skill))
        instructions = "\n\n".join(filter(None, [instructions, catalog.instructions]))
    return result, instructions


def _dynamic_delegate_tool(
    runtime: RuntimeConfig,
    parent_model: Model,
    parent_tools: Sequence[Tool],
    capabilities: Sequence[AgentCapability],
    max_tokens: int | None,
) -> tuple[Tool, AbstractCapability]:
    """Build the model-facing ``delegate_task`` tool with per-call child agents.

    The static ``SubAgents`` capability is intentionally not used here: its roster is
    materialized when the parent ``Agent`` is constructed. This tool validates a
    request and constructs a fresh child ``Agent`` inside the tool call, so the main
    model can choose a new role, instructions and safe reader set on every turn.
    """
    skill_catalog = SkillCatalog(runtime.skill_paths, max_bytes=runtime.skill_max_bytes)
    parent_tool_by_name = {
        tool.name: tool for tool in parent_tools if tool.name != "delegate_task"
    }
    for name, config in runtime.subagents.items():
        for tool_name in config.tools:
            if tool_name not in _CHILD_READERS:
                raise ValueError(
                    f"sub-agent {name!r} cannot use tool {tool_name!r}; "
                    f"only {sorted(_CHILD_READERS)} are supported"
                )
            if tool_name != "read_skill" and tool_name not in parent_tool_by_name:
                raise ValueError(
                    f"sub-agent {name!r} requests unavailable tool {tool_name!r}"
                )
            if tool_name == "read_skill" and not config.skill_paths:
                raise ValueError(
                    f"sub-agent {name!r} requests read_skill without skill_paths"
                )
    reader_lock = asyncio.Lock()

    class _DelegateRunState:
        def __init__(self) -> None:
            self.count = 0
            self.lock = asyncio.Lock()

    run_state: ContextVar[_DelegateRunState | None] = ContextVar(
        "roborsi_dynamic_delegate_state", default=None
    )

    class _DynamicDelegateRun(AbstractCapability):
        async def wrap_run(
            self, ctx: RunContext, *, handler: WrapRunHandler
        ) -> AgentRunResult:
            token = run_state.set(_DelegateRunState())
            try:
                return await handler()
            finally:
                run_state.reset(token)

    def _child_model(config: SubAgentConfig | None, requested: str | None) -> Model:
        if config is not None and config.llm is not None:
            if requested is not None:
                raise ValueError(
                    "a dynamic model override cannot be combined with child llm"
                )
            return RetryLoggingModel(
                config.llm.build_model(),
                policy=config.llm.retry,
                log_path=getattr(parent_model, "log_path", None),
            )
        model_name = requested or (config.model if config is not None else None)
        if model_name is None:
            return parent_model
        from roborsi.planner.base import build_api_model

        return RetryLoggingModel(
            build_api_model(model_name),
            policy=parent_model.policy
            if isinstance(parent_model, RetryLoggingModel)
            else None,
            log_path=getattr(parent_model, "log_path", None),
        )

    def _selected_skill_tool(names: tuple[str, ...]) -> Tool:
        def read_skill(name: str, resource: str = "SKILL.md") -> dict[str, Any]:
            if name not in names:
                return {
                    "name": name,
                    "resource": resource,
                    "error": {"code": "unknown_skill"},
                }
            return skill_catalog.read_skill(name, resource)

        read_skill.__name__ = "read_skill"
        read_skill.__doc__ = (
            "Read one resource from a skill selected for this child agent."
        )
        return Tool(read_skill)

    def _dynamic_config(
        agent_name: str,
        instructions: str | None,
        description: str | None,
        skills: list[str] | None,
        tools: list[str] | None,
    ) -> tuple[SubAgentConfig | None, str, list[Tool]]:
        config = runtime.subagents.get(agent_name)
        if config is not None:
            if any(
                value is not None
                for value in (instructions, description, skills, tools)
            ):
                raise ValueError(
                    f"configured sub-agent {agent_name!r} cannot be redefined in delegate_task"
                )
            for tool_name in config.tools:
                if tool_name not in _CHILD_READERS:
                    raise ValueError(
                        f"sub-agent {agent_name!r} cannot use tool {tool_name!r}; "
                        f"only {sorted(_CHILD_READERS)} are supported"
                    )
                if tool_name != "read_skill" and tool_name not in parent_tool_by_name:
                    raise ValueError(
                        f"sub-agent {agent_name!r} requests unavailable tool {tool_name!r}"
                    )
                if tool_name == "read_skill" and not config.skill_paths:
                    raise ValueError(
                        f"sub-agent {agent_name!r} requests read_skill without skill_paths"
                    )
            child_instructions = convert_planner_input(
                prompt=config.instructions,
                query="",
                skills=[load_skill(path) for path in config.skills],
            ).system_prompt
            child_tools, child_instructions = _skill_tools(
                config,
                [
                    parent_tool_by_name[name]
                    for name in config.tools
                    if name != "read_skill"
                ],
                child_instructions,
            )
            return config, child_instructions, child_tools
        if not _DYNAMIC_AGENT_NAME.fullmatch(agent_name):
            raise ValueError(
                "dynamic agent_name must start with a letter and contain only letters, digits, _ or -"
            )
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(
                "dynamic delegate_task requires non-empty instructions for a new agent"
            )
        selected_names = tuple(skills or ())
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("dynamic skills must be unique")
        missing_skills = [
            name for name in selected_names if name not in skill_catalog.entries
        ]
        if missing_skills:
            raise ValueError(f"unknown dynamic skills: {', '.join(missing_skills)}")
        selected_tools = tuple(tools or ())
        if len(selected_tools) != len(set(selected_tools)):
            raise ValueError("dynamic tools must be unique")
        invalid = [name for name in selected_tools if name not in _CHILD_READERS]
        if invalid:
            raise ValueError(
                f"dynamic child tools are restricted to {sorted(_CHILD_READERS)}; invalid: {invalid}"
            )
        unavailable = [
            name
            for name in selected_tools
            if name != "read_skill" and name not in parent_tool_by_name
        ]
        if unavailable:
            raise ValueError(f"dynamic child tools are unavailable: {unavailable}")
        child_tools = [
            parent_tool_by_name[name] for name in selected_tools if name != "read_skill"
        ]
        if "read_skill" in selected_tools:
            if not selected_names:
                raise ValueError(
                    "read_skill requires at least one selected dynamic skill"
                )
            child_tools.append(_selected_skill_tool(selected_names))
        child_instructions = instructions
        if selected_names:
            child_instructions = "\n\n".join(
                [
                    child_instructions,
                    "Selected skills (use read_skill to read them):\n"
                    + "\n".join(
                        f"- {name}: {skill_catalog.entries[name].description}"
                        for name in selected_names
                    ),
                ]
            )
        return None, child_instructions, child_tools

    async def delegate_task(
        ctx: RunContext,
        agent_name: str,
        task: str,
        instructions: str | None = None,
        description: str | None = None,
        model: str | None = None,
        skills: list[str] | None = None,
        tools: list[str] | None = None,
    ) -> str:
        """Run a configured or newly defined isolated child agent.

        Existing names use their configured definition. A new name requires
        ``instructions``; ``skills`` are names from the parent skill catalog and
        ``tools`` are restricted reader names. Arbitrary filesystem paths, robot
        tools and nested delegation are rejected.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("delegate_task task must be non-empty")
        state = run_state.get()
        if state is None:
            state = _DelegateRunState()
            run_state.set(state)
        async with state.lock:
            state.count += 1
            if state.count > runtime.max_dynamic_subagents:
                raise ValueError(
                    f"dynamic sub-agent limit exceeded ({runtime.max_dynamic_subagents} calls per run)"
                )
        config, child_instructions, child_tools = _dynamic_config(
            agent_name, instructions, description, skills, tools
        )
        child_model = _child_model(config, model)
        child_capabilities = (
            _context_capabilities(capabilities, config, runtime)
            if config is not None
            else _context_capabilities(capabilities, runtime)
        )
        child = Agent(
            child_model,
            name=agent_name,
            description=description or (config.description if config else None),
            instructions=child_instructions,
            model_settings=build_model_settings(child_model, max_tokens),
            toolsets=[_SharedToolset(child_tools, reader_lock)]
            if child_tools
            else None,
            capabilities=child_capabilities,
            retries={
                "output": config.output_retries if config else runtime.output_retries
            },
        )
        result = await child.run(
            task,
            deps=ctx.deps,
            usage=ctx.usage,
            usage_limits=ctx.usage_limits,
        )
        return str(result.output)

    return Tool(delegate_task), _DynamicDelegateRun()


class _InheritRequestLimit(AbstractCapability):
    """Forward the planner's request limit alongside the SDK's shared usage."""

    def __init__(self, limits: ContextVar[UsageLimits | None]) -> None:
        super().__init__()
        self._limits = limits

    async def wrap_run(
        self, ctx: RunContext, *, handler: WrapRunHandler
    ) -> AgentRunResult:
        parent_limits = self._limits.get()
        if parent_limits is not None:
            # SubAgents 0.34 shares usage but starts children with default limits.
            # Mutate the run-owned limit object also used by the SDK request loop.
            ctx.usage_limits.request_limit = parent_limits.request_limit
        token = self._limits.set(ctx.usage_limits)
        try:
            return await handler()
        finally:
            self._limits.reset(token)


class _SharedToolset(WrapperToolset):
    """Keep existing tool calls serial across otherwise concurrent agent runs."""

    def __init__(self, tools: Sequence[Tool], lock: asyncio.Lock) -> None:
        super().__init__(FunctionToolset(tools=tools))
        self._lock = lock

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        async with self._lock:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)


def build_runtime_agent(
    *,
    model: Model,
    system_prompt: str,
    tools: Sequence[Tool],
    max_tokens: int | None,
    capabilities: Sequence[AgentCapability],
    runtime: RuntimeConfig | None = None,
) -> Agent:
    """Build the existing parent agent with optional explicitly configured delegates.

    Child skill and model errors propagate before any model request. Child runs
    use the SDK's isolated histories, shared usage, and cancellation lifecycle.

    Args:
        model: Existing parent model, including request retry configuration.
        system_prompt: Already assembled parent instructions.
        tools: Existing planner tool wrappers.
        max_tokens: Output-token cap inherited by all configured agents.
        capabilities: Existing thinking and history-processing capabilities.
        runtime: Explicit delegates; omitted or empty retains a single agent.

    Returns:
        The parent Agent driven by the existing planner loop.

    Raises:
        ValueError: If a child requests a forbidden or unavailable tool, or the
            toolkit already contains the reserved delegation tool name.
        ImportError: If delegation is configured without the runtime extra.
    """
    if runtime is not None:
        tools, system_prompt = _skill_tools(runtime, tools, system_prompt)
        if any(tool.name == "delegate_task" for tool in tools):
            raise ValueError("delegate_task is reserved for runtime delegation")
        parent_capabilities = _context_capabilities(capabilities, runtime)
        if runtime.tool_names is not None:
            selected_names = frozenset(runtime.tool_names)

            def select_tools(ctx: RunContext, definitions: list) -> list:
                from roborsi.runtime.trace import current_trace

                missing = selected_names - {item.name for item in definitions}
                if missing:
                    raise ValueError(
                        f"selected tools are unavailable: {sorted(missing)}"
                    )
                selected = [item for item in definitions if item.name in selected_names]
                trace = current_trace()
                if trace is not None:
                    trace.emit(
                        "tool_selection",
                        {
                            "tools": [
                                {
                                    "name": item.name,
                                    "input_schema": item.parameters_json_schema,
                                }
                                for item in selected
                            ]
                        },
                    )
                return selected

            parent_capabilities.append(PrepareTools(select_tools))
        if runtime.dynamic_subagents:
            delegate_tool, delegate_capability = _dynamic_delegate_tool(
                runtime, model, tools, tuple(capabilities), max_tokens
            )
            tools.append(delegate_tool)
            parent_capabilities.append(delegate_capability)
            if runtime.subagents:
                configured = "\n".join(
                    f"- {name}: {config.description or config.instructions}"
                    for name, config in runtime.subagents.items()
                )
                system_prompt = "\n\n".join(
                    [
                        system_prompt,
                        "Configured sub-agents (you may also create a new one with delegate_task):\n"
                        + configured,
                    ]
                )
    else:
        parent_capabilities = list(capabilities)
    if runtime is None or runtime.dynamic_subagents or not runtime.subagents:
        return Agent(
            model,
            instructions=system_prompt or None,
            tools=tools,
            model_settings=build_model_settings(model, max_tokens),
            capabilities=parent_capabilities,
            retries={"output": runtime.output_retries if runtime is not None else 1},
        )

    try:
        from pydantic_ai_harness import SubAgent, SubAgents
    except ImportError as exc:
        raise ImportError(
            "Configured sub-agents require the runtime extra: pip install 'roborsi[runtime]'"
        ) from exc

    catalog = {tool.name: tool for tool in tools}
    if "delegate_task" in catalog:
        raise ValueError("delegate_task is reserved for configured sub-agents")
    lock = asyncio.Lock()
    limits: ContextVar[UsageLimits | None] = ContextVar(
        "roborsi_request_limits", default=None
    )
    run_capabilities = [*parent_capabilities, _InheritRequestLimit(limits)]
    children = []
    for name, config in runtime.subagents.items():
        for tool_name in config.tools:
            if tool_name not in _CHILD_READERS:
                raise ValueError(
                    f"sub-agent {name!r} cannot use tool {tool_name!r}; only artifact and text readers are supported"
                )
            if tool_name == "read_skill":
                if not config.skill_paths:
                    raise ValueError(
                        f"sub-agent {name!r} requests read_skill without skill_paths"
                    )
                continue
            if tool_name not in catalog:
                raise ValueError(
                    f"sub-agent {name!r} requests unavailable tool {tool_name!r}"
                )
        child_model = model
        if config.llm is not None:
            child_model = RetryLoggingModel(
                config.llm.build_model(),
                policy=config.llm.retry,
                log_path=model.log_path
                if isinstance(model, RetryLoggingModel)
                else None,
            )
        elif config.model is not None:
            from roborsi.planner.base import build_api_model

            child_model = RetryLoggingModel(
                build_api_model(config.model),
                policy=model.policy if isinstance(model, RetryLoggingModel) else None,
                log_path=model.log_path
                if isinstance(model, RetryLoggingModel)
                else None,
            )
        context = convert_planner_input(
            prompt=config.instructions,
            query="",
            skills=[load_skill(path) for path in config.skills],
        )
        child_tools, child_instructions = _skill_tools(
            config,
            [catalog[key] for key in config.tools if key != "read_skill"],
            context.system_prompt,
        )
        child_capabilities = [
            *_context_capabilities(capabilities, config, runtime),
            _InheritRequestLimit(limits),
        ]
        child = Agent(
            child_model,
            name=name,
            description=config.description,
            instructions=child_instructions,
            model_settings=build_model_settings(child_model, max_tokens),
            toolsets=[_SharedToolset(child_tools, lock)],
            capabilities=child_capabilities,
            retries={"output": config.output_retries},
        )
        children.append(SubAgent(child))

    return Agent(
        model,
        instructions=system_prompt or None,
        model_settings=build_model_settings(model, max_tokens),
        retries={"output": runtime.output_retries},
        toolsets=[_SharedToolset(tools, lock)],
        capabilities=[
            *run_capabilities,
            SubAgents(agents=children, agent_folders=None, inherit_tools=False),
        ],
    )
