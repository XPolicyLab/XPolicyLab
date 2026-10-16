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

"""Explicit sub-agent configuration for the API planner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roborsi.llm.client import LLMConfig
    from roborsi.runtime.context_engine import ContextEngine, ContextPolicy
    from roborsi.runtime.response import ResponseProcessor
    from roborsi.runtime.trace import TraceConfig


def _llm_config(value: Any) -> LLMConfig | None:
    if value is None:
        return None
    from roborsi.llm.client import LLMConfig
    from roborsi.llm.retry import RetryPolicy

    if isinstance(value, LLMConfig):
        if not isinstance(value.retry, RetryPolicy):
            raise ValueError("llm retry must be a RetryPolicy")
        return value
    if isinstance(value, Mapping):
        values = dict(value)
        try:
            if not isinstance(values.get("model"), str):
                raise ValueError("model must be non-empty text")
            if isinstance(values.get("retry"), Mapping):
                values["retry"] = RetryPolicy(**values["retry"])
            elif "retry" in values and not isinstance(values["retry"], RetryPolicy):
                raise ValueError("retry must be a RetryPolicy or mapping")
            return LLMConfig(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid llm configuration: {exc}") from exc
    raise ValueError("llm must be an LLMConfig or mapping")


def _paths(value: Any, field_name: str) -> tuple[Path, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(path, (str, Path)) or not str(path).strip() for path in value
    ):
        raise ValueError(f"{field_name} must be a sequence of paths")
    return tuple(Path(path) for path in value)


def _context_config(config: Any) -> None:
    if type(config.output_retries) is not int or config.output_retries < 0:
        raise ValueError("output_retries must be a non-negative integer")
    if config.context_engine is not None and config.context_engine_factory is not None:
        raise ValueError(
            "context_engine and context_engine_factory are mutually exclusive"
        )
    if config.context is not None:
        from roborsi.runtime.context_engine import ContextPolicy

        policy = config.context
        if isinstance(policy, Mapping):
            try:
                policy = ContextPolicy(**policy)
            except TypeError as exc:
                raise ValueError(f"invalid context policy: {exc}") from exc
        if not isinstance(policy, ContextPolicy):
            raise ValueError("context must be a ContextPolicy or mapping")
        if (
            config.context_engine is not None
            or config.context_engine_factory is not None
        ):
            raise ValueError(
                "context policy and custom context engine are mutually exclusive"
            )
        object.__setattr__(config, "context", policy)
    for name in ("context_engine", "context_engine_factory"):
        value = getattr(config, name)
        if value is not None and not callable(value):
            raise ValueError(f"{name} must be callable")
    object.__setattr__(config, "skill_paths", _paths(config.skill_paths, "skill_paths"))
    if (
        isinstance(config.skill_max_bytes, bool)
        or not isinstance(config.skill_max_bytes, int)
        or config.skill_max_bytes < 1
    ):
        raise ValueError("skill_max_bytes must be a positive integer")
    llm = _llm_config(config.llm)
    if (
        llm is not None
        and (llm.image_history_groups is not None or llm.preserve_initial_image_count)
        and (
            config.context_engine is not None
            or config.context_engine_factory is not None
        )
    ):
        raise ValueError(
            "image history settings must be handled by the custom context_engine"
        )
    if (
        llm is not None
        and (llm.image_history_groups is not None or llm.preserve_initial_image_count)
        and config.context is not None
        and (
            config.context.image_history_groups is not None
            or config.context.preserve_initial_image_count
        )
    ):
        raise ValueError("configure image history in context or LLMConfig, not both")
    object.__setattr__(config, "llm", llm)


def _default_trace() -> TraceConfig:
    from roborsi.runtime.trace import TraceConfig

    return TraceConfig()


@dataclass(frozen=True, slots=True)
class SubAgentConfig:
    """Instructions, model and selected readers for one named delegate.

    An omitted model reuses the parent's configured model, including its
    endpoint and retries. An explicit provider-prefixed model uses that
    provider's environment configuration. Skill files are read per episode.
    """

    instructions: str
    description: str | None = None
    model: str | None = None
    llm: LLMConfig | None = None
    skills: tuple[str | Path, ...] = ()
    tools: tuple[str, ...] = ()
    skill_paths: tuple[str | Path, ...] = ()
    skill_max_bytes: int = 262144
    context: ContextPolicy | None = None
    context_engine: ContextEngine | None = None
    context_engine_factory: Callable[[], ContextEngine] | None = None
    output_retries: int = 1

    def __post_init__(self) -> None:
        _context_config(self)
        if self.model is not None and self.llm is not None:
            raise ValueError("sub-agent model and llm are mutually exclusive")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("sub-agent instructions must be non-empty text")
        for name, value in (("description", self.description), ("model", self.model)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"sub-agent {name} must be non-empty text")
        if not isinstance(self.skills, (list, tuple)) or any(
            not isinstance(path, (str, Path)) or not str(path).strip()
            for path in self.skills
        ):
            raise ValueError("sub-agent skills must be a sequence of file paths")
        if not isinstance(self.tools, (list, tuple)) or any(
            not isinstance(name, str) or not name.strip() for name in self.tools
        ):
            raise ValueError("sub-agent tools must be a sequence of tool names")
        if len(self.tools) != len(set(self.tools)):
            raise ValueError("sub-agent tool names must be unique")
        object.__setattr__(self, "skills", tuple(Path(path) for path in self.skills))
        object.__setattr__(self, "tools", tuple(self.tools))


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Context, skills, model and dynamically delegable agents for one API planner."""

    subagents: Mapping[str, SubAgentConfig] = field(default_factory=dict)
    llm: LLMConfig | None = None
    skill_paths: tuple[str | Path, ...] = ()
    skill_max_bytes: int = 262144
    context: ContextPolicy | None = None
    context_engine: ContextEngine | None = None
    context_engine_factory: Callable[[], ContextEngine] | None = None
    trace: TraceConfig = field(default_factory=_default_trace)
    output_retries: int = 1
    dynamic_subagents: bool = False
    max_dynamic_subagents: int = 16
    response_processor: ResponseProcessor | None = None
    tool_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _context_config(self)
        if self.response_processor is not None and not callable(
            self.response_processor
        ):
            raise ValueError("response_processor must be callable")
        if self.tool_names is not None:
            if not isinstance(self.tool_names, (list, tuple)) or any(
                not isinstance(name, str) or not name.strip()
                for name in self.tool_names
            ):
                raise ValueError("tool_names must be a sequence of nonempty names")
            if len(set(self.tool_names)) != len(self.tool_names):
                raise ValueError("tool_names must be unique")
            object.__setattr__(self, "tool_names", tuple(self.tool_names))
        from roborsi.runtime.trace import TraceConfig

        if isinstance(self.trace, Mapping):
            object.__setattr__(self, "trace", TraceConfig(**self.trace))
        if not isinstance(self.trace, TraceConfig):
            raise ValueError("trace must be a TraceConfig or mapping")
        if not isinstance(self.subagents, Mapping):
            raise ValueError("subagents must map names to SubAgentConfig values")
        if not isinstance(self.dynamic_subagents, bool):
            raise ValueError("dynamic_subagents must be a boolean")
        if (
            isinstance(self.max_dynamic_subagents, bool)
            or not isinstance(self.max_dynamic_subagents, int)
            or self.max_dynamic_subagents < 1
        ):
            raise ValueError("max_dynamic_subagents must be a positive integer")
        for name, config in self.subagents.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(
                    "sub-agent names must be non-empty without surrounding whitespace"
                )
            if not isinstance(config, SubAgentConfig):
                raise ValueError(f"sub-agent {name!r} requires a SubAgentConfig")
        object.__setattr__(self, "subagents", dict(self.subagents))

    def validate_resources(self) -> None:
        """Validate explicit skill resources before starting external resources.

        Catalogs are rebuilt when agents are assembled, so resource content is
        still fresh for each episode. This check does not create provider clients.
        """
        from roborsi.runtime.skills import SkillCatalog, load_skill

        SkillCatalog(self.skill_paths, max_bytes=self.skill_max_bytes)
        for config in self.subagents.values():
            SkillCatalog(config.skill_paths, max_bytes=config.skill_max_bytes)
            for path in config.skills:
                load_skill(path)

    @classmethod
    def from_file(cls, path: str | Path) -> RuntimeConfig:
        """Load YAML or JSON, resolving skill paths against the config directory.

        File errors propagate to the caller.

        Args:
            path: Configuration file containing a ``subagents`` mapping.

        Returns:
            Validated configuration with absolute skill paths.

        Raises:
            ValueError: If configuration fields or values are invalid.
        """
        import yaml

        path = Path(path).resolve()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid runtime config {path}: {exc}") from exc
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, base_dir: str | Path
    ) -> RuntimeConfig:
        """Validate runtime settings and resolve skill paths against ``base_dir``.

        This is the same loader used by :meth:`from_file`, for applications
        embedding runtime settings in a larger configuration document.
        """
        directory = Path(base_dir).resolve()
        allowed = {
            "tool_names",
            "subagents",
            "dynamic_subagents",
            "max_dynamic_subagents",
            "llm",
            "context",
            "skill_paths",
            "skill_max_bytes",
            "trace",
            "output_retries",
        }
        if not isinstance(data, Mapping) or set(data) - allowed:
            raise ValueError("runtime config must be a mapping with known fields")
        definitions = data.get("subagents", {})
        if not isinstance(definitions, Mapping):
            raise ValueError("subagents must be a mapping")
        agents = {}
        for name, values in definitions.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"sub-agent {name!r} must be a mapping")
            try:
                config = SubAgentConfig(**values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid sub-agent {name!r}: {exc}") from exc
            agents[name] = replace(
                config,
                skills=tuple((directory / skill).resolve() for skill in config.skills),
                skill_paths=tuple(
                    (directory / skill).resolve() for skill in config.skill_paths
                ),
            )
        values = {key: value for key, value in data.items() if key != "subagents"}
        try:
            config = cls(subagents=agents, **values)
            config = replace(
                config,
                skill_paths=tuple(
                    (directory / skill).resolve() for skill in config.skill_paths
                ),
            )
            config.validate_resources()
            return config
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid runtime config: {exc}") from exc
