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

"""Validated, relocatable Agent presets with explicit code and resource inputs."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roborsi.agents.composition import compose
from roborsi.agents.composition import json_value as _json
from roborsi.agents.composition import reject_secrets as _reject_secrets

if TYPE_CHECKING:
    from roborsi.embodied_agent import EmbodiedAgent, LocalToolSpec, McpServer
    from roborsi.runtime.config import RuntimeConfig
    from roborsi.runtime.context_engine import ContextEngine
    from roborsi.runtime.response import ResponseProcessor

_DEFAULT_FACTORY = "roborsi.agents.config:make_embodied_agent"
_REFERENCE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")
_BUNDLE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_SKIP = {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache"}


def bundle_package(reference: str) -> str:
    """Validate an installed preset reference and return its package name."""
    prefix, separator, name = reference.partition(":")
    if prefix != "bundle" or not separator or not _BUNDLE_NAME.fullmatch(name):
        raise ValueError("installed Agent config must be bundle:lowercase_name")
    return f"roborsi_agent_bundle_{name}"


def _paths(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of nonempty paths")
    return value


def _files(path: Path, root: Path) -> list[Path]:
    if not path.is_relative_to(root):
        raise ValueError(f"Agent input is outside source_root: {path}")
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError(f"Agent inputs cannot contain symlinks: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Agent input is missing: {path}")
    result = []
    for child in sorted(path.iterdir()):
        if child.name in _SKIP or child.suffix in {".pyc", ".pyo"}:
            continue
        result.extend(_files(child, root))
    if not result:
        raise ValueError(f"Agent input directories must contain files: {path}")
    return result


@dataclass(frozen=True)
class AgentConfig:
    """One preset and the content identity of its declared inputs.

    Factories receive ``make_agent(case_dir, *, config=...)``. They can use
    :meth:`embodied_agent` to reuse the common settings with their own tools
    and context factory. Loading a preset never imports its custom factory.
    """

    path: Path
    source_root: Path
    document_json: str
    input_files: tuple[Path, ...]
    fingerprint: str
    overrides: tuple[str, ...] = ()

    @classmethod
    def from_file(
        cls, path: str | Path, *, overrides: Sequence[str] = ()
    ) -> AgentConfig:
        """Compose YAML/JSON layers, apply overrides and pin declared resources."""
        selected = Path(os.path.abspath(path))
        data, config_files, overrides = compose(selected, overrides)
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("Agent config requires schema_version: 1")
        _json(data)
        _reject_secrets(data)
        factory = data.get("factory", _DEFAULT_FACTORY)
        if not isinstance(factory, str) or not _REFERENCE.fullmatch(factory):
            raise ValueError("Agent factory must be module:function")
        for field in ("agent", "runtime", "parameters", "resources"):
            if not isinstance(data.get(field, {}), dict):
                raise ValueError(f"{field} must be a mapping")
        if factory == _DEFAULT_FACTORY and data.get("parameters"):
            raise ValueError("parameters require a custom Agent factory")
        root_value = data.get("source_root", ".")
        if not isinstance(root_value, str) or not root_value.strip():
            raise ValueError("source_root must be a nonempty path")
        root = (selected.parent / root_value).resolve()
        options = data.get("agent", {})
        allowed = {
            "planner",
            "model",
            "max_turns",
            "max_tokens",
            "planner_timeout_s",
            "action_timeout_s",
            "reasoning_effort",
            "system_prompt",
            "codex_cli",
            "provider_capture",
            "working_dir",
        }
        if set(options) - allowed:
            raise ValueError("unknown Agent constructor settings")
        planner = options.get("planner", "api")
        if not isinstance(planner, str) or planner not in {
            "api",
            "codex",
            "claude_code",
            "codex_cli",
        }:
            raise ValueError(
                "agent.planner must be api, codex, codex_cli or claude_code"
            )
        capture = None
        if "provider_capture" in options:
            if planner not in {"codex", "codex_cli"}:
                raise ValueError("agent.provider_capture requires codex or codex_cli")
            if options["provider_capture"] is not None:
                from roborsi.planner.codex_cli_config import validate_capture_provider
                from roborsi.planner.provider_capture import ProviderCaptureConfig

                capture = ProviderCaptureConfig.from_dict(options["provider_capture"])
                validate_capture_provider(capture)
        if planner == "codex_cli":
            from roborsi.planner.codex_cli_config import CodexCliConfig

            cli = CodexCliConfig.from_dict(options.get("codex_cli", {}))
            if capture is not None:
                if cli.provider_capture is not None:
                    raise ValueError(
                        "provider_capture cannot be set in both agent and codex_cli"
                    )
                # Check the selected provider before any factory or model starts.
                replace(cli, provider_capture=capture)
            if cli.tools == "external" and factory == _DEFAULT_FACTORY:
                raise ValueError(
                    "codex_cli.tools=external requires a custom Agent factory"
                )
            options.setdefault("max_turns", None)
            options.setdefault("max_tokens", None)
            if options["max_turns"] is not None:
                raise ValueError(
                    "codex_cli requires max_turns: null; CLI turns cannot be capped"
                )
            if options.get("reasoning_effort", "none") not in (
                "none",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            ):
                raise ValueError("unsupported codex_cli reasoning_effort")
        elif "codex_cli" in options:
            raise ValueError("agent.codex_cli requires the codex_cli planner")
        if options.get("working_dir") is not None and (
            not isinstance(options["working_dir"], str)
            or not options["working_dir"].strip()
        ):
            raise ValueError("agent.working_dir must be a nonempty path")
        model = options.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("agent.model must be nonempty text")
        if planner != "api":
            if data.get("runtime"):
                raise ValueError("runtime settings require the api planner")
            if options.get("max_tokens") is not None:
                raise ValueError("agent.max_tokens requires the api planner")
        if model is not None and data.get("runtime", {}).get("llm") is not None:
            raise ValueError("pass runtime.llm or agent.model, not both")
        for name in (
            "max_turns",
            "max_tokens",
            "planner_timeout_s",
            "action_timeout_s",
        ):
            if name not in options:
                continue
            value = options[name]
            if value is None and name in {"max_tokens", "planner_timeout_s"}:
                continue
            if planner == "codex_cli" and (
                value is None
                and name in {"max_turns", "action_timeout_s"}
                or type(value) is int
                and value == 0
                and name == "planner_timeout_s"
            ):
                continue
            numeric = (int, float) if name == "action_timeout_s" else (int,)
            if (
                type(value) not in numeric
                or (isinstance(value, float) and not math.isfinite(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        for name in ("reasoning_effort", "system_prompt"):
            if name in options and (
                not isinstance(options[name], str) or not options[name].strip()
            ):
                raise ValueError(f"{name} must be nonempty text")
        if data.get("prompt_file") is not None and "system_prompt" in options:
            raise ValueError("choose prompt_file or agent.system_prompt")
        paths = list(config_files)

        def resolve(value: str) -> Path:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Agent resource paths must be nonempty text")
            # Preserve symlink components until _files has checked them.
            return Path(os.path.abspath(selected.parent / value))

        for field in ("code", "skills"):
            for name in _paths(data.get(field, []), field):
                path = resolve(name)
                if field == "skills" and not path.is_file():
                    raise ValueError("skills must name Markdown files")
                paths.append(path)
        if data.get("prompt_file") is not None:
            prompt = resolve(data["prompt_file"])
            if not prompt.is_file() or not prompt.read_text().strip():
                raise ValueError("prompt_file must contain nonempty text")
            paths.append(prompt)
        paths.extend(resolve(value) for value in data.get("resources", {}).values())
        runtime = data.get("runtime", {})
        from roborsi.runtime.config import RuntimeConfig

        RuntimeConfig.from_dict(runtime, base_dir=selected.parent)
        for name in _paths(runtime.get("skill_paths", []), "runtime.skill_paths"):
            paths.append(resolve(name))
        for subagent in runtime.get("subagents", {}).values():
            for field in ("skills", "skill_paths"):
                paths.extend(
                    resolve(name) for name in _paths(subagent.get(field, []), field)
                )
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError("dependencies must be a list of requirement strings")
        from packaging.requirements import InvalidRequirement, Requirement

        for requirement in dependencies:
            try:
                Requirement(requirement)
            except (InvalidRequirement, TypeError) as exc:
                raise ValueError("invalid Agent dependency") from exc
        selected_files = {item for path in paths for item in _files(path, root)}
        for file in tuple(selected_files):
            for parent in file.parents:
                if parent == root:
                    break
                initializer = parent / "__init__.py"
                if initializer.is_file():
                    selected_files.update(_files(initializer, root))
        files = tuple(sorted(selected_files))
        identity = [
            [
                file.relative_to(root).as_posix(),
                hashlib.sha256(file.read_bytes()).hexdigest(),
            ]
            for file in files
        ]
        fingerprint = hashlib.sha256(
            _json(
                {"files": identity, "overrides": list(overrides)}
                if overrides
                else identity
            ).encode()
        ).hexdigest()
        return cls(selected, root, _json(data), files, fingerprint, overrides)

    @classmethod
    def load(
        cls, reference: str | Path, *, overrides: Sequence[str] = ()
    ) -> AgentConfig:
        """Load a file or an installed ``bundle:name`` and verify bundle contents."""
        if not str(reference).startswith("bundle:"):
            return cls.from_file(reference, overrides=overrides)
        from importlib.resources import files

        package = bundle_package(str(reference))
        directory = Path(str(files(package)))
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("name") != str(reference).removeprefix("bundle:")
        ):
            raise ValueError("invalid Agent bundle manifest")
        relative = manifest.get("config")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("invalid Agent bundle config path")
        config = cls.from_file(
            directory / "payload" / relative, overrides=manifest.get("overrides", ())
        )
        if (
            config.source_root != directory / "payload"
            or config.fingerprint != manifest.get("config_sha256")
        ):
            raise ValueError("installed Agent bundle contents changed")
        if overrides:
            selected = cls.from_file(
                config.path, overrides=(*config.overrides, *overrides)
            )
            if selected.source_root != config.source_root:
                raise ValueError("installed Agent overrides cannot change source_root")
            return selected
        return config

    @property
    def document(self) -> dict[str, Any]:
        """Return a fresh copy of the preset without deployment-specific paths."""
        return json.loads(self.document_json)

    @property
    def factory(self) -> str:
        return self.document.get("factory", _DEFAULT_FACTORY)

    @property
    def parameters(self) -> dict[str, Any]:
        """Custom policy parameters; the factory validates their own schema."""
        return self.document.get("parameters", {})

    @property
    def resources(self) -> dict[str, Path]:
        """Explicit custom resources, resolved relative to the preset file."""
        return {
            name: (self.path.parent / path).resolve()
            for name, path in self.document.get("resources", {}).items()
        }

    def runtime(self) -> RuntimeConfig:
        """Build fresh runtime settings for this case, using the shared loader."""
        from roborsi.runtime.config import RuntimeConfig

        return RuntimeConfig.from_dict(
            self.document.get("runtime", {}), base_dir=self.path.parent
        )

    def verify(self) -> None:
        """Reject changes to the selected preset, code or resources."""
        if (
            self.from_file(self.path, overrides=self.overrides).fingerprint
            != self.fingerprint
        ):
            raise ValueError("Agent config inputs changed")

    def create_agent(self, case_dir: Path) -> Any:
        """Create one Agent and record the actual selected configuration."""
        from roborsi.evaluation.result import write_json_atomic
        from roborsi.evaluation.service_host import load_factory

        self.verify()
        root = str(self.source_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        factory = load_factory(self.factory)
        module = importlib.import_module(factory.__module__)
        location = getattr(module, "__file__", None)
        if not self.factory.startswith("roborsi.") and (
            location is None or Path(location).resolve() not in self.input_files
        ):
            raise ValueError("custom Agent factory must be included in config.code")
        directory = Path(case_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError("Agent case directory must exist")
        write_json_atomic(
            directory / "agent-config.json",
            {
                "schema_version": 1,
                "config_sha256": self.fingerprint,
                "factory": self.factory,
                "configuration": self.document,
                "overrides": list(self.overrides),
                "files": [
                    file.relative_to(self.source_root).as_posix()
                    for file in self.input_files
                ],
            },
        )
        return factory(directory, config=self)

    def embodied_agent(
        self,
        case_dir: Path,
        *,
        local_tools: Sequence[LocalToolSpec] = (),
        mcp_servers: Sequence[McpServer] = (),
        context_engine_factory: Callable[[], ContextEngine] | None = None,
        response_processor: ResponseProcessor | None = None,
        working_dir: str | Path | None = None,
    ) -> EmbodiedAgent:
        """Apply common settings to EmbodiedAgent with factory-owned extensions."""
        from roborsi.embodied_agent import EmbodiedAgent

        options = self.document.get("agent", {})
        if options.get("provider_capture") is not None:
            from roborsi.planner.provider_capture import ProviderCaptureConfig

            options["provider_capture"] = ProviderCaptureConfig.from_dict(
                options["provider_capture"]
            )
        if options.get("codex_cli") is not None:
            from roborsi.planner.codex_cli_config import CodexCliConfig

            options["codex_cli"] = CodexCliConfig.from_dict(options["codex_cli"])
        if working_dir is not None:
            options["working_dir"] = Path(working_dir).resolve()
        elif options.get("working_dir") is not None:
            options["working_dir"] = (
                self.path.parent / options["working_dir"]
            ).resolve()
        if self.document.get("prompt_file") is not None:
            options["system_prompt"] = (
                self.path.parent / self.document["prompt_file"]
            ).read_text()
        api = options.get("planner", "api") == "api"
        runtime = self.runtime() if api else None
        if not api:
            options.setdefault("max_tokens", None)
        if context_engine_factory is not None or response_processor is not None:
            if not api:
                raise ValueError(
                    "context and response extensions require the api planner"
                )
            runtime = replace(
                runtime,
                context_engine_factory=context_engine_factory,
                response_processor=response_processor,
            )
        return EmbodiedAgent(
            output_dir=case_dir,
            local_tools=local_tools,
            mcp_servers=mcp_servers,
            runtime=runtime,
            evaluation_skills=tuple(
                (self.path.parent / path).resolve()
                for path in self.document.get("skills", [])
            ),
            **options,
        )


def make_embodied_agent(case_dir: Path, *, config: AgentConfig) -> EmbodiedAgent:
    """Default preset factory using RoboRSI's native-action EmbodiedAgent bridge."""
    return config.embodied_agent(case_dir)
