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

"""Validated execution settings for a pinned, externally installed Codex CLI."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roborsi.planner.provider_capture import ProviderCaptureConfig

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CONFIG_KEY = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\Z")
_FLAGS = {
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "--full-auto",
    "--dangerously-bypass-approvals-and-sandbox",
    "--ephemeral",
    "--oss",
    "--search",
}
_VALUE_FLAGS = {
    "--sandbox",
    "-s",
    "--ask-for-approval",
    "-a",
    "--profile",
    "-p",
    "--add-dir",
    "--local-provider",
    "--enable",
    "--disable",
}


def _secret_name(value: str) -> bool:
    name = value.lower().replace("-", "_")
    return name in {
        "authorization",
        "password",
        "secret",
        "api_key",
        "token",
        "access_key",
        "bearer",
    } or name.endswith(("_api_key", "_token", "_password", "_secret", "_access_key"))


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"codex_cli.{name} must be nonempty text without NUL bytes")
    return value


def _scalar(value: Any) -> bool:
    return type(value) in {str, int, bool} or (
        type(value) is float and math.isfinite(value)
    )


def validate_capture_provider(config: ProviderCaptureConfig) -> None:
    """Require a custom provider whose capture endpoint Codex can override."""
    if config.model_provider in {
        "openai",
        "ollama",
        "lmstudio",
        "amazon-bedrock",
        "amazon-bedrock-runtime",
    }:
        raise ValueError(
            "provider_capture requires a custom model_provider; Codex built-in "
            "provider definitions can ignore endpoint overrides"
        )


@dataclass(frozen=True, slots=True)
class CodexCliConfig:
    """Control CLI identity, native policy and selected environment variables.

    Credentials belong in inherited environment variables or ``env_vars``.
    ``env`` contains only nonsecret values and can be persisted with a preset.
    Construction validates and defensively copies all mutable input mappings.
    """

    expected_version: str
    executable: str = "codex"
    expected_sha256: str | None = None
    cli_args: tuple[str, ...] = ()
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    inherit_env: bool = True
    env: Mapping[str, str] = field(default_factory=dict)
    env_vars: tuple[str, ...] = ()
    tools: str = "mcp"
    provider_capture: ProviderCaptureConfig | None = None

    def __post_init__(self) -> None:
        _text(self.expected_version, "expected_version")
        _text(self.executable, "executable")
        if self.expected_sha256 is not None:
            if not isinstance(self.expected_sha256, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", self.expected_sha256
            ):
                raise ValueError("codex_cli.expected_sha256 must contain 64 hex digits")
            object.__setattr__(self, "expected_sha256", self.expected_sha256.lower())
        if type(self.inherit_env) is not bool:
            raise ValueError("codex_cli.inherit_env must be a boolean")
        if not isinstance(self.tools, str) or self.tools not in {"mcp", "external"}:
            raise ValueError("codex_cli.tools must be mcp or external")
        if not isinstance(self.cli_args, (tuple, list)):
            raise ValueError("codex_cli.cli_args must be a sequence of arguments")
        args = tuple(_text(arg, "cli_args") for arg in self.cli_args)
        index = 0
        while index < len(args):
            option, equals, value = args[index].partition("=")
            if option in _FLAGS and not equals:
                index += 1
                continue
            if option not in _VALUE_FLAGS:
                raise ValueError(
                    f"unsupported or reserved codex_cli.cli_args option: {option}"
                )
            if equals:
                _text(value, "cli_args option value")
                index += 1
            else:
                index += 1
                if index >= len(args) or args[index].startswith("-"):
                    raise ValueError(
                        f"codex_cli.cli_args option requires a value: {option}"
                    )
                index += 1
        object.__setattr__(self, "cli_args", args)
        if not isinstance(self.config_overrides, Mapping):
            raise ValueError("codex_cli.config_overrides must be a mapping")
        overrides = {}
        for key, value in self.config_overrides.items():
            if not isinstance(key, str) or not _CONFIG_KEY.fullmatch(key):
                raise ValueError(
                    "codex_cli.config_overrides keys must be dotted TOML names"
                )
            if (
                key in {"model", "model_reasoning_effort"}
                or key == "mcp_servers"
                or key == "mcp_servers.roborsi"
                or key.startswith("mcp_servers.roborsi.")
            ):
                raise ValueError(f"reserved codex_cli.config_overrides key: {key}")
            environment_reference = (
                key.endswith(".env_key") or ".env_http_headers." in key
            )
            if environment_reference and (
                not isinstance(value, str) or not _ENV_NAME.fullmatch(value)
            ):
                raise ValueError(
                    "Codex credential references must name an environment variable"
                )
            if not environment_reference and (
                any(_secret_name(part) for part in key.split("."))
                or ".http_headers" in key
            ):
                raise ValueError(
                    "Codex credentials must come from named environment variables"
                )
            if isinstance(value, (tuple, list)):
                if not all(_scalar(item) for item in value):
                    raise ValueError(
                        "codex_cli.config_overrides lists must contain TOML scalars"
                    )
                value = tuple(value)
            elif not _scalar(value):
                raise ValueError(
                    "codex_cli.config_overrides values must be TOML scalars or scalar lists"
                )
            overrides[key] = value
        object.__setattr__(self, "config_overrides", MappingProxyType(overrides))
        if not isinstance(self.env, Mapping):
            raise ValueError("codex_cli.env must be a mapping")
        environment = {}
        for name, value in self.env.items():
            if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
                raise ValueError(
                    "codex_cli.env keys must be environment variable names"
                )
            if _secret_name(name):
                raise ValueError(
                    "Codex credentials must use env_vars, not literal env values"
                )
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("codex_cli.env values must be text without NUL bytes")
            environment[name] = value
        object.__setattr__(self, "env", MappingProxyType(environment))
        if not isinstance(self.env_vars, (tuple, list)) or any(
            not isinstance(name, str) or not _ENV_NAME.fullmatch(name)
            for name in self.env_vars
        ):
            raise ValueError("codex_cli.env_vars must be environment variable names")
        if len(set(self.env_vars)) != len(self.env_vars):
            raise ValueError("codex_cli.env_vars must be unique")
        if set(self.env_vars) & set(environment):
            raise ValueError(
                "codex_cli.env and env_vars cannot define the same variable"
            )
        object.__setattr__(self, "env_vars", tuple(self.env_vars))
        if self.provider_capture is not None:
            from roborsi.planner.provider_capture import ProviderCaptureConfig

            capture = self.provider_capture
            if isinstance(capture, Mapping):
                capture = ProviderCaptureConfig.from_dict(capture)
            if not isinstance(capture, ProviderCaptureConfig):
                raise ValueError(
                    "codex_cli.provider_capture must be a capture configuration"
                )
            validate_capture_provider(capture)
            selected = overrides.get("model_provider")
            if selected is not None and selected != capture.model_provider:
                raise ValueError(
                    "provider_capture must use the selected model_provider"
                )
            upstream = overrides.get(
                f"model_providers.{capture.model_provider}.base_url"
            )
            if upstream is not None and (
                not isinstance(upstream, str)
                or upstream.rstrip("/") != capture.upstream_base_url.rstrip("/")
            ):
                raise ValueError(
                    "provider_capture must use the configured upstream_base_url"
                )
            object.__setattr__(self, "provider_capture", capture)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CodexCliConfig:
        """Validate YAML settings without importing the CLI or model SDK."""
        if not isinstance(value, Mapping):
            raise ValueError("codex_cli must be a mapping")
        if set(value) - {item.name for item in fields(cls)}:
            raise ValueError("unknown codex_cli settings")
        if "expected_version" not in value:
            raise ValueError("codex_cli.expected_version is required")
        return cls(**value)

    def subprocess_env(self) -> dict[str, str]:
        """Resolve selected environment values only when preparing a subprocess."""
        result = dict(os.environ) if self.inherit_env else {}
        for name in self.env_vars:
            if name not in os.environ:
                raise ValueError(
                    f"required Codex environment variable is missing: {name}"
                )
            result[name] = os.environ[name]
        result.update(self.env)
        return result
