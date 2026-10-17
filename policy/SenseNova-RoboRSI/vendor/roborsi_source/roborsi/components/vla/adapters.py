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

"""Explicit Agent tool and packaged Skill adapters for a VLA component."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import VLAConfigError
from .types import VLA, VLAPrediction

if TYPE_CHECKING:
    from roborsi.embodied_agent import LocalToolSpec


def make_vla_tool(
    component: VLA,
    *,
    observation_getter: Callable[[], Mapping[str, Any]],
    result_serializer: Callable[[VLAPrediction], Mapping[str, Any]],
    name: str = "vla_predict",
    options_schema: Mapping[str, Any] | None = None,
) -> LocalToolSpec:
    """Wrap inference using a fresh observation and caller-selected output.

    The caller retains ownership of the component, observations and any action
    queue. This tool does not reset the provider or execute predicted actions.
    """
    from jsonschema import Draft202012Validator

    from roborsi.embodied_agent import LocalToolSpec
    from roborsi.tools.toolkit import ToolResult
    from roborsi.utils.config_values import json_value

    if not callable(observation_getter) or not callable(result_serializer):
        raise TypeError("observation_getter and result_serializer must be callable")
    properties: dict[str, Any] = {"instruction": {"type": "string"}}
    if options_schema is not None:
        if not isinstance(options_schema, Mapping):
            raise TypeError("options_schema must be a JSON schema mapping")
        option_definition = copy.deepcopy(dict(options_schema))
        json_value(option_definition)
        Draft202012Validator.check_schema(option_definition)
        properties["options"] = {"allOf": [{"type": "object"}, option_definition]}
    schema = {"type": "object", "properties": properties, "additionalProperties": False}
    validator = Draft202012Validator(schema)

    def handler(arguments: dict[str, Any]) -> ToolResult:
        try:
            json_value(arguments)
        except ValueError:
            return ToolResult(
                name, {"error": "arguments must contain finite JSON", "retryable": True}
            )
        if not validator.is_valid(arguments):
            return ToolResult(
                name,
                {"error": "arguments do not match the tool schema", "retryable": True},
            )
        observation = observation_getter()
        try:
            prediction = component.predict(
                observation,
                instruction=arguments.get("instruction"),
                options=arguments.get("options"),
            )
        except VLAConfigError as exc:
            return ToolResult(name, {"error": str(exc), "retryable": True})
        result = result_serializer(prediction)
        if not isinstance(result, Mapping):
            raise ValueError("result_serializer must return a finite JSON object")
        result = dict(result)
        json_value(result)
        return ToolResult(name, result)

    return LocalToolSpec(
        name=name,
        description="Predict actions from the current observation without executing them.",
        input_schema=copy.deepcopy(schema),
        handler=handler,
        propagate_errors=True,
    )


def write_vla_skill(destination: Path, *, tool_name: str = "vla_predict") -> Path:
    """Write the installed Skill template to a new destination/SKILL.md file."""
    from roborsi.embodied_agent import LocalToolSpec

    # Use the existing tool-name boundary, including the reserved finish name.
    LocalToolSpec(tool_name, "", {"type": "object"})
    template = (
        files("roborsi.components.vla")
        .joinpath("skills/vla/SKILL.md")
        .read_text(encoding="utf-8")
    )
    destination = Path(destination)
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("Skill destination cannot contain symbolic links")
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / "SKILL.md"
    owned_file = False
    try:
        with target.open("x", encoding="utf-8") as stream:
            owned_file = True
            stream.write(template.replace("{{tool_name}}", tool_name))
    except Exception as write_error:
        try:
            if owned_file:
                target.unlink(missing_ok=True)
            destination.rmdir()
        except OSError as cleanup_error:
            raise write_error from cleanup_error
        raise
    return target
