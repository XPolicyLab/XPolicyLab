"""Model-independent tool schemas and results over the same Operations as skills."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from PhysicalRSI_core.contracts import Context, Operation


@dataclass(frozen=True)
class ToolReply:
    data: Any = None
    error: str | None = None
    # Transport adapters decide how to serialize text and optional media.
    media: tuple = ()


@dataclass(frozen=True)
class ToolBinding:
    operation: Operation
    description: str
    schema: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"value": {}},
            "additionalProperties": False,
        }
    )

    def __post_init__(self):
        Draft202012Validator.check_schema(self.schema)


class ToolGateway:
    def __init__(self, bindings, *, execution=None):
        bindings = tuple(bindings)
        self.bindings = {binding.operation.name: binding for binding in bindings}
        if len(self.bindings) != len(bindings):
            raise ValueError("Duplicate tool name")
        self.execution = execution

    def specifications(self):
        return [
            dict(name=name, description=binding.description, inputSchema=binding.schema)
            for name, binding in self.bindings.items()
        ]

    def call(self, name, arguments, *, context=None):
        try:
            binding = self.bindings[name]
            Draft202012Validator(binding.schema).validate(arguments)
            context = context or Context(
                uuid.uuid4().hex,
                execution=self.execution,
                harness_revision=binding.operation.revision,
            )
            result = binding.operation(arguments.get("value"), context)
            return result if isinstance(result, ToolReply) else ToolReply(data=result)
        except Exception as error:
            return ToolReply(error=type(error).__name__ + ": " + str(error))
