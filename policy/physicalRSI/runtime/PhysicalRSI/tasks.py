"""Versioned task-package boundary, shared by CLI, conversation and Python."""

import importlib
import inspect
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest, identifier, read_json

SCHEMA = "physicalrsi.task/v1"


def configuration_schema(binding):
    factory = _factory(binding["specification"]["entrypoint"])
    source = inspect.getsourcefile(factory)
    if source is None or file_digest(Path(source)) != binding["adapter_sha256"]:
        raise ValueError("Task adapter changed; start a new draft")
    return deepcopy(getattr(factory, "configuration_schema", None))


def merge_configuration(current, patch):
    """Update declared configuration fields; arrays are replaced as a whole."""
    if not isinstance(patch, dict):
        raise ValueError("Configuration patch must be a JSON object")
    result = deepcopy(current)
    for key, value in patch.items():
        if key not in current:
            raise ValueError("Unknown configuration field: " + key)
        if isinstance(current[key], dict):
            result[key] = merge_configuration(current[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _factory(entrypoint):
    module_name, factory_name = entrypoint.split(":")
    try:
        return getattr(importlib.import_module(module_name), factory_name)
    except (ImportError, AttributeError) as error:
        raise ValueError("Task adapter is unavailable: " + entrypoint) from error


def load_spec(path):
    """Load a trusted installed task adapter; this is executable Python configuration."""
    path = Path(path).expanduser().resolve()
    spec = read_json(path)
    if (
        set(spec) != {"schema", "name", "entrypoint", "configuration"}
        or spec["schema"] != SCHEMA
    ):
        raise ValueError("Expected a physicalrsi.task/v1 package specification")
    identifier(spec["name"])
    if not isinstance(spec["configuration"], dict):
        raise ValueError("Task configuration must be an object")
    module_name, separator, factory_name = spec["entrypoint"].partition(":")
    if not separator or not module_name or not factory_name.isidentifier():
        raise ValueError("Task entrypoint must be installed.module:factory")
    factory = _factory(spec["entrypoint"])
    if not callable(factory):
        raise ValueError("Task entrypoint is not callable")
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError("Task adapter must expose its implementation source")
    return dict(
        specification=spec,
        specification_sha256=digest(spec),
        adapter_sha256=file_digest(Path(source)),
        source_directory=str(path.parent),
    )


class TaskSession:
    def __init__(self, binding, workspace, *, language_model=None, rsi_plan=None, harness=None):
        spec = binding["specification"]
        if digest(spec) != binding["specification_sha256"]:
            raise ValueError("Task specification changed")
        factory = _factory(spec["entrypoint"])
        source = inspect.getsourcefile(factory)
        if source is None or file_digest(Path(source)) != binding["adapter_sha256"]:
            raise ValueError(
                "Task adapter changed; load the task package explicitly again"
            )
        schema = configuration_schema(binding)
        if schema is not None:
            from jsonschema import validate, ValidationError

            try:
                validate(spec["configuration"], schema)
            except ValidationError as error:
                raise ValueError(
                    "Invalid task configuration: " + error.message
                ) from None
        kwargs = dict(
            configuration=deepcopy(spec["configuration"]),
            workspace=Path(workspace),
            source_directory=Path(binding["source_directory"]),
            language_model=language_model,
        )
        # New task adapters may consume the application-level plan and
        # harness.  Keep v1 adapters source-compatible by only passing these
        # optional values when their factory declares the parameters (or a
        # catch-all **kwargs).
        parameters = inspect.signature(factory).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "rsi_plan" in parameters:
            kwargs["rsi_plan"] = deepcopy(rsi_plan)
        if accepts_kwargs or "harness" in parameters:
            kwargs["harness"] = deepcopy(harness)
        self.backend = factory(**kwargs)
        if getattr(self.backend, "protocol_version", None) != 1 or any(
            not callable(getattr(self.backend, method, None))
            for method in ("check", "run", "evolve", "status")
        ):
            raise ValueError("Task adapter must implement v1 check/run/evolve/status")
        self.binding = binding
        self.rsi_plan = deepcopy(rsi_plan)
        self.harness = deepcopy(harness)

    def status(self):
        result = dict(
            name=self.binding["specification"]["name"],
            specification_sha256=self.binding["specification_sha256"],
            **self.backend.status(),
        )
        if self.rsi_plan is not None:
            result["rsi_plan"] = deepcopy(self.rsi_plan)
        if self.harness is not None:
            result["harness"] = deepcopy(self.harness)
        return result
