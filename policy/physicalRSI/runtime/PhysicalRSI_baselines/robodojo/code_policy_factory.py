"""Compatibility implementation for an isolated code-policy skill assembly.

The installed primitive assembler receives channel, configuration, identity and
own(service). It registers each allocated service immediately with own so failed
initialization/reset retains or cleans up all owned resources.
"""

import importlib
import inspect
from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest, read_json, relative_path
from PhysicalRSI_core.self_harness.artifacts import verify_harness

from .isolated_actor import IsolatedActor
from .xpolicy_model import Model


class OwnedServices:
    def __init__(self):
        self.services = []

    def add(self, service):
        if not callable(getattr(service, "close", None)):
            raise ValueError("Owned service requires close")
        self.services.append(service)
        return service

    def close(self):
        errors = []
        for service in list(reversed(self.services)):
            try:
                service.close()
            except BaseException as error:
                errors.append(error)
            else:
                self.services.remove(service)
        if errors:
            raise RuntimeError(
                "Policy service cleanup failed; ownership retained"
            ) from errors[0]


def verify_spec(spec):
    if spec.get("schema") not in {
        "physicalrsi.robodojo.code-policy/v1",
        "physicalrsi.robodojo.code-policy/v2",
    }:
        raise ValueError("Unknown policy assembly specification")
    if spec["schema"] in {
        "physicalrsi.robodojo.code-policy/v2",
    }:
        configurations = spec.get("task_configurations")
        if (
            "configuration" in spec
            or not isinstance(configurations, dict)
            or not configurations
            or any(not isinstance(value, dict) for value in configurations.values())
        ):
            raise ValueError(
                "Code policy v2 requires explicit task configurations without a default"
            )
    elif "task_configurations" in spec:
        raise ValueError("Task configurations require Code policy v2")
    runtime = Path(spec["runtime"]).resolve()
    if file_digest(runtime / "manifest.json") != spec["runtime_manifest_sha256"]:
        raise ValueError(
            "Code policy isolation runtime manifest changed"
        )
    for name, sha in spec["files"].items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or file_digest(path) != sha:
            raise ValueError(
                "Code policy dependency changed: " + name
            )
    module, sep, name = spec["entrypoint"].partition(":")
    if not sep or not name.isidentifier():
        raise ValueError("Installed primitive assembler required")
    factory = getattr(importlib.import_module(module), name)
    source = inspect.getsourcefile(factory)
    if source is None or str(Path(source).resolve()) not in spec["files"]:
        raise ValueError("Primitive assembler source must be pinned")
    return factory


def code_policy_factory(spec, *, pool=None):
    frozen = deepcopy(spec)
    revision = digest(frozen)

    def create(*, harness, binding, output):
        if verify_harness(harness) != binding["harness_sha256"]:
            raise ValueError("Deployment harness identity mismatch")
        root = Path(harness["root"])
        assets = read_json(root / "assets.json")
        pinned = assets.get("policy_spec")
        if pinned != revision:
            raise ValueError(
                "Harness does not pin the Code policy runtime specification"
            )
        assembler = verify_spec(frozen)
        if frozen["schema"] in {
            "physicalrsi.robodojo.code-policy/v2",
        }:
            if binding["task"] not in frozen["task_configurations"]:
                raise ValueError(
                    (
                        "No frozen Code policy configuration for task: "
                    )
                    + binding["task"]
                )
            configuration = frozen["task_configurations"][binding["task"]]
        else:
            configuration = frozen["configuration"]
        requirement_fn = getattr(assembler, "resource_requirements", None)
        requirements = (
            requirement_fn(deepcopy(configuration)) if requirement_fn else None
        )

        def admit_batch(count):
            if requirements:
                if pool is None:
                    raise ValueError("Resource pool required for policy services")
                pool.require_capacity(
                    {key: count * value for key, value in requirements.items()}
                )

        expert = binding["expert"]
        skill = read_json(root / "skills.json")[expert["name"]]
        code_path = relative_path(skill["source"])
        if (
            skill["revision"] != expert["revision"]
            or harness["components"]["skills"].get(str(code_path)) != expert["revision"]
        ):
            raise ValueError("Selected code skill is not the frozen source")
        source = (root / code_path).read_text()
        memory = read_json(root / "memory_rules.json").get(binding["task"], {})

        def actor_factory(*, identity):
            if verify_harness(harness) != binding["harness_sha256"]:
                raise ValueError("Harness changed before actor creation")
            verify_spec(frozen)
            resources = OwnedServices()
            actor_output = Path(output) / identity["episode"] / identity["actor"]

            def primitives(channel):
                return assembler(
                    channel,
                    configuration=deepcopy(configuration),
                    identity=deepcopy(identity),
                    own=resources.add,
                    **({"pool": pool} if pool is not None else {}),
                )

            return IsolatedActor(
                identity=identity,
                source=source,
                memory=memory,
                primitive_revision=revision,
                primitive_factory=primitives,
                dimensions=frozen["dimensions"],
                runtime=frozen["runtime"],
                output=actor_output,
                timeout_s=frozen["timeout_s"],
                cameras=frozen.get("cameras"),
                completion_hold_steps=frozen.get("completion_hold_steps", 0),
                owned_services=[resources],
            )

        return Model(
            actor_factory=actor_factory,
            harness_revision=binding["harness_sha256"],
            max_actors=frozen["max_actors"],
            timeout_s=frozen["action_timeout_s"],
            admit_batch=admit_batch,
        )

    return create
