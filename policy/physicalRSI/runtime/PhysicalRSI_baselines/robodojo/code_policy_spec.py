"""Build a versioned Code policy specification from explicit host configuration."""

from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest

from .code_policy_factory import verify_spec

ASSEMBLER = "PhysicalRSI_baselines.robodojo.primitive_assembly:assemble"


def build(
    *,
    runtime,
    dimensions,
    configuration=None,
    task_configurations=None,
    timeout_s,
    max_actors=10,
    action_timeout_s,
    files=(),
):
    """Create and verify a Code policy v1/v2 spec without inventing calibration.

    The caller must provide either one shared configuration or a complete
    task-to-configuration mapping.  Every host dependency is pinned by digest;
    this function intentionally has no defaults for planner, perception or
    calibration settings.
    """
    if (configuration is None) == (task_configurations is None):
        raise ValueError("Provide exactly one Code policy configuration form")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("Code policy dimensions are required")
    if type(max_actors) is not int or not 1 <= max_actors <= 10:
        raise ValueError("Code policy max_actors must be between one and ten")
    if not isinstance(files, (list, tuple, set)):
        raise ValueError("Code policy dependency files must be a path collection")
    pinned = {}
    for value in files:
        path = Path(value).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Code policy dependency must be a concrete file: " + str(path))
        pinned[str(path)] = file_digest(path)
    if not pinned:
        raise ValueError("Code policy dependency files cannot be empty")
    if configuration is not None:
        schema = "physicalrsi.robodojo.code-policy/v1"
        body = {"configuration": deepcopy(configuration)}
    else:
        if not isinstance(task_configurations, dict) or not task_configurations:
            raise ValueError("Code policy task_configurations cannot be empty")
        schema = "physicalrsi.robodojo.code-policy/v2"
        body = {"task_configurations": deepcopy(task_configurations)}
    spec = dict(
        schema=schema,
        runtime=str(Path(runtime).resolve(strict=True)),
        runtime_manifest_sha256=file_digest(Path(runtime) / "manifest.json"),
        entrypoint=ASSEMBLER,
        files=pinned,
        dimensions=deepcopy(dimensions),
        timeout_s=timeout_s,
        max_actors=max_actors,
        action_timeout_s=action_timeout_s,
        **body,
    )
    verify_spec(spec)
    return spec


def identity(spec):
    """Return the frozen identity used by harness assets and deployment."""
    verify_spec(spec)
    return digest(spec)
